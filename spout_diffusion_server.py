"""
Spout-Based TouchDesigner Diffusion Server
Replaces socket communication with high-performance Spout video streaming + OSC control

Architecture:
- Receives video frames via Spout (60+ FPS)
- Receives control parameters via OSC  
- Processes frames with diffusion pipeline
- Sends processed frames back via Spout

Usage:
    python spout_diffusion_server.py [--model MODEL]
"""

import sys
import os
import time
import array
from itertools import repeat
import numpy as np
import cv2
import torch
from PIL import Image
import argparse
import threading
import queue
import signal
import gc
import random
import hashlib
from pythonosc import dispatcher
from pythonosc import osc_server
import SpoutGL
import SpoutGL.helpers
import OpenGL.GL as GL

# Import existing pipeline components
from pipeline import create_pipeline
from config_loader import get_config

# Global flag for clean shutdown
running = True

def signal_handler(sig, frame):
    global running
    print(f"\nCtrl+C detected - shutting down...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

class SpoutDiffusionServer:
    def __init__(self, model_id, input_sender="PythonOut", output_sender="TouchIn", 
                 osc_port=9998, model_type="sd15", inference_steps=4, guidance_scale=0.0,
                 use_fp16=None, enable_optimizations=True,
                 deterministic=False, seed: int = 42,
                 acceleration: str | None = None,
                 width: int | None = None,
                 height: int | None = None,
                 blend_frames_init: int | None = None,
                 blend_time_init: float | None = None):
        
        self.model_id = model_id
        self.input_sender = input_sender
        self.output_sender = output_sender
        self.osc_port = osc_port
        
        # Initialize Spout objects as None for proper lifecycle tracking
        self.receiver = None
        self.sender = None
        self._spout_initialized = False
        
        # Initialize Spout with improved detection
        self._initialize_spout()
        
        # Frame state - use single buffer like working example
        self.buffer = None  # Single buffer for both receive and send
        self.current_width = 0
        self.current_height = 0
        self.frame_count = 0
        
        # Processing state
        self.current_prompt = ""
        self.current_strength = 0.4
        self.running = True
        
        # Single-threaded processing - eliminate queues
        self.last_frame_data = None
        self.last_frame_width = 0
        self.last_frame_height = 0
        
        # Pre-allocated buffers for memory efficiency
        self.output_rgba_buffer = None
        self.current_buffer_size = (0, 0)
        
        # Performance tracking
        self.last_fps_time = time.time()
        self.last_fps_frame = 0
        
        # Performance settings
        self.use_fp16 = use_fp16
        self.enable_optimizations = enable_optimizations
        self.performance_mode = "balanced"  # Can be: fast, balanced, quality
        
        # Determinism settings
        self.deterministic = bool(deterministic)
        self.seed = int(seed)

        # Content-based caching (same input + params -> same output)
        self._last_signature = None
        self._last_output_bytes = None

        # Temporal blending controls/state
        self.blend_frames = 10            # frames to blend between changes (0 disables)
        self.blend_time_sec = 0.0         # if >0, use time-based blending instead of frames
        self._blend = {
            "active": False,
            "idx": 0,
            "total": 0,
            "start_rgb": None,   # uint8 (H,W,3)
            "target_rgb": None,  # uint8 (H,W,3)
        }
        self._active_signature = None     # signature we are currently blending toward
        self.fps_ema = 0.0                # FPS estimate for time->frames
        self._prev_time = time.time()
        # Override initial blending from CLI if provided
        if isinstance(blend_frames_init, int):
            self.blend_frames = max(0, min(int(blend_frames_init), 600))
        if isinstance(blend_time_init, (int, float)):
            self.blend_time_sec = max(0.0, min(float(blend_time_init), 30.0))
        
        # Apply global determinism environment early if requested
        if self.deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                # Disable TF32 for strict reproducibility
                if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
                    torch.backends.cuda.matmul.allow_tf32 = False
                if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = False
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        
        # Load diffusion pipeline with performance optimizations
        print(f"Loading diffusion model: {model_id}")
        try:
            # Auto-detect optimal precision if not specified
            if use_fp16 is None:
                use_fp16 = torch.cuda.is_available()  # Default to FP16 on CUDA
            
            # Determine acceleration setting (CLI overrides config)
            try:
                if acceleration is not None:
                    acceleration_type = str(acceleration)
                else:
                    config = get_config()
                    acceleration_type = config.get('diffusion', {}).get('acceleration', 'xformers')
                print(f"Using acceleration: {acceleration_type}")
            except Exception:
                acceleration_type = "xformers"  # Fallback
            
            # Force deterministic-friendly acceleration when determinism requested
            if self.deterministic and acceleration_type.lower() != "none":
                print("Deterministic mode: overriding acceleration to 'none' (disabling xformers/flash)")
                acceleration_type = "none"
                
            # Enable compilation for torch_compile acceleration
            use_compile = acceleration_type == "torch_compile"
                
            self.pipe = create_pipeline(
                model_id=model_id,
                model_type=model_type,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                use_fp16=use_fp16,
                acceleration=acceleration_type if enable_optimizations else "none",
                use_channels_last=enable_optimizations,
                compile_unet=use_compile,  # Enable UNet compilation
                compile_vae=False,   # Disable VAE compilation for stability
                width=width if isinstance(width, int) and width > 0 else 512,
                height=height if isinstance(height, int) and height > 0 else 512,
            )
            
            print(f"Diffusion pipeline loaded successfully!")
            print(f"  Acceleration: {acceleration_type}")
            print(f"  Precision: {'FP16' if use_fp16 else 'FP32'}")
            print(f"  Optimizations: {'Enabled' if enable_optimizations else 'Disabled'}")
            if use_compile:
                print(f"  torch.compile: UNet and VAE compilation enabled")
                print(f"  NOTE: First inference will be slower due to compilation (~30s)")
            
        except Exception as e:
            print(f"Failed to load diffusion pipeline: {e}")
            print("Falling back to pass-through mode")
            self.pipe = None
        
        # Put models into eval and configure determinism features if requested
        if self.pipe is not None:
            try:
                self.pipe.unet.eval()
                self.pipe.vae.eval()
                if hasattr(self.pipe, 'text_encoder') and self.pipe.text_encoder is not None:
                    self.pipe.text_encoder.eval()
                if hasattr(self.pipe, 'text_encoder_2') and self.pipe.text_encoder_2 is not None:
                    self.pipe.text_encoder_2.eval()
            except Exception:
                pass
            
            if self.deterministic:
                self._configure_deterministic_attention()
        
        # Setup OSC server
        self.setup_osc_server()
        
        print(f"Spout Diffusion Server Ready:")
        print(f"  Spout Input: '{input_sender}' -> Spout Output: '{output_sender}'")
        print(f"  OSC Control: localhost:{osc_port}")
        print(f"  Model: {model_id} ({'Loaded' if self.pipe else 'Pass-through mode'})")
        print()
        print("TouchDesigner Setup:")
        print(f"  1. Add Spout Out TOP - set Sender Name: '{input_sender}'")
        print(f"  2. Add OSC Out DAT - set Network Address: localhost:{osc_port}")
        print(f"  3. Add Spout In TOP - set Spout Name: '{output_sender}'")
        print(f"  4. Send OSC: /prompt 'your prompt', /strength 0.5, /steps 4")
        print(f"  5. Performance: /precision 16, /performance fast")
    
    def _initialize_spout(self):
        """Initialize Spout objects with improved detection"""
        if self._spout_initialized:
            return True
        
        try:
            print(f"Initializing Spout objects...")
            
            # Check what senders are available
            self._check_available_senders()
            
            # Create receiver
            self.receiver = SpoutGL.SpoutReceiver()
            self.receiver.setReceiverName(self.input_sender)
            print(f"Receiver created for '{self.input_sender}'")
            
            # Create sender
            self.sender = SpoutGL.SpoutSender()
            self.sender.setSenderName(self.output_sender)
            print(f"Sender created for '{self.output_sender}'")
            
            self._spout_initialized = True
            print(f"Spout initialization complete")
            return True
            
        except Exception as e:
            print(f"Failed to initialize Spout: {e}")
            self._cleanup_failed_spout_init()
            return False
    
    def _check_available_senders(self):
        """Check what senders are available using proper detection"""
        try:
            temp_receiver = SpoutGL.SpoutReceiver()
            sender_list = temp_receiver.getSenderList()
            active_sender = temp_receiver.getActiveSender()
            
            print(f"Available senders ({len(sender_list)}): {sender_list}")
            print(f"Active sender: '{active_sender}'")
            
            if self.input_sender not in sender_list:
                print(f"WARNING: Requested sender '{self.input_sender}' not in available list")
                print(f"Consider using one of: {sender_list}")
            else:
                print(f"OK: Requested sender '{self.input_sender}' found in list")
            
            temp_receiver.releaseReceiver()
            
        except Exception as e:
            print(f"Could not check available senders: {e}")
    
    def _cleanup_failed_spout_init(self):
        """Clean up after failed Spout initialization"""
        if self.receiver:
            try:
                self.receiver.releaseReceiver()
            except Exception:
                pass
            self.receiver = None
        
        if self.sender:
            try:
                self.sender.releaseSender()
            except Exception:
                pass
            self.sender = None
        
        self._spout_initialized = False
    
    def setup_osc_server(self):
        """Setup OSC server for parameter control"""
        disp = dispatcher.Dispatcher()
        disp.map("/prompt", self.handle_prompt)
        disp.map("/strength", self.handle_strength)
        disp.map("/steps", self.handle_steps)
        disp.map("/guidance", self.handle_guidance)
        disp.map("/precision", self.handle_precision)
        disp.map("/performance", self.handle_performance_mode)
        # Temporal blending controls
        disp.map("/blendframes", self.handle_blend_frames)
        disp.map("/blendtime", self.handle_blend_time)
        disp.map("/blendreset", self.handle_blend_reset)
        # Determinism and seed controls
        disp.map("/deterministic", self.handle_deterministic)
        disp.map("/seed", self.handle_seed)
        
        # Start OSC server in background thread
        self.osc_server = osc_server.ThreadingOSCUDPServer(("localhost", self.osc_port), disp)
        self.osc_thread = threading.Thread(target=self.osc_server.serve_forever, daemon=True)
        self.osc_thread.start()
        print(f"OSC server listening on port {self.osc_port}")

    # Temporal blending OSC handlers
    def handle_blend_frames(self, unused_addr, frames):
        try:
            n = int(frames)
            self.blend_frames = max(0, min(n, 600))
            print(f"OSC: Blend frames set to {self.blend_frames}")
        except Exception:
            print(f"OSC: Invalid /blendframes '{frames}', expected integer")

    def handle_blend_time(self, unused_addr, seconds):
        try:
            t = float(seconds)
            self.blend_time_sec = max(0.0, min(t, 30.0))
            mode = "time" if self.blend_time_sec > 0 else "frames"
            print(f"OSC: Blend time set to {self.blend_time_sec:.2f}s (mode={mode})")
        except Exception:
            print(f"OSC: Invalid /blendtime '{seconds}', expected float seconds")

    def handle_blend_reset(self, unused_addr, *_):
        self._blend["active"] = False
        self._blend["idx"] = 0
        self._blend["total"] = 0
        self._blend["start_rgb"] = None
        self._blend["target_rgb"] = None
        print("OSC: Blend reset")

    # Determinism OSC handlers
    def handle_deterministic(self, unused_addr, value):
        try:
            v = int(value)
            enable = bool(v)
            if enable == self.deterministic:
                print(f"OSC: Deterministic already {'on' if enable else 'off'}")
                return
            self.deterministic = enable
            if enable:
                print("OSC: Deterministic ON - disabling TF32/xFormers and enforcing deterministic algos")
                try:
                    # Strict reproducibility settings
                    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
                        torch.backends.cuda.matmul.allow_tf32 = False
                    if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'allow_tf32'):
                        torch.backends.cudnn.allow_tf32 = False
                    torch.use_deterministic_algorithms(True)
                except Exception:
                    pass
                # Disable memory-efficient attention
                if self.pipe is not None:
                    self._configure_deterministic_attention()
            else:
                print("OSC: Deterministic OFF - enabling fast kernels (TF32, cudnn.benchmark). xFormers if available.")
                try:
                    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
                        torch.backends.cuda.matmul.allow_tf32 = True
                    if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'allow_tf32'):
                        torch.backends.cudnn.allow_tf32 = True
                    torch.use_deterministic_algorithms(False)
                    torch.backends.cudnn.benchmark = True
                except Exception:
                    pass
                # Try enabling xFormers attention for speed
                try:
                    if self.pipe is not None and hasattr(self.pipe, '_enable_xformers'):
                        self.pipe._enable_xformers()
                except Exception as e:
                    print(f"OSC: Could not enable xFormers: {e}")
        except Exception:
            print(f"OSC: Invalid /deterministic '{value}', expected 0 or 1")

    def handle_seed(self, unused_addr, seed):
        try:
            s = int(seed)
            self.seed = s
            print(f"OSC: Seed set to {self.seed}")
        except Exception:
            print(f"OSC: Invalid /seed '{seed}', expected integer")
    
    def handle_prompt(self, unused_addr, prompt):
        """Handle OSC prompt updates"""
        self.current_prompt = str(prompt)
        print(f"OSC: Prompt updated: '{self.current_prompt[:50]}...'")
    
    def handle_strength(self, unused_addr, strength):
        """Handle OSC strength updates"""
        self.current_strength = float(strength)
        print(f"OSC: Strength updated: {self.current_strength}")
    
    def handle_steps(self, unused_addr, steps):
        """Handle OSC steps updates"""
        new_steps = int(steps)
        if hasattr(self.pipe, 'config'):
            self.pipe.config.num_inference_steps = new_steps
        print(f"OSC: Steps updated: {new_steps}")
    
    def handle_guidance(self, unused_addr, guidance):
        """Handle OSC guidance scale updates"""
        try:
            guidance_val = float(guidance)
            if hasattr(self.pipe, 'config'):
                self.pipe.config.guidance_scale = guidance_val
            print(f"OSC: Guidance scale updated: {guidance_val}")
        except Exception:
            print(f"OSC: Invalid /guidance '{guidance}', expected float")
    
    def handle_precision(self, unused_addr, precision):
        """Handle OSC precision updates (16 or 32)"""
        precision_val = int(precision)
        if precision_val == 16:
            print("OSC: Switching to FP16 precision - restart server to apply")
        elif precision_val == 32:
            print("OSC: Switching to FP32 precision - restart server to apply")
        else:
            print(f"OSC: Invalid precision {precision_val}, use 16 or 32")
    
    def handle_performance_mode(self, unused_addr, mode):
        """Handle OSC performance mode updates"""
        mode_str = str(mode).lower()
        if mode_str in ["fast", "balanced", "quality"]:
            self.performance_mode = mode_str
            print(f"OSC: Performance mode: {mode_str}")
            self._apply_performance_mode()
        else:
            print(f"OSC: Invalid mode '{mode_str}', use: fast, balanced, quality")
    
    def _apply_performance_mode(self):
        """Apply performance mode settings"""
        if not self.pipe:
            return
            
        if self.performance_mode == "fast":
            # Fastest settings - may reduce quality slightly
            if hasattr(self.pipe, 'config'):
                # Prefer 1 step for SD-Turbo, 2 for others
                model_lower = self.model_id.lower()
                if "sd-turbo" in model_lower:
                    self.pipe.config.num_inference_steps = 1
                else:
                    self.pipe.config.num_inference_steps = min(2, self.pipe.config.num_inference_steps)
            # Enable speed-oriented settings only when not in deterministic mode
            if not self.deterministic:
                try:
                    # Prefer faster matmul/cuDNN paths when allowed
                    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
                        torch.backends.cuda.matmul.allow_tf32 = True
                    if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'allow_tf32'):
                        torch.backends.cudnn.allow_tf32 = True
                    torch.use_deterministic_algorithms(False)
                    torch.backends.cudnn.benchmark = True
                except Exception:
                    pass
                # Try enabling xFormers for memory-efficient attention
                try:
                    if hasattr(self.pipe, '_enable_xformers'):
                        self.pipe._enable_xformers()
                except Exception as e:
                    print(f"Could not enable xFormers in FAST mode: {e}")
            else:
                print("FAST mode active, but keeping deterministic constraints (no TF32/xFormers)")
        elif self.performance_mode == "balanced":
            # Default balanced settings
            pass  # Keep current settings
        elif self.performance_mode == "quality":
            # Higher quality settings - slower
            if hasattr(self.pipe, 'config'):
                self.pipe.config.num_inference_steps = max(4, self.pipe.config.num_inference_steps)
        
        print(f"Applied {self.performance_mode} performance mode")

    def _reset_rng(self):
        """Reset RNGs to the configured seed each frame for determinism."""
        if not self.deterministic:
            return
        s = self.seed
        try:
            torch.manual_seed(s)
            torch.cuda.manual_seed_all(s)
        except Exception:
            pass
        np.random.seed(s)
        random.seed(s)

    def _configure_deterministic_attention(self):
        """Disable memory-efficient attention/xformers for deterministic compute."""
        try:
            # Prefer default attention processors (diffusers API)
            if hasattr(self.pipe.unet, 'set_default_attn_processor'):
                self.pipe.unet.set_default_attn_processor()
            else:
                # Fallback for older versions
                from diffusers.models.attention_processor import AttnProcessor
                self.pipe.unet.set_attn_processor(AttnProcessor())
        except Exception:
            pass
    
    def update_buffer(self, width, height):
        """Update buffer when resolution changes - single buffer approach like working example"""
        if width != self.current_width or height != self.current_height:
            print(f"Resolution change: {self.current_width}x{self.current_height} -> {width}x{height}")
            
            self.current_width = width
            self.current_height = height
            
            # Single buffer for both receive and send (like working example)
            buffer_size = width * height * 4  # RGBA
            self.buffer = array.array('B', repeat(0, buffer_size))
            
            print(f"Created buffer: {buffer_size} bytes for {width}x{height} RGBA")
            return True
        return False
    
    def process_frame(self, frame_data, width, height):
        """Process frame with diffusion or pass-through"""
        try:
            if self.pipe is None:
                # Pass-through mode with green dot indicator
                output_data = frame_data.copy()
                
                if width > 20 and height > 20:
                    for y in range(5, 10):
                        for x in range(5, 10):
                            pixel_idx = (y * width + x) * 4
                            if pixel_idx + 3 < len(output_data):
                                output_data[pixel_idx + 0] = 0    # Red
                                output_data[pixel_idx + 1] = 255  # Green  
                                output_data[pixel_idx + 2] = 0    # Blue
                                output_data[pixel_idx + 3] = 255  # Alpha
                
                return output_data
            
            else:
                # Real diffusion processing
                # Reset RNG each frame for deterministic behavior
                self._reset_rng()
                # Convert RGBA buffer to numpy array (view, no copy)
                frame_array = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 4))
                
                # Convert RGBA to RGB (view, no copy until flip)
                frame_rgb = frame_array[:, :, :3]

                # FPS EMA for time-based blending estimation
                now = time.time()
                dt = max(1e-3, now - getattr(self, "_prev_time", now))
                inst_fps = 1.0 / dt
                self.fps_ema = inst_fps if self.fps_ema == 0.0 else (0.9 * self.fps_ema + 0.1 * inst_fps)
                self._prev_time = now

                # Determine total blend frames (time overrides frames)
                if self.blend_time_sec > 0.0:
                    approx_fps = max(1.0, self.fps_ema if self.fps_ema > 0 else 30.0)
                    total_blend_frames = int(round(self.blend_time_sec * approx_fps))
                else:
                    total_blend_frames = int(self.blend_frames)
                total_blend_frames = max(0, min(total_blend_frames, 600))

                # Build a fast content+params signature to detect unchanged input
                try:
                    h = hashlib.blake2b(digest_size=16)
                    # Hash frame content as-is (no copies via memoryview)
                    h.update(memoryview(frame_rgb))
                    # Include key parameters that affect output
                    h.update(self.current_prompt.encode('utf-8'))
                    h.update(str(self.current_strength).encode('utf-8'))
                    steps_val = getattr(self.pipe.config, 'num_inference_steps', 0)
                    h.update(str(steps_val).encode('utf-8'))
                    h.update(str(self.deterministic).encode('utf-8'))
                    h.update(str(self.seed).encode('utf-8'))
                    h.update(self.model_id.encode('utf-8'))
                    signature = (width, height, h.hexdigest())
                except Exception:
                    signature = None

                # If mid-blend toward the same signature, emit next blended frame
                if self._blend["active"] and signature == self._active_signature:
                    start = self._blend["start_rgb"]
                    target = self._blend["target_rgb"]
                    if start is not None and target is not None and start.shape == target.shape == (height, width, 3):
                        i = self._blend["idx"]
                        N = max(1, self._blend["total"]) if self._blend["total"] > 0 else max(1, total_blend_frames)
                        alpha = float(i + 1) / float(N)
                        blended = cv2.addWeighted(target, alpha, start, 1.0 - alpha, 0.0)
                        self._blend["idx"] += 1
                        if self.output_rgba_buffer is None or self.current_buffer_size != (height, width):
                            self.output_rgba_buffer = np.empty((height, width, 4), dtype=np.uint8)
                            self.current_buffer_size = (height, width)
                        self.output_rgba_buffer[:, :, :3] = blended
                        self.output_rgba_buffer[:, :, 3] = 255
                        # Finish blend when done; update cache with final target
                        if self._blend["idx"] >= N:
                            self._blend["active"] = False
                            self._blend["idx"] = 0
                            self._blend["total"] = 0
                            self._blend["start_rgb"] = None
                            self._blend["target_rgb"] = None
                            if self._prev_output_rgb is None or self._prev_output_rgb.shape != target.shape:
                                self._prev_output_rgb = target.copy()
                            else:
                                self._prev_output_rgb[:] = target
                            final_bytes = self.output_rgba_buffer.tobytes()
                            # Cache final result once blending completes
                            self._last_signature = signature
                            self._last_output_bytes = final_bytes
                        return self.output_rgba_buffer.tobytes()
                    else:
                        # Reset invalid blend state
                        self._blend["active"] = False
                        self._blend["idx"] = 0
                        self._blend["total"] = 0
                        self._blend["start_rgb"] = None
                        self._blend["target_rgb"] = None

                # If signature unchanged and we have a cached output, return it
                if signature is not None and signature == self._last_signature and self._last_output_bytes is not None:
                    return self._last_output_bytes
                
                # Flip for TouchDesigner coordinates (first actual copy)
                #frame_rgb_flipped = np.flipud(frame_rgb)
                
                # Convert to PIL Image (uses view when possible)
                pil_frame = Image.fromarray(frame_rgb)
                
                # Apply model-specific parameter mapping (from working server)
                effective_strength, actual_steps = self._map_parameters_for_model(
                    self.current_strength, 
                    self.pipe.config.num_inference_steps
                )
                
                # Update pipeline steps if changed
                if hasattr(self.pipe, 'config') and self.pipe.config.num_inference_steps != actual_steps:
                    self.pipe.config.num_inference_steps = actual_steps
                
                # Process with diffusion pipeline
                for output_image in self.pipe.stream_generate(
                    prompt=self.current_prompt,
                    image=pil_frame,
                    strength=effective_strength,
                    seed=self.seed if self.deterministic else None
                ):
                    # Convert back to numpy array
                    output_array = np.array(output_image, dtype=np.uint8)
                    
                    # Flip back for TouchDesigner (in-place when possible)
                    #output_flipped = np.flipud(output_array)
                    
                    # Reuse pre-allocated RGBA buffer when possible
                    if self.output_rgba_buffer is None or self.current_buffer_size != (height, width):
                        self.output_rgba_buffer = np.empty((height, width, 4), dtype=np.uint8)
                        self.current_buffer_size = (height, width)
                    
                    # Fill buffer efficiently (no allocation)
                    self.output_rgba_buffer[:, :, :3] = output_array  # RGB channels
                    self.output_rgba_buffer[:, :, 3] = 255  # Alpha channel
                    
                    # If new signature and we have previous output and blending enabled, start a blend
                    if signature != self._active_signature and self._prev_output_rgb is not None and total_blend_frames > 0:
                        self._active_signature = signature
                        self._blend["active"] = True
                        self._blend["idx"] = 0
                        self._blend["total"] = total_blend_frames
                        self._blend["start_rgb"] = self._prev_output_rgb.copy()
                        self._blend["target_rgb"] = output_array.copy()
                        # Emit first blended frame
                        N = max(1, self._blend["total"])
                        alpha = 1.0 / float(N)
                        blended = cv2.addWeighted(self._blend["target_rgb"], alpha, self._blend["start_rgb"], 1.0 - alpha, 0.0)
                        self._blend["idx"] += 1
                        if self.output_rgba_buffer is None or self.current_buffer_size != (height, width):
                            self.output_rgba_buffer = np.empty((height, width, 4), dtype=np.uint8)
                            self.current_buffer_size = (height, width)
                        self.output_rgba_buffer[:, :, :3] = blended
                        self.output_rgba_buffer[:, :, 3] = 255
                        return self.output_rgba_buffer.tobytes()

                    # Otherwise, no blending: output target and cache
                    self._active_signature = signature
                    self._prev_output_rgb = output_array.copy()

                    # Return bytes view (final unavoidable copy)
                    output_bytes = self.output_rgba_buffer.tobytes()
                    # Cache result for identical subsequent frames/params
                    self._last_signature = signature
                    self._last_output_bytes = output_bytes
                    return output_bytes
                
                # Fallback if no output generated
                print(f"[WARNING] No output from diffusion pipeline")
                return frame_data
                
        except Exception as e:
            print(f"Processing error: {e}")
            import traceback
            traceback.print_exc()
            return frame_data
    
    def _map_parameters_for_model(self, client_strength, client_steps):
        """Map strength/steps to model-appropriate ranges (from working server)"""
        model_lower = self.model_id.lower()
        
        if "sd-turbo" in model_lower:
            # SD-Turbo: 1-4 steps optimal; allow 1-step for max speed
            recommended_steps = max(1, min(4, client_steps))
            min_strength = 1.0 / recommended_steps
            max_strength = 1.0
        elif "sdxl-turbo" in model_lower:
            # SDXL-Turbo: 1-4 steps, very sensitive to strength
            recommended_steps = max(1, min(4, client_steps))
            min_strength = 1.0 / recommended_steps if recommended_steps > 1 else 0.1
            max_strength = 1.0
        elif "sdxl" in model_lower:
            # SDXL Base: 4-10 steps optimal, higher strength tolerance
            recommended_steps = max(4, min(10, client_steps))
            min_strength = max(0.3, 1.0 / recommended_steps)  # SDXL needs higher minimum
            max_strength = 1.0
        else:
            # SD 1.5 and others: standard mapping
            recommended_steps = max(4, min(8, client_steps))
            min_strength = 1.0 / recommended_steps
            max_strength = 1.0
        
        # Map strength 0-1 to model-appropriate range
        effective_strength = min_strength + client_strength * (max_strength - min_strength)
        
        # Ensure we always get at least 1 timestep
        mapped_timesteps = int(recommended_steps * effective_strength)
        if mapped_timesteps < 1:
            mapped_timesteps = 1
            effective_strength = 1.0 / recommended_steps
        
        return effective_strength, recommended_steps
    
    def calculate_fps(self):
        """Calculate and display performance stats"""
        current_time = time.time()
        time_diff = current_time - self.last_fps_time
        
        if time_diff >= 3.0:  # Update every 3 seconds
            frame_diff = self.frame_count - self.last_fps_frame
            fps = frame_diff / time_diff
            
            print(f"Performance: {fps:.1f} FPS | Frames: {self.frame_count} | Prompt: '{self.current_prompt[:30]}...'")
            
            self.last_fps_time = current_time
            self.last_fps_frame = self.frame_count
    
    def process_single_frame(self):
        """Single-threaded frame processing - receive, process, send in one go"""
        try:
            # Check for frame updates
            if self.receiver and self.receiver.isUpdated():
                new_width = self.receiver.getSenderWidth()
                new_height = self.receiver.getSenderHeight()
                
                if new_width > 0 and new_height > 0:
                    self.update_buffer(new_width, new_height)
            
            # Verify buffer before receive
            if not self.buffer:
                return
                
            expected_size = self.current_width * self.current_height * 4
            if len(self.buffer) < expected_size:
                self.update_buffer(self.current_width, self.current_height)
                return
            
            # Receive frame
            try:
                result = self.receiver.receiveImage(self.buffer, GL.GL_RGBA, False, 0)
                
                if not result:
                    return
                
                # Convert to numpy (view, no copy)
                frame_data = np.frombuffer(self.buffer, dtype=np.uint8)
                
                # Quick validation
                if np.count_nonzero(frame_data) == 0:
                    return
                
                # Process frame immediately (pass view directly, no copy)
                processed_data = self.process_frame(frame_data, self.current_width, self.current_height)
                
                # Send processed frame immediately (reuse buffer when possible)
                if isinstance(processed_data, bytes):
                    send_buffer = processed_data  # No copy needed
                elif hasattr(processed_data, 'data_ptr'):
                    # Use GPU memory directly if available
                    send_buffer = processed_data.tobytes()
                else:
                    send_buffer = bytes(processed_data)
                
                self.sender.sendImage(send_buffer, self.current_width, self.current_height, GL.GL_RGBA, False, 0)
                
                # Update counters
                self.frame_count += 1
                
                # Calculate FPS periodically
                if self.frame_count % 30 == 0:
                    self.calculate_fps()
                    
            except Exception as receive_error:
                if "Buffer not large enough" in str(receive_error):
                    # Force buffer recreation
                    current_width = self.receiver.getSenderWidth()
                    current_height = self.receiver.getSenderHeight()
                    if current_width > 0 and current_height > 0:
                        self.update_buffer(current_width, current_height)
                    else:
                        self.update_buffer(1920, 1080)  # Fallback
                elif self.frame_count % 60 == 1:  # Log occasionally
                    print(f"Receive error: {receive_error}")
                    
        except Exception as e:
            print(f"Frame processing error: {e}")
    
    # Removed - replaced by single-threaded processing
    
    # Removed - replaced by single-threaded processing
    
    def run(self):
        """Start the server with single-threaded processing"""
        print("Starting Spout diffusion server (single-threaded mode)...")
        
        # Force initialize buffer
        self.update_buffer(512, 512)
        
        print("Server running! Press Ctrl+C to stop.")
        
        try:
            while running and self.running:
                self.process_single_frame()
                time.sleep(0.001)  # Minimal delay to prevent 100% CPU
        except KeyboardInterrupt:
            print("\nShutting down server...")
            self.running = False
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up server...")
        
        # Stop OSC server
        if hasattr(self, 'osc_server'):
            self.osc_server.shutdown()
        
        # Release Spout resources
        try:
            self.receiver.releaseReceiver()
            self.sender.releaseSender()
        except:
            pass
        
        print(f"Processed {self.frame_count} total frames")
        print("Server shutdown complete")

def main():
    # Load configuration
    try:
        config = get_config()
        diffusion_config = config.get('diffusion', {})
        server_config = config.get('server', {})
        
        default_model = diffusion_config.get('model_id', 'stabilityai/sd-turbo')
        default_steps = diffusion_config.get('steps', 4)
        default_strength = diffusion_config.get('strength', 0.4)
        
    except Exception as e:
        print(f"Config loading failed: {e}, using defaults")
        default_model = 'stabilityai/sd-turbo'
        default_steps = 4
        default_strength = 0.4
    
    parser = argparse.ArgumentParser(description='Spout-based TouchDesigner Diffusion Server')
    parser.add_argument('--model', default=default_model, help='Diffusion model to use')
    parser.add_argument('--input', default='PythonOut', help='Input Spout sender name')
    parser.add_argument('--output', default='TouchIn', help='Output Spout sender name')
    parser.add_argument('--osc-port', type=int, default=9998, help='OSC control port')
    parser.add_argument('--steps', type=int, default=default_steps, help='Inference steps')
    parser.add_argument('--strength', type=float, default=default_strength, help='Default strength')
    parser.add_argument('--guidance', type=float, help='Guidance scale (0=no guidance, 7.5=typical, 20=max)')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 precision (faster, may reduce quality)')
    parser.add_argument('--fp32', action='store_true', help='Force FP32 precision (slower, higher quality)')
    parser.add_argument('--no-optimizations', action='store_true', help='Disable performance optimizations')
    parser.add_argument('--performance', choices=['fast', 'balanced', 'quality'], default='balanced', 
                        help='Performance mode (fast/balanced/quality)')
    parser.add_argument('--deterministic', action='store_true', help='Enable deterministic output for static inputs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for deterministic mode')
    # Non-live attributes: set at start
    parser.add_argument('--acceleration', choices=['none','xformers','flash_attention','tensorrt','torch_compile','onnx'],
                        help='Attention/engine acceleration (non-live)')
    parser.add_argument('--width', type=int, help='Working width for the diffusion pipeline (non-live)')
    parser.add_argument('--height', type=int, help='Working height for the diffusion pipeline (non-live)')
    parser.add_argument('--blend-frames', type=int, dest='blend_frames', help='Initial blend frames (can still change live)')
    parser.add_argument('--blend-time', type=float, dest='blend_time', help='Initial blend time in seconds (can still change live)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SPOUT DIFFUSION SERVER")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Spout Input: '{args.input}' -> Output: '{args.output}'")
    print(f"OSC Control Port: {args.osc_port}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Determine precision setting
    use_fp16 = None
    if args.fp16:
        use_fp16 = True
        print(f"Precision: FP16 (forced)")
    elif args.fp32:
        use_fp16 = False
        print(f"Precision: FP32 (forced)")
    else:
        use_fp16 = None  # Auto-detect
        print(f"Precision: Auto-detect ({'FP16' if torch.cuda.is_available() else 'FP32'})")
    
    enable_optimizations = not args.no_optimizations
    print(f"Optimizations: {'Enabled' if enable_optimizations else 'Disabled'}")
    print(f"Performance Mode: {args.performance.title()}")
    print(f"Deterministic: {'ON' if args.deterministic else 'OFF'} (seed={args.seed})")
    print()
    
    # Determine model type
    model_type = "sd15" if ("sd-turbo" in args.model.lower() or "stable-diffusion-v1" in args.model.lower()) else "sdxl_turbo" if "sdxl-turbo" in args.model.lower() else "sdxl"
    # Use provided guidance or default based on model type
    if args.guidance is not None:
        guidance_scale = args.guidance
    else:
        guidance_scale = 0.0 if "turbo" in args.model.lower() else 7.5
    
    # Create and run server
    server = SpoutDiffusionServer(
        model_id=args.model,
        input_sender=args.input,
        output_sender=args.output,
        osc_port=args.osc_port,
        model_type=model_type,
        inference_steps=args.steps,
        guidance_scale=guidance_scale,
        use_fp16=use_fp16,
        enable_optimizations=enable_optimizations,
        deterministic=args.deterministic,
        seed=args.seed,
        acceleration=args.acceleration,
        width=args.width,
        height=args.height,
        blend_frames_init=args.blend_frames,
        blend_time_init=args.blend_time,
    )
    
    # Set initial parameters
    server.current_strength = args.strength
    server.current_prompt = "artistic painting, vibrant colors, masterpiece"
    server.performance_mode = args.performance
    server._apply_performance_mode()
    
    server.run()

if __name__ == "__main__":
    main()