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

# Set CUBLAS workspace config BEFORE importing torch for deterministic mode
# This must be done before any CUDA operations
if '--deterministic' in sys.argv or any('deterministic' in arg.lower() for arg in sys.argv):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print("Pre-import: Set CUBLAS_WORKSPACE_CONFIG for deterministic mode")

import time
import array
from itertools import repeat
import numpy as np
import torch
import argparse
import threading
import queue
import signal
import gc
import random
from pythonosc import dispatcher
from pythonosc import osc_server
import SpoutGL
import SpoutGL.helpers
import OpenGL.GL as GL

# Import existing pipeline components
from pipeline import create_pipeline
# from config_loader import get_config  # Not used in minimal build

# Global flag for clean shutdown
running = True

def signal_handler(sig, frame):
    global running
    print(f"\nCtrl+C detected - shutting down...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

class SpoutDiffusionServer:
    def __init__(self, input_sender="PythonOut", output_sender="TouchIn", 
                 osc_port=9998, inference_steps=1, 
                 use_fp16=None, enable_optimizations=True,
                 deterministic=False, seed: int = 42,
                 acceleration: str | None = None,
                 width: int | None = None,
                 height: int | None = None,
                 compile_vae: bool = False):
        
        # Hardcoded to SD-Turbo for optimal performance
        self.model_id = "stabilityai/sd-turbo"
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
        # Minimal: no negative prompt handling
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
        
        # Performance profiling
        self.timing_enabled = False  # Set to True for detailed timing
        self.timing_data = {}
        
        # Performance settings (minimal)
        self.use_fp16 = use_fp16
        self.enable_optimizations = enable_optimizations
        
        # Determinism settings
        self.deterministic = bool(deterministic)
        self.seed = int(seed)

        # Blending and signature caching removed for simplicity
        
        # Apply global determinism environment early if requested
        if self.deterministic:
            # CUBLAS_WORKSPACE_CONFIG should already be set before torch import
            cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            if not cublas_config:
                print("WARNING: CUBLAS_WORKSPACE_CONFIG not set before torch import")
                print("WARNING: Deterministic mode may not work properly")
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            else:
                print(f"SUCCESS: CUBLAS_WORKSPACE_CONFIG={cublas_config}")
                
            try:
                # Strict determinism settings
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
                    torch.backends.cuda.matmul.allow_tf32 = False
                if hasattr(torch.backends, 'cudnn'):
                    torch.backends.cudnn.allow_tf32 = False
                try:
                    torch.use_deterministic_algorithms(True)
                except Exception:
                    pass
                print("SUCCESS: Deterministic mode configured (cudnn.deterministic=True, benchmark=False, TF32 off)")
            except Exception as e:
                print(f"WARNING: Deterministic mode setup failed: {e}")
        
        # Load SD-Turbo pipeline with optimal settings
        print(f"Loading SD-Turbo model: {self.model_id}")
        try:
            # Auto-detect optimal precision if not specified
            if use_fp16 is None:
                use_fp16 = torch.cuda.is_available()  # Default to FP16 on CUDA
            
            # Determine acceleration setting (no external config)
            acceleration_type = str(acceleration) if acceleration is not None else "xformers"
            print(f"Using acceleration: {acceleration_type}")
            
            # Allow optimizations in video-deterministic mode for better performance
            if self.deterministic and acceleration_type.lower() in ["torch_compile"]:
                print("Deterministic mode: disabling torch_compile (not video-compatible)")
                acceleration_type = "none"
            # Keep xformers/flash_attention for video determinism - they're reasonably stable
                
            # Enable compilation for torch_compile acceleration
            use_compile = acceleration_type == "torch_compile"
                
            self.pipe = create_pipeline(
                model_id=self.model_id,
                model_type="sd_turbo",
                num_inference_steps=inference_steps,
                guidance_scale=0.0,  # SD-Turbo doesn't use guidance
                use_fp16=use_fp16,
                acceleration=acceleration_type if enable_optimizations else "none",
                use_channels_last=enable_optimizations,
                compile_unet=use_compile,
                compile_vae=bool(compile_vae),
                width=width if isinstance(width, int) and width > 0 else 512,
                height=height if isinstance(height, int) and height > 0 else 512,
            )
            
            print(f"Diffusion pipeline loaded successfully!")
            print(f"  Acceleration: {acceleration_type}")
            print(f"  Precision: {'FP16' if use_fp16 else 'FP32'}")
            print(f"  Optimizations: {'Enabled' if enable_optimizations else 'Disabled'}")
            if use_compile or compile_vae:
                compiled_parts = []
                if use_compile:
                    compiled_parts.append("UNet")
                if compile_vae:
                    compiled_parts.append("VAE")
                print(f"  torch.compile: {' and '.join(compiled_parts)} compilation enabled")
                print(f"  NOTE: First inference may be slower due to compilation")
            
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
            else:
                # Global fast-path settings for non-deterministic mode
                if torch.cuda.is_available():
                    try:
                        if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
                            torch.backends.cuda.matmul.allow_tf32 = True
                        if hasattr(torch.backends, 'cudnn'):
                            torch.backends.cudnn.allow_tf32 = True
                            torch.backends.cudnn.benchmark = True
                        try:
                            torch.set_float32_matmul_precision('high')
                        except Exception:
                            pass
                    except Exception:
                        pass
        
        # Setup OSC server
        self.setup_osc_server()
        
        print(f"Spout Diffusion Server Ready:")
        print(f"  Spout Input: '{input_sender}' -> Spout Output: '{output_sender}'")
        print(f"  OSC Control: localhost:{osc_port}")
        print(f"  Model: SD-Turbo ({'Loaded' if self.pipe else 'Pass-through mode'})")
        print()
        print("TouchDesigner Setup:")
        print(f"  1. Add Spout Out TOP - set Sender Name: '{input_sender}'")
        print(f"  2. Add OSC Out DAT - set Network Address: localhost:{osc_port}")
        print(f"  3. Add Spout In TOP - set Spout Name: '{output_sender}'")
        print(f"  4. Send OSC: /prompt 'your prompt', /strength 0.5, /steps 1")
        print(f"  5. Control: /seed 42")
        print(f"  6. Advanced: --acceleration torch_compile, or lower resolution --width 448 --height 448")
        if self.deterministic:
            print(f"  7. Deterministic mode: ENABLED (seed changes via /seed)")
        else:
            print(f"  7. Deterministic mode: DISABLED (use --deterministic flag to enable)")
    
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
        # Minimal set: prompt, strength, steps, seed
        # Seed control (deterministic mode must be set at startup)
        disp.map("/seed", self.handle_seed)
        
        # Start OSC server in background thread
        self.osc_server = osc_server.ThreadingOSCUDPServer(("localhost", self.osc_port), disp)
        self.osc_thread = threading.Thread(target=self.osc_server.serve_forever, daemon=True)
        self.osc_thread.start()
        print(f"OSC server listening on port {self.osc_port}")

    # Blending handlers removed

    # Seed control handler

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
    
    # Removed: negative prompt, guidance, precision, performance mode handlers for minimal build

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

                # Blending/caching removed: process every frame directly

                # Blending removed

                # OPTIMIZATION: Skip PIL conversion entirely for speed
                # Pass numpy RGB array directly to pipeline
                
                # SD-Turbo parameter mapping - allow OSC step control
                effective_strength, recommended_steps = self._map_parameters_for_model(
                    self.current_strength, 
                    self.pipe.config.num_inference_steps
                )
                
                # Debug logging removed for performance
                
                # Update pipeline steps if changed (OSC control)
                if hasattr(self.pipe, 'config') and self.pipe.config.num_inference_steps != recommended_steps:
                    self.pipe.config.num_inference_steps = recommended_steps
                
                # Process with diffusion pipeline
                for output_image in self.pipe.stream_generate(
                    prompt=self.current_prompt,
                    negative_prompt="",
                    image=frame_rgb,
                    strength=effective_strength,
                    seed=self.seed if self.deterministic else None,
                    return_numpy=True
                ):
                    # Already numpy array (H, W, 3)
                    output_array = output_image
                    
                    # Flip back for TouchDesigner (in-place when possible)
                    #output_flipped = np.flipud(output_array)
                    
                    # Reuse pre-allocated RGBA buffer when possible
                    if self.output_rgba_buffer is None or self.current_buffer_size != (height, width):
                        self.output_rgba_buffer = np.empty((height, width, 4), dtype=np.uint8)
                        # Pre-fill alpha once
                        self.output_rgba_buffer[:, :, 3] = 255
                        self.current_buffer_size = (height, width)
                    
                    # Fill buffer efficiently (no allocation)
                    self.output_rgba_buffer[:, :, :3] = output_array  # RGB channels
                    
                    # Return buffer directly (no blending)
                    return self.output_rgba_buffer
                
                # Fallback if no output generated
                print(f"[WARNING] No output from diffusion pipeline")
                return frame_data
                
        except Exception as e:
            print(f"Processing error: {e}")
            import traceback
            traceback.print_exc()
            return frame_data
    
    def _map_parameters_for_model(self, client_strength, client_steps):
        """SD-Turbo parameter mapping with improved strength handling"""
        # SD-Turbo optimal range: 1-4 steps
        recommended_steps = max(1, min(4, client_steps))
        
        # For SD-Turbo, use client strength directly without modification
        # Let user have full control over strength range 0.0-1.0
        effective_strength = max(0.0, min(1.0, client_strength))
        
        # Remove automatic strength boosting - let user control this via OSC
        
        return effective_strength, recommended_steps
    
    def calculate_fps(self):
        """Calculate and display performance stats"""
        current_time = time.time()
        time_diff = current_time - self.last_fps_time
        
        if time_diff >= 3.0:  # Update every 3 seconds
            frame_diff = self.frame_count - self.last_fps_frame
            fps = frame_diff / time_diff
            
            print(f"Performance: {fps:.1f} FPS | Frames: {self.frame_count} | Steps: {self.pipe.config.num_inference_steps} | Prompt: '{self.current_prompt[:25]}...'")
            
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
                
                # Process frame immediately (pass view directly, no copy)
                processed_data = self.process_frame(frame_data, self.current_width, self.current_height)
                
                # Send processed frame immediately (optimized for numpy arrays)
                if isinstance(processed_data, np.ndarray):
                    if processed_data.flags['C_CONTIGUOUS']:
                        send_buffer = memoryview(processed_data)
                    else:
                        send_buffer = memoryview(np.ascontiguousarray(processed_data))
                elif isinstance(processed_data, (bytes, bytearray, memoryview)):
                    send_buffer = processed_data
                else:
                    send_buffer = bytes(processed_data)
                
                self.sender.sendImage(send_buffer, self.current_width, self.current_height, GL.GL_RGBA, False, 0)
                
                # Update counters
                self.frame_count += 1
                
                # Calculate FPS periodically (reduce frequency for max speed)
                if self.frame_count % 60 == 0:  # Reduced from 30 to 60
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
                # Remove sleep for maximum speed - let GPU processing be the bottleneck
                # time.sleep(0.001)  # Removed: was limiting to ~1000 FPS max
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
    # Minimal defaults
    default_steps = 1
    default_strength = 0.4
    
    parser = argparse.ArgumentParser(description='SD-Turbo TouchDesigner Spout Server (minimal)')
    # Hardcoded to SD-Turbo; minimal CLI
    parser.add_argument('--input', default='PythonOut', help='Input Spout sender name')
    parser.add_argument('--output', default='TouchIn', help='Output Spout sender name')
    parser.add_argument('--osc-port', type=int, default=9998, help='OSC control port')
    parser.add_argument('--steps', type=int, default=default_steps, help='Inference steps')
    parser.add_argument('--strength', type=float, default=default_strength, help='Default strength')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 precision (faster, may reduce quality)')
    parser.add_argument('--fp32', action='store_true', help='Force FP32 precision (slower, higher quality)')
    parser.add_argument('--no-optimizations', action='store_true', help='Disable performance optimizations')
    parser.add_argument('--deterministic', action='store_true', help='Enable deterministic output for static inputs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for deterministic mode')
    # Non-live attributes: set at start
    parser.add_argument('--acceleration', choices=['none','xformers','flash_attention','tensorrt','torch_compile','onnx'],
                        help='Attention/engine acceleration (non-live)')
    parser.add_argument('--width', type=int, help='Working width for the diffusion pipeline (non-live)')
    parser.add_argument('--height', type=int, help='Working height for the diffusion pipeline (non-live)')
    parser.add_argument('--compile-vae', action='store_true', help='Compile VAE decoder (torch.compile)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SD-TURBO SPOUT SERVER (minimal)")
    print("=" * 60)
    print(f"Model: SD-Turbo (stabilityai/sd-turbo)")
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
    print(f"Deterministic: {'ON' if args.deterministic else 'OFF'} (seed={args.seed})")
    print()
    
    # Hardcoded SD-Turbo configuration
    # Create and run SD-Turbo server
    server = SpoutDiffusionServer(
        input_sender=args.input,
        output_sender=args.output,
        osc_port=args.osc_port,
        inference_steps=args.steps,
        use_fp16=use_fp16,
        enable_optimizations=enable_optimizations,
        deterministic=args.deterministic,
        seed=args.seed,
        acceleration=args.acceleration,
        width=args.width,
        height=args.height,
        compile_vae=args.compile_vae,
    )
    
    # Set initial parameters
    server.current_strength = args.strength
    server.current_prompt = "artistic painting, vibrant colors, masterpiece"
    
    server.run()

if __name__ == "__main__":
    main()
