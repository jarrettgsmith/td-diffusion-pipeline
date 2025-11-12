# TouchDesigner Diffusion Pipeline

A clean, modular diffusion pipeline built from scratch with full control over every component. Supports latest models (SDXL, Turbo, Lightning) with optional TensorRT acceleration.

## 🎯 Why This Pipeline?

- **🔧 Full Control** - Every line of code is yours to modify
- **🚀 No Dependency Hell** - Clean separation between core and optional components  
- **⚡ Modern Models** - Built for SDXL, Turbo, Lightning, LCM, and future models
- **🎨 Flexible Acceleration** - TensorRT/ONNX are optional, not required
- **🎭 TouchDesigner Ready** - Async socket server for real-time streaming
- **✅ Actually Works** - Tested and debugged, produces proper images

## Features

✅ **Model Support**
- Stable Diffusion 1.5/2.1
- SDXL and variants
- SDXL Turbo (1-step generation!)
- SDXL Lightning (2-4 steps)
- LCM models
- Easy to add new architectures

✅ **Optimizations** (all optional)
- xFormers memory-efficient attention
- Flash Attention 2
- torch.compile (PyTorch 2.0+)
- TensorRT conversion (when available)
- ONNX export
- FP16/BF16 mixed precision
- Channels-last memory format

✅ **Schedulers**
- LCM (Latent Consistency Models)
- DDIM (Denoising Diffusion Implicit Models)
- DPM++ (Diffusion Probabilistic Models)
- Euler/Euler Ancestral
- Automatic selection based on model

## Quick Start

### 1. Install Dependencies

```bash
# Create conda environment
conda create -n diffusion python=3.10
conda activate diffusion

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt
```

### 2. Test the Pipeline

```python
from pipeline import create_pipeline

# SDXL Turbo - blazing fast 1-step generation
pipe = create_pipeline(
    "stabilityai/sdxl-turbo",
    model_type="sdxl_turbo",
    num_inference_steps=1,
    width=512,
    height=512
)

# Generate image
for image in pipe.stream_generate("a cat wearing a hat"):
    image.save("output.png")
```

### 3. Run Tests

```bash
# Test all components
python test_pipeline.py

# Test specific features
python test_pipeline.py basic    # Basic generation
python test_pipeline.py turbo    # SDXL Turbo speed test
python test_pipeline.py i2i      # Image-to-image
```

### 4. Start Server for TouchDesigner

```bash
python server.py
```

## Configuration

Edit `config.json` to customize:

```json
{
  "host": "localhost",
  "port": 9999,
  "model": {
    "id": "stabilityai/sdxl-turbo",
    "type": "sdxl_turbo",
    "prompt": "highly detailed, 8k",
    "num_inference_steps": 1,
    "guidance_scale": 0.0
  },
  "performance": {
    "width": 512,
    "height": 512,
    "acceleration": "xformers",
    "use_fp16": true,
    "warmup_steps": 3
  }
}
```

## Model Examples

### SDXL Turbo (Fastest - 1 step)
```python
pipe = create_pipeline(
    "stabilityai/sdxl-turbo",
    model_type="sdxl_turbo",
    num_inference_steps=1,
    guidance_scale=0.0  # No CFG needed
)
```

### SDXL Lightning (Fast - 2-4 steps)
```python
pipe = create_pipeline(
    "ByteDance/SDXL-Lightning",
    model_type="sdxl_lightning",
    num_inference_steps=4,
    scheduler_type="dpm"
)
```

### LCM (Fast - 4 steps)
```python
pipe = create_pipeline(
    "SimianLuo/LCM_Dreamshaper_v7",
    model_type="lcm",
    num_inference_steps=4,
    scheduler_type="lcm"
)
```

### Standard SDXL (Quality)
```python
pipe = create_pipeline(
    "stabilityai/stable-diffusion-xl-base-1.0",
    model_type="sdxl",
    num_inference_steps=30,
    guidance_scale=7.5
)
```

## CV Window Examples

For displaying output images in real-time instead of saving to disk:

### Basic CV Window Display
```bash
python examples/cv_window_example.py
```
Displays generated images in OpenCV windows. Press 'q' to quit, any other key for next image.

### Streaming CV Window (Real-time)
```bash
python examples/streaming_cv_window.py
```
Continuously generates new images in real-time with FPS counter. Controls:
- `q`: Quit
- `s`: Save current image
- `p`: Change prompt
- Any other key: Continue

### Movie Stream Simulation
```bash
python examples/movie_stream_simulation.py
```
Simulates processing a movie stream through diffusion pipeline. Shows both input and output streams side-by-side, demonstrating the exact workflow you want for movie processing. Each input frame gets processed and displayed in real-time.

## Performance Tips

### Without TensorRT
- Use `acceleration="xformers"` for 30-50% speedup
- Enable `use_fp16=true` for half memory usage
- Set `use_channels_last=true` for better memory layout
- Use `torch.compile` on PyTorch 2.0+ for 10-20% speedup

### With TensorRT (Optional)
```bash
# Install TensorRT (optional, complex setup)
pip install tensorrt torch-tensorrt

# Enable in config
"acceleration": "tensorrt"
```

First run will take 30-60 minutes to compile, then:
- 2-4x faster inference
- Cached engines for instant startup
- Best for production deployments

## Architecture

```
pipeline.py              # Core diffusion pipeline
├── Model Loading        # Automatic model detection
├── Scheduler Setup      # Flexible scheduler system
├── Optimization Layer   # Optional accelerations
└── Streaming Interface  # Real-time generation

tensorrt_optimizer.py    # Optional TensorRT module
├── TensorRT Converter   # When available
├── ONNX Exporter       # Fallback option
└── Cache Management    # Compiled model storage

server.py               # TouchDesigner server
├── Async Architecture  # Non-blocking I/O
├── Queue Management    # Frame buffering
└── Socket Interface    # Binary protocol
```

## Troubleshooting

### "CUDA out of memory"
- Reduce resolution to 512×512 or 256×256
- Enable `use_fp16=true`
- Reduce `batch_size` to 1

### "Module not found: xformers"
```bash
# xformers is optional but recommended
pip install xformers
# Or disable: acceleration="none"
```

### "TensorRT not available"
- TensorRT is completely optional
- Pipeline works fine with `acceleration="xformers"` or `"none"`
- Only install TensorRT if you need maximum performance

### Slow first run
- First run downloads model weights (several GB)
- Subsequent runs use cached models
- TensorRT compilation (if enabled) takes 30-60 min first time

## Adding New Models

The pipeline automatically detects model architecture. Just specify the model ID:

```python
# Any HuggingFace diffusion model
pipe = create_pipeline(
    "your-username/your-model",
    model_type="sdxl",  # or sd15, sd21, etc.
    # ... other options
)
```

## Contributing

This is designed to be hackable! Key files to modify:

- `pipeline.py` - Core generation logic
- `server.py` - TouchDesigner communication
- `tensorrt_optimizer.py` - Acceleration options

## License

MIT - Use freely in your projects!

## Credits

Built with:
- [Diffusers](https://github.com/huggingface/diffusers) - Model loading
- [PyTorch](https://pytorch.org) - Deep learning framework
- [xFormers](https://github.com/facebookresearch/xformers) - Efficient attention

Inspired by StreamDiffusion but rebuilt for full control and modern models.