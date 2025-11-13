# TouchDesigner Diffusion Pipeline

Real-time diffusion processing for TouchDesigner using Spout and OSC.

## Requirements

- Windows with NVIDIA GPU
- [TouchDesigner](https://derivative.ca/download)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- NVIDIA driver 525.60+ (for CUDA 12.x support)

## Setup

```bash
git clone git@github.com:jarrettgsmith/td-diffusion-pipeline.git
cd td-diffusion-pipeline
uv sync
```

## Usage

### Basic Usage

1. Start the diffusion server:
```bash
uv run python spout_diffusion_server.py
```

2. Open `Diffuser.toe` in TouchDesigner

3. The system communicates via:
   - **Spout**: Video streaming between TD and Python
   - **OSC**: Control messages (port 9998)

### Deterministic Mode

For 100% reproducible outputs with the same prompt + frame + seed:
```bash
uv run python spout_diffusion_server.py --deterministic --seed 42
```

### Performance

The minimal server targets SD‑Turbo with low latency. For maximum speed, prefer FP16 on CUDA and keep steps low (1–4).

### Advanced Options

```bash
# Force FP16 precision (faster, CUDA only)
uv run python spout_diffusion_server.py --fp16

# Force FP32 precision (higher quality, slower)
uv run python spout_diffusion_server.py --fp32

# Disable all optimizations (for debugging)
uv run python spout_diffusion_server.py --no-optimizations

# Model is fixed to SD‑Turbo in the minimal server

# Set inference steps
uv run python spout_diffusion_server.py --steps 4

# Set default strength
uv run python spout_diffusion_server.py --strength 0.5

# (SD‑Turbo ignores guidance by default)

# Custom Spout names
uv run python spout_diffusion_server.py --input PythonOut --output TouchIn

# Custom OSC port
uv run python spout_diffusion_server.py --osc-port 9998
```

## OSC Control

The server accepts OSC commands on port 9998 (by default):

### Live Controls (minimal)
- `/prompt "your text"` - Set generation prompt
- `/strength 0.5` - Set image influence (0.0–1.0)
- `/steps 1` - Set inference steps (1–4 recommended)
- `/seed 123` - Set random seed (effective in deterministic mode)
  

## Configuration

Edit `config.yaml` to change:
- Model (`stabilityai/sd-turbo` by default)
- Processing parameters (steps, strength, size)
- Performance presets

## Models Tested

- `stabilityai/sd-turbo` - Fast, 4 steps
- `stabilityai/sdxl-turbo` - Higher quality, 4 steps
- Other Hugging Face diffusion models supported

## License

MIT
