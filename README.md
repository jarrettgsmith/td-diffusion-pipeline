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

### Performance Modes

**Fast mode** (lowest latency, may reduce quality):
```bash
uv run python spout_diffusion_server.py --performance fast
```

**Quality mode** (best results, slower):
```bash
uv run python spout_diffusion_server.py --performance quality
```

**Balanced mode** (default):
```bash
uv run python spout_diffusion_server.py --performance balanced
```

### Advanced Options

```bash
# Force FP16 precision (faster, CUDA only)
uv run python spout_diffusion_server.py --fp16

# Force FP32 precision (higher quality, slower)
uv run python spout_diffusion_server.py --fp32

# Disable all optimizations (for debugging)
uv run python spout_diffusion_server.py --no-optimizations

# Custom model
uv run python spout_diffusion_server.py --model stabilityai/sdxl-turbo

# Set inference steps
uv run python spout_diffusion_server.py --steps 4

# Set default strength
uv run python spout_diffusion_server.py --strength 0.5

# Enable frame blending (smooth transitions)
uv run python spout_diffusion_server.py --blend-frames 10
uv run python spout_diffusion_server.py --blend-time 0.5

# Custom Spout names
uv run python spout_diffusion_server.py --input PythonOut --output TouchIn

# Custom OSC port
uv run python spout_diffusion_server.py --osc-port 9998
```

## OSC Control

The server accepts OSC commands on port 9998 (by default):

### Live Controls
- `/prompt "your text"` - Set generation prompt
- `/strength 0.5` - Set image influence (0.0-1.0)
- `/steps 4` - Set inference steps
- `/performance fast|balanced|quality` - Change performance mode
- `/deterministic 0|1` - Toggle deterministic mode
- `/seed 123` - Set random seed (for deterministic mode)
- `/blendframes 10` - Set frame blending count
- `/blendtime 0.5` - Set time-based blending (seconds)
- `/blendreset` - Reset blending state

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