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

1. Start the diffusion server:
```bash
uv run python spout_diffusion_server.py
```

2. Open `Diffuser.toe` in TouchDesigner

3. The system communicates via:
   - **Spout**: Video streaming between TD and Python
   - **OSC**: Control messages (port 9999)

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