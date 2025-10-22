# Tensor 

## Overview
Manim scene to visualise 2D vectors stored as PyTorch tensors. Includes utilities to convert tensors to numpy and draw arrows with labels and coordinates.

## Requirements
- Python 3.8 or newer  
- manim (Community edition)  
- torch  
- numpy  
- ffmpeg (installed and reachable from PATH)

## Installation (example)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install -U pip
pip install manim torch numpy
```

## Files
- `vector_manim_scene_no_tex.py` — Manim scene file containing `TensorVectorsScene`
- `README.md` — This file

## Usage (preview, low quality)
```powershell
manim -pql vector_manim_scene_no_tex.py TensorVectorsScene
```

## Usage (high quality)
```powershell
manim -pqh vector_manim_scene_no_tex.py TensorVectorsScene
```

## Notes
- Ensure `FFMPEG_BINARY` and `FFPROBE_BINARY` in `vector_manim_scene_no_tex.py` point to your ffmpeg installation if not on PATH.  
- Activate the virtual environment before running manim to ensure correct dependencies are used.
