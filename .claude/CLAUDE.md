# Workout Logger — Project Guide

## Purpose

CLI tool that takes a workout video, classifies the exercise being performed against an exercise bank, and counts repetitions using a quantized InternVL3-8B vision-language model.

## Tech Stack

- **Python 3.10+** with **uv** for environment management
- **InternVL3-8B** (4-bit NF4 quantized via bitsandbytes) for vision-language inference
- **OpenCV** for video loading
- **PyTorch + Transformers** for model inference

## Project Structure

All source code is in the `video_reps/` package:

- `main.py` — CLI entry point (argparse)
- `pipeline.py` — orchestrates the full inference pipeline
- `video_loader.py` — loads video frames with OpenCV
- `frame_sampler.py` — uniform and motion-based frame sampling
- `preprocess.py` — converts frames to PIL images for the model
- `exercise_bank.py` — loads and formats the exercise bank JSON
- `model_runner.py` — loads quantized InternVL3-8B, handles multi-image inference
- `prompt.py` — builds classification and rep-counting prompts
- `utils.py` — JSON extraction from model output, logging setup

Data files live in `data/`:
- `data/exercise_bank/exercise_bank.json` — exercise definitions
- `data/videos/` — user workout videos (not committed)

## Key Architecture Decisions

- **Two-call inference**: First call classifies the exercise, second call counts reps using the exercise's specific rep definition. This is more debuggable than a single combined call.
- **InternVL3 multi-image input**: Each frame becomes an `<image>` tag in the prompt. The model's `.chat()` method requires `pixel_values` tensor and `num_patches_list` (list of tile counts per image) for correct multi-image handling.
- **Dynamic preprocessing**: Images are tiled using InternVL3's dynamic resolution strategy (in `model_runner.py`), with tiles of 448x448 pixels.

## Running

```bash
uv sync
uv run video-reps --video data/videos/example.mp4 --mode uniform --num_frames 16
```

## Adding Exercises

Edit `data/exercise_bank/exercise_bank.json`. Each exercise needs: `name`, `aliases`, `description`, `rep_definition`.
