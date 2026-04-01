# Workout Logger

CLI tool for workout video exercise classification and rep counting using a quantized InternVL3-8B vision-language model.

Given a workout video, the tool:
1. Samples frames (uniform or motion-based)
2. Classifies the exercise against an exercise bank
3. Counts the number of repetitions

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- CUDA-capable GPU recommended (CPU mode available but slow)

## Setup

```bash
# Install dependencies
uv sync

# Or install in editable mode
uv pip install -e .
```

## Usage

```bash
# Basic usage
uv run python -m video_reps.main --video data/videos/pushups.mp4

# With options
uv run python -m video_reps.main \
  --video data/videos/pushups.mp4 \
  --mode motion \
  --num_frames 16 \
  --model_path OpenGVLab/InternVL3-8B \
  --exercise_bank data/exercise_bank/exercise_bank.json \
  --device cuda

# Using the installed CLI entry point
uv run video-reps --video data/videos/pushups.mp4

# JSON output
uv run video-reps --video data/videos/squats.mp4 --json

# Debug mode (shows timings, raw model output)
uv run video-reps --video data/videos/squats.mp4 --debug
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--video` | (required) | Path to the workout video file |
| `--mode` | `uniform` | Frame sampling: `uniform` or `motion` |
| `--num_frames` | `16` | Number of frames to sample |
| `--model_path` | `OpenGVLab/InternVL3-8B` | HuggingFace model ID or local path |
| `--exercise_bank` | `data/exercise_bank/exercise_bank.json` | Path to exercise bank JSON |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--top_k_exercises` | all | Limit exercise bank entries in prompt |
| `--debug` | off | Print timings and raw model output |
| `--json` | off | Output result as JSON |

### Example Output

```
Detected exercise: push-up
Detected repetitions: 12
Confidence: 0.88
```

## Exercise Bank

The exercise bank is a JSON file at `data/exercise_bank/exercise_bank.json`. Each exercise has:

- **name**: canonical name
- **aliases**: alternative names
- **description**: what the exercise looks like
- **rep_definition**: what counts as one complete rep

Add or modify exercises by editing the JSON file. The model uses this bank to classify which exercise is being performed.

## Project Structure

```
video_reps/
├── main.py          # CLI entry point
├── pipeline.py      # Orchestrates the full pipeline
├── video_loader.py  # OpenCV video loading
├── frame_sampler.py # Uniform and motion-based sampling
├── preprocess.py    # Frame preprocessing (resize, PIL conversion)
├── exercise_bank.py # Exercise bank loading and formatting
├── model_runner.py  # Quantized InternVL3-8B inference
├── prompt.py        # Prompt construction
└── utils.py         # JSON extraction, logging
```

## How It Works

1. **Load video** — reads all frames using OpenCV
2. **Sample frames** — selects `num_frames` frames via uniform spacing or motion scoring
3. **Preprocess** — resizes and converts frames to PIL images
4. **Classify exercise** — sends frames + exercise bank to InternVL3-8B, gets best match
5. **Count reps** — sends frames + exercise rep definition to the model, gets rep count
