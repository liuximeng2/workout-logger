"""CLI entry point for the workout video rep counter."""

import argparse
import json
import sys

from .pipeline import run_pipeline
from .utils import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Classify exercises and count reps from workout videos using InternVL3-8B"
    )
    parser.add_argument("--video", required=True, help="Path to the workout video file")
    parser.add_argument(
        "--mode",
        choices=["uniform", "motion"],
        default="uniform",
        help="Frame sampling strategy (default: uniform)",
    )
    parser.add_argument(
        "--sample_fps",
        type=float,
        default=1.0,
        help="Target sampling rate in frames per second from video FPS (default: 4). "
        "Set to 0 to use --num_frames with uniform spacing across the whole clip instead.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="With --sample_fps 0: total frames to sample. With --mode motion and "
        "--sample_fps > 0: max frames after fps thinning (default: 16).",
    )
    parser.add_argument(
        "--model_path",
        default="models/InternVL3-8B",
        help="Local directory or HuggingFace model ID (default: models/InternVL3-8B; "
        "downloads OpenGVLab/InternVL3-8B into that folder on first run)",
    )
    parser.add_argument(
        "--exercise_bank",
        default=None,
        help="Path to exercise_bank.json (default: data/exercise_bank/exercise_bank.json)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for inference: cuda or cpu (default: cuda)",
    )
    parser.add_argument(
        "--top_k_exercises",
        type=int,
        default=None,
        help="Limit exercise bank entries in prompt",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output (timings, raw model responses)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )

    args = parser.parse_args()
    setup_logging(debug=args.debug)

    try:
        sample_fps = None if args.sample_fps == 0 else args.sample_fps
        result = run_pipeline(
            video_path=args.video,
            mode=args.mode,
            num_frames=args.num_frames,
            sample_fps=sample_fps,
            model_path=args.model_path,
            exercise_bank_path=args.exercise_bank,
            device=args.device,
            debug=args.debug,
            top_k_exercises=args.top_k_exercises,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Detected exercise: {result['exercise']}")
        print(f"Detected repetitions: {result['reps']}")
        print(f"Confidence: {result['confidence']:.2f}")


if __name__ == "__main__":
    main()
