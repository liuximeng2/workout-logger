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
        "--num_frames",
        type=int,
        default=16,
        help="Number of frames to sample (default: 16)",
    )
    parser.add_argument(
        "--model_path",
        default="OpenGVLab/InternVL3-8B",
        help="HuggingFace model ID or local path (default: OpenGVLab/InternVL3-8B)",
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
        result = run_pipeline(
            video_path=args.video,
            mode=args.mode,
            num_frames=args.num_frames,
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
