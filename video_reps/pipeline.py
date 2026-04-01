"""Pipeline orchestration for exercise classification and rep counting."""

import logging
import time

from .exercise_bank import format_exercise_bank_for_prompt, get_exercise, load_exercise_bank
from .frame_sampler import fps_sample, motion_sample, uniform_sample
from .model_runner import ModelRunner
from .preprocess import preprocess_frames
from .prompt import build_exercise_classification_prompt, build_rep_count_prompt
from .utils import extract_json
from .video_loader import get_video_metadata, load_video

logger = logging.getLogger("video_reps")


def run_pipeline(
    video_path: str,
    mode: str = "uniform",
    num_frames: int = 16,
    sample_fps: float | None = 4.0,
    model_path: str = "models/InternVL3-8B",
    exercise_bank_path: str | None = None,
    device: str = "cuda",
    debug: bool = False,
    top_k_exercises: int | None = None,
) -> dict:
    """Run the full exercise classification and rep counting pipeline.

    Args:
        video_path: Path to the input workout video.
        mode: Frame sampling strategy ('uniform' or 'motion').
        num_frames: When ``sample_fps`` is 0 or None, number of frames to sample (uniform/motion).
        sample_fps: Target frames per second from video FPS; 0 or None uses ``num_frames`` instead.
        model_path: Local directory (default ``models/InternVL3-8B``) or HF model ID.
        exercise_bank_path: Path to exercise_bank.json.
        device: Device for inference ('cuda' or 'cpu').
        debug: Enable debug output.
        top_k_exercises: Limit exercise bank entries in prompt.

    Returns:
        Dict with exercise, confidence, reps, mode, num_frames, and sample_fps when used.
    """
    timings = {}

    # Step 1: Load video
    t0 = time.time()
    metadata = get_video_metadata(video_path)
    frames = load_video(video_path)
    timings["load_video"] = time.time() - t0
    logger.info("Loaded %d frames (%.1fs, %dx%d)", len(frames), metadata["duration_sec"], metadata["width"], metadata["height"])

    # Step 2: Sample frames
    t0 = time.time()
    vfps = float(metadata.get("fps") or 0.0)
    use_fps = sample_fps is not None and sample_fps > 0
    if use_fps:
        thinned = fps_sample(
            frames, vfps, sample_fps, duration_sec=float(metadata.get("duration_sec") or 0) or None
        )
        if mode == "motion":
            sampled = motion_sample(thinned, min(num_frames, len(thinned)))
        else:
            sampled = thinned
    else:
        if mode == "motion":
            sampled = motion_sample(frames, num_frames)
        else:
            sampled = uniform_sample(frames, num_frames)
    timings["sample_frames"] = time.time() - t0
    logger.info(
        "Sampled %d frames using '%s' strategy%s",
        len(sampled),
        mode,
        f" (~{sample_fps} fps)" if use_fps else "",
    )

    # Step 3: Preprocess frames
    t0 = time.time()
    images = preprocess_frames(sampled)
    timings["preprocess"] = time.time() - t0

    # Step 4: Load exercise bank
    bank = load_exercise_bank(exercise_bank_path)
    exercise_text = format_exercise_bank_for_prompt(bank, top_k=top_k_exercises)
    logger.info("Exercise bank loaded with %d exercises", len(bank["exercises"]))

    # Step 5: Load model
    t0 = time.time()
    runner = ModelRunner(model_path, device)
    timings["load_model"] = time.time() - t0

    # Step 6: Classify exercise
    t0 = time.time()
    classify_prompt = build_exercise_classification_prompt(exercise_text, len(images))
    classify_response = runner.infer(images, classify_prompt)
    timings["classify"] = time.time() - t0

    if debug:
        logger.debug("Classification raw output: %s", classify_response)

    classify_result = extract_json(classify_response)
    exercise_name = classify_result.get("exercise", "unknown")
    confidence = classify_result.get("confidence", 0.0)
    logger.info("Detected exercise: %s (confidence: %.2f)", exercise_name, confidence)

    # Step 7: Look up exercise for rep definition
    exercise_entry = get_exercise(bank, exercise_name)
    if exercise_entry is None:
        logger.warning("Exercise '%s' not found in bank, using generic rep counting", exercise_name)
        rep_definition = "One full rep is a complete cycle of the movement from start position back to start position."
    else:
        rep_definition = exercise_entry["rep_definition"]

    # Step 8: Count reps
    t0 = time.time()
    rep_prompt = build_rep_count_prompt(exercise_name, rep_definition, len(images))
    rep_response = runner.infer(images, rep_prompt)
    timings["count_reps"] = time.time() - t0

    if debug:
        logger.debug("Rep count raw output: %s", rep_response)

    rep_result = extract_json(rep_response)
    reps = rep_result.get("reps", 0)
    logger.info("Detected reps: %d", reps)

    if debug:
        logger.debug("Timings: %s", {k: f"{v:.2f}s" for k, v in timings.items()})

    out: dict = {
        "exercise": exercise_name,
        "confidence": confidence,
        "reps": reps,
        "mode": mode,
        "num_frames": len(sampled),
    }
    if use_fps:
        out["sample_fps"] = sample_fps
    return out
