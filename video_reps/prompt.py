"""Prompt builders for exercise classification and rep counting."""


def build_exercise_classification_prompt(
    exercise_bank_text: str, num_images: int
) -> str:
    """Build a prompt for classifying the exercise from sampled frames.

    Args:
        exercise_bank_text: Formatted exercise bank string.
        num_images: Number of frame images being sent.

    Returns:
        The classification prompt with <image> tags.
    """
    image_tags = "\n".join(["<image>"] * num_images)
    return f"""{image_tags}
These are {num_images} sequential frames sampled from a workout video.

Identify which exercise is being performed by choosing the single best match from the exercise bank below.

Exercise bank:
{exercise_bank_text}

Return JSON only:
{{"exercise": "<best exercise name from the bank>", "confidence": <number between 0.0 and 1.0>}}"""


def build_rep_count_prompt(
    exercise_name: str, rep_definition: str, num_images: int
) -> str:
    """Build a prompt for counting reps of a detected exercise.

    Args:
        exercise_name: The detected exercise name.
        rep_definition: What constitutes one complete rep.
        num_images: Number of frame images being sent.

    Returns:
        The rep counting prompt with <image> tags.
    """
    image_tags = "\n".join(["<image>"] * num_images)
    return f"""{image_tags}
These are {num_images} sequential frames sampled from a workout video.

The detected exercise is: {exercise_name}

Rep definition:
{rep_definition}

Count the number of complete repetitions shown across these frames.

Rules:
- Count only full reps
- Do not double count
- Ignore partial reps at the beginning or end unless clearly complete

Return JSON only:
{{"exercise": "{exercise_name}", "reps": <integer>}}"""
