"""Load and manage the exercise bank."""

import json
from pathlib import Path


_DEFAULT_BANK_PATH = Path(__file__).resolve().parent.parent / "data" / "exercise_bank" / "exercise_bank.json"


def load_exercise_bank(path: str | None = None) -> dict:
    """Load the exercise bank from a JSON file.

    Args:
        path: Path to exercise_bank.json. Defaults to data/exercise_bank/exercise_bank.json.

    Returns:
        The parsed exercise bank dict with an "exercises" key.
    """
    bank_path = Path(path) if path else _DEFAULT_BANK_PATH
    if not bank_path.exists():
        raise FileNotFoundError(f"Exercise bank not found: {bank_path}")

    with open(bank_path) as f:
        bank = json.load(f)

    # Validate structure
    if "exercises" not in bank:
        raise ValueError("Exercise bank must contain an 'exercises' key")
    for ex in bank["exercises"]:
        for field in ("name", "description", "rep_definition"):
            if field not in ex:
                raise ValueError(f"Exercise missing required field '{field}': {ex}")

    return bank


def format_exercise_bank_for_prompt(bank: dict, top_k: int | None = None) -> str:
    """Format the exercise bank as a numbered list for prompt insertion.

    Args:
        bank: The exercise bank dict.
        top_k: Limit to first top_k exercises. None for all.

    Returns:
        A formatted string listing exercises with descriptions.
    """
    exercises = bank["exercises"]
    if top_k is not None:
        exercises = exercises[:top_k]

    lines = []
    for i, ex in enumerate(exercises, 1):
        aliases = ""
        if ex.get("aliases"):
            aliases = f" (also: {', '.join(ex['aliases'])})"
        lines.append(f"{i}. {ex['name']}{aliases}: {ex['description']}")

    return "\n".join(lines)


def get_exercise(bank: dict, name: str) -> dict | None:
    """Look up an exercise by name (case-insensitive).

    Also checks aliases for a match.

    Returns:
        The matching exercise dict, or None if not found.
    """
    name_lower = name.lower().strip()
    for ex in bank["exercises"]:
        if ex["name"].lower() == name_lower:
            return ex
        for alias in ex.get("aliases", []):
            if alias.lower() == name_lower:
                return ex
    return None
