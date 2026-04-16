"""Convert user feedback into training data for model improvement.

Supports:
- DPO-style preference pairs (chosen vs rejected suggestions)
- Few-shot example curation from top-rated suggestions
- Incremental training data generation from feedback DB
"""

import argparse
import json
import logging
import sqlite3
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DB = Path(__file__).parent.parent / "feedback.db"


def load_feedback_db(db_path: str = None) -> list[dict]:
    """Load all feedback entries from SQLite."""
    db = Path(db_path) if db_path else DEFAULT_DB
    if not db.exists():
        logger.warning(f"Feedback DB not found: {db}")
        return []

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM feedback ORDER BY created_at DESC"
    ).fetchall()
    conn.close()

    entries = []
    for row in rows:
        entries.append({
            "id": row["id"],
            "session_id": row["session_id"],
            "input_progression": json.loads(row["input_progression"]),
            "genre": row["genre"],
            "suggestion": json.loads(row["suggestion_shown"]),
            "rank": row["suggestion_rank"],
            "action": row["user_action"],
            "rating": row["rating"],
            "created_at": row["created_at"],
        })

    logger.info(f"Loaded {len(entries)} feedback entries from {db}")
    return entries


def feedback_to_preference_pairs(entries: list[dict]) -> list[dict]:
    """Convert feedback into DPO-style (chosen, rejected) preference pairs.

    A preference pair consists of:
    - The same input (chord progression + genre)
    - A chosen response (accepted by user)
    - A rejected response (rejected by user)
    """
    # Group by session + input progression
    sessions: dict[str, list[dict]] = {}
    for entry in entries:
        key = f"{entry['session_id']}_{json.dumps(entry['input_progression'])}"
        sessions.setdefault(key, []).append(entry)

    pairs = []
    for key, session_entries in sessions.items():
        accepted = [e for e in session_entries if e["action"] == "accepted"]
        rejected = [e for e in session_entries if e["action"] == "rejected"]

        for acc in accepted:
            for rej in rejected:
                pairs.append({
                    "input": {
                        "progression": acc["input_progression"],
                        "genre": acc["genre"],
                    },
                    "chosen": acc["suggestion"],
                    "rejected": rej["suggestion"],
                    "chosen_rating": acc.get("rating"),
                    "rejected_rating": rej.get("rating"),
                })

    logger.info(f"Generated {len(pairs)} DPO preference pairs")
    return pairs


def export_dpo_pairs(pairs: list[dict], output_path: str):
    """Export preference pairs as JSONL for fine-tuning."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    logger.info(f"Exported {len(pairs)} DPO pairs to {out}")


def curate_few_shot_examples(
    entries: list[dict],
    max_per_genre: int = 3,
) -> dict[str, list[dict]]:
    """Curate the best few-shot examples from feedback data.

    Selects suggestions that were:
    1. Accepted by users
    2. Highest rated
    3. Most frequently accepted for similar progressions
    """
    # Group by genre
    by_genre: dict[str, list[dict]] = {}
    for entry in entries:
        if entry["action"] == "accepted":
            by_genre.setdefault(entry["genre"], []).append(entry)

    curated = {}
    for genre, genre_entries in by_genre.items():
        # Sort by rating (descending), then by frequency
        sorted_entries = sorted(
            genre_entries,
            key=lambda e: (e.get("rating") or 0, 1),
            reverse=True,
        )

        # Deduplicate by input progression
        seen_inputs = set()
        unique = []
        for entry in sorted_entries:
            input_key = json.dumps(entry["input_progression"])
            if input_key not in seen_inputs:
                seen_inputs.add(input_key)
                unique.append({
                    "input_progression": entry["input_progression"],
                    "suggestion": entry["suggestion"],
                    "rating": entry.get("rating"),
                })

        curated[genre] = unique[:max_per_genre]

    logger.info(f"Curated few-shot examples: {', '.join(f'{g}={len(v)}' for g, v in curated.items())}")
    return curated


def save_few_shot_examples(curated: dict, output_path: str):
    """Save curated examples as a JSON file for prompt injection."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(curated, f, indent=2)
    logger.info(f"Few-shot examples saved to {out}")


def generate_incremental_training_data(
    entries: list[dict],
    output_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate additional training data from accepted chord suggestions.

    When users accept a chord suggestion, we know:
    - The input chroma → the accepted chord label is a positive sample
    - The rejected alternatives are negative signals

    This creates (chroma_vector, chord_label) pairs for incremental training.
    """
    from chord_classifier import label_to_idx

    chromas = []
    labels = []

    for entry in entries:
        if entry["action"] != "accepted":
            continue

        suggestion = entry["suggestion"]
        progression = suggestion.get("progression", [])

        for chord in progression:
            # Generate a synthetic chroma for this chord
            chord_idx = label_to_idx(chord)
            root_idx = chord_idx // 8

            # Create idealized chroma
            chroma = np.zeros(12, dtype=np.float32)
            chroma[root_idx] = 1.0

            # Add noise for variation
            chroma += np.random.randn(12).astype(np.float32) * 0.1
            chroma = np.maximum(chroma, 0)
            total = chroma.sum()
            if total > 0:
                chroma /= total

            chromas.append(chroma)
            labels.append(chord_idx)

    if chromas:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(str(out / "feedback_chroma.npy"), np.array(chromas))
        np.save(str(out / "feedback_labels.npy"), np.array(labels))
        logger.info(f"Generated {len(chromas)} incremental training samples")

    return np.array(chromas), np.array(labels)


def main():
    parser = argparse.ArgumentParser(description="Convert feedback to training data")
    parser.add_argument("--db_path", default=None, help="Path to feedback.db")
    parser.add_argument("--output_dir", default="data", help="Output directory")
    parser.add_argument("--dpo_output", default="data/dpo_pairs.jsonl")
    parser.add_argument("--few_shot_output", default="data/few_shot_examples.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    entries = load_feedback_db(args.db_path)
    if not entries:
        logger.warning("No feedback data found. Run the app and collect feedback first.")
        return

    # DPO pairs
    pairs = feedback_to_preference_pairs(entries)
    if pairs:
        export_dpo_pairs(pairs, args.dpo_output)

    # Few-shot curation
    curated = curate_few_shot_examples(entries)
    save_few_shot_examples(curated, args.few_shot_output)

    # Incremental training data
    generate_incremental_training_data(entries, args.output_dir)

    logger.info("Feedback processing complete!")


if __name__ == "__main__":
    main()
