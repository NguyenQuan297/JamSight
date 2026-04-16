"""Feedback collection and adaptive prompt improvement for piano.

Simplified: log feedback + retrieve top examples for few-shot injection.
No DPO — just learn from what pianists accept.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "jamsight_feedback.db"

CREATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    genre TEXT NOT NULL,
    input_chords TEXT NOT NULL,
    suggestion_rank INTEGER NOT NULL,
    suggestion TEXT NOT NULL,
    action TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_fb_genre ON feedback(genre);
CREATE INDEX IF NOT EXISTS idx_fb_action ON feedback(action);
"""


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(CREATE_SCHEMA)
    return conn


def log_feedback(
    session_id: str,
    genre: str,
    input_chords: list[str],
    suggestion_rank: int,
    suggestion: dict,
    action: str,
) -> int:
    """Log user feedback. action = 'accepted' | 'rejected' | 'played'."""
    conn = _get_conn()
    cursor = conn.execute(
        """INSERT INTO feedback
           (session_id, genre, input_chords, suggestion_rank, suggestion, action)
           VALUES (?,?,?,?,?,?)""",
        (session_id, genre, json.dumps(input_chords),
         suggestion_rank, json.dumps(suggestion), action),
    )
    conn.commit()
    fid = cursor.lastrowid
    conn.close()
    logger.info(f"Feedback logged: id={fid}, action={action}, genre={genre}")
    return fid


def get_accepted_for_genre(genre: str, limit: int = 5) -> list[dict]:
    """Get top accepted suggestions for a genre — used as few-shot examples."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            """SELECT input_chords, suggestion, COUNT(*) as n
               FROM feedback
               WHERE genre = ? AND action = 'accepted'
               GROUP BY input_chords, suggestion
               ORDER BY n DESC
               LIMIT ?""",
            (genre, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()

    return [
        {
            "input": json.loads(row["input_chords"]),
            "output": json.loads(row["suggestion"]),
            "accept_count": row["n"],
        }
        for row in rows
    ]


def build_adaptive_prompt(base_prompt: str, genre: str) -> str:
    """Inject top-rated real user examples into prompt."""
    examples = get_accepted_for_genre(genre, limit=3)
    if not examples:
        return base_prompt

    shots = "\n\n".join(
        f"REAL PIANIST PREFERRED (accepted {ex['accept_count']}x):\n"
        f"Input: {' - '.join(ex['input'])}\n"
        f"Output: {json.dumps(ex['output'], indent=2)}"
        for ex in examples
    )

    return (
        f"{base_prompt}\n\n"
        f"The following are piano suggestions that real musicians accepted. "
        f"Learn from these preferences:\n\n{shots}\n\n"
        f"Now generate suggestions for the new input:"
    )


def get_feedback_stats() -> dict:
    """Get feedback analytics."""
    conn = _get_conn()
    try:
        total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        by_action = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT action, COUNT(*) FROM feedback GROUP BY action"
            ).fetchall()
        }
        by_genre = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT genre, COUNT(*) FROM feedback GROUP BY genre"
            ).fetchall()
        }
    except sqlite3.OperationalError:
        return {"total_feedback": 0, "by_action": {}, "by_genre": {}}
    finally:
        conn.close()

    return {
        "total_feedback": total,
        "by_action": by_action,
        "by_genre": by_genre,
    }
