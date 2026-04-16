"""Piano genre context assembly using ChoCo few-shot examples.

Two sources of context for Claude prompts:
1. ChoCo chord_examples.json → real chord progressions as few-shot
2. Static genre descriptions → piano voicing style notes

If ChoCo data is not available, falls back to hardcoded exemplars.
"""

import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

_CHORD_EXAMPLES: list[dict] | None = None
_EXAMPLES_PATH = Path(__file__).parent.parent / "train" / "data" / "chord_examples.json"


def _load_chord_examples() -> list[dict]:
    """Load ChoCo-derived chord progressions (lazy, cached)."""
    global _CHORD_EXAMPLES
    if _CHORD_EXAMPLES is None:
        if _EXAMPLES_PATH.exists():
            with open(_EXAMPLES_PATH) as f:
                _CHORD_EXAMPLES = json.load(f)
            logger.info(f"Loaded {len(_CHORD_EXAMPLES)} chord examples from ChoCo")
        else:
            _CHORD_EXAMPLES = []
            logger.info("ChoCo examples not found — using hardcoded fallback")
    return _CHORD_EXAMPLES


# ──────────────────────────────────────────────
# Genre keyword mapping for ChoCo filtering
# ──────────────────────────────────────────────

GENRE_KEYWORDS = {
    "blues": ["blues", "jazz", "soul", "gospel"],
    "jazz": ["jazz", "bebop", "swing", "weimar", "bossa"],
    "pop": ["billboard", "pop", "rock", "wikifonia", "folk"],
    "rock": ["rock", "billboard", "metal", "punk", "grunge"],
    "funk": ["funk", "soul", "r&b", "disco", "motown"],
    "classical": ["classical", "schubert", "beethoven", "mozart", "chopin", "rwc-classical"],
}

# ──────────────────────────────────────────────
# Static genre context (piano-specific voicings)
# ──────────────────────────────────────────────

GENRE_CONTEXT = {
    "blues": {
        "description": (
            "Blues piano uses dominant 7th chords throughout (even on tonic), "
            "boogie-woogie left hand bass patterns (root-5th-6th-5th), "
            "and blue notes (b3, b5, b7). Call-and-response phrasing. "
            "Think Ray Charles, Otis Spann, Pinetop Perkins."
        ),
        "voicing_tip": "LH: walking bass or boogie pattern. RH: 3rd + b7 + 9th.",
        "techniques": ["boogie-woogie bass", "grace notes", "tremolo", "crushed notes"],
    },
    "jazz": {
        "description": (
            "Jazz piano uses rootless voicings, shell chords (root+7th in LH), "
            "and chromatic approach notes. ii-V-I is the fundamental cadence. "
            "Voice lead by half steps between chords. "
            "Think Bill Evans, Ahmad Jamal, Keith Jarrett, Herbie Hancock."
        ),
        "voicing_tip": "LH: root only or shell (root+7th). RH: rootless (3-7-9-5 or 7-3-5-9).",
        "techniques": ["rootless voicings", "voice leading", "block chords", "quartal harmony"],
    },
    "pop": {
        "description": (
            "Pop piano uses diatonic progressions (I-V-vi-IV), arpeggiated patterns, "
            "sustained chords, and melodic simplicity. Inversions for smooth bass motion. "
            "Think Adele, Coldplay, Sam Smith, Ben Folds."
        ),
        "voicing_tip": "LH: octave bass (C2+C3). RH: arpeggiated triads or add9 chords.",
        "techniques": ["arpeggiation", "octave bass", "add9 voicings", "dynamic build"],
    },
    "rock": {
        "description": (
            "Rock piano uses driving octaves in LH, block chords with staccato accents "
            "on 2 and 4 in RH. Power comes from rhythm, not extensions. "
            "Think Billy Joel, Elton John, Ben Folds Five."
        ),
        "voicing_tip": "LH: driving octaves every beat. RH: triads, staccato accents.",
        "techniques": ["octave bass", "block chords", "staccato", "glissando"],
    },
    "funk": {
        "description": (
            "Funk piano uses 16th-note comping, ghost notes, staccato percussive hits. "
            "Clavinet-style playing. Emphasize 'and' of beats. One-chord vamps common. "
            "Think Stevie Wonder, Herbie Hancock Headhunters, Nile Rodgers."
        ),
        "voicing_tip": "LH: root on downbeat only. RH: staccato 9th/13th chords, mute between.",
        "techniques": ["16th note comping", "ghost notes", "staccato", "syncopation"],
    },
}

# Hardcoded fallback examples (used when ChoCo is not available)
FALLBACK_EXAMPLES = {
    "blues": [
        ["A7", "A7", "D7", "A7"],
        ["Am7", "Dm7", "Am7", "E7b9"],
        ["C7", "F7", "C7", "G7"],
    ],
    "jazz": [
        ["Dm7", "G7", "Cmaj7", "Cmaj7"],
        ["Am7", "D7", "Gmaj7", "Cmaj7"],
        ["Fm7", "Bb7", "Ebmaj7", "Abmaj7"],
    ],
    "pop": [
        ["C", "G", "Am", "F"],
        ["Am", "F", "C", "G"],
        ["G", "Em", "C", "D"],
    ],
    "rock": [
        ["Am", "G", "F", "C"],
        ["E", "B", "C#m", "A"],
        ["A", "G", "D", "A"],
    ],
    "funk": [
        ["E9", "E9", "E9", "E9"],
        ["Am7", "D7", "Am7", "D7"],
        ["Cm7", "F9", "Cm7", "F9"],
    ],
}


def get_few_shot_examples(genre: str, n: int = 3) -> str:
    """Get n chord progressions matching the genre for Claude prompt injection."""
    examples = _load_chord_examples()
    keywords = GENRE_KEYWORDS.get(genre.lower(), [genre.lower()])

    if examples:
        # Filter by genre keywords
        filtered = [
            e for e in examples
            if any(kw in e.get("source", "").lower() or kw in e.get("genre", "").lower()
                   for kw in keywords)
        ]
        if not filtered:
            filtered = examples  # fallback to all

        chosen = random.sample(filtered, min(n, len(filtered)))
        lines = []
        for i, ex in enumerate(chosen, 1):
            prog = " → ".join(ex["progression"])
            src = ex.get("source", "")
            lines.append(f"  {i}. {prog}  (from {src})" if src else f"  {i}. {prog}")
        return "\n".join(lines)
    else:
        # Hardcoded fallback
        fallback = FALLBACK_EXAMPLES.get(genre.lower(), FALLBACK_EXAMPLES["pop"])
        chosen = random.sample(fallback, min(n, len(fallback)))
        return "\n".join(f"  {i}. {' → '.join(prog)}" for i, prog in enumerate(chosen, 1))


def get_exemplar(genre: str) -> str:
    """Build full context string for Claude prompt: few-shot + genre description."""
    genre_lower = genre.lower()
    ctx = GENRE_CONTEXT.get(genre_lower, GENRE_CONTEXT["pop"])
    few_shots = get_few_shot_examples(genre, n=4)

    return f"""Genre: {genre.capitalize()}

Style: {ctx['description']}

Piano voicings: {ctx['voicing_tip']}
Techniques: {', '.join(ctx['techniques'])}

Real chord progressions from {genre} recordings:
{few_shots}"""


def assemble_context(analysis: dict, genre: str) -> str:
    """Assemble full context for AI prompts."""
    exemplar = get_exemplar(genre)
    return (
        f"=== DETECTED PIANO PERFORMANCE ===\n"
        f"Chords: {' - '.join(analysis['chords'])}\n"
        f"Key: {analysis['key']}\n"
        f"Tempo: {analysis['bpm']} BPM\n"
        f"Time Signature: {analysis.get('time_sig', '4/4')}\n"
        f"Duration: {analysis['duration']}s\n"
        f"Instrument: Piano\n\n"
        f"=== GENRE CONTEXT ===\n"
        f"{exemplar}"
    )


def get_available_genres() -> list[dict]:
    """Return available genres."""
    return [
        {"id": genre, "name": genre.capitalize()}
        for genre in GENRE_CONTEXT
    ]
