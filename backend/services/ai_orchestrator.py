"""Claude API integration for piano chord suggestions and solo generation."""

import json
import logging
import anthropic
from typing import Optional

logger = logging.getLogger(__name__)

client: Optional[anthropic.Anthropic] = None


def get_client() -> anthropic.Anthropic:
    global client
    if client is None:
        client = anthropic.Anthropic()
    return client


# ──────────────────────────────────────────────
# Piano-specific system prompts (cached)
# ──────────────────────────────────────────────

CHORD_SYSTEM_PROMPT = """You are a professional pianist and jazz harmony expert.
You analyze piano chord progressions and suggest idiomatic piano reharmonizations —
voicings that are natural to play on piano, with proper voice leading.

Piano-specific rules for chord suggestions:
- Left hand: root position or shell voicings (root + 7th), range C2-B3
- Right hand: 3rd, 5th, 7ths, extensions (9th, 11th, 13th), range C4-B5
- Avoid parallel 5ths in voice leading
- Prefer smooth voice leading — common tones stay, others move by step
- For jazz: rootless voicings in right hand, bass in left hand

Your knowledge spans:
- Jazz piano: Bill Evans voicings, rootless voicings, quartal harmony
- Blues piano: boogie-woogie bass, gospel voicings, dominant 7th extensions
- Pop piano: arpeggiated patterns, pedal tones, slash chords for bass motion
- Classical crossover: Chopin-style voice leading, chromatic mediants

Always respond with valid JSON only. No markdown fences, no explanation outside the JSON."""

SOLO_SYSTEM_PROMPT = """You are a virtuoso pianist composing an 8-bar improvised solo.
You write music idiomatic for piano — exploiting two hands, wide range,
sustain pedal, and keyboard-specific ornaments.

Piano solo rules:
- RIGHT HAND melody: MIDI range 60-96 (C4-C7), monophonic or with 3rds/6ths
- LEFT HAND support: MIDI range 36-60 (C2-C4), chord shells or bass movement
- Velocity: right hand 65-100, left hand 45-72 (always subordinate)
- Techniques: grace notes (duration 0.04s), trills, ascending/descending runs
- Group right-hand notes into 2-4 bar phrases with breathing gaps (0.2-0.5s)
- Left hand rhythm: whole notes, half notes, or Alberti bass — not busy

Always respond with valid JSON only. No markdown fences, no explanation outside the JSON."""


# ──────────────────────────────────────────────
# Piano-focused few-shot examples
# ──────────────────────────────────────────────

FEW_SHOT_CHORDS = """
EXAMPLE 1:
Input: Am - F - C - G, key A minor, genre blues, tempo 80, instrument piano
Output:
{
  "original_progression": ["Am", "F", "C", "G"],
  "key": "A minor",
  "suggestions": [
    {
      "rank": 1,
      "label": "Jazzy shell voicings",
      "difficulty": "beginner",
      "progression": ["Am9", "Fmaj7", "Cmaj7", "G7"],
      "left_hand": ["A2+G3", "F2+E3", "C2+B2", "G2+F3"],
      "right_hand": ["C4+E4+B4", "A3+C4+E4", "E4+G4+B4", "B3+D4+F4"],
      "voice_leading": "Common tone E held between Am9 and Fmaj7. Smooth stepwise motion in right hand.",
      "changes": [{"position": 0, "original": "Am", "replacement": "Am9", "reason": "Adding 9th (B) creates color; shell voicing (A+G) in left hand is standard jazz piano"}],
      "overall_effect": "Sophisticated but accessible — works for jazz and neo-soul piano",
      "genre_note": "Shell voicings (root+7th) in left hand — standard jazz piano comp"
    },
    {
      "rank": 2,
      "label": "Tritone substitution",
      "difficulty": "intermediate",
      "progression": ["Am7", "Fmaj7", "Eb7", "G7"],
      "left_hand": ["A2+G3", "F2+E3", "Eb2+Db3", "G2+F3"],
      "right_hand": ["C4+E4+G4", "A3+C4+E4", "G3+Bb3+Db4", "B3+D4+F4"],
      "voice_leading": "Eb7 replaces A7 — tritone relationship. Chromatic bass descent Eb-D resolves to G.",
      "changes": [{"position": 2, "original": "C", "replacement": "Eb7", "reason": "Tritone sub creates chromatic voice leading, Bb-A resolution is satisfying on piano"}],
      "overall_effect": "Strong chromatic tension with bebop sophistication",
      "genre_note": "Classic bebop piano move — Eb7 has same tritone (G-Db) as A7"
    },
    {
      "rank": 3,
      "label": "Modal reharmonization",
      "difficulty": "advanced",
      "progression": ["Am9", "Bm7b5", "E7alt", "Am"],
      "left_hand": ["A2+G3", "B2+A3", "E2+D3", "A2+E3"],
      "right_hand": ["C4+E4+B4", "D4+F4+A4", "G#3+D4+F4", "C4+E4+A4"],
      "voice_leading": "ii-V-i in A minor. Bm7b5-E7alt creates maximum tension before resolution.",
      "changes": [{"position": 1, "original": "F", "replacement": "Bm7b5", "reason": "Natural ii chord in A minor, sets up V-i cadence. Herbie Hancock style."}],
      "overall_effect": "Full jazz ii-V-i cadence — dramatic tension and release",
      "genre_note": "Herbie Hancock / Bill Evans territory — altered dominant (E7alt) is peak jazz piano"
    }
  ],
  "theory_note": "Your vi-IV-I-V in C major is versatile. The ii-V-i reharmonization transforms it into a proper jazz minor cadence with voice leading that sits perfectly under piano fingers."
}

EXAMPLE 2:
Input: C - G - Am - F, key C major, genre pop, tempo 120, instrument piano
Output:
{
  "original_progression": ["C", "G", "Am", "F"],
  "key": "C major",
  "suggestions": [
    {
      "rank": 1,
      "label": "Smooth inversions",
      "difficulty": "beginner",
      "progression": ["Cmaj7", "G/B", "Am7", "Fmaj9"],
      "left_hand": ["C3+G3", "B2+G3", "A2+G3", "F2+E3"],
      "right_hand": ["E4+G4+B4", "D4+G4+B4", "C4+E4+G4", "A3+C4+E4"],
      "voice_leading": "Descending bass line C-B-A-F. Right hand G4 sustains across first 3 chords.",
      "changes": [{"position": 1, "original": "G", "replacement": "G/B", "reason": "Bass note B creates smooth descending line, natural on piano"}],
      "overall_effect": "Elegant, singer-songwriter feel. Bass motion makes it feel like it flows.",
      "genre_note": "Classic piano ballad technique — inversions for smooth bass"
    },
    {
      "rank": 2,
      "label": "Modal mixture",
      "difficulty": "intermediate",
      "progression": ["Cmaj7", "Eb", "Ab", "Fm7"],
      "left_hand": ["C3+G3", "Eb3+Bb3", "Ab2+Eb3", "F2+Eb3"],
      "right_hand": ["E4+G4+B4", "G4+Bb4+Eb5", "C4+Eb4+Ab4", "Ab3+C4+Eb4"],
      "voice_leading": "Borrowed bIII and bVI from C minor. Dramatic shift from major to minor territory.",
      "changes": [{"position": 1, "original": "G", "replacement": "Eb", "reason": "Borrowed from parallel minor — Radiohead-esque emotional weight"}],
      "overall_effect": "Emotionally heavy, cinematic. The major-to-minor shift is powerful on piano.",
      "genre_note": "Radiohead, Billie Eilish territory — modal mixture on piano creates depth"
    },
    {
      "rank": 3,
      "label": "Neo-soul reharmonization",
      "difficulty": "advanced",
      "progression": ["Cmaj9", "Bm7-E7", "Am9-D13", "Fmaj7#11"],
      "left_hand": ["C2+B2", "B2+A2-E2+D2", "A2+G2-D2+C3", "F2+E3"],
      "right_hand": ["E4+G4+B4+D5", "D4+F#4+A4-G#3+B3+D4", "C4+E4+G4+B4-F#4+A4+C5", "A4+C5+E5+G#5"],
      "voice_leading": "Chain of secondary ii-V resolutions. Each chord flows into the next via half-step motion.",
      "changes": [{"position": 1, "original": "G", "replacement": "Bm7-E7", "reason": "Secondary ii-V targeting Am. D'Angelo-style constant motion."}],
      "overall_effect": "Lush, D'Angelo / Robert Glasper inspired. Constant harmonic motion, very pianistic.",
      "genre_note": "Neo-soul piano — Robert Glasper, Erykah Badu — rootless voicings with extensions"
    }
  ],
  "theory_note": "The I-V-vi-IV axis is pop's backbone. On piano, inversions and extensions transform it. The neo-soul version adds secondary dominants — each resolution chains into the next."
}
"""


def build_chord_prompt(analysis: dict, exemplar: str) -> str:
    """Build piano chord suggestion prompt with chain-of-thought and few-shot."""
    chords_str = " - ".join(analysis["chords"])
    genre = analysis.get("genre", "blues")

    return f"""{FEW_SHOT_CHORDS}

Now analyze this real piano performance:

Detected progression: {chords_str}
Key: {analysis['key']}  |  Tempo: {analysis['bpm']} BPM
Time signature: {analysis.get('time_sig', '4/4')}
Genre: {genre}  |  Instrument: Piano

Genre context from a reference piece:
---
{exemplar[:600]}
---

Before giving suggestions, reason through these steps internally:
1. What is the tonal center and mode of this progression?
2. Which chords are stable (tonic) vs unstable (dominant)?
3. What voice leading connects the chords smoothly on piano?
4. What substitution fits this genre — tritone sub, modal interchange, secondary dominant?

Suggest 3 piano-specific reharmonizations ranked from beginner to advanced.
Include left_hand and right_hand voicing suggestions for each.

Return ONLY the JSON structure as shown in the examples above.
The "suggestions" array must have exactly 3 items with ranks 1, 2, 3.
Difficulties must be: "beginner", "intermediate", "advanced" respectively.
Every suggestion must include: left_hand, right_hand, voice_leading, genre_note fields."""


def build_solo_prompt(analysis: dict, exemplar: str) -> str:
    """Build piano-specific solo generation prompt with two-hand writing."""
    chords_str = " - ".join(analysis["chords"])
    genre = analysis.get("genre", "blues")
    bpm = analysis["bpm"]
    key = analysis["key"]
    spb = 60.0 / bpm

    return f"""Compose an 8-bar piano solo over this progression:

Chords: {chords_str} (repeating to fill 8 bars)
Key: {key}  |  Tempo: {bpm} BPM  |  Genre: {genre}
Time signature: {analysis.get('time_sig', '4/4')}

Beat duration: {spb:.3f} seconds

Piano idiomatic techniques to include:
- Bar 1-2: establish melodic motif in right hand (C4-G5 range), simple left hand (half notes)
- Bar 3-4: develop motif, add passing tones, left hand moves with chord changes
- Bar 5-6: CLIMAX — ascending run (8-10 notes, 0.08s each), reach C6 area
- Bar 7-8: descend back, resolve to tonic, end with sustained chord both hands

Grace notes: pitch = target-1 or target+2, duration = 0.04, beat = beat - 0.04
Alberti bass: left hand pattern root(1)-5th(1.5)-3rd(2)-5th(2.5) per bar

Genre reference:
---
{exemplar[:500]}
---

Return ONLY this JSON:
{{
  "title": "JamSight Piano Solo — {key} {genre}",
  "tempo": {bpm},
  "time_signature": "{analysis.get('time_sig', '4/4')}",
  "instrument": "piano",
  "bars": 8,
  "notes": [
    {{
      "bar": 1, "beat": 1.0,
      "pitch": 64, "duration": 0.5, "velocity": 82,
      "note_name": "E4", "hand": "right",
      "function": "melodic motif start",
      "technique": "normal"
    }},
    {{
      "bar": 1, "beat": 1.0,
      "pitch": 48, "duration": 1.9, "velocity": 58,
      "note_name": "C3", "hand": "left",
      "function": "bass root",
      "technique": "sustain"
    }}
  ],
  "phrase_notes": [
    "Bar 1-2: opening statement — tonic feel, simple motif",
    "Bar 3-4: development — passing tones, rhythmic variation",
    "Bar 5-6: climax — ascending run to high register",
    "Bar 7-8: resolution — descend to tonic, final chord"
  ]
}}

Requirements:
- Generate 30-50 notes total (both hands combined)
- RIGHT HAND: 20-35 melodic notes, MIDI 60-96, velocity 65-100
- LEFT HAND: 10-20 support notes, MIDI 36-60, velocity 45-72
- Every note MUST have: bar, beat, pitch, duration, velocity, note_name, hand, function, technique
- "hand" must be "right" or "left"
- "technique" must be one of: "normal", "sustain", "grace", "staccato", "legato"
- Include at least 2 grace notes (duration 0.04) and 1 run (4+ consecutive notes with duration 0.08-0.12)"""


def _parse_json_response(text: str) -> dict:
    """Parse JSON from Claude's response, handling common formatting issues."""
    cleaned = text.strip()

    # Remove markdown code fences if present
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start:end])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from response: {cleaned[:200]}...")


def get_chord_suggestions(analysis: dict, exemplar: str) -> dict:
    """Get piano chord substitution suggestions from Claude."""
    prompt = build_chord_prompt(analysis, exemplar)
    cl = get_client()

    response = cl.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=3000,
        system=[{
            "type": "text",
            "text": CHORD_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": prompt}],
    )

    result = _parse_json_response(response.content[0].text)
    logger.info(f"Piano chord suggestions generated: {len(result.get('suggestions', []))} options")
    return result


def get_solo(analysis: dict, exemplar: str) -> dict:
    """Generate a piano solo from Claude with two-hand writing."""
    prompt = build_solo_prompt(analysis, exemplar)
    cl = get_client()

    response = cl.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=5000,
        system=[{
            "type": "text",
            "text": SOLO_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": prompt}],
    )

    result = _parse_json_response(response.content[0].text)
    logger.info(f"Piano solo generated: {len(result.get('notes', []))} notes, {result.get('bars', 0)} bars")
    return result
