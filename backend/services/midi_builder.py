"""Piano-specific MIDI file generation with two-hand rendering."""

import logging
from pathlib import Path

import pretty_midi

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Chord → MIDI notes
# ──────────────────────────────────────────────

_ROOTS = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
    "A#": 10, "Bb": 10, "B": 11,
}

_INTERVALS = {
    "":     [0, 4, 7],
    "m":    [0, 3, 7],
    "7":    [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "m7":   [0, 3, 7, 10],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
    "sus4": [0, 5, 7],
    "sus2": [0, 2, 7],
    "9":    [0, 4, 7, 10, 14],
    "m9":   [0, 3, 7, 10, 14],
    "maj9": [0, 4, 7, 11, 14],
    "6":    [0, 4, 7, 9],
    "m6":   [0, 3, 7, 9],
    "7b9":  [0, 4, 7, 10, 13],
    "7#9":  [0, 4, 7, 10, 15],
    "13":   [0, 4, 7, 10, 21],
    "m7b5": [0, 3, 6, 10],
    "dim7": [0, 3, 6, 9],
    "7alt": [0, 4, 6, 10, 13],
    "#11":  [0, 4, 6, 7, 11],
}


def _parse_chord(chord_name: str) -> list[int]:
    """Parse chord name into MIDI pitches for piano voicing."""
    chord_name = chord_name.strip()
    if not chord_name or chord_name == "N":
        return [48, 52, 55]  # C3 E3 G3

    # Find root
    root_note = -1
    root_len = 0
    for name, midi_val in sorted(_ROOTS.items(), key=lambda x: -len(x[0])):
        if chord_name.startswith(name):
            root_note = midi_val
            root_len = len(name)
            break

    if root_note == -1:
        return [48, 52, 55]

    suffix = chord_name[root_len:]
    suffix_map = {
        "min": "m", "minor": "m", "M7": "maj7", "Maj7": "maj7",
        "mi7": "m7", "min7": "m7", "dom7": "7", "alt": "7alt",
        "7alt": "7alt",
    }
    suffix = suffix_map.get(suffix, suffix)
    intervals = _INTERVALS.get(suffix, _INTERVALS[""])

    # Piano voicing: left hand bass (octave 2-3), right hand chord (octave 4)
    base_left = 36 + root_note   # C2 area for bass
    base_right = 60 + root_note  # C4 area for chord tones

    notes = [base_left]  # bass root in left hand
    for iv in intervals[1:]:  # skip root, put extensions in right hand
        notes.append(base_right + iv - intervals[0])

    return notes


def solo_json_to_midi(solo_data: dict, output_path: str) -> str:
    """Convert AI-generated piano solo JSON to MIDI with two-hand tracks."""
    tempo = solo_data.get("tempo", 120)
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    spb = 60.0 / tempo  # seconds per beat

    # Two separate instruments for right and left hand (both Acoustic Grand Piano)
    right_hand = pretty_midi.Instrument(program=0, name="Piano Right Hand")
    left_hand = pretty_midi.Instrument(program=0, name="Piano Left Hand")

    for note_data in solo_data.get("notes", []):
        bar = note_data.get("bar", 1) - 1    # 0-indexed
        beat = note_data.get("beat", 1.0) - 1  # 0-indexed
        start = (bar * 4 + beat) * spb
        end = start + note_data.get("duration", 0.5) * spb

        hand = note_data.get("hand", "right")
        pitch = note_data.get("pitch", 60)
        velocity = note_data.get("velocity", 80)

        # Clamp by hand range
        if hand == "left":
            pitch = max(36, min(60, pitch))   # C2-C4
            velocity = max(1, min(72, velocity))  # softer
        else:
            pitch = max(48, min(96, pitch))   # C3-C7
            velocity = max(1, min(127, velocity))

        pn = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=max(0.0, start),
            end=max(start + 0.04, end),
        )

        if hand == "left":
            left_hand.notes.append(pn)
        else:
            right_hand.notes.append(pn)

    midi.instruments.extend([right_hand, left_hand])
    midi.write(output_path)
    logger.info(
        f"Piano solo MIDI: {output_path} "
        f"(RH={len(right_hand.notes)}, LH={len(left_hand.notes)} notes)"
    )
    return output_path


def chords_to_midi(chord_progression: list[str], tempo: int, output_path: str,
                   bars: int = 8) -> str:
    """Generate piano chord backing track with proper voicings."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    spb = 60.0 / tempo

    left_hand = pretty_midi.Instrument(program=0, name="Piano LH - Bass")
    right_hand = pretty_midi.Instrument(program=0, name="Piano RH - Chords")

    num_chords = len(chord_progression)
    if num_chords == 0:
        midi.write(output_path)
        return output_path

    seconds_per_bar = 4 * spb

    for bar_idx in range(bars):
        chord_name = chord_progression[bar_idx % num_chords]
        notes = _parse_chord(chord_name)

        start = bar_idx * seconds_per_bar
        end = start + seconds_per_bar - 0.05

        # First note = bass (left hand), rest = chord (right hand)
        if notes:
            left_hand.notes.append(pretty_midi.Note(
                velocity=60, pitch=notes[0], start=start, end=end,
            ))
            for pitch in notes[1:]:
                right_hand.notes.append(pretty_midi.Note(
                    velocity=55, pitch=pitch, start=start, end=end,
                ))

    midi.instruments.extend([right_hand, left_hand])
    midi.write(output_path)
    logger.info(f"Piano chord MIDI: {output_path} ({bars} bars)")
    return output_path


def combined_midi(solo_data: dict, chord_progression: list[str],
                  output_path: str) -> str:
    """Create combined MIDI: solo (2 hands) + chord backing (2 hands)."""
    tempo = solo_data.get("tempo", 120)
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    spb = 60.0 / tempo
    seconds_per_bar = 4 * spb

    # Track 1 & 2: Solo right + left hand
    solo_rh = pretty_midi.Instrument(program=0, name="Solo RH")
    solo_lh = pretty_midi.Instrument(program=0, name="Solo LH")

    for note_data in solo_data.get("notes", []):
        bar = note_data.get("bar", 1) - 1
        beat = note_data.get("beat", 1.0) - 1
        start = (bar * 4 + beat) * spb
        end = start + note_data.get("duration", 0.5) * spb
        hand = note_data.get("hand", "right")
        pitch = note_data.get("pitch", 60)

        if hand == "left":
            pitch = max(36, min(60, pitch))
        else:
            pitch = max(48, min(96, pitch))

        pn = pretty_midi.Note(
            velocity=max(1, min(127, note_data.get("velocity", 80))),
            pitch=pitch,
            start=max(0, start),
            end=max(start + 0.04, end),
        )
        if hand == "left":
            solo_lh.notes.append(pn)
        else:
            solo_rh.notes.append(pn)

    midi.instruments.extend([solo_rh, solo_lh])

    # Track 3 & 4: Chord backing (softer)
    chord_rh = pretty_midi.Instrument(program=0, name="Backing RH")
    chord_lh = pretty_midi.Instrument(program=0, name="Backing LH")

    bars = solo_data.get("bars", 8)
    num_chords = len(chord_progression)
    for bar_idx in range(bars):
        chord_name = chord_progression[bar_idx % num_chords] if num_chords > 0 else "C"
        notes = _parse_chord(chord_name)
        start = bar_idx * seconds_per_bar
        end = start + seconds_per_bar - 0.05

        if notes:
            chord_lh.notes.append(pretty_midi.Note(
                velocity=45, pitch=notes[0], start=start, end=end,
            ))
            for pitch in notes[1:]:
                chord_rh.notes.append(pretty_midi.Note(
                    velocity=40, pitch=pitch, start=start, end=end,
                ))

    midi.instruments.extend([chord_rh, chord_lh])
    midi.write(output_path)
    logger.info(f"Combined piano MIDI: {output_path}")
    return output_path
