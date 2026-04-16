"""Build piano chord classification dataset from MAESTRO MIDI files.

Pipeline: MIDI files → 2-second windows → 36-dim feature vectors → JSON dataset

Features (36-dim):
  [0:12]  Chroma histogram (pitch class distribution)
  [12:24] Velocity-weighted chroma
  [24:30] Piano-specific: hand balance, range, density, sustain, dynamics, richness
  [30:36] Temporal: note intervals, durations, run detection
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DATA = Path(__file__).parent / "data"

# 96 chord classes = 12 roots x 8 types
ROOTS = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
CHORD_TYPES = ["maj", "min", "7", "maj7", "min7", "dim", "aug", "sus4"]
CLASSES = [
    f"{r}" if t == "maj" else f"{r}{t}"
    for r in ROOTS for t in CHORD_TYPES
]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Piano MIDI note range
PIANO_LO = 48  # C3
PIANO_HI = 96  # C7


def midi_to_windows(midi_path: str, window_sec: float = 2.0,
                    hop_sec: float = 0.5) -> list[dict]:
    """Parse one MIDI file into 2-second analysis windows.

    Each window produces a 36-dim feature vector + estimated chord label.
    """
    import pretty_midi

    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        logger.debug(f"Skipping {midi_path}: {e}")
        return []

    # Collect all piano notes (program 0-7 = keyboard instruments)
    notes = []
    for inst in midi.instruments:
        if inst.program < 8 and not inst.is_drum:
            notes.extend(inst.notes)

    if len(notes) < 8:
        return []

    notes.sort(key=lambda n: n.start)
    duration = midi.get_end_time()
    windows = []

    t = 0.0
    while t + window_sec <= duration:
        win_notes = [
            n for n in notes
            if n.start >= t and n.start < t + window_sec
            and PIANO_LO <= n.pitch <= PIANO_HI
        ]

        if len(win_notes) >= 4:
            feat = extract_features(win_notes)
            chord = estimate_chord(win_notes)
            if feat is not None and chord in LABEL_TO_IDX:
                windows.append({
                    "features": feat.tolist(),
                    "label_idx": LABEL_TO_IDX[chord],
                    "label_str": chord,
                    "track_id": Path(midi_path).stem,
                    "t_start": round(t, 2),
                })
        t += hop_sec

    return windows


def extract_features(notes: list) -> np.ndarray | None:
    """Extract 36-dim piano feature vector from a window of MIDI notes."""
    if not notes:
        return None

    pitches = np.array([n.pitch for n in notes])
    vels = np.array([n.velocity for n in notes], dtype=float)
    durs = np.array([n.end - n.start for n in notes])

    # [0:12] Chroma histogram
    chroma = np.zeros(12)
    for p in pitches:
        chroma[p % 12] += 1
    total = chroma.sum()
    if total > 0:
        chroma /= total

    # [12:24] Velocity-weighted chroma
    vchroma = np.zeros(12)
    for n, v in zip(notes, vels):
        vchroma[n.pitch % 12] += v
    vtotal = vchroma.sum()
    if vtotal > 0:
        vchroma /= vtotal

    # [24:30] Piano-specific features
    lh = pitches[pitches < 60]  # left hand proxy: below C4
    rh = pitches[pitches >= 60]  # right hand proxy: C4+

    piano_feats = np.array([
        len(lh) / (len(pitches) + 1e-8),         # left-hand ratio
        len(rh) / (len(pitches) + 1e-8),         # right-hand ratio
        (pitches.max() - pitches.min()) / 48.0,   # range span (normalized)
        (durs > 0.4).mean(),                       # sustained note ratio
        vels.std() / 64.0,                         # dynamic variation
        len(set(pitches % 12)) / 12.0,            # harmonic richness
    ])

    # [30:36] Temporal features
    starts = np.array([n.start for n in notes])
    intervals = np.diff(starts) if len(starts) > 1 else np.array([0.0])

    temporal = np.array([
        np.clip(intervals.mean(), 0, 1),           # avg note interval
        np.clip(intervals.std(), 0, 1),            # interval variation
        (intervals < 0.08).mean(),                  # rapid succession (run detection)
        np.clip(durs.mean(), 0, 2) / 2.0,          # avg note duration
        np.clip(durs.std(), 0, 1),                 # duration variation
        np.clip(len(notes) / 30.0, 0, 1),          # note density
    ])

    feat = np.concatenate([chroma, vchroma, piano_feats, temporal])
    return feat.astype(np.float32)


def estimate_chord(notes: list) -> str:
    """Heuristic chord estimation from MIDI notes."""
    pitch_classes = [n.pitch % 12 for n in notes]
    counts = Counter(pitch_classes)
    root = counts.most_common(1)[0][0]

    has_minor3 = (root + 3) % 12 in counts
    has_major3 = (root + 4) % 12 in counts
    has_dim5 = (root + 6) % 12 in counts
    has_perf5 = (root + 7) % 12 in counts
    has_aug5 = (root + 8) % 12 in counts
    has_minor7 = (root + 10) % 12 in counts
    has_major7 = (root + 11) % 12 in counts

    if has_major7 and has_major3:
        quality = "maj7"
    elif has_minor7 and has_minor3:
        quality = "min7"
    elif has_minor7 and has_major3:
        quality = "7"
    elif has_dim5 and has_minor3:
        quality = "dim"
    elif has_aug5 and has_major3:
        quality = "aug"
    elif has_minor3 and not has_major3:
        quality = "min"
    elif not has_major3 and not has_minor3 and has_perf5:
        quality = "sus4"
    else:
        quality = "maj"

    root_name = ROOTS[root]
    return root_name if quality == "maj" else f"{root_name}{quality}"


def generate_synthetic_supplement(samples_per_chord: int = 200) -> list[dict]:
    """Generate synthetic chroma vectors for underrepresented chords."""
    INTERVALS = {
        "maj": [0, 4, 7], "min": [0, 3, 7],
        "7": [0, 4, 7, 10], "maj7": [0, 4, 7, 11],
        "min7": [0, 3, 7, 10], "dim": [0, 3, 6],
        "aug": [0, 4, 8], "sus4": [0, 5, 7],
    }

    samples = []
    for root_idx in range(12):
        for type_idx, (ctype, ivs) in enumerate(INTERVALS.items()):
            class_idx = root_idx * len(CHORD_TYPES) + type_idx
            label = CLASSES[class_idx]

            for _ in range(samples_per_chord):
                # Base chroma from intervals
                chroma = np.zeros(12, dtype=np.float32)
                for iv in ivs:
                    chroma[(root_idx + iv) % 12] = 1.0
                chroma[root_idx] *= 1.5  # root emphasis

                # Add noise
                noise = np.random.randn(12).astype(np.float32) * np.random.uniform(0.05, 0.25)
                chroma = np.maximum(chroma + noise, 0)
                total = chroma.sum()
                if total > 0:
                    chroma /= total

                # Duplicate as velocity-weighted (slightly different noise)
                vchroma = chroma + np.random.randn(12).astype(np.float32) * 0.05
                vchroma = np.maximum(vchroma, 0)
                vtotal = vchroma.sum()
                if vtotal > 0:
                    vchroma /= vtotal

                # Random piano features
                piano_feats = np.array([
                    np.random.uniform(0.2, 0.5),  # LH ratio
                    np.random.uniform(0.5, 0.8),  # RH ratio
                    np.random.uniform(0.3, 0.8),  # range
                    np.random.uniform(0.2, 0.7),  # sustain
                    np.random.uniform(0.1, 0.5),  # dynamics
                    np.random.uniform(0.3, 0.9),  # richness
                ], dtype=np.float32)

                # Random temporal features
                temporal = np.array([
                    np.random.uniform(0.05, 0.3),
                    np.random.uniform(0.02, 0.2),
                    np.random.uniform(0.0, 0.3),
                    np.random.uniform(0.1, 0.5),
                    np.random.uniform(0.05, 0.3),
                    np.random.uniform(0.3, 0.8),
                ], dtype=np.float32)

                feat = np.concatenate([chroma, vchroma, piano_feats, temporal])
                samples.append({
                    "features": feat.tolist(),
                    "label_idx": class_idx,
                    "label_str": label,
                    "track_id": f"synthetic_{label}",
                    "t_start": 0.0,
                })

    return samples


def build_dataset():
    """Build piano_dataset.json from MAESTRO + optional GiantMIDI + synthetic."""
    all_samples = []

    # MAESTRO
    maestro_dir = DATA / "maestro" / "maestro-v3.0.0"
    if maestro_dir.exists():
        midi_files = list(maestro_dir.rglob("*.midi"))
        print(f"Processing MAESTRO: {len(midi_files)} files...")
        for i, mf in enumerate(midi_files):
            samples = midi_to_windows(str(mf))
            all_samples.extend(samples)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(midi_files)} files, {len(all_samples)} samples so far...")
    else:
        print("MAESTRO not found — run: python download_data.py maestro")

    # GiantMIDI (optional)
    giant_dir = DATA / "giantmidi"
    if giant_dir.exists():
        midi_files = list(giant_dir.glob("*.mid")) + list(giant_dir.glob("*.midi"))
        print(f"Processing GiantMIDI: {len(midi_files)} files...")
        for i, mf in enumerate(midi_files):
            all_samples.extend(midi_to_windows(str(mf)))
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(midi_files)} files...")

    # Check class coverage and add synthetic for missing/rare chords
    class_counts = Counter(s["label_str"] for s in all_samples)
    min_count = 50
    underrepresented = [c for c in CLASSES if class_counts.get(c, 0) < min_count]

    if underrepresented:
        print(f"\n{len(underrepresented)} chord classes underrepresented, adding synthetic data...")
        synthetic = generate_synthetic_supplement(samples_per_chord=200)
        # Only add for underrepresented classes
        synthetic = [s for s in synthetic if s["label_str"] in underrepresented]
        all_samples.extend(synthetic)
        print(f"  Added {len(synthetic)} synthetic samples")

    # Summary
    class_counts = Counter(s["label_str"] for s in all_samples)
    unique_tracks = len(set(s["track_id"] for s in all_samples))

    print(f"\n{'=' * 50}")
    print(f"Dataset built:")
    print(f"  Total samples:  {len(all_samples):,}")
    print(f"  Unique tracks:  {unique_tracks}")
    print(f"  Chord classes:  {len(class_counts)}")
    print(f"\nTop 20 chords:")
    max_count = class_counts.most_common(1)[0][1] if class_counts else 1
    for chord, n in class_counts.most_common(20):
        bar_len = int(n / max_count * 30)
        bar = "█" * bar_len
        print(f"  {chord:8s} {n:6,}  {bar}")

    # Save
    output_path = DATA / "piano_dataset.json"
    DATA.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "instrument": "piano",
            "feature_dim": 36,
            "classes": CLASSES,
            "n_classes": len(CLASSES),
            "n_samples": len(all_samples),
            "samples": all_samples,
        }, f)

    size_mb = output_path.stat().st_size / 1_000_000
    print(f"\nSaved: {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build piano chord dataset from MIDI")
    parser.add_argument("--window", type=float, default=2.0, help="Window size in seconds")
    parser.add_argument("--hop", type=float, default=0.5, help="Hop size in seconds")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    build_dataset()
