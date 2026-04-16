"""Build piano chord dataset from REAL AUDIO (MAPS) + MIDI fallback.

Key difference from v1: features come from audio signal processing, not MIDI.
This closes the domain gap between training data and user video recordings.

Feature vector (36-dim):
  [0:12]  CREPE pitch-based chroma (from neural pitch detection)
  [12:24] librosa CQT chroma (from raw audio spectrogram)
  [24:36] Spectral features: centroid, bandwidth, rolloff, ZCR, 8 MFCCs

Data sources:
  1. MAPS (primary)    — real piano audio + aligned MIDI → ground truth chords
  2. MAESTRO (fallback) — MIDI only → synthesize features (weaker, but available)
  3. Synthetic          — augment rare chord classes
"""

import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_BASE = Path(os.environ.get("JAMSIGHT_TRAIN_DIR", Path(__file__).parent))
DATA = _BASE / "data"

ROOTS = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
CHORD_TYPES = ["maj", "min", "7", "maj7", "min7", "dim", "aug", "sus4"]
CLASSES = [
    f"{r}" if t == "maj" else f"{r}{t}"
    for r in ROOTS for t in CHORD_TYPES
]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
FEATURE_DIM = 36


# ══════════════════════════════════════════════════════════════════
# Source 1: MAPS — real audio + aligned MIDI (solves domain gap)
# ══════════════════════════════════════════════════════════════════

def process_maps(maps_dir: Path) -> list[dict]:
    """Process MAPS dataset: real piano audio → features, MIDI → chord labels."""
    samples = []

    for piano_dir in sorted(maps_dir.iterdir()):
        if not piano_dir.is_dir():
            continue

        wav_files = list(piano_dir.rglob("*.wav"))
        if not wav_files:
            continue

        piano_type = _get_maps_piano_type(piano_dir.name)
        logger.info(f"Processing MAPS/{piano_dir.name} ({piano_type}): {len(wav_files)} files")

        for wav_path in wav_files:
            midi_path = wav_path.with_suffix(".mid")
            if not midi_path.exists():
                midi_path = wav_path.with_suffix(".midi")
            if not midi_path.exists():
                continue

            file_samples = _process_maps_file(str(wav_path), str(midi_path), piano_type)
            samples.extend(file_samples)

        logger.info(f"  → {len(samples)} total samples so far")

    return samples


def _process_maps_file(audio_path: str, midi_path: str, piano_type: str) -> list[dict]:
    """Extract features from real audio, chord labels from aligned MIDI."""
    import librosa
    import pretty_midi

    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception as e:
        logger.debug(f"Skipping {audio_path}: {e}")
        return []

    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return []

    # CREPE pitch detection
    crepe_time, crepe_freq, crepe_conf = _run_crepe(y, sr)

    duration = len(y) / sr
    window_sec = 2.0
    hop_sec = 0.5
    samples = []

    t = 0.0
    while t + window_sec <= duration:
        # Ground truth chord from aligned MIDI
        chord = _midi_window_to_chord(midi, t, t + window_sec)
        if chord not in LABEL_TO_IDX:
            t += hop_sec
            continue

        # Audio features
        feat = _extract_audio_features(
            y, sr, t, window_sec,
            crepe_time, crepe_freq, crepe_conf,
        )
        if feat is not None:
            samples.append({
                "features": feat.tolist(),
                "label_idx": LABEL_TO_IDX[chord],
                "label_str": chord,
                "track_id": Path(audio_path).stem,
                "source": "maps",
                "piano_type": piano_type,
            })
        t += hop_sec

    return samples


def _run_crepe(y: np.ndarray, sr: int) -> tuple:
    """Run CREPE pitch detection, fallback to pyin if unavailable."""
    try:
        import crepe
        time, freq, conf, _ = crepe.predict(y, sr, viterbi=True, step_size=50)
        return time, freq, conf
    except ImportError:
        logger.info("CREPE not available, using librosa pyin")
        import librosa
        f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=2000, sr=sr,
                                      hop_length=int(sr * 0.05))
        time = np.arange(len(f0)) * 0.05
        freq = np.where(np.isnan(f0), 0, f0)
        conf = voiced.astype(float)
        return time, freq, conf


def _extract_audio_features(
    y: np.ndarray, sr: int, t_start: float, window: float,
    crepe_time: np.ndarray, crepe_freq: np.ndarray, crepe_conf: np.ndarray,
) -> np.ndarray | None:
    """Extract 36-dim feature vector from an audio window.

    [0:12]  CREPE-based chroma (neural pitch → pitch class distribution)
    [12:24] librosa CQT chroma (spectrogram-based, complementary)
    [24:36] Spectral: centroid, bandwidth, rolloff, ZCR, 8 MFCCs
    """
    import librosa

    # Slice audio window
    start_sample = int(t_start * sr)
    end_sample = start_sample + int(window * sr)
    y_win = y[start_sample:end_sample]

    if len(y_win) < sr * 0.5:  # too short
        return None

    # 2. librosa CQT chroma (12-dim) — compute first so we can fallback
    chroma_cqt = librosa.feature.chroma_cqt(y=y_win, sr=sr, hop_length=512)
    chroma_mean = chroma_cqt.mean(axis=1)
    chroma_max = chroma_mean.max()
    if chroma_max > 0:
        chroma_mean /= chroma_max

    # 1. CREPE-based chroma (12-dim)
    mask = (crepe_time >= t_start) & (crepe_time < t_start + window) & (crepe_conf > 0.6)
    win_freq = crepe_freq[mask]
    win_conf = crepe_conf[mask]

    crepe_chroma = np.zeros(12)
    if len(win_freq) >= 1:
        for f, c in zip(win_freq, win_conf):
            if f > 0:
                midi_note = int(round(librosa.hz_to_midi(f)))
                crepe_chroma[midi_note % 12] += c
        total = crepe_chroma.sum()
        if total > 0:
            crepe_chroma /= total
        else:
            crepe_chroma = chroma_mean.copy()  # fallback to CQT
    else:
        crepe_chroma = chroma_mean.copy()  # fallback to CQT when no pitch data

    # 3. Spectral features (12-dim)
    centroid = librosa.feature.spectral_centroid(y=y_win, sr=sr).mean() / 8000
    bandwidth = librosa.feature.spectral_bandwidth(y=y_win, sr=sr).mean() / 4000
    rolloff = librosa.feature.spectral_rolloff(y=y_win, sr=sr).mean() / 8000
    zcr = librosa.feature.zero_crossing_rate(y_win).mean()
    mfccs = librosa.feature.mfcc(y=y_win, sr=sr, n_mfcc=8).mean(axis=1)
    # Normalize MFCCs to [0,1] range roughly
    mfccs = np.clip((mfccs + 50) / 100, 0, 1)

    spectral = np.array([centroid, bandwidth, rolloff, zcr, *mfccs])
    spectral = np.clip(spectral, 0, 1).astype(np.float32)

    return np.concatenate([crepe_chroma, chroma_mean, spectral]).astype(np.float32)


def _midi_window_to_chord(midi, t_start: float, t_end: float) -> str:
    """Estimate chord label from MIDI notes in a time window."""
    notes = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if n.start >= t_start and n.start < t_end:
                notes.append(n)

    if len(notes) < 3:
        return "N"

    pitch_classes = [n.pitch % 12 for n in notes]
    counts = Counter(pitch_classes)
    root = counts.most_common(1)[0][0]

    has = {iv: (root + iv) % 12 in counts for iv in [3, 4, 6, 7, 8, 10, 11]}

    if has[11] and has[4]:
        quality = "maj7"
    elif has[10] and has[3]:
        quality = "min7"
    elif has[10] and has[4]:
        quality = "7"
    elif has[6] and has[3]:
        quality = "dim"
    elif has[8] and has[4]:
        quality = "aug"
    elif has[3] and not has[4]:
        quality = "min"
    elif not has[3] and not has[4] and has[7]:
        quality = "sus4"
    else:
        quality = "maj"

    root_name = ROOTS[root]
    return root_name if quality == "maj" else f"{root_name}{quality}"


def _get_maps_piano_type(dir_name: str) -> str:
    """MAPS encodes recording condition in directory name."""
    ambient_suffixes = ("Cl", "Bsdf", "BGCl", "Stgb")
    return "ambient" if dir_name.endswith(ambient_suffixes) else "studio"


# ══════════════════════════════════════════════════════════════════
# Source 2: MAESTRO MIDI fallback (weaker — no real audio)
# ══════════════════════════════════════════════════════════════════

def process_maestro_midi(maestro_dir: Path, max_files: int = 200) -> list[dict]:
    """Fallback: extract features from MIDI (no audio domain gap coverage).

    Uses MIDI note info to synthesize chroma-like features.
    Weaker than MAPS but available immediately.
    """
    import pretty_midi

    samples = []
    midi_files = list(maestro_dir.rglob("*.midi"))[:max_files]
    logger.info(f"Processing MAESTRO MIDI fallback: {len(midi_files)} files")

    for mf in midi_files:
        try:
            midi = pretty_midi.PrettyMIDI(str(mf))
        except Exception:
            continue

        notes = []
        for inst in midi.instruments:
            if inst.program < 8 and not inst.is_drum:
                notes.extend(inst.notes)
        if len(notes) < 10:
            continue

        notes.sort(key=lambda n: n.start)
        duration = midi.get_end_time()

        t = 0.0
        while t + 2.0 <= duration:
            win_notes = [n for n in notes if n.start >= t and n.start < t + 2.0]
            if len(win_notes) >= 4:
                feat = _midi_notes_to_features(win_notes)
                chord = _midi_window_to_chord(midi, t, t + 2.0)
                if feat is not None and chord in LABEL_TO_IDX:
                    samples.append({
                        "features": feat.tolist(),
                        "label_idx": LABEL_TO_IDX[chord],
                        "label_str": chord,
                        "track_id": mf.stem,
                        "source": "maestro_midi",
                        "piano_type": "midi",
                    })
            t += 0.5

    return samples


def _midi_notes_to_features(notes: list) -> np.ndarray | None:
    """Synthesize 36-dim features from MIDI notes (no audio available)."""
    pitches = np.array([n.pitch for n in notes])
    vels = np.array([n.velocity for n in notes], dtype=float)

    # Simulate CREPE chroma from MIDI pitches
    crepe_chroma = np.zeros(12)
    for p in pitches:
        crepe_chroma[p % 12] += 1
    total = crepe_chroma.sum()
    if total > 0:
        crepe_chroma /= total

    # Simulate CQT chroma (add slight noise to differentiate from CREPE)
    cqt_chroma = crepe_chroma.copy()
    cqt_chroma += np.random.randn(12) * 0.03
    cqt_chroma = np.maximum(cqt_chroma, 0)
    cmax = cqt_chroma.max()
    if cmax > 0:
        cqt_chroma /= cmax

    # Approximate spectral features from MIDI (deterministic, not random)
    durs = np.array([n.end - n.start for n in notes])
    mean_pitch = np.mean(pitches)
    pitch_std = np.std(pitches)
    n_pitch_classes = len(set(pitches % 12))
    spectral = np.array([
        mean_pitch / 127,                                # ~ centroid
        pitch_std / 30,                                  # ~ bandwidth
        np.max(pitches) / 127,                           # ~ rolloff
        np.clip(len(notes) / 30, 0, 1),                  # ~ ZCR proxy
        np.min(pitches) / 127,                           # MFCC1 ~ register
        n_pitch_classes / 12,                            # MFCC2 ~ harmonic richness
        np.clip(durs.mean(), 0, 2) / 2,                  # MFCC3 ~ sustain
        vels.std() / 64 if len(vels) > 1 else 0.5,      # MFCC4 ~ dynamics
        np.clip(durs.std(), 0, 1),                       # MFCC5 ~ rhythmic variety
        (pitches > 60).mean(),                           # MFCC6 ~ RH ratio
        (durs > 0.4).mean(),                             # MFCC7 ~ sustained ratio
        vels.mean() / 127,                               # MFCC8 ~ avg loudness
    ])
    spectral = np.clip(spectral, 0, 1).astype(np.float32)

    return np.concatenate([crepe_chroma, cqt_chroma, spectral]).astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# Source 3: Synthetic data for rare chord classes
# ══════════════════════════════════════════════════════════════════

def generate_synthetic(samples_per_chord: int = 100) -> list[dict]:
    """Generate synthetic features for underrepresented chord classes."""
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

            root_norm = root_idx / 11.0

            for sample_i in range(samples_per_chord):
                # CREPE-like chroma
                chroma1 = np.zeros(12, dtype=np.float32)
                for iv in ivs:
                    chroma1[(root_idx + iv) % 12] = 1.0 + np.random.uniform(-0.1, 0.2)
                chroma1[root_idx] *= 1.3
                chroma1 = np.maximum(chroma1 + np.random.randn(12) * 0.08, 0)
                s = chroma1.sum()
                if s > 0:
                    chroma1 /= s

                # CQT-like chroma (slightly different noise)
                chroma2 = chroma1 + np.random.randn(12).astype(np.float32) * 0.05
                chroma2 = np.maximum(chroma2, 0)
                m = chroma2.max()
                if m > 0:
                    chroma2 /= m

                # Deterministic spectral based on chord properties
                spectral = np.array([
                    (root_idx * 4 + 40) / 127,                    # centroid ~ root register
                    len(ivs) / 5,                                  # bandwidth ~ chord size
                    (root_idx * 4 + 60) / 127,                    # rolloff ~ upper register
                    np.random.uniform(0.2, 0.5),                   # ZCR
                    root_norm,                                     # MFCC1 ~ tonal center
                    float(len(ivs)) / 5,                           # MFCC2 ~ complexity
                    1.0 if "maj" in ctype else 0.3,                # MFCC3 ~ brightness
                    0.7 if "7" in ctype else 0.4,                  # MFCC4 ~ dissonance
                    np.random.uniform(0.35, 0.65),                 # MFCC5
                    np.random.uniform(0.35, 0.65),                 # MFCC6
                    np.random.uniform(0.35, 0.65),                 # MFCC7
                    np.random.uniform(0.35, 0.65),                 # MFCC8
                ], dtype=np.float32)
                spectral = np.clip(spectral, 0, 1)

                feat = np.concatenate([chroma1, chroma2, spectral])
                samples.append({
                    "features": feat.tolist(),
                    "label_idx": class_idx,
                    "label_str": label,
                    "track_id": f"synthetic_{label}_{sample_i % 5}",
                    "source": "synthetic",
                    "piano_type": "synthetic",
                })

    return samples


# ══════════════════════════════════════════════════════════════════
# ChoCo → few-shot examples for Claude prompts
# ══════════════════════════════════════════════════════════════════

def process_choco_for_prompts():
    """Extract chord progressions from ChoCo for Claude prompt few-shot examples."""
    choco_path = DATA / "choco" / "piano_choco.json"
    if not choco_path.exists():
        logger.info("ChoCo not found — skipping few-shot extraction")
        return

    with open(choco_path) as f:
        entries = json.load(f)

    examples = []
    for entry in entries:
        chords = entry.get("chords", [])
        if not isinstance(chords, list) or len(chords) < 4:
            continue

        # Extract 4-chord progressions
        chord_labels = []
        for c in chords:
            if isinstance(c, dict) and "label" in c:
                chord_labels.append(c["label"])
            elif isinstance(c, str):
                chord_labels.append(c)

        for i in range(0, len(chord_labels) - 3, 2):
            prog = chord_labels[i:i + 4]
            if len(prog) == 4 and all(prog):
                examples.append({
                    "progression": prog,
                    "source": entry.get("source", ""),
                    "genre": entry.get("genre", ""),
                    "title": entry.get("title", ""),
                })

    # Deduplicate by progression
    seen = set()
    unique = []
    for ex in examples:
        key = tuple(ex["progression"])
        if key not in seen:
            seen.add(key)
            unique.append(ex)

    output = DATA / "chord_examples.json"
    with open(output, "w") as f:
        json.dump(unique[:2000], f, indent=1)

    logger.info(f"ChoCo few-shot examples: {len(unique)} unique → saved {min(len(unique), 2000)}")


# ══════════════════════════════════════════════════════════════════
# Main build pipeline
# ══════════════════════════════════════════════════════════════════

def build_dataset():
    """Build training dataset from all available sources."""
    all_samples = []

    # 1. MAPS — real audio (highest priority)
    maps_dir = DATA / "maps"
    if maps_dir.exists() and any(maps_dir.iterdir()):
        maps_samples = process_maps(maps_dir)
        all_samples.extend(maps_samples)
        logger.info(f"MAPS: {len(maps_samples)} samples")
    else:
        logger.info("MAPS not found — will use MAESTRO MIDI fallback")

    # 2. MAESTRO MIDI fallback (if MAPS insufficient)
    maestro_dir = DATA / "maestro" / "maestro-v3.0.0"
    if maestro_dir.exists():
        # Use more MAESTRO samples if MAPS is missing/small
        max_files = 100 if len(all_samples) > 5000 else 600
        maestro_samples = process_maestro_midi(maestro_dir, max_files=max_files)
        all_samples.extend(maestro_samples)
        logger.info(f"MAESTRO fallback: {len(maestro_samples)} samples")

    # 3. Synthetic supplement for rare chords
    class_counts = Counter(s["label_str"] for s in all_samples)
    min_threshold = max(50, len(all_samples) // 500)
    underrepresented = [c for c in CLASSES if class_counts.get(c, 0) < min_threshold]
    if underrepresented:
        synthetic = generate_synthetic(samples_per_chord=150)
        synthetic = [s for s in synthetic if s["label_str"] in underrepresented]
        all_samples.extend(synthetic)
        logger.info(f"Synthetic supplement: {len(synthetic)} samples for {len(underrepresented)} rare classes")

    # Summary
    class_counts = Counter(s["label_str"] for s in all_samples)
    source_counts = Counter(s["source"] for s in all_samples)
    unique_tracks = len(set(s["track_id"] for s in all_samples))

    print(f"\n{'=' * 55}")
    print(f"Dataset built:")
    print(f"  Total samples:  {len(all_samples):,}")
    print(f"  Unique tracks:  {unique_tracks}")
    print(f"  Chord classes:  {len(class_counts)} / {len(CLASSES)}")
    print(f"  Sources:        {dict(source_counts)}")
    print(f"\nTop 15 chords:")
    max_n = class_counts.most_common(1)[0][1] if class_counts else 1
    for chord, n in class_counts.most_common(15):
        bar = "█" * int(n / max_n * 25)
        print(f"  {chord:8s} {n:6,}  {bar}")

    # Track-level split
    _save_splits(all_samples)

    # 4. ChoCo → few-shot examples (separate file)
    process_choco_for_prompts()

    print(f"\nDone!")


def _save_splits(samples: list[dict]):
    """Split by track ID to prevent data leakage."""
    tracks = list(set(s["track_id"] for s in samples))
    rng = np.random.default_rng(42)
    rng.shuffle(tracks)
    n = len(tracks)

    val_tracks = set(tracks[:int(n * 0.15)])
    test_tracks = set(tracks[int(n * 0.15):int(n * 0.25)])

    splits = {
        "train": [s for s in samples if s["track_id"] not in val_tracks | test_tracks],
        "val": [s for s in samples if s["track_id"] in val_tracks],
        "test": [s for s in samples if s["track_id"] in test_tracks],
    }

    DATA.mkdir(parents=True, exist_ok=True)
    for split_name, split_data in splits.items():
        output = DATA / f"piano_{split_name}.json"
        with open(output, "w") as f:
            json.dump({
                "feature_dim": FEATURE_DIM,
                "n_classes": len(CLASSES),
                "classes": CLASSES,
                "n_samples": len(split_data),
                "samples": split_data,
            }, f)
        size_mb = output.stat().st_size / 1_000_000
        print(f"  {split_name}: {len(split_data):,} samples ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build piano chord dataset")
    parser.add_argument("--maps_only", action="store_true", help="Only process MAPS")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    build_dataset()
