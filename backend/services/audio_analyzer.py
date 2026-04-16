"""Audio analysis for piano performances.

Pipeline: Video → FFmpeg → WAV → librosa + CREPE → chords/key/BPM
Optional: ONNX model for trained chord detection (replaces heuristic).
"""

import subprocess
import logging
from collections import Counter
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Chord profiles: 12-dim chroma templates
CHORD_TEMPLATES = {
    "maj":  [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    "min":  [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    "maj7": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    "min7": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    "7":    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    "dim":  [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    "sus4": [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    "aug":  [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
}

ROOT_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

SUFFIX_DISPLAY = {
    "maj": "", "min": "m", "maj7": "maj7", "min7": "m7",
    "7": "7", "dim": "dim", "sus4": "sus4", "aug": "aug",
}

# Key profiles (Krumhansl-Kessler)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# ──────────────────────────────────────────────
# ONNX Chord Predictor (loads trained piano model)
# ──────────────────────────────────────────────

class PianoChordPredictor:
    """Singleton ONNX chord predictor. Loads once, predicts many times."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self):
        """Try to load the ONNX model. Gracefully skip if not available."""
        if self._loaded:
            return True

        model_path = Path(__file__).parent.parent / "train" / "models" / "piano_model.onnx"
        classes_path = Path(__file__).parent.parent / "train" / "data" / "piano_dataset.json"

        if not model_path.exists():
            logger.info("ONNX model not found — using heuristic chord detection")
            return False

        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )

            if classes_path.exists():
                import json
                with open(classes_path) as f:
                    data = json.load(f)
                self.classes = data["classes"]
            else:
                # Fallback class list
                chord_types = ["maj", "min", "7", "maj7", "min7", "dim", "aug", "sus4"]
                self.classes = [
                    f"{r}" if t == "maj" else f"{r}{t}"
                    for r in ROOT_NAMES for t in chord_types
                ]

            self._loaded = True
            logger.info(f"Piano ONNX model loaded: {len(self.classes)} chord classes")
            return True
        except ImportError:
            logger.info("onnxruntime not installed — using heuristic chord detection")
            return False
        except Exception as e:
            logger.warning(f"Failed to load ONNX model: {e}")
            return False

    def predict(self, feature_vec: np.ndarray) -> tuple[str, float]:
        """Predict chord from 36-dim feature vector. Returns (label, confidence)."""
        x = feature_vec[None].astype(np.float32)
        logits = self.session.run(["logits"], {"features": x})[0][0]
        probs = _softmax(logits)
        idx = int(probs.argmax())
        return self.classes[idx], float(probs[idx])

    def predict_top3(self, feature_vec: np.ndarray) -> list[tuple[str, float]]:
        """Predict top-3 chords."""
        x = feature_vec[None].astype(np.float32)
        logits = self.session.run(["logits"], {"features": x})[0][0]
        probs = _softmax(logits)
        top3 = probs.argsort()[-3:][::-1]
        return [(self.classes[i], float(probs[i])) for i in top3]

    @property
    def available(self) -> bool:
        return self._loaded


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# Global predictor instance
_predictor = PianoChordPredictor()


# ──────────────────────────────────────────────
# Audio extraction and analysis
# ──────────────────────────────────────────────

def extract_audio(video_path: str) -> str:
    """Extract mono 22050Hz WAV from video using FFmpeg."""
    wav_path = str(Path(video_path).with_suffix(".wav"))
    cmd = [
        "ffmpeg", "-i", video_path,
        "-ac", "1", "-ar", "22050", "-vn",
        "-y", wav_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Install FFmpeg to process video files.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg error: {e.stderr.decode()[:500]}")
    logger.info(f"Extracted audio: {wav_path}")
    return wav_path


def analyze_audio(wav_path: str) -> dict:
    """Analyze piano audio: detect chords, key, BPM, MIDI notes."""
    import librosa

    y, sr = librosa.load(wav_path, sr=22050)
    duration = float(len(y) / sr)
    logger.info(f"Loaded audio: {duration:.1f}s at {sr}Hz")

    # BPM
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = int(round(float(np.atleast_1d(tempo)[0])))

    # Chroma features
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)

    # Try ONNX model first, fallback to heuristic
    _predictor.load()
    if _predictor.available:
        chords = _detect_chords_onnx(y, sr)
    else:
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        chords = _detect_chords_heuristic(chroma, sr, beat_times, y)

    chords = _deduplicate_consecutive(chords)

    key = _estimate_key(chroma)
    time_sig = _estimate_time_signature(y, sr)
    midi_notes = _detect_pitches(y, sr)

    return {
        "bpm": bpm,
        "key": key,
        "chords": chords,
        "time_sig": time_sig,
        "duration": round(duration, 2),
        "midi_notes": midi_notes,
        "instrument": "piano",
    }


# ──────────────────────────────────────────────
# ONNX-based chord detection
# ──────────────────────────────────────────────

def _extract_window_features(notes_in_window: list[dict]) -> np.ndarray | None:
    """Extract 36-dim features from a list of note dicts (pitch, velocity, start, end)."""
    if len(notes_in_window) < 4:
        return None

    pitches = np.array([n["pitch"] for n in notes_in_window])
    vels = np.array([n["velocity"] for n in notes_in_window], dtype=float)
    durs = np.array([n["end"] - n["start"] for n in notes_in_window])

    # Chroma (12-dim)
    chroma = np.zeros(12)
    for p in pitches:
        chroma[p % 12] += 1
    total = chroma.sum()
    if total > 0:
        chroma /= total

    # Velocity-weighted chroma (12-dim)
    vchroma = np.zeros(12)
    for n, v in zip(notes_in_window, vels):
        vchroma[n["pitch"] % 12] += v
    vtotal = vchroma.sum()
    if vtotal > 0:
        vchroma /= vtotal

    # Piano features (6-dim)
    lh = pitches[pitches < 60]
    rh = pitches[pitches >= 60]
    piano_feats = np.array([
        len(lh) / (len(pitches) + 1e-8),
        len(rh) / (len(pitches) + 1e-8),
        (pitches.max() - pitches.min()) / 48.0,
        (durs > 0.4).mean(),
        vels.std() / 64.0,
        len(set(pitches % 12)) / 12.0,
    ])

    # Temporal features (6-dim)
    starts = np.array([n["start"] for n in notes_in_window])
    intervals = np.diff(starts) if len(starts) > 1 else np.array([0.0])
    temporal = np.array([
        np.clip(intervals.mean(), 0, 1),
        np.clip(intervals.std(), 0, 1),
        (intervals < 0.08).mean(),
        np.clip(durs.mean(), 0, 2) / 2.0,
        np.clip(durs.std(), 0, 1),
        np.clip(len(notes_in_window) / 30.0, 0, 1),
    ])

    return np.concatenate([chroma, vchroma, piano_feats, temporal]).astype(np.float32)


def _detect_chords_onnx(y: np.ndarray, sr: int) -> list[str]:
    """Detect chords using trained ONNX model on audio windows."""
    import librosa

    # Detect onsets to get note-like events
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)

    # Use pyin for pitch detection
    f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=2000, sr=sr, hop_length=512)
    f0_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=512)

    # Build pseudo-notes from pitch detection
    notes = []
    for i in range(len(f0) - 1):
        if voiced_flag[i] and not np.isnan(f0[i]):
            pitch = int(round(librosa.hz_to_midi(f0[i])))
            if 36 <= pitch <= 96:
                notes.append({
                    "pitch": pitch,
                    "velocity": 80,
                    "start": float(f0_times[i]),
                    "end": float(f0_times[i + 1]),
                })

    if len(notes) < 8:
        return _detect_chords_heuristic_from_chroma(y, sr)

    # Sliding window analysis
    duration = len(y) / sr
    window_sec = 2.0
    hop_sec = 1.0
    chords = []
    prev = None

    t = 0.0
    while t + window_sec <= duration:
        win_notes = [n for n in notes if n["start"] >= t and n["start"] < t + window_sec]
        feat = _extract_window_features(win_notes)

        if feat is not None:
            chord, conf = _predictor.predict(feat)
            if conf > 0.3 and chord != prev:
                chords.append(chord)
                prev = chord
        t += hop_sec

    return chords if chords else ["C"]


def _detect_chords_heuristic_from_chroma(y: np.ndarray, sr: int) -> list[str]:
    """Simple fallback using chroma directly."""
    import librosa
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    beat_times = librosa.frames_to_time(
        librosa.beat.beat_track(y=y, sr=sr)[1], sr=sr
    )
    return _detect_chords_heuristic(chroma, sr, beat_times, y)


# ──────────────────────────────────────────────
# Heuristic chord detection (fallback)
# ──────────────────────────────────────────────

def _detect_chords_heuristic(chroma: np.ndarray, sr: int,
                              beat_times: np.ndarray, y: np.ndarray) -> list[str]:
    """Detect chords from chroma features using template matching."""
    import librosa

    if len(beat_times) < 2:
        total_frames = chroma.shape[1]
        hop = 512
        segment_frames = int(2.0 * sr / hop)
        segments = []
        for i in range(0, total_frames, segment_frames):
            seg_chroma = chroma[:, i:i + segment_frames].mean(axis=1)
            segments.append(seg_chroma)
    else:
        beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=512)
        segments = []
        for i in range(len(beat_frames) - 1):
            start, end = beat_frames[i], beat_frames[i + 1]
            if end > start:
                seg_chroma = chroma[:, start:end].mean(axis=1)
                segments.append(seg_chroma)

    chords = [_chroma_to_chord(seg) for seg in segments]

    # Group every 4 beats into a measure
    measure_chords = []
    for i in range(0, len(chords), 4):
        measure = chords[i:i + 4]
        if measure:
            most_common = Counter(measure).most_common(1)[0][0]
            measure_chords.append(most_common)

    return measure_chords if measure_chords else chords[:8]


def _chroma_to_chord(chroma_vec: np.ndarray) -> str:
    """Match chroma vector to best chord template."""
    if np.sum(chroma_vec) < 1e-6:
        return "N"

    chroma_norm = chroma_vec / (np.linalg.norm(chroma_vec) + 1e-8)
    best_chord = "C"
    best_score = -1.0

    for root_idx in range(12):
        for chord_type, template in CHORD_TEMPLATES.items():
            rotated = np.roll(template, root_idx)
            rotated_norm = rotated / (np.linalg.norm(rotated) + 1e-8)
            score = float(np.dot(chroma_norm, rotated_norm))
            if score > best_score:
                best_score = score
                suffix = SUFFIX_DISPLAY[chord_type]
                best_chord = f"{ROOT_NAMES[root_idx]}{suffix}"

    return best_chord


def _estimate_key(chroma: np.ndarray) -> str:
    """Estimate key using Krumhansl-Kessler profiles."""
    mean_chroma = chroma.mean(axis=1)
    best_key = "C major"
    best_corr = -1.0

    for root_idx in range(12):
        for profile, mode in [(MAJOR_PROFILE, "major"), (MINOR_PROFILE, "minor")]:
            rotated = np.roll(profile, root_idx)
            corr = float(np.corrcoef(mean_chroma, rotated)[0, 1])
            if corr > best_corr:
                best_corr = corr
                best_key = f"{ROOT_NAMES[root_idx]} {mode}"

    return best_key


def _estimate_time_signature(y: np.ndarray, sr: int) -> str:
    """Estimate time signature from beat accent pattern."""
    import librosa

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    _, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    if len(beats) < 8:
        return "4/4"

    onset_at_beats = onset_env[beats]
    groups_of_3 = [onset_at_beats[i] for i in range(0, len(onset_at_beats) - 3, 3)]
    groups_of_4 = [onset_at_beats[i] for i in range(0, len(onset_at_beats) - 4, 4)]

    avg_3 = np.std(groups_of_3) if groups_of_3 else 999
    avg_4 = np.std(groups_of_4) if groups_of_4 else 999

    return "3/4" if avg_3 < avg_4 * 0.7 else "4/4"


def _detect_pitches(y: np.ndarray, sr: int) -> list[float]:
    """Detect prominent pitches as MIDI note numbers."""
    import librosa

    try:
        import crepe
        _, frequency, confidence, _ = crepe.predict(y, sr, viterbi=True, step_size=50)
        valid = frequency[confidence > 0.7]
        if len(valid) > 0:
            midi = librosa.hz_to_midi(valid)
            return [round(float(m), 1) for m in midi[:200]]
    except ImportError:
        logger.info("CREPE not available, using librosa pyin")

    f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=2000, sr=sr)
    valid_f0 = f0[voiced_flag & ~np.isnan(f0)]
    if len(valid_f0) > 0:
        midi = librosa.hz_to_midi(valid_f0)
        return [round(float(m), 1) for m in midi[:200]]

    return []


def _deduplicate_consecutive(chords: list[str]) -> list[str]:
    """Remove consecutive duplicate chords."""
    if not chords:
        return []

    result = [chords[0]]
    for chord in chords[1:]:
        if chord != result[-1] and chord != "N":
            result.append(chord)

    result = [c for c in result if c != "N"]
    return result if result else ["C"]
