"""Model evaluation and comparison with baseline."""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

ROOTS = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def _chroma_to_chord_baseline(chroma_vec: np.ndarray) -> str:
    """Simple baseline: strongest chroma bin = root, assume major."""
    root_idx = int(chroma_vec[:12].argmax())
    return ROOTS[root_idx]


def load_pytorch_model(path: str, device: str = "cpu"):
    """Load a trained PyTorch chord classifier."""
    from chord_classifier import PianoMLP

    ckpt = torch.load(path, map_location=device, weights_only=False)
    feature_dim = ckpt.get("feature_dim", 36)
    hidden_dim = ckpt.get("hidden_dim", 256)
    classes = ckpt.get("classes", [])
    num_classes = len(classes) if classes else 96

    model = PianoMLP(in_dim=feature_dim, num_classes=num_classes, hidden_dim=hidden_dim)
    model.load_state_dict(ckpt["state"])
    model.eval()
    model.to(device)
    return model, ckpt


def load_onnx_model(path: str):
    """Load an ONNX model for inference."""
    import onnxruntime as ort
    return ort.InferenceSession(path)


def evaluate_on_audio(model, audio_path: str, ckpt: dict) -> list[dict]:
    """Run chord detection on a real audio file using the trained model."""
    import librosa
    from chord_classifier import idx_to_label

    y, sr = librosa.load(audio_path, sr=16000)
    feature_dim = ckpt.get("feature_dim", 36)

    # Use CREPE/pyin for pitch detection
    try:
        import crepe
        crepe_time, crepe_freq, crepe_conf, _ = crepe.predict(
            y, sr, viterbi=True, step_size=50
        )
    except ImportError:
        f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=2000, sr=sr,
                                      hop_length=int(sr * 0.05))
        crepe_time = np.arange(len(f0)) * 0.05
        crepe_freq = np.where(np.isnan(f0), 0, f0)
        crepe_conf = voiced.astype(float)

    duration = len(y) / sr
    results = []

    t = 0.0
    while t + 2.0 <= duration:
        feat = _extract_eval_features(y, sr, t, 2.0,
                                       crepe_time, crepe_freq, crepe_conf)
        if feat is not None and len(feat) == feature_dim:
            with torch.no_grad():
                x = torch.tensor(feat).unsqueeze(0)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0]
                pred_idx = probs.argmax().item()
                confidence = probs[pred_idx].item()

            results.append({
                "time_start": round(t, 2),
                "time_end": round(t + 2.0, 2),
                "chord": idx_to_label(pred_idx),
                "confidence": round(confidence, 3),
            })
        t += 1.0

    return results


def _extract_eval_features(y, sr, t_start, window,
                            crepe_time, crepe_freq, crepe_conf) -> np.ndarray | None:
    """Extract 36-dim features matching the training pipeline."""
    import librosa

    start_sample = int(t_start * sr)
    end_sample = start_sample + int(window * sr)
    y_win = y[start_sample:end_sample]
    if len(y_win) < sr * 0.5:
        return None

    # CREPE chroma (12-dim)
    mask = (crepe_time >= t_start) & (crepe_time < t_start + window) & (crepe_conf > 0.6)
    win_freq = crepe_freq[mask]
    win_conf = crepe_conf[mask]

    crepe_chroma = np.zeros(12)
    if len(win_freq) >= 3:
        for f, c in zip(win_freq, win_conf):
            if f > 0:
                midi_note = int(round(librosa.hz_to_midi(f)))
                crepe_chroma[midi_note % 12] += c
        s = crepe_chroma.sum()
        if s > 0:
            crepe_chroma /= s
    else:
        return None

    # CQT chroma (12-dim)
    chroma_cqt = librosa.feature.chroma_cqt(y=y_win, sr=sr, hop_length=512)
    chroma_mean = chroma_cqt.mean(axis=1)
    cmax = chroma_mean.max()
    if cmax > 0:
        chroma_mean /= cmax

    # Spectral (12-dim)
    centroid = librosa.feature.spectral_centroid(y=y_win, sr=sr).mean() / 8000
    bandwidth = librosa.feature.spectral_bandwidth(y=y_win, sr=sr).mean() / 4000
    rolloff = librosa.feature.spectral_rolloff(y=y_win, sr=sr).mean() / 8000
    zcr = librosa.feature.zero_crossing_rate(y_win).mean()
    mfccs = librosa.feature.mfcc(y=y_win, sr=sr, n_mfcc=8).mean(axis=1)
    mfccs = np.clip((mfccs + 50) / 100, 0, 1)

    spectral = np.array([centroid, bandwidth, rolloff, zcr, *mfccs])
    spectral = np.clip(spectral, 0, 1).astype(np.float32)

    return np.concatenate([crepe_chroma, chroma_mean, spectral]).astype(np.float32)


def compare_with_baseline(model, device: str = "cpu") -> dict:
    """Compare trained model accuracy vs argmax-chroma baseline on test set."""
    from chord_classifier import PianoChordDataset, load_split, label_to_idx

    test_s, _ = load_split("test")
    test_loader = DataLoader(
        PianoChordDataset(test_s, augment=False),
        batch_size=64, shuffle=False,
    )

    model.eval()
    model_correct = 0
    baseline_correct = 0
    total = 0

    for features, labels in test_loader:
        # Model prediction
        with torch.no_grad():
            logits = model(features.to(device))
            model_preds = logits.argmax(dim=1).cpu()
            model_correct += (model_preds == labels).sum().item()

        # Baseline: argmax of first 12 dims (CREPE chroma) → assume major
        for i in range(len(features)):
            chroma_vec = features[i].numpy()
            baseline_chord = _chroma_to_chord_baseline(chroma_vec)
            baseline_idx = label_to_idx(baseline_chord)
            if baseline_idx == labels[i].item():
                baseline_correct += 1

        total += labels.size(0)

    model_acc = model_correct / total * 100 if total > 0 else 0
    baseline_acc = baseline_correct / total * 100 if total > 0 else 0

    return {
        "model_accuracy": round(model_acc, 2),
        "baseline_accuracy": round(baseline_acc, 2),
        "improvement": round(model_acc - baseline_acc, 2),
        "total_samples": total,
    }


def per_class_metrics(model, device: str = "cpu") -> dict:
    """Compute per-chord-class precision, recall, F1."""
    from chord_classifier import PianoChordDataset, load_split, idx_to_label, NUM_CLASSES

    test_s, _ = load_split("test")
    test_loader = DataLoader(
        PianoChordDataset(test_s, augment=False),
        batch_size=64, shuffle=False,
    )

    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for features, labels in test_loader:
            logits = model(features.to(device))
            preds = logits.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    from sklearn.metrics import classification_report
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(NUM_CLASSES)),
        target_names=[idx_to_label(i) for i in range(NUM_CLASSES)],
        output_dict=True,
        zero_division=0,
    )

    return report


def generate_report(model, output_path: str = "evaluation_report.txt", device: str = "cpu"):
    """Generate a full evaluation report."""
    comparison = compare_with_baseline(model, device)
    metrics = per_class_metrics(model, device)

    lines = [
        "=" * 60,
        "JamSight Chord Classifier — Evaluation Report",
        "=" * 60,
        "",
        f"Total test samples: {comparison['total_samples']}",
        f"Model accuracy:    {comparison['model_accuracy']}%",
        f"Baseline accuracy: {comparison['baseline_accuracy']}%",
        f"Improvement:       +{comparison['improvement']}%",
        "",
        "Per-class metrics (top 20 by support):",
        "-" * 50,
    ]

    class_metrics = [
        (name, data)
        for name, data in metrics.items()
        if name not in ("accuracy", "macro avg", "weighted avg")
    ]
    class_metrics.sort(key=lambda x: x[1].get("support", 0), reverse=True)

    for name, data in class_metrics[:20]:
        lines.append(
            f"  {name:8s}  P={data['precision']:.2f}  R={data['recall']:.2f}  "
            f"F1={data['f1-score']:.2f}  n={int(data['support'])}"
        )

    lines.extend([
        "",
        "-" * 50,
        f"Macro avg:    P={metrics['macro avg']['precision']:.3f}  "
        f"R={metrics['macro avg']['recall']:.3f}  F1={metrics['macro avg']['f1-score']:.3f}",
        f"Weighted avg: P={metrics['weighted avg']['precision']:.3f}  "
        f"R={metrics['weighted avg']['recall']:.3f}  F1={metrics['weighted avg']['f1-score']:.3f}",
    ])

    report_text = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report_text)

    print(report_text)
    logger.info(f"Report saved to {output_path}")
    return report_text


def main():
    parser = argparse.ArgumentParser(description="Evaluate chord classifier")
    parser.add_argument("--model_path", required=True, help="Path to .pt model")
    parser.add_argument("--audio", default=None, help="Evaluate on a single audio file")
    parser.add_argument("--report", default="evaluation_report.txt")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, ckpt = load_pytorch_model(args.model_path, device)

    if args.audio:
        results = evaluate_on_audio(model, args.audio, ckpt)
        for r in results:
            print(f"  [{r['time_start']:6.2f}s - {r['time_end']:6.2f}s] "
                  f"{r['chord']:8s} (conf={r['confidence']:.1%})")
    else:
        generate_report(model, args.report, device)


if __name__ == "__main__":
    main()
