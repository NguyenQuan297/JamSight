"""Model evaluation and comparison with baseline."""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def load_pytorch_model(path: str, device: str = "cpu"):
    """Load a trained PyTorch chord classifier."""
    from chord_classifier import ChordCNN, NUM_CLASSES
    model = ChordCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def load_onnx_model(path: str):
    """Load an ONNX model for inference."""
    import onnxruntime as ort
    return ort.InferenceSession(path)


def evaluate_on_audio(model, audio_path: str) -> list[dict]:
    """Run chord detection on a real audio file using the trained model."""
    import librosa
    from chord_classifier import idx_to_label

    y, sr = librosa.load(audio_path, sr=22050)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)

    # Segment by 2-second windows
    hop = 512
    window_frames = int(2.0 * sr / hop)
    results = []

    for i in range(0, chroma.shape[1], window_frames):
        segment = chroma[:, i:i + window_frames]
        if segment.shape[1] == 0:
            continue

        mean_chroma = segment.mean(axis=1).astype(np.float32)
        total = mean_chroma.sum()
        if total > 0:
            mean_chroma /= total

        # Predict
        with torch.no_grad():
            x = torch.tensor(mean_chroma).unsqueeze(0)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()

        time_start = i * hop / sr
        time_end = min((i + window_frames) * hop / sr, len(y) / sr)

        results.append({
            "time_start": round(time_start, 2),
            "time_end": round(time_end, 2),
            "chord": idx_to_label(pred_idx),
            "confidence": round(confidence, 3),
        })

    return results


def compare_with_baseline(model, test_dir: str, device: str = "cpu") -> dict:
    """Compare trained model accuracy vs librosa chroma baseline."""
    from chord_classifier import ChordDataset, idx_to_label

    test_ds = ChordDataset(test_dir, "test")
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # Model predictions
    model.eval()
    model_correct = 0
    baseline_correct = 0
    total = 0

    # Simple baseline: argmax of chroma vector → root, assume major
    from services.audio_analyzer import _chroma_to_chord
    from chord_classifier import label_to_idx

    for chroma_batch, labels in test_loader:
        # Model
        with torch.no_grad():
            logits = model(chroma_batch.to(device))
            model_preds = logits.argmax(dim=1).cpu()
            model_correct += (model_preds == labels).sum().item()

        # Baseline: just pick the strongest chroma bin → assume major chord
        for i in range(len(chroma_batch)):
            chroma_vec = chroma_batch[i].numpy()
            baseline_chord = _chroma_to_chord(chroma_vec)
            baseline_idx = label_to_idx(baseline_chord)
            if baseline_idx == labels[i].item():
                baseline_correct += 1

        total += labels.size(0)

    model_acc = model_correct / total * 100
    baseline_acc = baseline_correct / total * 100

    return {
        "model_accuracy": round(model_acc, 2),
        "baseline_accuracy": round(baseline_acc, 2),
        "improvement": round(model_acc - baseline_acc, 2),
        "total_samples": total,
    }


def per_class_metrics(model, test_dir: str, device: str = "cpu") -> dict:
    """Compute per-chord-class precision, recall, F1."""
    from chord_classifier import ChordDataset, NUM_CLASSES, idx_to_label

    test_ds = ChordDataset(test_dir, "test")
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for chroma, labels in test_loader:
            logits = model(chroma.to(device))
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


def generate_report(model, test_dir: str, output_path: str = "evaluation_report.txt", device: str = "cpu"):
    """Generate a full evaluation report."""
    comparison = compare_with_baseline(model, test_dir, device)
    metrics = per_class_metrics(model, test_dir, device)

    lines = [
        "=" * 60,
        "JamSight Chord Classifier — Evaluation Report",
        "=" * 60,
        "",
        f"Total test samples: {comparison['total_samples']}",
        f"Model accuracy:    {comparison['model_accuracy']}%",
        f"Baseline accuracy: {comparison['baseline_accuracy']}%",
        f"Improvement:       {comparison['improvement']}%",
        "",
        "Per-class metrics (top 20 by support):",
        "-" * 50,
    ]

    # Sort classes by support
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
    parser.add_argument("--data_dir", default="data", help="Test data directory")
    parser.add_argument("--audio", default=None, help="Evaluate on a single audio file")
    parser.add_argument("--report", default="evaluation_report.txt")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_pytorch_model(args.model_path, device)

    if args.audio:
        results = evaluate_on_audio(model, args.audio)
        for r in results:
            print(f"  [{r['time_start']:6.2f}s - {r['time_end']:6.2f}s] "
                  f"{r['chord']:8s} (conf={r['confidence']:.1%})")
    else:
        generate_report(model, args.data_dir, args.report, device)


if __name__ == "__main__":
    main()
