"""Piano chord classifier: 36-dim feature vector → 96 chord classes.

Model: PianoMLP with residual connections, LayerNorm, GELU.
Training: weighted sampling for class balance, cosine LR, label smoothing.
Ablation: runs 4 configs and picks the best.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

logger = logging.getLogger(__name__)

_BASE = Path(os.environ.get("JAMSIGHT_TRAIN_DIR", Path(__file__).parent))
DATA_DIR = _BASE / "data"
MODEL_DIR = _BASE / "models"

ROOTS = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
CHORD_TYPES = ["maj", "min", "7", "maj7", "min7", "dim", "aug", "sus4"]
NUM_CLASSES = len(ROOTS) * len(CHORD_TYPES)  # 96
CLASSES = [
    r if t == "maj" else f"{r}{t}"
    for r in ROOTS for t in CHORD_TYPES
]


def label_to_idx(label: str) -> int:
    """Convert chord label to class index."""
    label = label.strip()
    root_idx = -1
    root_len = 0
    for i, r in enumerate(ROOTS):
        if label.startswith(r) and len(r) > root_len:
            root_idx = i
            root_len = len(r)
    if root_idx == -1:
        return 0

    suffix = label[root_len:]
    suffix_map = {"": "maj", "m": "min", "M7": "maj7", "min7": "min7",
                  "m7": "min7", "dom7": "7"}
    suffix = suffix_map.get(suffix, suffix)
    if suffix not in CHORD_TYPES:
        suffix = "maj"

    return root_idx * len(CHORD_TYPES) + CHORD_TYPES.index(suffix)


def idx_to_label(idx: int) -> str:
    """Convert class index to chord label."""
    root_idx = idx // len(CHORD_TYPES)
    type_idx = idx % len(CHORD_TYPES)
    suffix = {"maj": "", "min": "m"}.get(CHORD_TYPES[type_idx], CHORD_TYPES[type_idx])
    return f"{ROOTS[root_idx]}{suffix}"


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class PianoChordDataset(Dataset):
    """Dataset from piano split JSON files. Supports online augmentation."""

    def __init__(self, samples: list[dict], augment: bool = False):
        self.X = [np.array(s["features"], dtype=np.float32) for s in samples]
        self.y = torch.tensor(
            [s["label_idx"] for s in samples], dtype=torch.long
        )
        self.augmenter = None
        if augment:
            from augment import PianoAugmentation
            self.augmenter = PianoAugmentation()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        if self.augmenter is not None:
            x = self.augmenter(x)
        return x, self.y[idx]


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────

class PianoMLP(nn.Module):
    """Piano chord classifier with residual connection.

    Input:  (batch, 36) piano feature vector
    Output: (batch, 96) logits
    """

    def __init__(self, in_dim: int = 36, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.hidden = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        return self.head(h + self.hidden(h))  # residual


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def load_split(split: str) -> tuple[list[dict], dict]:
    """Load a pre-split dataset file (piano_train.json, piano_val.json, etc.)."""
    path = DATA_DIR / f"piano_{split}.json"
    if not path.exists():
        # Fallback to single-file format
        single = DATA_DIR / "piano_dataset.json"
        if single.exists():
            logger.info(f"Split file not found, falling back to {single}")
            return _fallback_split(str(single), split)
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run prepare_data.py first."
        )

    with open(path) as f:
        data = json.load(f)
    return data["samples"], data


def _fallback_split(dataset_path: str, split: str) -> tuple[list, dict]:
    """Fallback: split single dataset file by track."""
    with open(dataset_path) as f:
        data = json.load(f)
    samples = data["samples"]
    tracks = list(set(s["track_id"] for s in samples))
    rng = np.random.default_rng(42)
    rng.shuffle(tracks)
    n = len(tracks)
    val_tracks = set(tracks[:int(n * 0.15)])
    test_tracks = set(tracks[int(n * 0.15):int(n * 0.25)])

    if split == "train":
        return [s for s in samples if s["track_id"] not in val_tracks | test_tracks], data
    elif split == "val":
        return [s for s in samples if s["track_id"] in val_tracks], data
    else:
        return [s for s in samples if s["track_id"] in test_tracks], data


def make_weighted_sampler(samples: list[dict], num_classes: int) -> WeightedRandomSampler:
    """Create weighted sampler to handle class imbalance."""
    labels = [s["label_idx"] for s in samples]
    counts = np.bincount(labels, minlength=num_classes)
    weights = 1.0 / (counts[labels] + 1)
    return WeightedRandomSampler(weights, len(weights))


def train_model(config: dict) -> float:
    """Train one model with given config. Returns best validation accuracy."""
    train_s, train_meta = load_split("train")
    val_s, _ = load_split("val")
    classes = train_meta.get("classes", CLASSES)
    num_classes = len(classes)
    # Detect feature_dim from actual data, fallback to metadata, then 36
    if train_s:
        feature_dim = len(train_s[0]["features"])
    else:
        feature_dim = train_meta.get("feature_dim", 36)
    logger.info(f"Split: train={len(train_s)}, val={len(val_s)}, feature_dim={feature_dim}, classes={num_classes}")

    # Weighted sampler
    sampler = make_weighted_sampler(train_s, num_classes)
    train_loader = DataLoader(
        PianoChordDataset(train_s, augment=config.get("augment", True)),
        batch_size=config.get("batch_size", 64),
        sampler=sampler, num_workers=0,
    )
    val_loader = DataLoader(
        PianoChordDataset(val_s, augment=False),
        batch_size=256, shuffle=False, num_workers=0,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PianoMLP(in_dim=feature_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"],
    )
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config.get("label_smoothing", 0.1),
    )

    # Incremental training
    model_path = MODEL_DIR / f"{config['name']}_best.pt"
    if config.get("incremental") and model_path.exists():
        logger.info(f"Loading existing model for incremental training: {model_path}")
        ckpt = torch.load(str(model_path), map_location=device)
        model.load_state_dict(ckpt["state"])

    best_acc = 0.0
    patience_counter = 0
    max_patience = config.get("patience", 10)

    for epoch in range(config["epochs"]):
        # Train
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validate
        model.eval()
        correct = top3_correct = total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                correct += (logits.argmax(1) == y).sum().item()
                top3_correct += (logits.topk(3, 1).indices == y.unsqueeze(1)).any(1).sum().item()
                total += len(y)

        acc = correct / total if total > 0 else 0
        top3_acc = top3_correct / total if total > 0 else 0

        if epoch % 5 == 0 or epoch == config["epochs"] - 1:
            logger.info(
                f"  [{config['name']}] Epoch {epoch:3d}: "
                f"loss={total_loss / len(train_loader):.4f}  "
                f"val_acc={acc:.1%}  top3={top3_acc:.1%}  "
                f"lr={optimizer.param_groups[0]['lr']:.1e}"
            )

        # Early stopping
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"state": model.state_dict(), "classes": classes,
                 "config": config, "best_acc": best_acc,
                 "feature_dim": feature_dim},
                str(model_path),
            )
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"  Early stopping at epoch {epoch}. Best val_acc: {best_acc:.1%}")
                break

    logger.info(f"  [{config['name']}] Training complete. Best accuracy: {best_acc:.1%}")
    return best_acc


def export_onnx(model_path: str, output_path: str = None):
    """Export trained model to ONNX for fast inference."""
    ckpt = torch.load(model_path, map_location="cpu")
    feature_dim = ckpt.get("feature_dim", 36)
    num_classes = len(ckpt["classes"])

    model = PianoMLP(in_dim=feature_dim, num_classes=num_classes)
    model.load_state_dict(ckpt["state"])
    model.eval()

    if output_path is None:
        output_path = str(Path(model_path).with_suffix(".onnx"))

    dummy = torch.randn(1, feature_dim)
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    logger.info(f"ONNX exported: {output_path}")


def run_ablation():
    """Run 4 training configs and pick the best."""
    configs = [
        {"name": "piano_mlp_full",
         "lr": 3e-4, "epochs": 60, "label_smoothing": 0.1, "batch_size": 64},
        {"name": "piano_mlp_no_smooth",
         "lr": 3e-4, "epochs": 60, "label_smoothing": 0.0, "batch_size": 64},
        {"name": "piano_mlp_high_lr",
         "lr": 1e-3, "epochs": 60, "label_smoothing": 0.1, "batch_size": 64},
        {"name": "piano_mlp_low_lr",
         "lr": 1e-4, "epochs": 60, "label_smoothing": 0.1, "batch_size": 64},
    ]

    results = []
    for cfg in configs:
        print(f"\n{'=' * 50}")
        print(f"Training: {cfg['name']}")
        print(f"{'=' * 50}")
        acc = train_model(cfg)
        results.append((cfg["name"], acc))

    # Summary
    print(f"\n{'=' * 50}")
    print("ABLATION RESULTS")
    print(f"{'=' * 50}")
    best_name, best_acc = max(results, key=lambda x: x[1])
    for name, acc in sorted(results, key=lambda x: -x[1]):
        marker = " <-- BEST" if name == best_name else ""
        print(f"  {name:<28} {acc:.1%}{marker}")

    # Export best to ONNX
    best_path = str(MODEL_DIR / f"{best_name}_best.pt")
    onnx_path = str(MODEL_DIR / "piano_model.onnx")
    export_onnx(best_path, onnx_path)
    print(f"\nBest model exported to ONNX: {onnx_path}")

    return best_name, best_acc


def main():
    parser = argparse.ArgumentParser(description="Train JamSight piano chord classifier")
    parser.add_argument("--name", default="piano_mlp_full", help="Run name")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--ablation", action="store_true", help="Run full ablation study")
    parser.add_argument("--export_onnx", type=str, default=None, help="Export model to ONNX")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )

    if args.export_onnx:
        export_onnx(args.export_onnx)
        return

    if args.ablation:
        run_ablation()
        return

    config = {
        "name": args.name,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "label_smoothing": args.label_smoothing,
        "incremental": args.incremental,
    }
    acc = train_model(config)
    print(f"\nFinal best validation accuracy: {acc:.1%}")

    # Auto-export ONNX
    model_path = str(MODEL_DIR / f"{args.name}_best.pt")
    onnx_path = str(MODEL_DIR / "piano_model.onnx")
    if Path(model_path).exists():
        export_onnx(model_path, onnx_path)


if __name__ == "__main__":
    main()
