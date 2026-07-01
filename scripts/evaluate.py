import sys
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.config import MODEL_PATH, CLASSES, TEST_DIR, DEVICE
from ultralytics import YOLO


def collect_predictions(model, test_dir: Path):
    """Run model on every test image, return (y_true, y_pred, y_proba)."""
    image_paths = []
    y_true = []
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = test_dir / cls_name
        if not cls_dir.exists():
            print(f"WARNING: {cls_dir} does not exist, skipping")
            continue
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                image_paths.append(img_path)
                y_true.append(cls_idx)
    y_true = np.array(y_true)

    print(f"Running inference on {len(image_paths)} test images...")
    y_pred = np.zeros(len(image_paths), dtype=int)
    y_proba = np.zeros((len(image_paths), len(CLASSES)), dtype=np.float32)
    t0 = time.perf_counter()
    for i, img_path in enumerate(image_paths):
        result = model.predict(source=str(img_path), device=DEVICE, verbose=False)[0]
        probs = result.probs.data.cpu().numpy()
        y_proba[i] = probs
        y_pred[i] = int(np.argmax(probs))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(image_paths)} done")
    elapsed = time.perf_counter() - t0
    print(f"Inference done in {elapsed:.1f}s ({elapsed/len(image_paths)*1000:.1f} ms/image)")
    return y_true, y_pred, y_proba, elapsed


def compute_metrics(y_true, y_pred):
    """Compute accuracy, per-class P/R/F1, macro & weighted F1."""
    metrics = {}
    metrics["overall_accuracy"] = float(accuracy_score(y_true, y_pred))

    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASSES))), zero_division=0
    )

    metrics["per_class"] = {}
    for i, cls in enumerate(CLASSES):
        metrics["per_class"][cls] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1_score": float(f[i]),
            "support": int(support[i]),
        }

    metrics["macro_avg"] = {
        "precision": float(np.mean(p)),
        "recall": float(np.mean(r)),
        "f1_score": float(np.mean(f)),
    }
    n = support.sum()
    metrics["weighted_avg"] = {
        "precision": float(np.sum(p * support) / n),
        "recall": float(np.sum(r * support) / n),
        "f1_score": float(np.sum(f * support) / n),
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0])
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].set_ylabel("True class")
    axes[0].set_xlabel("Predicted class")
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[1])
    axes[1].set_title("Confusion Matrix (normalized: row = 100%)")
    axes[1].set_ylabel("True class")
    axes[1].set_xlabel("Predicted class")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_per_class_metrics(metrics, out_path: Path):
    classes = CLASSES
    p = [metrics["per_class"][c]["precision"] for c in classes]
    r = [metrics["per_class"][c]["recall"] for c in classes]
    f1 = [metrics["per_class"][c]["f1_score"] for c in classes]

    x = np.arange(len(classes))
    w = 0.27
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, p, w, label="Precision", color="#1976D2")
    ax.bar(x, r, w, label="Recall", color="#388E3C")
    ax.bar(x + w, f1, w, label="F1-Score", color="#F57C00")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Precision / Recall / F1")
    ax.axhline(metrics["macro_avg"]["f1_score"], color="red", ls="--", lw=1, label=f"Macro F1 = {metrics['macro_avg']['f1_score']:.3f}")
    ax.legend(loc="lower right")
    for i, v in enumerate(f1):
        ax.text(i + w, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def print_report(metrics, elapsed):
    total_support = sum(m["support"] for m in metrics["per_class"].values())
    print("\n" + "=" * 60)
    print(" CLASSIFICATION REPORT")
    print("=" * 60)
    print(f"{'':>12} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
    for cls in CLASSES:
        m = metrics["per_class"][cls]
        print(f"{cls:>12} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1_score']:>10.4f} {m['support']:>10}")
    print()
    print(f"{'accuracy':>12} {'':>10} {'':>10} {metrics['overall_accuracy']:>10.4f} {total_support:>10}")
    print(f"{'macro avg':>12} {metrics['macro_avg']['precision']:>10.4f} {metrics['macro_avg']['recall']:>10.4f} {metrics['macro_avg']['f1_score']:>10.4f} {'':>10}")
    print(f"{'weighted avg':>12} {metrics['weighted_avg']['precision']:>10.4f} {metrics['weighted_avg']['recall']:>10.4f} {metrics['weighted_avg']['f1_score']:>10.4f} {'':>10}")
    print("=" * 60)
    print(f"Inference: {elapsed:.1f}s total ({elapsed/total_support*1000:.1f} ms/image)")
    print("=" * 60)


def main():
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model not found at {MODEL_PATH}. Run train.py first.")
        sys.exit(1)
    if not TEST_DIR.exists():
        print(f"Error: Test dir not found at {TEST_DIR}.")
        sys.exit(1)

    print(f"Loading model from: {MODEL_PATH}")
    print(f"Test directory:     {TEST_DIR}")
    model = YOLO(str(MODEL_PATH))

    y_true, y_pred, y_proba, elapsed = collect_predictions(model, TEST_DIR)
    metrics = compute_metrics(y_true, y_pred)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(MODEL_PATH).parent / "runs" / f"eval-{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(y_true, y_pred, out_dir / "confusion_matrix.png")
    plot_per_class_metrics(metrics, out_dir / "per_class_metrics.png")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print_report(metrics, elapsed)
    print(f"\nPlots and metrics saved to: {out_dir}")


if __name__ == "__main__":
    main()
