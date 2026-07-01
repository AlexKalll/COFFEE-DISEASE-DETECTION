import sys
import json
import itertools
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.config import PRETRAINED_MODEL, DATASET_DIR, IMG_SIZE, RUNS_DIR
from ultralytics import YOLO

GRID = {
    "lr0": [0.0001],
    "batch": [8, 16],
    }
EPOCHS = 1
PROJECT = str(Path(RUNS_DIR) / "grid_search")
NAME = "trial"

def run_one(lr, batch):
    model = YOLO(str(PRETRAINED_MODEL))
    results = model.train(
        data=str(DATASET_DIR),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=batch,
        lr0=lr,
        project=PROJECT,
        name=NAME,
        exist_ok=False,
        verbose=False,
        save=False,
        plots=False,)

    save_dir = Path(results.save_dir)
    new_name = f"{save_dir.name}_lr{lr}_b{batch}"
    new_path = save_dir.parent / new_name
    save_dir.rename(new_path)
    return float(results.results_dict.get("metrics/accuracy_top1", 0.0)), new_path

def main():
    combos = list(itertools.product(GRID["lr0"], GRID["batch"]))
    print(f"Grid search: {len(combos)} combinations, {EPOCHS} epochs each")
    print(f"Saving runs under: {PROJECT}/\n")

    rows = []
    for i, (lr, batch) in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] lr={lr}, batch={batch} ...", end=" ", flush=True)
        acc, run_path = run_one(lr, batch)
        print(f"top1_acc={acc:.4f}  ->  {run_path.name}")
        rows.append({"trial": run_path.name, "lr0": lr, "batch": batch, "top1_acc": acc})

    n_runs = len(rows)
    out_dir = Path(PROJECT) / f"run-{n_runs}-grid-results"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows).sort_values("top1_acc", ascending=False)
    print("\n" + "=" * 60)
    print(" GRID SEARCH RESULTS (sorted by top1_acc)")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)
    print(f"\nBest config: lr0={df.iloc[0]['lr0']}, batch={int(df.iloc[0]['batch'])}")
    print(f"Best top1_acc: {df.iloc[0]['top1_acc']:.4f}")

    df.to_csv(out_dir / "grid_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for batch in sorted(GRID["batch"]):
        sub = df[df["batch"] == batch].sort_values("lr0")
        ax.plot(sub["lr0"], sub["top1_acc"], "o-", label=f"batch={batch}")
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate (log scale)")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title(f"Grid Search ({EPOCHS} epochs each, {len(combos)} runs)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "grid_search.png", dpi=120)
    plt.close()
    print(f"\nResults saved to: {out_dir}")

if __name__ == "__main__":
    main()
