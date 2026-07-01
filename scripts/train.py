from ultralytics import YOLO
from ultralytics import settings
import sys
from pathlib import Path
import shutil

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.config import (
    PRETRAINED_MODEL,
    DATASET_DIR,
    IMG_SIZE,
    EPOCHS,
    BATCH_SIZE,
    MODEL_PATH,
    LAST_MODEL_PATH,
    RUNS_DIR,
    RUN_NAME,
    DEVICE,
)


def train():
    settings.update({"runs_dir": str(RUNS_DIR)})
    print(f"Using device: {DEVICE}")

    model = YOLO(PRETRAINED_MODEL)

    results = model.train(
        data=str(DATASET_DIR),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=RUNS_DIR,
        name=RUN_NAME,
        verbose=True,
        save=True,
        plots=True,
    )

    save_dir = Path(results.save_dir)
    best_model_path = save_dir / "weights" / "best.pt"
    last_model_path = save_dir / "weights" / "last.pt"
    print(f"Run directory: {save_dir.name}")

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

    if best_model_path.exists():
        shutil.copy(best_model_path, MODEL_PATH)
        print(f"best.pt saved to {MODEL_PATH}")
    else:
        print(f"WARNING: best.pt not found at {best_model_path}")

    if last_model_path.exists():
        shutil.copy(last_model_path, LAST_MODEL_PATH)
        print(f"last.pt saved to {LAST_MODEL_PATH}")
    else:
        print(f"WARNING: last.pt not found at {last_model_path}")

    print(results)


if __name__ == "__main__":
    train()
