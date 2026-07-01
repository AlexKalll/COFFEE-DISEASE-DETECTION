from ultralytics import YOLO
import sys
from pathlib import Path
import shutil

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.config import (
    PRETRAINED_MODEL,
    TRAIN_DIR,
    IMG_SIZE,
    EPOCHS,
    BATCH_SIZE,
    PROJECT_NAME,
    RUN_NAME,
    MODEL_PATH,
    LAST_MODEL_PATH,
)


def train():
    model = YOLO(PRETRAINED_MODEL)

    results = model.train(
        data=str(TRAIN_DIR),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_NAME,
        name=RUN_NAME,
        verbose=True,
        save=True,
        plots=True,
    )

    run_base = Path(PROJECT_NAME) / RUN_NAME
    run_folders = sorted(run_base.parent.glob(f"{RUN_NAME}*"))
    actual_run = run_folders[-1] if run_folders else run_base

    best_model_path = actual_run / "weights" / "best.pt"
    last_model_path = actual_run / "weights" / "last.pt"

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

    if best_model_path.exists():
        shutil.copy(best_model_path, MODEL_PATH)
        print(f"best.pt saved to {MODEL_PATH}")
    if last_model_path.exists():
        shutil.copy(last_model_path, LAST_MODEL_PATH)
        print(f"last.pt saved to {LAST_MODEL_PATH}")

    print(results)


if __name__ == "__main__":
    train()
