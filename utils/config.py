from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"

MODEL_NAME = "yolo11n-cls"
PRETRAINED_MODEL = BASE_DIR / f"{MODEL_NAME}.pt"

MODELS_DIR = BASE_DIR / "models-1"
MODEL_PATH = MODELS_DIR / "best.pt"
LAST_MODEL_PATH = MODELS_DIR / "last.pt"

RUNS_DIR = MODELS_DIR / "runs"
RUN_NAME = "train"

CLASSES = ["Healthy", "Rust", "Phoma", "Miner"]
NUM_CLASSES = len(CLASSES)

IMG_SIZE = 224
EPOCHS = 30
BATCH_SIZE = 16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_YAML = BASE_DIR / "data.yaml"
