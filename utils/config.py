from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"

TRAIN_DIR = DATASET_DIR / "Train"
VAL_DIR = DATASET_DIR / "valid"
TEST_DIR = DATASET_DIR / "test"

MODEL_NAME = "yolo11n-cls"
MODEL_PATH = BASE_DIR / "models" / "train" / "best.pt"
LAST_MODEL_PATH = BASE_DIR / "models" / "train" / "last.pt"
PRETRAINED_MODEL = BASE_DIR / "yolo11n-cls.pt"

CLASSES = ["Healthy", "Rust", "Phoma", "Miner"]
NUM_CLASSES = len(CLASSES)

IMG_SIZE = 224
EPOCHS = 30
BATCH_SIZE = 16

PROJECT_NAME = "models/runs"
RUN_NAME = "train"

DATA_YAML = BASE_DIR / "data.yaml"
