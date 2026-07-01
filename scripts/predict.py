from ultralytics import YOLO
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.config import MODEL_PATH, CLASSES, DEVICE


def predict(image_path, top_k=3):
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run train.py first to train the model.")
        return None

    model = YOLO(str(MODEL_PATH))

    results = model.predict(source=image_path, device=DEVICE, verbose=False)

    if not results:
        print("No predictions returned.")
        return None

    result = results[0]
    probs = result.probs

    top_indices = probs.top5[:top_k]
    top_confidences = probs.top5conf[:top_k].cpu().numpy()
    top_classes = [result.names[i] for i in top_indices]

    predictions = []
    for cls, conf in zip(top_classes, top_confidences):
        predictions.append({"class": cls, "confidence": round(float(conf), 4)})

    return predictions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Coffee Leaf Disease Inference")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument(
        "--top-k", type=int, default=3, help="Number of top predictions to return"
    )
    args = parser.parse_args()

    results = predict(args.image_path, args.top_k)
    if results:
        print(f"\nTop {args.top_k} Predictions:")
        for i, pred in enumerate(results, 1):
            print(f"  {i}. {pred['class']}: {pred['confidence']:.4f}")
