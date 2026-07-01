import cv2
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.config import MODEL_PATH, CLASSES
from ultralytics import YOLO


def webcam_inference(camera_index=0, display=True):
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run train.py first to train the model.")
        return

    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_index}")
        return

    print("Press 'q' to quit")
    print("Press 's' to save screenshot")

    frame_count = 0
    fps = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_count += 1
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            fps = 10 / elapsed
            start_time = time.time()

        results = model.predict(source=frame, verbose=False)
        result = results[0]

        probs = result.probs
        top_idx = probs.top1
        top_class = result.names[top_idx]
        top_conf = probs.top1conf.item()

        label = f"{top_class}: {top_conf:.2f}"

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, label, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if display:
            cv2.imshow("Coffee Disease Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_path = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(screenshot_path, frame)
            print(f"Screenshot saved: {screenshot_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Webcam Inference")
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Don't display window"
    )
    args = parser.parse_args()

    webcam_inference(args.camera, display=not args.no_display)
