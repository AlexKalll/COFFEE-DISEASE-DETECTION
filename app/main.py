import gradio as gr
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.config import MODEL_PATH, CLASSES
from ultralytics import YOLO

model = None


def load_model():
    global model
    if model is None:
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Please run train.py first."
            )
        model = YOLO(str(MODEL_PATH))
    return model


def predict_image(image, top_k=3):
    if image is None:
        return {}

    clf = load_model()
    results = clf.predict(source=image, verbose=False)
    result = results[0]

    probs = result.probs
    top_indices = probs.top5[:top_k]
    top_confidences = probs.top5conf[:top_k].cpu().numpy().tolist()
    classes = [result.names[i] for i in top_indices]
    confidences = top_confidences

    output = {}
    for cls, conf in zip(classes, confidences):
        output[cls] = round(conf, 4)

    return output


def main():
    load_model()

    with gr.Blocks(title="Coffee Leaf Disease Detection") as demo:
        gr.Markdown("# Coffee Leaf Disease Detection")
        gr.Markdown(
            "Upload an image or use your webcam to detect diseases in coffee leaves. "
            "Supported classes: **Healthy**, **Rust**, **Phoma**, **Miner**."
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    sources=["upload", "webcam"],
                    type="numpy",
                    label="Input Image",
                )
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=4,
                    value=3,
                    step=1,
                    label="Number of predictions",
                )
                predict_btn = gr.Button("Predict", variant="primary")

            with gr.Column():
                output = gr.Label(
                    label="Predictions",
                    num_top_classes=4,
                )

        examples = gr.Examples(
            examples=[],
            inputs=[image_input],
        )

        predict_btn.click(
            fn=predict_image,
            inputs=[image_input, top_k_slider],
            outputs=output,
        )

        image_input.change(
            fn=predict_image,
            inputs=[image_input, top_k_slider],
            outputs=output,
        )

    demo.launch()


if __name__ == "__main__":
    main()
