import time
import gradio as gr
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.config import MODEL_PATH, CLASSES
from ultralytics import YOLO

model = None
CLASS_COLORS = {"Healthy": "#2E7D32", "Rust": "#E65100", "Phoma": "#4A148C", "Miner": "#B71C1C"}

CUSTOM_CSS = """
.banner { padding: 1rem; background: linear-gradient(135deg, #2E7D32, #1B5E20);
  color: white; border-radius: 12px; text-align: center; margin-bottom: 0.75rem; }
.banner h1 { margin: 0; font-size: 1.6rem; }
.top1 { padding: 0.9rem 1.1rem; border-radius: 10px; border-left: 5px solid var(--c);
  background: rgba(0,0,0,0.04); margin-bottom: 0.5rem; }
.top1 .cls { font-size: 1.3rem; font-weight: 700; color: var(--c); margin: 0; }
.top1 .conf { font-size: 2rem; font-weight: 800; margin: 0.1rem 0 0 0; }
.footer { text-align: center; opacity: 0.7; font-size: 0.8rem; margin-top: 0.75rem; }
"""


def load_model():
    global model
    if model is None:
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
        model = YOLO(str(MODEL_PATH))
    return model


def predict_image(image, top_k=3):
    if image is None:
        return "<div class='top1' style='--c:#888'><p class='cls'>No image</p></div>", {}
    clf = load_model()
    t0 = time.perf_counter()
    results = clf.predict(source=image, verbose=False)
    ms = (time.perf_counter() - t0) * 1000
    probs = results[0].probs
    idxs = probs.top5[:top_k]
    confs = probs.top5conf[:top_k].cpu().numpy().tolist()
    classes = [results[0].names[i] for i in idxs]
    c1 = classes[0]
    card = f"<div class='top1' style='--c:{CLASS_COLORS.get(c1, '#2E7D32')}'><p class='cls'>{c1}</p><p class='conf' style='color:{CLASS_COLORS.get(c1, '#2E7D32')}'>{confs[0]*100:.1f}%</p></div>"
    bars = {classes[i]: round(float(confs[i]), 4) for i in range(len(classes))}
    return card + f"<div style='text-align:center;opacity:0.6;font-size:0.8rem'>inference: {ms:.0f} ms</div>", bars


def main():
    load_model()
    with gr.Blocks(title="Coffee Leaf Disease Detection") as demo:
        gr.HTML("<div class='banner'><h1>Coffee Leaf Disease Detection</h1></div>")
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(sources=["upload", "webcam"], type="numpy", label="Input Image", height=340)
                with gr.Row():
                    predict_btn = gr.Button("Predict", variant="primary", scale=3)
                    clear_btn = gr.Button("✕ Clear", scale=1)
                top_k_slider = gr.Slider(1, 4, value=3, step=1, label="Top-K predictions")
            with gr.Column():
                top1_out = gr.HTML(value="<div class='top1' style='--c:#888'><p class='cls' style='color:#888'>Awaiting input</p></div>")
                output = gr.Label(label="Upload or take a photo of coffee leaf and click the 'Predict' button", num_top_classes=4)
        gr.Markdown("<div class='footer'>Note: AI predictions may not be accurate.</div>")
        predict_btn.click(fn=predict_image, inputs=[image_input, top_k_slider], outputs=[top1_out, output], show_progress="full")
        clear_btn.click(fn=lambda: (None, "<div class='top1' style='--c:#888'><p class='cls' style='color:#888'>Awaiting input</p></div>", {}), inputs=[], outputs=[image_input, top1_out, output])
    demo.launch(theme=gr.themes.Soft(primary_hue="green"), css=CUSTOM_CSS)


if __name__ == "__main__":
    main()