"""Brain tumor MRI classifier with Grad-CAM — Gradio app."""
import json
import os
from pathlib import Path

import gradio as gr
import keras
import numpy as np
from PIL import Image

from gradcam import (CLASS_NAMES, build_grad_model, make_gradcam_heatmap,
                     overlay_heatmap, preprocess)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "brain_tumor_vgg16_final.keras"))
META_PATH = Path("metadata.json")

model = keras.models.load_model(MODEL_PATH, compile=False)
grad_model = build_grad_model(model)

class_names = CLASS_NAMES
if META_PATH.exists():  # keep class order in sync with training
    class_names = json.loads(META_PATH.read_text())["class_names"]

DISCLAIMER = (
    "⚠️ **Research/educational demo only. Not a medical device and not for diagnosis.** "
    "Grad-CAM highlights regions that influenced the model; it does not prove the model "
    "is looking at the tumor. Always consult a qualified clinician."
)


def predict(image, alpha, target_class):
    if image is None:
        raise gr.Error("Please upload an MRI image.")
    x, rgb = preprocess(Image.fromarray(image))
    idx = None if target_class == "Predicted class" else class_names.index(target_class)
    heatmap, probs, used = make_gradcam_heatmap(grad_model, x, idx)
    overlay = overlay_heatmap(rgb, heatmap, alpha=alpha, out_size=(480, 480))
    label = {c: float(p) for c, p in zip(class_names, probs)}
    return label, overlay


with gr.Blocks(title="Brain Tumor MRI Classifier + Grad-CAM") as demo:
    gr.Markdown("# 🧠 Brain Tumor MRI Classifier with Grad-CAM")
    gr.Markdown(DISCLAIMER)
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="MRI image")
            target = gr.Dropdown(["Predicted class"] + class_names, value="Predicted class",
                                 label="Explain which class?")
            alpha = gr.Slider(0.1, 0.9, value=0.4, step=0.05, label="Heatmap opacity")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            out_label = gr.Label(num_top_classes=4, label="Prediction")
            out_cam = gr.Image(label="Grad-CAM overlay")
    btn.click(predict, [inp, alpha, target], [out_label, out_cam])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
