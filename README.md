---
title: Brain Tumor MRI Classifier
emoji: 🧠
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# Brain Tumor MRI Classifier + Grad-CAM

VGG16 transfer-learning classifier (glioma / meningioma / notumor / pituitary) with Grad-CAM
explanations on `block5_conv3`. **Educational demo only — not for clinical use.**

## Files
| File | Purpose |
|---|---|
| `app.py` | Gradio web UI |
| `gradcam.py` | Preprocessing + Grad-CAM logic |
| `brain_tumor_vgg16_final.keras` | Your trained model (from Kaggle output) |
| `metadata.json` | Class order (from Kaggle output) |

## Run locally
```bash
pip install -r requirements.txt
python app.py        # http://localhost:7860
```

## Deploy to Hugging Face Spaces (free)
1. Create a Space at huggingface.co/new-space → SDK **Gradio**, hardware **CPU basic**.
2. Download `brain_tumor_vgg16_final.keras` and `metadata.json` from
   `/kaggle/working/brain_tumor_vgg16_output/` and put them in this folder.
3. Push (the `.gitattributes` already routes `.keras` through Git LFS):
```bash
git lfs install
git clone https://huggingface.co/spaces/<you>/<space-name>
cp -r * .gitattributes <space-name>/   # copy this folder's contents in
cd <space-name> && git add . && git commit -m "Deploy" && git push
```
The Space builds automatically and gives you a public URL.
