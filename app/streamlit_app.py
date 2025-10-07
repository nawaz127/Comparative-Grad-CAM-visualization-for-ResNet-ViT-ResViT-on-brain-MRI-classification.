import os
import sys
import tempfile
from io import BytesIO
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image

# make src importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.resnet import ResNetClassifier
from src.models.vit import ViTClassifier
from src.models.resvit import ResViTClassifier
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["brain_glioma", "brain_tumor", "brain_cancer"]

st.set_page_config(page_title="MRI 3-Model Grad-CAM Comparison", layout="wide")
st.title("🧠 MRI Classification & Grad-CAM Comparison")

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def preprocess_pil(img: Image.Image, size: int = 224):
    img224 = img.convert("RGB").resize((size, size))
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])
    x = tf(img224).unsqueeze(0)
    return img224, x


@st.cache_resource
def load_model(model_name: str, ckpt_path: str, num_classes: int):
    if model_name == "resnet":
        model = ResNetClassifier(num_classes=num_classes)
        target_layers = [model.backbone.layer4[-1]]
    elif model_name == "vit":
        model = ViTClassifier(num_classes=num_classes)
        target_layers = [model.backbone.vit_last]
    else:
        model = ResViTClassifier(num_classes=num_classes)
        target_layers = [model.backbone.res4]

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval().to(DEVICE)
    return model, target_layers


def predict_and_cam(model, x, target_layers, img224):
    with torch.no_grad():
        logits = model(x.to(DEVICE))
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_label = CLASSES[pred_idx]
    conf = probs[pred_idx]

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    grayscale_cam = cam(input_tensor=x)[0]
    rgb_img = np.array(img224).astype(np.float32) / 255.0
    cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return pred_label, conf, cam_img


def to_png_bytes(np_img_uint8: np.ndarray) -> bytes:
    im = Image.fromarray(np_img_uint8.astype(np.uint8))
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

# -------------------------------------------------
# UI
# -------------------------------------------------
st.subheader("🔧 Select checkpoints")

col1, col2, col3 = st.columns(3)

ckpt_resnet = col1.text_input(
    "ResNet checkpoint path",
    value="experiments/exp01_resnet/best.pt",
    help="Type or paste the full path to your trained ResNet model (.pt)"
)
ckpt_vit = col2.text_input(
    "ViT checkpoint path",
    value="experiments/exp02_vit/best.pt",
    help="Type or paste the full path to your trained ViT model (.pt)"
)
ckpt_resvit = col3.text_input(
    "ResViT checkpoint path",
    value="experiments/exp03_resvit/best.pt",
    help="Type or paste the full path to your trained ResViT model (.pt)"
)

# Optional browse buttons (local path)
st.caption("💡 You can manually edit the paths above or drag-and-drop .pt files below:")

resnet_file = st.file_uploader("Upload ResNet model", type=["pt"], key="resnet_file")
if resnet_file:
    temp_resnet = os.path.join(tempfile.gettempdir(), resnet_file.name)
    with open(temp_resnet, "wb") as f:
        f.write(resnet_file.read())
    ckpt_resnet = temp_resnet
    st.success(f"✅ Loaded custom ResNet checkpoint: {ckpt_resnet}")

vit_file = st.file_uploader("Upload ViT model", type=["pt"], key="vit_file")
if vit_file:
    temp_vit = os.path.join(tempfile.gettempdir(), vit_file.name)
    with open(temp_vit, "wb") as f:
        f.write(vit_file.read())
    ckpt_vit = temp_vit
    st.success(f"✅ Loaded custom ViT checkpoint: {ckpt_vit}")

resvit_file = st.file_uploader("Upload ResViT model", type=["pt"], key="resvit_file")
if resvit_file:
    temp_resvit = os.path.join(tempfile.gettempdir(), resvit_file.name)
    with open(temp_resvit, "wb") as f:
        f.write(resvit_file.read())
    ckpt_resvit = temp_resvit
    st.success(f"✅ Loaded custom ResViT checkpoint: {ckpt_resvit}")


uploaded = st.file_uploader("Upload an MRI image", type=["png", "jpg", "jpeg"])
if uploaded:
    img = Image.open(uploaded)
    img224, x = preprocess_pil(img, 224)

    st.write("Running models… please wait ⏳")

    results = []
    for name, ckpt, key in [
        ("ResNet", ckpt_resnet, "resnet"),
        ("ViT", ckpt_vit, "vit"),
        ("ResViT", ckpt_resvit, "resvit")
    ]:
        if not os.path.exists(ckpt):
            st.warning(f"⚠️ {name} checkpoint not found at: {ckpt}")
            continue
        model, target = load_model(key, ckpt, len(CLASSES))
        pred, conf, cam = predict_and_cam(model, x, target, img224)
        results.append((name, pred, conf, cam))

    # Show all CAMs side by side
    if results:
        cols = st.columns(len(results))
        for c, (name, pred, conf, cam) in zip(cols, results):
            c.image(cam, caption=f"{name}: {pred} ({conf*100:.1f}%)", use_container_width=True)
        # Combined download
        fig = np.hstack([r[3] for r in results])
        st.download_button(
            "📥 Download Combined Grad-CAM Panel",
            data=to_png_bytes(fig),
            file_name="MRI_GradCAM_Comparison.png",
            mime="image/png"
        )
else:
    st.info("Upload an MRI image to begin.")
