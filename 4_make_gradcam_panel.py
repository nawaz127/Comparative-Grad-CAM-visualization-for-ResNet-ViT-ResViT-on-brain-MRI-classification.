import argparse, glob, torch
from PIL import Image
import numpy as np, cv2, os
import random
from src.utils.gradcam_utils import get_cam, overlay_cam
from src.models.resnet import build_resnet
from src.models.vit import build_vit
from src.models.resvit import ResViTClassifier
import torchvision.transforms as T


# ------------------------------
# Load model + pick correct Grad-CAM layer
# ------------------------------
def load_model(name, num_classes, ckpt):
    name = name.lower()
    if name == 'resnet':
        m = build_resnet(num_classes, pretrained=False)
        target_layers = [m.layer4[-1]]
        is_vit = False

    elif name == 'vit':
        m = build_vit(num_classes, pretrained=False)
        if hasattr(m, 'encoder'):
            target_layers = [m.encoder.layers[-1]]
        else:
            target_layers = [m.transformer.encoder.layers[-1]]
        is_vit = True

    elif name == 'resvit':
        m = ResViTClassifier(num_classes=num_classes, pretrained=False)
        target_layers = [m.backbone.res4]  # final CNN block
        is_vit = False

    else:
        raise ValueError(f"Unknown model: {name}")

    sd = torch.load(ckpt, map_location='cpu')
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m, target_layers, is_vit


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--model', required=True, choices=['resnet', 'vit', 'resvit'])
    ap.add_argument('--images_glob', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--num_images', type=int, default=10)
    args = ap.parse_args()

    imgs = sorted(glob.glob(args.images_glob))
    assert imgs, f"No images found at {args.images_glob}"

    # Select random subset if too many images
    if len(imgs) > args.num_images:
        imgs = random.sample(imgs, args.num_images)

    # Always use fixed number of classes for MRI dataset
    num_classes = 3  # brain_glioma, brain_menin, brain_tumor
    print(f"🧠 Using {num_classes} classes for model loading")

    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])

    model, target_layers, is_vit = load_model(args.model, num_classes, args.ckpt)
    cam = get_cam(model, target_layers, is_vit=is_vit)

    overlays = []
    for p in imgs:
        pil = Image.open(p).convert("RGB").resize((224, 224))
        x = tf(pil).unsqueeze(0)
        grayscale_cam = cam(input_tensor=x)[0]
        overlay = overlay_cam(np.array(pil), grayscale_cam)
        overlays.append(overlay)

    # Combine overlays into a grid
    cols = min(5, len(overlays))
    rows = []
    for i in range(0, len(overlays), cols):
        row = np.hstack(overlays[i:i + cols])
        rows.append(row)
    panel = np.vstack(rows)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    print(f"✅ Saved Grad-CAM grid to: {args.out}")


if __name__ == "__main__":
    main()
