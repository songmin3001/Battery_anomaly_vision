"""
gradcam.py
Grad-CAM 히트맵 시각화
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from pathlib import Path

from dataset import VAL_TRANSFORM, CLASS_NAMES


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or model.features[-1]
        self._gradients = None
        self._activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, img_tensor, target_class=None):
        self.model.zero_grad()
        img_tensor.requires_grad_(True)

        logits = self.model(img_tensor)
        probs = torch.softmax(logits, dim=-1).squeeze()
        pred_class = probs.argmax().item()

        if target_class is None:
            target_class = pred_class

        logits[0, target_class].backward()

        weights = self._gradients.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam, pred_class, probs.detach().cpu().numpy()


def overlay_heatmap(img_pil, heatmap, alpha=0.45):
    img_bgr = cv2.cvtColor(np.array(img_pil.resize((224, 224))), cv2.COLOR_RGB2BGR)
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)


def visualize_single(model, img_path, save_path=None, device="cpu"):
    img_pil = Image.open(img_path).convert("RGB")
    img_tensor = VAL_TRANSFORM(img_pil).unsqueeze(0).to(device)

    gradcam = GradCAM(model)
    heatmap, pred_class, probs = gradcam.generate(img_tensor)
    overlay = overlay_heatmap(img_pil, heatmap)

    print(f"[예측] {CLASS_NAMES[pred_class]}")
    for i, (name, p) in enumerate(zip(CLASS_NAMES, probs)):
        print(f"  {name:<12} {p:.4f}  {'█' * int(p * 30)}")
    print(f"\n[이상 스코어] {1.0 - probs[0]:.4f}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, overlay)

    return overlay, pred_class, 1.0 - probs[0]


def batch_visualize(model, image_paths, save_dir, device="cpu"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for img_path in image_paths:
        name = Path(img_path).stem
        _, pred_class, anomaly_score = visualize_single(
            model, img_path, str(save_dir / f"{name}_gradcam.jpg"), device
        )
        results.append({"image": img_path, "pred_class": CLASS_NAMES[pred_class],
                        "anomaly_score": float(anomaly_score)})
    return results


if __name__ == "__main__":
    import argparse
    from model import build_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--save", default="gradcam_result.jpg")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(device)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    visualize_single(model, args.image, args.save, device)
    