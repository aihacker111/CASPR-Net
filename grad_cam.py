"""
gradcam_viz.py — Grad-CAM heatmap visualization for CASPR-Net (or any CNN/ViT)
Usage:
    python grad_cam.py --image cat.jpg --model casprnet_n --checkpoint model.pth
    python grad_cam.py --image dog.jpg --model casprnet_n --checkpoint model.pth --layer norm
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from timm.models import create_model as timm_create_model
CLASSES = ["cat", "dog"]   # index 0 = cat, index 1 = dog


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Works on any layer that outputs a spatial tensor [B, C, H, W]
    or a sequence tensor [B, N, C] (ViT-style — auto-detected).

    Usage:
        cam = GradCAM(model, target_layer=model.stages[-1][-1].norm1)
        heatmap = cam(image_tensor, class_idx)   # None = argmax
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._feats: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None

        self._handles = [
            target_layer.register_forward_hook(self._save_feats),
            target_layer.register_full_backward_hook(self._save_grads),
        ]

    def _save_feats(self, _, __, output):
        self._feats = output.detach()

    def _save_grads(self, _, __, grad_output):
        self._grads = grad_output[0].detach()

    def remove(self):
        for h in self._handles:
            h.remove()

    @torch.no_grad()
    def _to_spatial(self, t: torch.Tensor) -> torch.Tensor:
        """Convert [B, N, C] token tensor to [B, C, H, W] by assuming square grid."""
        if t.dim() == 4:
            return t                              # already spatial [B, C, H, W]
        B, N, C = t.shape
        H = W = int(N ** 0.5)
        return t.permute(0, 2, 1).reshape(B, C, H, W)

    def __call__(
        self,
        x: torch.Tensor,         # [1, 3, H, W]
        class_idx: int | None = None,
    ) -> np.ndarray:
        """
        Returns heatmap as float32 numpy array [H, W] in [0, 1].
        """
        self.model.eval()
        x = x.requires_grad_(True)

        # Forward
        logits = self.model(x)
        if isinstance(logits, tuple):
            logits = logits[0]

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backward for target class
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Grad-CAM: weight channels by global-average-pooled gradients
        feats = self._to_spatial(self._feats)  # [1, C, h, w]
        grads = self._to_spatial(self._grads)  # [1, C, h, w]

        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam     = (weights * feats).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam     = F.relu(cam)

        cam = cam.squeeze().cpu().numpy().astype(np.float32)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ---------------------------------------------------------------------------
# Overlay heatmap onto image
# ---------------------------------------------------------------------------

def overlay_heatmap(
    image: np.ndarray,           # [H, W, 3] uint8 BGR
    heatmap: np.ndarray,         # [h, w] float32  0..1
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    H, W = image.shape[:2]
    heat  = cv2.resize(heatmap, (W, H))
    heat8 = np.uint8(255 * heat)
    color = cv2.applyColorMap(heat8, colormap)
    blend = cv2.addWeighted(image, 1 - alpha, color, alpha, 0)
    return blend


def save_result(
    image_path: str,
    heatmap: np.ndarray,
    class_name: str,
    score: float,
    out_dir: str = "gradcam_output",
) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.imread(image_path)
    blended = overlay_heatmap(img_bgr, heatmap)

    # Side-by-side: original | heatmap overlay
    orig_resized = cv2.resize(img_bgr, (blended.shape[1], blended.shape[0]))
    canvas = np.hstack([orig_resized, blended])

    # Label
    label = f"Pred: {class_name}  ({score * 100:.1f}%)"
    cv2.putText(canvas, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 0), 1, cv2.LINE_AA)

    stem     = Path(image_path).stem
    out_path = out / f"{stem}_gradcam.jpg"
    cv2.imwrite(str(out_path), canvas)
    print(f"Saved → {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------------
# Batch visualize over a folder
# ---------------------------------------------------------------------------

def visualize_folder(
    folder: str,
    model: nn.Module,
    target_layer: nn.Module,
    out_dir: str = "gradcam_output",
    img_size: int = 224,
    device: str = "cpu",
):
    """Run Grad-CAM on every image in a folder."""
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    cam = GradCAM(model, target_layer)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    paths = [p for p in Path(folder).iterdir() if p.suffix.lower() in exts]
    print(f"Found {len(paths)} images in '{folder}'")

    for img_path in paths:
        pil   = Image.open(img_path).convert("RGB")
        x     = tf(pil).unsqueeze(0).to(device)

        with torch.enable_grad():
            heatmap = cam(x)

        with torch.no_grad():
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs      = torch.softmax(logits, dim=1)[0]
            class_idx  = probs.argmax().item()
            class_name = CLASSES[class_idx]
            score      = probs[class_idx].item()

        save_result(str(img_path), heatmap, class_name, score, out_dir)

    cam.remove()
    print("Done.")


# ---------------------------------------------------------------------------
# Quick single-image function (for notebook / training loop)
# ---------------------------------------------------------------------------

def show_gradcam(
    image_path: str,
    model: nn.Module,
    target_layer: nn.Module,
    class_idx: int | None = None,
    img_size: int = 224,
    device: str = "cpu",
    save_path: str | None = None,
):
    """
    One-liner for a single image. Returns (blended_bgr, class_name, score).
 
    Example:
        blended, label, score = show_gradcam("cat.jpg", model, model.stages[-1][-1].norm1)
        cv2.imshow("Grad-CAM", blended)
        cv2.waitKey(0)
    """
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
 
    pil = Image.open(image_path).convert("RGB")
    x   = tf(pil).unsqueeze(0).to(device)
 
    cam = GradCAM(model, target_layer)
 
    with torch.enable_grad():
        heatmap = cam(x, class_idx)
 
    with torch.no_grad():
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs      = torch.softmax(logits, dim=1)[0]
        pred_idx   = probs.argmax().item()
        class_name = CLASSES[pred_idx]
        score      = probs[pred_idx].item()
 
    cam.remove()
 
    img_bgr = cv2.imread(image_path)
    blended = overlay_heatmap(img_bgr, heatmap)
 
    # Side-by-side: original | heatmap overlay
    canvas = np.hstack([img_bgr, blended])
    label  = f"Pred: {class_name}  ({score * 100:.1f}%)"
    cv2.putText(canvas, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 0),       1, cv2.LINE_AA)
 
    # Auto-generate save_path next to the source image if not given
    if save_path is None:
        save_path = str(Path(image_path).parent / f"{Path(image_path).stem}_gradcam.jpg")
 
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, canvas)
    print(f"Saved → {save_path}")
 
    return canvas, class_name, score


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_model(model_name: str, checkpoint: str | None, device: str) -> nn.Module:
    """Load a named CASPR-Net variant and an optional checkpoint."""
    import model as _model_pkg  # ensure timm registrations are loaded
    _ = _model_pkg

    state = None
    if checkpoint:
        raw = torch.load(checkpoint, map_location="cpu")
        state = raw.get("model") or raw.get("state_dict") or raw
        ckpt_num_classes = state["head.fc.weight"].shape[0]
        print(f"Loading variant={model_name}, ckpt_num_classes={ckpt_num_classes}")
    else:
        ckpt_num_classes = 2

    net = timm_create_model(model_name, num_classes=ckpt_num_classes)

    if state is not None:
        net.load_state_dict(state, strict=True)
        print(f"Loaded weights from {checkpoint}")

    # Replace head with 2-class head for cat/dog inference
    if ckpt_num_classes != 2:
        net.reset_classifier(2)

    return net.to(device).eval()


def _pick_layer(model: nn.Module, layer_hint: str) -> nn.Module:
    """
    Select target layer by hint string:
      'last'  → final sparse channel level of the last CASPR block
      'norm'  → final backbone norm
      'stage0'..'stage3' → last block of that stage
      or pass a dotted attr path like 'stages.2.4.attn'
    """
    if layer_hint == "last":
        return model.stages[-1][-1].channel_mixer.post.levels[-1]
    if layer_hint == "norm":
        return model.norm
    if layer_hint.startswith("stage"):
        idx = int(layer_hint.replace("stage", ""))
        return model.stages[idx][-1].channel_mixer.post.levels[-1]
    # Dotted path fallback
    parts = layer_hint.split(".")
    obj   = model
    for p in parts:
        obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
    return obj


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM visualizer for dog/cat model")
    parser.add_argument("--image",      required=True,            help="Path to image or folder")
    parser.add_argument("--model",      default="casprnet_n", help="timm model name")
    parser.add_argument("--checkpoint", default=None,             help="Path to .pth checkpoint")
    parser.add_argument("--layer",      default="last",           help="Target layer hint")
    parser.add_argument("--class_idx",  type=int, default=None,   help="Force class index (0=cat,1=dog)")
    parser.add_argument("--img_size",   type=int, default=224)
    parser.add_argument("--out_dir",    default="gradcam_output")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model  = _build_model(args.model, args.checkpoint, args.device)
    layer  = _pick_layer(model, args.layer)
    print(f"Target layer: {layer.__class__.__name__}  |  device: {args.device}")

    p = Path(args.image)
    if p.is_dir():
        visualize_folder(str(p), model, layer, args.out_dir, args.img_size, args.device)
    else:
        blended, class_name, score = show_gradcam(
            str(p), model, layer,
            class_idx=args.class_idx,
            img_size=args.img_size,
            device=args.device,
            save_path=str(Path(args.out_dir) / f"{p.stem}_gradcam.jpg"),
        )
        print(f"Prediction: {class_name}  ({score * 100:.1f}%)")


if __name__ == "__main__":
    main()
