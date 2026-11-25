"""
MedEViT-SAM: Single-Image Promptable Segmentation Demo.

Runs promptable segmentation on an input 2D medical slice using a bounding box prompt.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch

ROOT_DIR = Path(__file__).resolve().parent
for p in [str(ROOT_DIR), str(ROOT_DIR / 'efficientvit')]:
    if p not in sys.path:
        sys.path.insert(0, p)

from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="MedEViT-SAM Single-Image Inference Demo")
    parser.add_argument("--image", type=str, required=True, help="Path to input 2D image (.png, .jpg, or .npz slice)")
    parser.add_argument("--box", type=str, required=True, help="Bounding box coordinates 'xmin,ymin,xmax,ymax' (e.g. '120,150,380,410')")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to fine-tuned MedEViT-SAM weights (.pth)")
    parser.add_argument("--image_size", type=int, default=512, help="Input resolution (default: 512)")
    parser.add_argument("--output", type=str, default="prediction_overlay.png", help="Path to save output visualization")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    return parser.parse_args()


def load_image(image_path: str, target_size: int = 512):
    path = Path(image_path)
    if path.suffix == ".npz":
        data = np.load(path)
        img = data["image"]
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
    else:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Could not load image from {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    orig_h, orig_w = img.shape[:2]

    # Resize preserving aspect ratio with zero padding
    scale = target_size / max(orig_h, orig_w)
    new_h, new_w = int(round(orig_h * scale)), int(round(orig_w * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left

    padded = np.pad(resized, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant", constant_values=0)
    meta = {"scale": scale, "pad_top": pad_top, "pad_left": pad_left, "orig_h": orig_h, "orig_w": orig_w}
    return padded, meta


def main():
    args = parse_args()

    print(f"Loading MedEViT-SAM-L0 on {args.device}...")
    model = create_efficientvit_sam_model(name="efficientvit-sam-l0", pretrained=True)

    if args.checkpoint:
        print(f"Loading fine-tuned weights from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)

    model.to(args.device)
    model.eval()

    predictor = EfficientViTSamPredictor(model)

    print(f"Loading input: {args.image}")
    padded_img, meta = load_image(args.image, target_size=args.image_size)

    predictor.set_image(padded_img)

    # Parse bounding box
    coords = [float(x.strip()) for x in args.box.split(",")]
    if len(coords) != 4:
        raise ValueError("Bounding box must be 4 numbers: xmin,ymin,xmax,ymax")
    xmin, ymin, xmax, ymax = coords

    # Scale box coordinates into padded 512x512 space
    box_scaled = np.array([
        xmin * meta["scale"] + meta["pad_left"],
        ymin * meta["scale"] + meta["pad_top"],
        xmax * meta["scale"] + meta["pad_left"],
        ymax * meta["scale"] + meta["pad_top"]
    ])

    print(f"Running inference with box prompt: {box_scaled}...")
    masks, iou_predictions, _ = predictor.predict(
        box=box_scaled[None, :],
        multimask_output=False
    )

    pred_mask = masks[0]  # Shape: (512, 512)

    # Create visualization
    overlay = padded_img.copy()
    overlay[pred_mask > 0] = overlay[pred_mask > 0] * 0.4 + np.array([46, 204, 113]) * 0.6  # Emerald green

    # Draw bounding box
    cv2.rectangle(
        overlay,
        (int(box_scaled[0]), int(box_scaled[1])),
        (int(box_scaled[2]), int(box_scaled[3])),
        (255, 191, 0),  # Amber gold
        2
    )

    cv2.imwrite(args.output, cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"[Done] Predicted mask and prompt overlay saved to: {args.output}")


if __name__ == "__main__":
    main()
