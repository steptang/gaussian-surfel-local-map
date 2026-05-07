"""SAM3 automask preprocessing: image -> per-pixel region ID map.

Run in an env with SAM3 + torch installed. See preprocess/README.md.

Per image, writes:
  <output_dir>/<image_stem>_regions.png  -- (H, W) uint16, region IDs (0=bg, 1..R)
  <output_dir>/<image_stem>_meta.json    -- per-region area + SAM3 score

Region IDs are dense (1..R, no gaps). When SAM3 produces overlapping masks the
larger-area mask wins per pixel, then smaller masks are layered above (this
matches LangSplat / Gaussian Grouping conventions).
"""

import argparse
import json
import os
from glob import glob

import numpy as np
from PIL import Image


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def list_images(input_dir):
    files = []
    for ext in IMAGE_EXTS:
        files.extend(glob(os.path.join(input_dir, f"*{ext}")))
    files.sort()
    return files


def build_region_map(masks, H, W, min_area=64):
    """Stack a list of SAM3 binary masks into a single uint16 ID map.

    Sorted small-to-large so larger masks are written first and small
    masks overwrite (i.e., small parts are preferred over the big object
    they sit on). Empty pixels remain 0.
    """
    keep = []
    for m in masks:
        seg = m.get("segmentation") if isinstance(m, dict) else m
        score = float(m.get("stability_score", m.get("predicted_iou", 1.0))) if isinstance(m, dict) else 1.0
        if seg is None:
            continue
        seg = np.asarray(seg, dtype=bool)
        area = int(seg.sum())
        if area < min_area:
            continue
        keep.append((area, score, seg))

    # Largest first so small masks win the per-pixel write order.
    keep.sort(key=lambda t: -t[0])

    region_map = np.zeros((H, W), dtype=np.uint16)
    meta = []
    for i, (area, score, seg) in enumerate(keep, start=1):
        if i >= np.iinfo(np.uint16).max:
            print(f"  warn: >65k regions, dropping the rest")
            break
        region_map[seg] = i
        meta.append({"id": int(i), "area": int(area), "score": float(score)})
    return region_map, meta


def load_sam3_automask():
    """Construct a SAM3 automatic mask generator. Import is lazy so this
    module can be imported in environments without SAM3 (e.g., for testing).

    Adjust the import / model loading below to match your SAM3 install.
    The returned object must have a .generate(np.ndarray HxWx3 uint8) ->
    list[dict|ndarray] interface (compatible with SAM2's
    SAM2AutomaticMaskGenerator).
    """
    try:
        from sam3 import SAM3AutomaticMaskGenerator, build_sam3
    except ImportError:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator as SAM3AutomaticMaskGenerator
        from sam2.build_sam import build_sam2 as build_sam3
    import torch
    ckpt = os.environ.get("SAM3_CHECKPOINT", "checkpoints/sam3_hiera_large.pt")
    cfg = os.environ.get("SAM3_CONFIG", "sam3_hiera_l.yaml")
    sam = build_sam3(cfg, ckpt, device="cuda")
    return SAM3AutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.85,
        min_mask_region_area=64,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="dir of training images")
    ap.add_argument("--output_dir", required=True, help="dir for *_regions.png")
    ap.add_argument("--min_area", type=int, default=64, help="drop masks below this pixel area")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    images = list_images(args.input_dir)
    if not images:
        raise SystemExit(f"no images found in {args.input_dir}")

    print(f"loading SAM3...")
    mask_gen = load_sam3_automask()

    for path in images:
        stem = os.path.splitext(os.path.basename(path))[0]
        out_png = os.path.join(args.output_dir, f"{stem}_regions.png")
        out_json = os.path.join(args.output_dir, f"{stem}_meta.json")
        if os.path.exists(out_png) and not args.overwrite:
            print(f"skip {stem} (exists)")
            continue

        img = np.array(Image.open(path).convert("RGB"))
        H, W = img.shape[:2]

        masks = mask_gen.generate(img)
        region_map, meta = build_region_map(masks, H, W, min_area=args.min_area)

        Image.fromarray(region_map, mode="I;16").save(out_png)
        with open(out_json, "w") as f:
            json.dump({"H": H, "W": W, "R": len(meta), "regions": meta}, f)
        print(f"  {stem}: {len(meta)} regions -> {out_png}")


if __name__ == "__main__":
    main()
