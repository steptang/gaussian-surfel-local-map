"""SigLIP2 per-region embedding preprocessing.

For each region produced by sam3_masks.py, run masked-then-cropped encoding
through SigLIP2's vision encoder and store the pooled, text-aligned
image_embed. Result is loaded at training time as supervision target for
each surfel's K=32 feature (after a learned 32 -> K_target projection).

Per image, writes:
  <output_dir>/<image_stem>_embeds.npy  -- (R+1, K_target) float16
                                            row 0 = zeros (background)
                                            rows 1..R = pooled SigLIP2 embeds

K_target depends on the variant chosen (see preprocess/README.md). It must
match `train.py --K_target` and is the output dim of SigLIP2's pooled head.
"""

import argparse
import os
from glob import glob

import numpy as np
import torch
from PIL import Image


# Encoder-side input geometry (square). 512 for the patch16-512 variants
# we use; SigLIP2 documentation lists the supported sizes per checkpoint.
ENCODER_INPUT_SIZE = 512
PADDING_PIXELS = 8                    # padding around region bbox
MIN_REGION_PIXELS = 64                # skip tiny regions (also filtered upstream)
BATCH_SIZE = 32


def list_images(input_dir, exts=(".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")):
    files = []
    for e in exts:
        files.extend(glob(os.path.join(input_dir, f"*{e}")))
    files.sort()
    return files


def masked_crop(image_np, region_mask, pad=PADDING_PIXELS, fill_value=0):
    """Mask non-region pixels to fill_value, crop to bbox + pad, return PIL.

    image_np:    (H, W, 3) uint8
    region_mask: (H, W) bool
    """
    H, W = region_mask.shape
    ys, xs = np.where(region_mask)
    if ys.size < MIN_REGION_PIXELS:
        return None
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, H)
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, W)

    crop_mask = region_mask[y0:y1, x0:x1]                 # (h, w)
    crop_img = image_np[y0:y1, x0:x1].copy()              # (h, w, 3)
    crop_img[~crop_mask] = fill_value
    return Image.fromarray(crop_img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="dir of training images")
    ap.add_argument("--regions_dir", required=True, help="dir with *_regions.png from sam3_masks.py")
    ap.add_argument("--output_dir", required=True, help="dir for *_embeds.npy")
    ap.add_argument("--variant", default="google/siglip2-base-patch16-512")
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from transformers import AutoModel, AutoProcessor
    print(f"loading {args.variant}...")
    model = AutoModel.from_pretrained(args.variant, torch_dtype=torch.float16).cuda().eval()
    processor = AutoProcessor.from_pretrained(args.variant)

    # K_target = output dim of model.get_image_features. Probe once.
    with torch.no_grad():
        probe = Image.new("RGB", (ENCODER_INPUT_SIZE, ENCODER_INPUT_SIZE), (0, 0, 0))
        probe_in = processor(images=[probe], return_tensors="pt").to(model.device)
        K_target = int(model.get_image_features(**probe_in).shape[-1])
    print(f"K_target = {K_target}")

    images = list_images(args.input_dir)
    if not images:
        raise SystemExit(f"no images found in {args.input_dir}")

    for path in images:
        stem = os.path.splitext(os.path.basename(path))[0]
        out_npy = os.path.join(args.output_dir, f"{stem}_embeds.npy")
        if os.path.exists(out_npy) and not args.overwrite:
            print(f"skip {stem} (exists)")
            continue

        regions_path = os.path.join(args.regions_dir, f"{stem}_regions.png")
        if not os.path.exists(regions_path):
            print(f"  {stem}: no regions file, skipping")
            continue

        image_np = np.array(Image.open(path).convert("RGB"))
        region_map = np.array(Image.open(regions_path))            # uint16
        R = int(region_map.max())
        embeds = np.zeros((R + 1, K_target), dtype=np.float16)     # row 0 = bg

        # Build masked-then-cropped images, batch through the encoder.
        crops = []
        ids = []
        for rid in range(1, R + 1):
            crop = masked_crop(image_np, region_map == rid)
            if crop is None:
                continue
            crops.append(crop)
            ids.append(rid)

        for start in range(0, len(crops), args.batch_size):
            batch = crops[start:start + args.batch_size]
            batch_ids = ids[start:start + args.batch_size]
            inputs = processor(images=batch, return_tensors="pt").to(model.device)
            with torch.no_grad():
                feats = model.get_image_features(**inputs)         # (B, K_target)
            feats = feats.float().cpu().numpy().astype(np.float16)
            for i, rid in enumerate(batch_ids):
                embeds[rid] = feats[i]

        np.save(out_npy, embeds)
        print(f"  {stem}: {len(crops)}/{R} regions -> {out_npy}")


if __name__ == "__main__":
    main()
