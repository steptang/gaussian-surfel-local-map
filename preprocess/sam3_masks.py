"""SAM3 region-segmentation preprocessing: image -> per-pixel region ID map.

SAM3 is concept-prompted (no automask "everything" mode). To approximate
exhaustive segmentation, we run SAM3 with `set_text_prompt` once per
concept in a fixed list; SAM3's exhaustive mode returns all instances of
the named concept per image. We union the returned masks across concepts,
deduplicate near-duplicates by IoU, and write a single uint16 region map.

Per image, writes:
  <output_dir>/<image_stem>_regions.png  -- (H, W) uint16, region IDs (0=bg, 1..R)
  <output_dir>/<image_stem>_meta.json    -- per-region area, score, source concept

Concept list: COCO-80 by default. Override via --concepts <file> (one per
line) or --concept_list "cls1,cls2,...". For indoor robot navigation, COCO-80
covers most objects; for finer-grained scenes consider LVIS (~1203 classes,
~10x slower) or a hand-curated list.

Prerequisites:
  pip install -e <path-to-sam3-repo>
  hf auth login                      # SAM3 weights are gated
"""

import argparse
import json
import os
from glob import glob

import numpy as np
import torch
from PIL import Image


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# COCO 80 categories. Reasonable default for natural / indoor scenes.
COCO_80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


def list_images(input_dir):
    files = []
    for ext in IMAGE_EXTS:
        files.extend(glob(os.path.join(input_dir, f"*{ext}")))
    files.sort()
    return files


def load_concepts(args):
    """Resolve the concept list from CLI flags. Precedence: file > literal > default."""
    if args.concepts:
        with open(args.concepts) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    if args.concept_list:
        return [c.strip() for c in args.concept_list.split(",") if c.strip()]
    return COCO_80


def build_sam3():
    """Construct SAM3 image model + processor. Lazy import so this module is
    importable in environments without sam3 installed."""
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    import sam3
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")
    model = build_sam3_image_model(bpe_path=bpe_path)
    return model, Sam3Processor


def _to_bool_mask(mask, H, W):
    """Coerce SAM3's per-instance mask to a (H, W) bool ndarray.

    SAM3 returns masks of shape (1, H, W) -- there's a leading channel dim
    from the `interpolate(out_masks.unsqueeze(1), (img_h, img_w))` inside
    Sam3Processor._forward_grounding. PIL.Image.fromarray treats anything
    with shape[0] != H_image as a multi-channel image, so we have to
    squeeze the leading singleton(s) before any 2D processing.

    Under torch.autocast bf16, mask logits may also come through as a
    bfloat16 tensor; numpy doesn't natively support bf16 so we cast first.
    """
    if isinstance(mask, torch.Tensor):
        if mask.dtype in (torch.bfloat16, torch.float16):
            mask = mask.float()
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    # Strip leading singleton dims: (1, H, W) or (1, 1, H, W) -> (H, W).
    while mask.ndim > 2 and mask.shape[0] == 1:
        mask = mask[0]
    # SAM3 typically returns bool already (state["masks"] = out_masks > 0.5);
    # threshold defensively in case logits leak through.
    if mask.dtype != bool:
        mask = mask > 0.5
    if mask.ndim != 2:
        # Genuinely degenerate -- caller should drop this prediction.
        return None
    if mask.shape != (H, W):
        # Resize at full uint8 range so PIL picks 'L' mode unambiguously.
        mask = np.array(
            Image.fromarray((mask.astype(np.uint8) * 255), mode='L').resize((W, H), Image.NEAREST)
        ).astype(bool)
    return mask


def collect_concept_masks(processor, inference_state, concepts, image_size, confidence):
    """Run SAM3 once per concept, return list of (mask, score, concept_name).

    Each entry in the returned list represents a single instance. SAM3's
    exhaustive mode returns multiple instances per concept when present.
    """
    H, W = image_size
    out = []
    for c in concepts:
        processor.reset_all_prompts(inference_state)
        result = processor.set_text_prompt(state=inference_state, prompt=c)
        # set_text_prompt returns a dict-like state with masks/boxes/scores;
        # accept either return-as-dict or stored-on-state APIs.
        d = result if isinstance(result, dict) else inference_state
        masks = d.get("masks", [])
        scores = d.get("scores", [1.0] * len(masks))
        for m, s in zip(masks, scores):
            score = float(s.item() if isinstance(s, torch.Tensor) else s)
            if score < confidence:
                continue
            bool_mask = _to_bool_mask(m, H, W)
            if bool_mask is None or bool_mask.sum() < 64:
                continue
            out.append((bool_mask, score, c))
    return out


def dedup_masks(masks, iou_threshold=0.7):
    """Greedy IoU dedup: sort by score desc, keep masks whose IoU with all
    previously-kept masks is below the threshold. Avoids counting the same
    region twice when overlapping concepts (e.g., 'chair' and 'furniture')
    fire on the same instance.
    """
    if not masks:
        return []
    masks_sorted = sorted(masks, key=lambda t: -t[1])  # by score desc
    kept = []
    for m, s, c in masks_sorted:
        ok = True
        for km, _, _ in kept:
            inter = np.logical_and(m, km).sum()
            union = np.logical_or(m, km).sum()
            if union > 0 and inter / union > iou_threshold:
                ok = False
                break
        if ok:
            kept.append((m, s, c))
    return kept


def build_region_map(masks, H, W):
    """Stack masks into uint16 region IDs. Larger area first so smaller
    parts overwrite the big object they sit on (e.g., 'wheel' on top of
    'car'). 0 stays as background.
    """
    masks_sorted = sorted(masks, key=lambda t: -t[0].sum())  # largest first
    region_map = np.zeros((H, W), dtype=np.uint16)
    meta = []
    for i, (m, s, c) in enumerate(masks_sorted, start=1):
        if i >= np.iinfo(np.uint16).max:
            print(f"  warn: >65k regions, dropping the rest")
            break
        region_map[m] = i
        meta.append({"id": i, "area": int(m.sum()), "score": s, "concept": c})
    return region_map, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="dir of training images")
    ap.add_argument("--output_dir", required=True, help="dir for *_regions.png")
    ap.add_argument("--concepts", default=None, help="path to concept list (one per line)")
    ap.add_argument("--concept_list", default=None, help="comma-separated concepts; overrides --concepts")
    ap.add_argument("--confidence", type=float, default=0.5, help="SAM3 detection threshold")
    ap.add_argument("--iou_dedup", type=float, default=0.7, help="IoU threshold for cross-concept dedup")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    images = list_images(args.input_dir)
    if not images:
        raise SystemExit(f"no images found in {args.input_dir}")

    concepts = load_concepts(args)
    print(f"using {len(concepts)} concepts (first 5: {concepts[:5]}{'...' if len(concepts)>5 else ''})")

    # SAM3's ViT runs in bf16 -- some internal paths (flash-attn-3 in
    # particular) emit bf16 activations while Linear weights stay fp32,
    # which mismatches without autocast. The example notebooks enter the
    # autocast permanently; we do the same here so every model call inside
    # the per-image loop is covered.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    print("loading SAM3...")
    model, ProcessorCls = build_sam3()
    processor = ProcessorCls(model, confidence_threshold=args.confidence)

    for path in images:
        stem = os.path.splitext(os.path.basename(path))[0]
        out_png = os.path.join(args.output_dir, f"{stem}_regions.png")
        out_json = os.path.join(args.output_dir, f"{stem}_meta.json")
        if os.path.exists(out_png) and not args.overwrite:
            print(f"skip {stem} (exists)")
            continue

        image = Image.open(path).convert("RGB")
        W, H = image.size
        inference_state = processor.set_image(image)

        raw = collect_concept_masks(processor, inference_state, concepts,
                                     (H, W), confidence=args.confidence)
        deduped = dedup_masks(raw, iou_threshold=args.iou_dedup)
        region_map, meta = build_region_map(deduped, H, W)

        Image.fromarray(region_map, mode="I;16").save(out_png)
        with open(out_json, "w") as f:
            json.dump({"H": H, "W": W, "R": len(meta),
                       "concepts": concepts, "regions": meta}, f)
        coverage = (region_map > 0).sum() / (H * W)
        print(f"  {stem}: {len(meta)} regions, {coverage:.1%} pixel coverage -> {out_png}")


if __name__ == "__main__":
    main()
