# Preprocessing: SAM3 regions + SigLIP2 region embeddings

Two-stage offline preprocessing producing per-image `*_regions.png` (SAM3 mask
IDs) and `*_embeds.npy` (per-region SigLIP2 text-aligned embeddings) for
semantic supervision of the surfels.

The two stages run in **separate conda environments** because SAM3 and the
HuggingFace SigLIP2 model often pin incompatible torch/CUDA versions. Output
artifacts are written to `<source>/sam3/` alongside the COLMAP `images/`
directory and are loaded automatically by `scene/dataset_readers.py` if
`--sam_dir sam3` is set (default).

## Stage 1 — SAM3 regions (env A)

SAM3 has no automask "everything" mode — every prediction call needs a
text concept. The script approximates exhaustive segmentation by running
SAM3 once per concept in a list (defaulting to COCO-80) and unioning the
results, then deduplicating overlapping detections by IoU. ~80 SAM3
forward passes per image at the default settings.

```bash
conda activate sam3
hf auth login                              # SAM3 weights are gated on HF
python preprocess/sam3_masks.py \
    --input_dir   <scene>/images \
    --output_dir  <scene>/sam3
```

Override the concept list with `--concept_list "chair,desk,monitor,..."`
or `--concepts path/to/concepts.txt` (one per line, `#` for comments).
Other knobs:
- `--confidence 0.5` — SAM3 detection threshold
- `--iou_dedup 0.7` — drop a mask if its IoU with a higher-scored mask
  exceeds this; relevant when two concepts both fire on the same instance
  (e.g., "chair" and "furniture")
- `--overwrite` — re-run on images that already have a regions.png

Per image, produces:
- `<image_stem>_regions.png` — `(H, W) uint16`, region IDs. 0 = background.
  Stored at the original image resolution. Downsampled with
  nearest-neighbour at training time to match `--resolution`.
- `<image_stem>_meta.json` — per-region area, SAM3 confidence, source concept,
  and the concept list used (so you can audit later why a region got that ID).

## Stage 2 — SigLIP2 region embeddings (env B)

```bash
conda activate siglip2
python preprocess/siglip2_embeddings.py \
    --input_dir   <scene>/images \
    --regions_dir <scene>/sam3 \
    --output_dir  <scene>/sam3 \
    --variant     google/siglip2-base-patch16-512
```

Per image, produces:
- `<image_stem>_embeds.npy` — `(R+1, K_target) float16`. Row 0 = zeros for
  background; rows 1..R = pooled SigLIP2 image embedding for region i,
  obtained via masked-then-cropped encoding (mask non-region pixels to zero,
  crop to bbox + padding, resize to encoder input, encode, take pooled
  `image_embed`).

K_target depends on the variant:
- `siglip2-base-patch16-512`  → 768
- `siglip2-large-patch16-512` → 1024
- `siglip2-so400m-patch16-512` → 1152

Make sure the value passed to `train.py --K_target` matches.
