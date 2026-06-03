"""Open-vocabulary text query against a trained semantic surfel model.

Encodes a free-form text query through SigLIP2's text encoder, then renders
the trained model from a chosen training viewpoint and computes per-pixel
cosine similarity between the projected rendered surfel features and the
text embedding. Saves the heatmap as a PNG overlay.

Example:
    python scripts/text_query.py \
        --model_path output/scan105 \
        --source_path data/dtu/scan105 \
        --query "chair" \
        --view_idx 12 \
        --variant google/siglip2-base-patch16-512 \
        --out heatmap.png

The viewpoint is the view_idx-th training camera. Use --view_idx -1 to scan
through every viewpoint and average the heatmaps (slow).
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Allow running from repo root without installing as a package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import Scene, GaussianModel
from scene.gaussian_model import SEMANTIC_DIM
from utils.semantic_loss import SemanticHead


def _unwrap_pooled(out):
    """Coerce a SigLIP2 (image|text)-features call result to a (B, K_target) tensor.

    Matches the helper in preprocess/siglip2_embeddings.py. Different
    transformers releases have shipped Siglip2Model.get_*_features
    returning either a torch.Tensor or a BaseModelOutputWithPooling; we
    accept both. The same unwrap order works for image and text since
    BaseModelOutputWithPooling exposes the same attribute names on either
    tower.
    """
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "text_embeds") and out.text_embeds is not None:
        return out.text_embeds
    if hasattr(out, "image_embeds") and out.image_embeds is not None:
        return out.image_embeds
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state.mean(dim=1)
    raise RuntimeError(f"unexpected SigLIP2 output type: {type(out).__name__}")


def encode_text_query(query: str, variant: str):
    """Returns (1, K_target) text embedding from SigLIP2's text encoder."""
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(variant, torch_dtype=torch.float32).cuda().eval()
    processor = AutoProcessor.from_pretrained(variant)
    inputs = processor(text=[query], padding="max_length", return_tensors="pt").to("cuda")
    with torch.no_grad():
        text_emb = _unwrap_pooled(model.get_text_features(**inputs))   # (1, K_target)
    return text_emb


def overlay_heatmap(rgb_chw: np.ndarray, heat_hw: np.ndarray, alpha: float = 0.5):
    """Compose a turbo-coloured heatmap onto the rendered image."""
    import matplotlib.cm as cm
    h = (heat_hw - heat_hw.min()) / (heat_hw.max() - heat_hw.min() + 1e-8)
    colored = cm.get_cmap("turbo")(h)[..., :3]              # (H, W, 3)
    rgb = rgb_chw.transpose(1, 2, 0)
    out = np.clip((1 - alpha) * rgb + alpha * colored, 0, 1)
    return (out * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--query", required=True)
    parser.add_argument("--variant", default="google/siglip2-base-patch16-512")
    parser.add_argument("--view_idx", type=int, default=0)
    parser.add_argument("--iteration", type=int, default=-1, help="-1 = latest")
    parser.add_argument("--out", default="heatmap.png")
    parser.add_argument("--checkpoint", default=None,
                        help="path to chkpnt*.pth produced by train.py; required for the SemanticHead")

    # Mirror render.py: load training settings (resolution, sh_degree,
    # K_target, data_device, ...) from <model_path>/cfg_args so the user
    # only needs to pass -s, -m, --query, etc. Without this, every field
    # ModelParams declares with sentinel=True comes through as None and
    # downstream code (camera_utils, gaussian_model, semantic_loss) blows
    # up with confusing NoneType errors.
    args = get_combined_args(parser)

    dataset = lp.extract(args)
    pipe = pp.extract(args)

    # Build the scene + load the trained surfel model from PLY.
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    # Restore the SemanticHead from checkpoint. The PLY by itself doesn't
    # store it because it isn't per-surfel. Without the head, queries make
    # no sense (rendered features live in arbitrary surfel-space).
    # NB: read via getattr because get_combined_args drops cmdline keys
    # whose value is None (arguments/__init__.py:134-136), and the saved
    # cfg_args from training doesn't carry --checkpoint either; without
    # the guard, args.checkpoint raises AttributeError when the flag is
    # omitted instead of falling through to the untrained-head warning.
    semantic_head = SemanticHead(SEMANTIC_DIM, dataset.K_target).cuda().eval()
    ckpt_path = getattr(args, "checkpoint", None)
    if ckpt_path is not None:
        # weights_only=False: the chkpnt*.pth tuple contains the gaussians
        # capture (with optimizer state -> numpy scalars) plus the
        # SemanticHead state. PyTorch 2.6+ rejects numpy types under the
        # safe-load default; we trust our own checkpoint files.
        ckpt = torch.load(ckpt_path, weights_only=False)
        head_state = ckpt[1]
        if head_state is None:
            raise SystemExit("checkpoint has no semantic head; was lambda_semantic > 0 during training?")
        semantic_head.load_state_dict(head_state)
    else:
        print("warning: no --checkpoint given; using untrained projection head (heatmap will be noise)")

    # Encode text query.
    text_emb = encode_text_query(args.query, args.variant)
    text_emb = F.normalize(text_emb, dim=-1)                # (1, K_target)
    if text_emb.shape[1] != dataset.K_target:
        raise SystemExit(
            f"text encoder produced K_target={text_emb.shape[1]} but model trained "
            f"with K_target={dataset.K_target}; pass --variant matching preprocessing"
        )

    # Render the chosen viewpoint.
    cams = scene.getTrainCameras()
    if args.view_idx >= len(cams):
        raise SystemExit(f"--view_idx {args.view_idx} out of range (have {len(cams)})")
    cam = cams[args.view_idx]

    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        pkg = render(cam, gaussians, pipe, bg)
        rendered_rgb = pkg["render"].clamp(0, 1)             # (3, H, W)
        rendered_sem = pkg["rendered_semantic"]              # (K, H, W)

        # Project per-pixel features to K_target and cosine-sim against query.
        K, H, W = rendered_sem.shape
        flat = rendered_sem.reshape(K, -1).T                 # (H*W, K)
        proj = semantic_head(flat)                           # (H*W, K_target)
        proj = F.normalize(proj, dim=-1)
        sim = (proj @ text_emb.squeeze(0)).reshape(H, W)     # (H, W)

    rgb_np = rendered_rgb.cpu().numpy()
    sim_np = sim.cpu().numpy()
    img = overlay_heatmap(rgb_np, sim_np)
    Image.fromarray(img).save(args.out)
    print(f"saved {args.out} (sim range [{sim_np.min():.3f}, {sim_np.max():.3f}])")


if __name__ == "__main__":
    main()
