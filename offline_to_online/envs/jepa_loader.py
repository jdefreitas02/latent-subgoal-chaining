"""JEPA loader and image transform — copies of the working helpers in
sac_env_train.py / eval_ogbench.py, parameterised so they can live in
this offline-online package without importing the heavy training scripts."""

import os
import sys
from pathlib import Path

import torch
from torchvision.transforms import v2 as transforms


def _add_leworldmodel_to_path():
    """Make stable_pretraining, stable_worldmodel, jepa.py, module.py importable.

    IMPORTANT: append (not prepend) so we don't shadow this package's `utils/`
    subpackage with ~/leworldmodel/utils.py.
    """
    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),  # leworldmodel/
    ]
    for p in candidates:
        if p not in sys.path and os.path.isdir(p):
            sys.path.append(p)


def _build_jepa_from_arch(img_size, patch_size):
    """Build the JEPA architecture that matches our trained checkpoints."""
    _add_leworldmodel_to_path()
    import stable_pretraining as spt
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    encoder = spt.backbone.utils.vit_hf(
        "tiny", patch_size=patch_size, image_size=img_size,
        pretrained=False, use_mask_token=False,
    )
    predictor = ARPredictor(
        num_frames=3, input_dim=192, hidden_dim=192, output_dim=192,
        depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.0,
    )
    action_encoder = Embedder(input_dim=25, emb_dim=192)
    projector = MLP(input_dim=192, output_dim=192, hidden_dim=2048,
                    norm_fn=torch.nn.BatchNorm1d)
    pred_proj = MLP(input_dim=192, output_dim=192, hidden_dim=2048,
                    norm_fn=torch.nn.BatchNorm1d)
    return JEPA(encoder=encoder, predictor=predictor,
                action_encoder=action_encoder, projector=projector, pred_proj=pred_proj)


def _load_from_state_dict(state_dict_path, device, img_size, patch_size):
    model = _build_jepa_from_arch(img_size=img_size, patch_size=patch_size)
    ckpt = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        raw_sd = {k[len("model."):]: v for k, v in ckpt["state_dict"].items()
                  if k.startswith("model.")}
    else:
        raw_sd = dict(ckpt)
    model.load_state_dict(raw_sd, strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_jepa(ckpt_path, device="cuda", img_size=224, patch_size=14):
    """Load a frozen JEPA model.

    Resolution order:
      1. If ``{ckpt_path}_state_dict.pt`` exists, build the architecture
         from scratch and load that state dict. This bypasses any
         class-pickle issues (e.g., classes defined under ``__main__`` in
         a different script).
      2. If ``ckpt_path`` itself is a ``.pt`` state-dict file, same.
      3. img_size == 224: ``swm.policy.AutoCostModel(ckpt_path)``.
      4. img_size != 224: build arch + load Lightning ``.ckpt``.
    """
    _add_leworldmodel_to_path()
    import stable_pretraining as spt
    import stable_worldmodel as swm

    # Prefer state-dict alongside if it exists -- robust against
    # __main__.<Class> pickle entries in object.ckpt files.
    sd_candidate = str(ckpt_path) + "_state_dict.pt"
    if os.path.exists(sd_candidate):
        return _load_from_state_dict(sd_candidate, device, img_size, patch_size)
    if str(ckpt_path).endswith(".pt") and os.path.exists(ckpt_path):
        return _load_from_state_dict(ckpt_path, device, img_size, patch_size)

    if img_size == 224:
        model = swm.policy.AutoCostModel(ckpt_path)
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    return _load_from_state_dict(ckpt_path, device, img_size, patch_size)


def make_img_transform():
    """ImageNet normalisation only — env renders natively at the correct resolution."""
    _add_leworldmodel_to_path()
    import stable_pretraining as spt
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
    ])


def encode_pixels_to_latent(jepa_model, pixels_hwc_uint8, device, img_transform=None):
    """Encode a single (H, W, 3) uint8 image to a 192-D latent (numpy float32).

    Mirrors the per-step encoding used in eval_ogbench.py.
    """
    if img_transform is None:
        img_transform = make_img_transform()
    img = torch.from_numpy(pixels_hwc_uint8).permute(2, 0, 1).contiguous()  # (3, H, W) uint8
    img = img_transform(img).to(device)  # (3, H, W) float
    info = {"pixels": img.unsqueeze(0).unsqueeze(0)}  # (1, 1, 3, H, W)
    with torch.no_grad():
        info = jepa_model.encode(info)
    return info["emb"].squeeze(0).squeeze(0).cpu().numpy().astype("float32")  # (192,)
