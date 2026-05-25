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


def _fix_vit_config_compat(model):
    """Patch models pickled with a newer transformers that changed internal
    attribute naming or removed submodules.

    Two known issues:
    1. ViTConfig: attributes like output_attentions stored as _output_attentions
       with empty attribute_map → accessing config.output_attentions raises
       AttributeError.  Fix: add attribute_map entries for each _-prefixed key.
    2. ViTSelfAttention: newer transformers replaced self.dropout (nn.Dropout
       module) with self.dropout_prob (float) + F.dropout.  When the checkpoint
       was saved with the newer version and loaded into 4.44.x that still calls
       self.dropout(...), the module is missing.  Fix: add nn.Dropout back.
    """
    import torch.nn as _nn
    try:
        from transformers.models.vit.modeling_vit import ViTSelfAttention as _VSA
    except ImportError:
        _VSA = None

    for module in model.modules():
        # --- Fix 1: ViTConfig _-prefixed attributes ---
        cfg = getattr(module, "config", None)
        if cfg is not None:
            try:
                d = object.__getattribute__(cfg, "__dict__")
            except Exception:
                d = {}
            for raw_key in list(d.keys()):
                if not raw_key.startswith("_"):
                    continue
                public_key = raw_key[1:]
                if public_key in d:
                    continue
                amap = getattr(cfg, "attribute_map", {})
                if public_key not in amap:
                    amap[public_key] = raw_key
                    try:
                        cfg.attribute_map = amap
                    except Exception:
                        pass

        # --- Fix 2: ViTSelfAttention missing dropout module ---
        if _VSA is not None and isinstance(module, _VSA):
            if "dropout" not in module._modules and hasattr(module, "dropout_prob"):
                module.dropout = _nn.Dropout(module.dropout_prob)


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
    _fix_vit_config_compat(model)
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
    # Fall through to AutoCostModel if the state dict keys don't match the
    # current architecture (e.g., after a stable_pretraining version bump).
    sd_candidate = str(ckpt_path) + "_state_dict.pt"
    for sd_path in [sd_candidate if os.path.exists(sd_candidate) else None,
                    ckpt_path if str(ckpt_path).endswith(".pt") and os.path.exists(ckpt_path) else None]:
        if sd_path is None:
            continue
        try:
            return _load_from_state_dict(sd_path, device, img_size, patch_size)
        except (RuntimeError, KeyError, ImportError) as e:
            print(f"[load_jepa] _load_from_state_dict({sd_path}) failed ({type(e).__name__}: "
                  f"{str(e)[:120]}); falling through to AutoCostModel.", flush=True)

    if img_size == 224:
        # Inject backwards-compat stubs before torch.load unpickles them.
        # 1. ViTEncoder was removed in transformers 5.x; needed for lejepa_object.ckpt.
        import torch.nn as _nn
        import transformers.models.vit.modeling_vit as _vit_mod
        if not hasattr(_vit_mod, "ViTEncoder"):
            class _ViTEncoderCompat(_nn.Module):
                pass
            _vit_mod.ViTEncoder = _ViTEncoderCompat

        # 2. CostShim was defined in __main__ of finetune_wm_on_play.py; needed
        #    for lejepa_play_ft_full_object.ckpt. Import the real class so pickle
        #    can resolve it without rerunning __main__.
        import __main__ as _main
        if not hasattr(_main, "CostShim"):
            # finetune_wm_on_play.py does `from envs.jepa_loader import ...`
            # so offline_to_online/ must be in sys.path when we import it.
            _o2o_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            if _o2o_dir not in sys.path:
                sys.path.append(_o2o_dir)
            try:
                from finetune_wm_on_play import CostShim as _CostShim
                _main.CostShim = _CostShim
            except (ImportError, Exception):
                pass  # falls through; torch.load will raise if it's truly needed

        model = swm.policy.AutoCostModel(ckpt_path)
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _fix_vit_config_compat(model)
        return model

    model = _load_from_state_dict(ckpt_path, device, img_size, patch_size)
    _fix_vit_config_compat(model)
    return model


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
