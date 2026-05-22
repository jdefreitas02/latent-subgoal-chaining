"""Export a joint-trained WM (held in a JAX TrainState inside a saved
ACFQLAgent checkpoint) into a PyTorch _state_dict.pt file that
``load_jepa(..., img_size=224)`` can read.

The resulting file is a HYBRID:
  - encoder + projector: from the original PyTorch WM (unchanged; we never
    train these in joint mode).
  - predictor + action_encoder + pred_proj: converted from JAX joint-trained
    params back to PyTorch keys.

Usage:
  python export_jointtrained_wm.py \\
      --agent_ckpt path/to/sd000s_XXX.../offline_final/params_500000.pkl \\
      --base_wm_ckpt /path/to/lejepa_play_ft_full \\
      --out path/to/lejepa_play_jointtrained_state_dict.pt
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_OTO = os.path.join(_ROOT, "offline_to_online")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _OTO not in sys.path:
    sys.path.insert(0, _OTO)

from envs.jepa_loader import load_jepa
from wm_jax import jax_params_to_torch_state_dict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent_ckpt", required=True,
                   help="Path to a saved agent .pkl produced by joint training.")
    p.add_argument("--base_wm_ckpt", required=True,
                   help="Base WM whose encoder + projector we keep (PyTorch).")
    p.add_argument("--out", required=True,
                   help="Output _state_dict.pt path. Must end with '.pt'.")
    args = p.parse_args()

    if not args.out.endswith(".pt"):
        raise ValueError("--out must end with .pt so load_jepa picks the "
                         "state-dict path.")
    if os.path.exists(args.out):
        raise FileExistsError(f"Refusing to overwrite {args.out}.")

    # 1. Load the agent ckpt and pull out the JAX WM params
    print(f"[load] agent ckpt from {args.agent_ckpt}", flush=True)
    with open(args.agent_ckpt, "rb") as f:
        ckpt = pickle.load(f)
    agent_state = ckpt["agent"]
    if "wm_state" not in agent_state or agent_state["wm_state"] is None:
        raise RuntimeError(
            "Agent checkpoint does not contain a joint-trained wm_state. "
            "Did you run with --use_joint_wm_training?"
        )
    wm_state = agent_state["wm_state"]
    if "params" not in wm_state:
        raise RuntimeError(f"wm_state has no 'params' key; keys: {list(wm_state.keys())}")
    jax_params = wm_state["params"]
    print(f"  step in wm_state: {wm_state.get('step')}")
    print(f"  top-level JAX wm modules: {list(jax_params.keys())}")

    # 2. Convert to PyTorch state_dict (only predictor + action_encoder + pred_proj)
    print(f"[convert] JAX wm params -> torch state_dict", flush=True)
    jt_sd = jax_params_to_torch_state_dict(jax_params)
    print(f"  produced {len(jt_sd)} keys")

    # 3. Load the base PyTorch WM (full state_dict) and overlay joint-trained keys
    print(f"[load] base PyTorch WM from {args.base_wm_ckpt}", flush=True)
    base_wm = load_jepa(args.base_wm_ckpt, device="cpu", img_size=224, patch_size=14)
    if hasattr(base_wm, "state_dict"):
        base_sd = base_wm.state_dict()
    else:
        base_sd = base_wm.jepa.state_dict()

    # Check that all the keys we want to overlay actually exist in the base
    missing_in_base = [k for k in jt_sd if k not in base_sd]
    if missing_in_base:
        print(f"  WARN: keys produced by export not in base WM: {missing_in_base[:5]}{'...' if len(missing_in_base) > 5 else ''}")
    # Also report shape mismatches
    shape_mismatches = []
    for k, v in jt_sd.items():
        if k in base_sd and tuple(v.shape) != tuple(base_sd[k].shape):
            shape_mismatches.append((k, tuple(base_sd[k].shape), tuple(v.shape)))
    if shape_mismatches:
        print("  SHAPE MISMATCHES:")
        for k, s_base, s_jt in shape_mismatches:
            print(f"    {k}: base={s_base} jt={s_jt}")
        raise RuntimeError("Aborting due to shape mismatches.")

    # Overlay
    hybrid = dict(base_sd)
    overlaid = 0
    for k, v in jt_sd.items():
        if k in hybrid:
            hybrid[k] = v
            overlaid += 1
    print(f"  overlaid {overlaid}/{len(jt_sd)} keys onto base state_dict")

    # 4. Save
    print(f"[save] writing {args.out}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(hybrid, args.out)
    sz_mb = os.path.getsize(args.out) / 1e6
    print(f"  wrote {sz_mb:.1f} MB")
    print("Done. Use this file as --wm_ckpt for the online phase (load_jepa "
          "will detect the .pt suffix and load it as a state_dict).")


if __name__ == "__main__":
    main()
