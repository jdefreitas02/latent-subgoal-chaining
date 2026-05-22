"""
Hybrid diagnostic eval: HIQL-baseline HL + WGSP LL + decoder.

Loads:
  - HIQL baseline HL actor (well-trained, 62% baseline)
  - WGSP LL actor + adapter + decoder (from row1 checkpoint)

If this scores near the baseline, the WGSP LL distillation worked and
only the WGSP HL is broken. If it also scores 0%, the LL was damaged too.

Usage:
    python latent_hindsight_rl/eval_hybrid_wgsp_hl.py
"""

import os, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

_parent = os.path.abspath(os.path.dirname(__file__))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import stable_worldmodel as swm
from sklearn import preprocessing

# Re-use all the model classes and helpers from eval_ogbench.py
sys.path.insert(0, str(Path(__file__).parent))
from eval_ogbench import (
    _LewmGaussianActor, _ActionChunkDecoder, _LatentAdapter, _load_adapter,
    _WGSPHierarchicalPolicy, _HIQLBaselineHighLevelWrapper,
    _BaselineGaussianActor, encode_and_project, load_jepa,
    _make_img_transform, run_task,
    _IMG_TRANSFORM,
)
import eval_ogbench as _eval_mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wgsp_ckpt',     default='checkpoints_hiql_wgsp_row1_headline_s0')
    parser.add_argument('--baseline_ckpt', default='checkpoints_hiql_baseline_k10_rep10_adapter256')
    parser.add_argument('--decoder_ckpt',  default='checkpoints_action_decoder/action_decoder.pth')
    parser.add_argument('--dataset_path',  default=None)
    parser.add_argument('--ckpt_path',     default=None)
    parser.add_argument('--num_episodes',  type=int, default=10)
    parser.add_argument('--rep_dim',       type=int, default=10)
    parser.add_argument('--img_size',      type=int, default=224)
    parser.add_argument('--patch_size',    type=int, default=14)
    parser.add_argument('--diagnose',      action='store_true')
    parser.add_argument('--ogbench_dir',   default=None)
    args = parser.parse_args()

    stablewm_home = os.environ.get('STABLEWM_HOME',
                                   os.path.join(os.path.expanduser('~'), 'stable_wm_data'))
    if args.dataset_path is None:
        args.dataset_path = os.path.join(stablewm_home, 'ogbench', 'visual-cube-single-play-v0_224')
    if args.ckpt_path is None:
        args.ckpt_path = os.path.join(stablewm_home, 'cube', 'lejepa')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if args.ogbench_dir:
        sys.path.insert(0, os.path.abspath(args.ogbench_dir))
    import ogbench
    import gymnasium

    env = gymnasium.make('swm/OGBCube-v0', ob_type='pixels', env_type='single', visualize_info=False)
    task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos

    print("Fitting action scaler...")
    dataset = swm.data.HDF5Dataset(args.dataset_path, keys_to_cache=['action'],
                                   cache_dir=str(Path(args.dataset_path).parent))
    action_data = dataset.get_col_data('action')
    action_data = action_data[~np.isnan(action_data).any(axis=1)]
    action_scaler = preprocessing.StandardScaler()
    action_scaler.fit(action_data)
    print(f"  Action scaler fit on {len(action_data):,} steps.")

    _eval_mod._IMG_TRANSFORM = _make_img_transform(args.img_size)
    print(f"Loading JEPA from {args.ckpt_path}...")
    jepa_model = load_jepa(args.ckpt_path, device, img_size=args.img_size, patch_size=args.patch_size)

    # --- WGSP adapter + LL actor + decoder ---
    wgsp_dir = Path(args.wgsp_ckpt)
    adapter_ckpt = wgsp_dir / 'adapter.pth'
    wgsp_adapter, _, policy_dim = _load_adapter(adapter_ckpt, device)
    print(f"WGSP adapter: 192 → {policy_dim}")

    ll_actor = _LewmGaussianActor(
        state_dim=policy_dim, goal_dim=args.rep_dim, output_dim=5,
        hidden_dims=(512, 512, 512), tanh_squash=True, action_scale=3.0,
    ).to(device)
    ll_actor.load_state_dict(torch.load(wgsp_dir / 'll_actor.pth', map_location=device))
    ll_actor.eval()
    for p in ll_actor.parameters(): p.requires_grad_(False)
    print(f"WGSP LL actor loaded (state={policy_dim}, goal={args.rep_dim}, out=5)")

    dec_ckpt = wgsp_dir / 'action_decoder.pth'
    if not dec_ckpt.exists():
        dec_ckpt = Path(args.decoder_ckpt)
    decoder = _ActionChunkDecoder(in_dim=5, out_dim=25, latent_dim=192, hidden_dims=(256, 256)).to(device)
    decoder.load_state_dict(torch.load(dec_ckpt, map_location=device))
    decoder.eval()
    for p in decoder.parameters(): p.requires_grad_(False)
    print(f"WGSP decoder loaded from {dec_ckpt}")

    # --- HIQL baseline HL actor ---
    bl_dir = Path(args.baseline_ckpt)
    bl_adapter_ckpt = bl_dir / 'adapter.pth'
    bl_adapter, _, bl_policy_dim = _load_adapter(bl_adapter_ckpt, device)
    assert bl_policy_dim == policy_dim, (
        f"Adapter dim mismatch: baseline {bl_policy_dim} vs WGSP {policy_dim}")
    print(f"Baseline adapter: 192 → {bl_policy_dim}")

    # Baseline HL: input_dim = policy_dim * 2, output_dim = rep_dim (no log_stds)
    hl_actor_net = _BaselineGaussianActor(
        input_dim=policy_dim * 2, output_dim=args.rep_dim,
        hidden_dims=(512, 512, 512), tanh_squash=False,
    ).to(device)
    hl_actor_net.load_state_dict(torch.load(bl_dir / 'hl_actor.pth', map_location=device))
    hl_actor_net.eval()
    for p in hl_actor_net.parameters(): p.requires_grad_(False)
    print(f"HIQL baseline HL loaded (input={policy_dim*2}, out={args.rep_dim})")

    # Wrap baseline HL in the same wrapper used by eval_ogbench
    hl_model = _HIQLBaselineHighLevelWrapper(hl_actor_net)

    # Build hybrid policy
    policy = _WGSPHierarchicalPolicy(
        jepa_model, ll_actor, action_scaler, hl_model,
        gap=8, device=device, decoder=decoder,
        subgoal_reached_threshold=0.0,
        latent_adapter=wgsp_adapter,
        native_eval=True,
    )

    print("\n" + "="*60)
    print("Hybrid policy: HIQL-baseline HL + WGSP LL")
    print(f"  {len(task_infos)} tasks × {args.num_episodes} eps each")
    print("="*60)

    t0 = time.time()
    per_task = {}
    all_successes = []
    for task_id in range(1, len(task_infos) + 1):
        task_name = task_infos[task_id - 1].get('task_name', f'task{task_id}')
        mean_sr, eps = run_task(
            env, policy, task_id, task_name, args.num_episodes, 200,
            diagnose=args.diagnose, goal_info_key='target')
        per_task[task_name] = mean_sr
        all_successes.extend(eps)

    overall = np.mean(all_successes)
    print(f"\n{'='*60}")
    print("HYBRID RESULTS: baseline-HL + WGSP-LL")
    print(f"{'='*60}")
    for name, sr in per_task.items():
        print(f"  {name:<30s} {sr*100:5.1f}%")
    print(f"  {'─'*36}")
    print(f"  {'overall':<30s} {overall*100:5.1f}%")
    print(f"  elapsed: {time.time()-t0:.0f}s")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
