#!/usr/bin/env python3
"""
Check whether leworldmodel expert episodes demonstrate goal-reaching for all 5 OGBench cube tasks.

For each of the 5 tasks (horizontal, vertical×2, diagonal×2):
  - Encodes the task goal image with the frozen 224×224 LeWM encoder
  - Computes, for each cached episode:
      * start_dist  : L2(z_enc[0],  z_goal)
      * final_dist  : L2(z_enc[-1], z_goal)
      * improvement : start_dist - final_dist  (positive → moving toward goal)
  - Reports: fraction of episodes within done_threshold at start/final,
    and percentile breakdown of final distances.

Use this to determine whether 200-step expert demonstrations cover all 5 eval goals.
A task with <5% final coverage likely has too few successful demos to learn from.

Usage:
    python latent_hindsight_rl/check_dataset_episode_length.py
    python latent_hindsight_rl/check_dataset_episode_length.py --done_threshold 2.406 --n_episodes 2000
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2 as transforms

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import stable_pretraining as spt


def load_jepa(ckpt_path, device, img_size=224, patch_size=14):
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    encoder = spt.backbone.utils.vit_hf(
        "tiny", patch_size=patch_size, image_size=img_size,
        pretrained=False, use_mask_token=False)
    predictor = ARPredictor(
        num_frames=3, input_dim=192, hidden_dim=192, output_dim=192,
        depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.0)
    action_encoder = Embedder(input_dim=25, emb_dim=192)
    projector = MLP(input_dim=192, output_dim=192, hidden_dim=2048, norm_fn=nn.BatchNorm1d)
    pred_proj  = MLP(input_dim=192, output_dim=192, hidden_dim=2048, norm_fn=nn.BatchNorm1d)
    model = JEPA(encoder=encoder, predictor=predictor,
                 action_encoder=action_encoder, projector=projector, pred_proj=pred_proj)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_sd = {k[len("model."):]: v for k, v in ckpt["state_dict"].items()
              if k.startswith("model.")} if "state_dict" in ckpt else dict(ckpt)
    model.load_state_dict(raw_sd, strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def encode_img(jepa, img_hwc, img_transform, device):
    """(H, W, 3) uint8 → [1, 192] float32 tensor."""
    t = img_transform(img_hwc).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = jepa.encode({"pixels": t})["emb"]
    return emb[:, -1].cpu()   # [1, 192]


def main():
    parser = argparse.ArgumentParser(description="Check episode coverage of OGBench cube tasks.")
    parser.add_argument('--cache_path',     default=None)
    parser.add_argument('--ckpt_path',      default=None)
    parser.add_argument('--done_threshold', type=float, default=2.406,
                        help="L2 threshold for 'goal reached' (default 2.406 = 1.5× mean 1-WM-step dist)")
    parser.add_argument('--img_size',       type=int, default=224)
    parser.add_argument('--patch_size',     type=int, default=14)
    parser.add_argument('--device',         default='cuda')
    parser.add_argument('--n_episodes',     type=int, default=None,
                        help="Sample this many episodes (default: all ~10 000)")
    parser.add_argument('--ogbench_dir',    default=None,
                        help="Root of cloned ogbench repo (needed to register gymnasium envs)")
    args = parser.parse_args()

    stablewm = os.environ.get("STABLEWM_HOME", os.path.expanduser("~/stable_wm_data"))
    if args.cache_path is None:
        args.cache_path = os.path.join(stablewm, "lewm_224_latents_cache.pt")
    if args.ckpt_path is None:
        args.ckpt_path  = os.path.join(stablewm, "cube", "lejepa_weights.ckpt")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Load encoder ────────────────────────────────────────────────────────────
    print("Loading LeWM encoder ...")
    jepa = load_jepa(args.ckpt_path, device, args.img_size, args.patch_size)
    img_transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
    ])

    # ── Load cache ──────────────────────────────────────────────────────────────
    print(f"Loading cache from {args.cache_path} ...")
    cache = torch.load(args.cache_path, map_location="cpu", weights_only=False)
    all_latents = cache["all_latents"]   # list of [T_raw, 192] tensors (T_raw ≈ 201 frames)
    n_total = len(all_latents)
    print(f"  {n_total} episodes, {all_latents[0].shape[0]} latent steps each")

    if args.n_episodes is not None:
        rng = np.random.RandomState(42)
        idxs = rng.choice(n_total, size=min(args.n_episodes, n_total), replace=False)
        all_latents = [all_latents[i] for i in idxs]
    n_eps = len(all_latents)

    # Stack start and final latents ── [N_ep, 192]
    start_latents = torch.stack([ep[0]  for ep in all_latents])
    final_latents = torch.stack([ep[-1] for ep in all_latents])

    ep_len_wm    = all_latents[0].shape[0]
    ep_len_frames = ep_len_wm  # latents are stored at raw-frame granularity in this cache

    # ── Create env and encode task goals ────────────────────────────────────────
    import stable_worldmodel  # registers swm/ gymnasium namespace
    if args.ogbench_dir:
        sys.path.insert(0, os.path.abspath(args.ogbench_dir))
    import ogbench        # registers ogbench gymnasium environments
    import gymnasium

    print("Creating swm/OGBCube-v0 to read task goals ...")
    env = gymnasium.make("swm/OGBCube-v0", ob_type="pixels",
                         env_type="single", visualize_info=False)
    task_infos = env.unwrapped.task_infos
    n_tasks    = len(task_infos)
    print(f"  {n_tasks} tasks: {[t.get('task_name', '?') for t in task_infos]}\n")

    print(f"done_threshold = {args.done_threshold}")
    print(f"Episodes analysed: {n_eps}  "
          f"(ep_len = {ep_len_frames} latent steps = ~{ep_len_frames} physical frames)\n")

    W = 70
    print("─" * W)
    print(f"{'Task':<28s}  {'%start<thr':>10}  {'%final<thr':>10}  "
          f"{'med_final':>10}  {'p90_final':>10}")
    print("─" * W)

    all_results = {}
    for task_id in range(1, n_tasks + 1):
        name = task_infos[task_id - 1].get("task_name", f"task{task_id}")
        _, info = env.reset(options=dict(task_id=task_id, variation=[]))
        z_goal = encode_img(jepa, info["target"], img_transform, device)  # [1, 192]

        start_dists = torch.norm(start_latents - z_goal, dim=-1).numpy()   # [N_ep]
        final_dists = torch.norm(final_latents - z_goal, dim=-1).numpy()   # [N_ep]
        impr        = start_dists - final_dists   # positive = moving toward goal

        pct_start = np.mean(start_dists < args.done_threshold) * 100
        pct_final = np.mean(final_dists < args.done_threshold) * 100
        med_final = np.median(final_dists)
        p90_final = np.percentile(final_dists, 90)
        med_impr  = np.median(impr)

        flag = "✓" if pct_final > 5.0 else "✗"
        print(f"{flag} {name:<26s}  {pct_start:>9.1f}%  {pct_final:>9.1f}%  "
              f"{med_final:>10.3f}  {p90_final:>10.3f}")
        all_results[name] = dict(
            pct_start=pct_start, pct_final=pct_final,
            med_final=med_final, p90_final=p90_final, med_impr=med_impr,
            start_dists=start_dists, final_dists=final_dists,
        )

    print("─" * W)

    # ── Histogram of improvements per task ──────────────────────────────────────
    print("\nMedian distance improvement over episode (positive = toward goal):")
    for name, r in all_results.items():
        bar_len = int(max(0, r["med_impr"]) / 3.0 * 20)
        bar = "█" * bar_len
        print(f"  {name:<28s}  impr={r['med_impr']:+6.2f}  {bar}")

    # ── Verdict ────────────────────────────────────────────────────────────────
    print("\n── Verdict ──────────────────────────────────────────────────────")
    any_fail = False
    for name, r in all_results.items():
        if r["pct_final"] < 1.0:
            print(f"  FAIL  {name}: {r['pct_final']:.1f}% final coverage — "
                  f"episodes NEVER reach this goal. RL cannot learn task.")
            any_fail = True
        elif r["pct_final"] < 5.0:
            print(f"  WARN  {name}: {r['pct_final']:.1f}% final coverage — "
                  f"very sparse signal. May still work with enough data.")
        else:
            print(f"  OK    {name}: {r['pct_final']:.1f}% final coverage")

    if not any_fail:
        print("\n  All tasks have > 1% final coverage. Dataset is viable for RL training.")
    else:
        print("\n  Some tasks have 0% coverage. The expert demos never complete those tasks.")
        print("  Either the data is insufficient, or done_threshold is too tight.")
        print(f"  Try --done_threshold {args.done_threshold * 3:.2f} (3× current) to check.")

    env.close()


if __name__ == "__main__":
    main()
