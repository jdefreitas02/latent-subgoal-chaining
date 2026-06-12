"""Report-grade WM diagnostics, reproducing the cube-single foundations figures
for any env's world model:

  D1  Predictor drift vs horizon  (Table 3.1 / Fig 3.1 "predictor_drift.png")
      mean ||z_hat_{t+d} - z_{t+d}||_2 as the predictor f_psi is rolled forward
      d raw frames. Re-encodes ground-truth frames from pixels (faithful to the
      verify_predictor.py methodology), so this REQUIRES the pixel HDF5.

  D2  Encoder latent geometry "L2 vs frame gap"  (summary.txt Test 1)
      mean ||z_t - z_{t+k}||_2 over k, from encoded latents.

  D3  Latent-norm histogram  (Fig 3.2 "goal_norms_ood.png")
      distribution of ||z|| over dataset states + the 5 rendered task goals,
      showing SIGReg concentration around one scale.

Outputs a summary.txt, an .npz of raw arrays, and two PNGs into --out_dir.

Usage:
    python wm_report_diagnostics.py \
        --jepa  $STABLEWM_HOME/scene/lejepa_scene/lejepa_scene \
        --hdf5  $STABLEWM_HOME/ogbench/visual-scene-play-v0_224 \
        --cache $STABLEWM_HOME/ogbench/lewm_scene_latents_cache.pt \
        --env_family scene \
        --out_dir diagnostics_scene
"""
import argparse
import os
import sys

import numpy as np
import torch
import h5py
import hdf5plugin  # noqa: F401

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_REPO)
sys.path.append(os.path.join(_REPO, 'offline_to_online'))

HISTORY = 3
FRAMESKIP = 5
LATENT_DIM = 192
DONE_THRESHOLD = 2.0


def _img_transform():
    import stable_pretraining as spt
    from torchvision.transforms import v2 as transforms
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
    ])


def predictor_drift(jepa, h5_path, device, n_traj, horizons_chunks):
    """D1: roll the predictor forward and compare to encoded ground truth.

    For each sampled start, encode HISTORY context frames (frameskip apart) and
    the action chunks, then autoregressively predict the next chunked latent and
    compare to the *encoded* ground-truth latent at that horizon. Returns a dict
    {raw_frame_delta: array_of_l2_errors}.
    """
    tf = _img_transform()
    max_chunks = max(horizons_chunks)
    errs = {c: [] for c in horizons_chunks}

    with h5py.File(h5_path, 'r') as f:
        ep_offset = f['ep_offset'][...].astype(np.int64)
        ep_len = f['ep_len'][...].astype(np.int64)
        n_eps = len(ep_len)
        # need room for HISTORY context + max_chunks prediction steps, all frameskip apart
        span = (HISTORY + max_chunks) * FRAMESKIP
        rng = np.random.default_rng(0)
        tried = 0
        while sum(len(v) for v in errs.values()) < n_traj * len(horizons_chunks) and tried < n_traj * 4:
            tried += 1
            ep = int(rng.integers(0, n_eps))
            off, length = int(ep_offset[ep]), int(ep_len[ep])
            if length <= span + 1:
                continue
            t0 = int(rng.integers(0, length - span - 1))

            # context: HISTORY frames + (max_chunks) future frames, all frameskip apart
            n_frames = HISTORY + max_chunks
            positions = [t0 + j * FRAMESKIP for j in range(n_frames)]
            frames, acts = [], []
            for p in positions:
                pix = f['pixels'][off + p]
                frames.append(tf(torch.from_numpy(pix.transpose(2, 0, 1))))
                a = f['action'][off + p: off + p + FRAMESKIP].reshape(-1)
                acts.append(torch.from_numpy(a.astype(np.float32)))
            pixels = torch.stack(frames, 0).unsqueeze(0).to(device)   # (1, n_frames, 3, H, W)
            actions = torch.stack(acts, 0).unsqueeze(0).to(device)    # (1, n_frames, 25)
            actions = torch.nan_to_num(actions, 0.0)

            with torch.no_grad():
                out = jepa.encode({"pixels": pixels, "action": actions})
                emb = out["emb"]            # (1, n_frames, 192) ground-truth encoded
                act_emb = out["act_emb"]    # (1, n_frames, 192)
                # autoregressive rollout from the first HISTORY context latents
                ctx = emb[:, :HISTORY].clone()        # (1, HISTORY, 192)
                ctx_act = act_emb[:, :HISTORY].clone()
                pred_chain = ctx
                for step in range(max_chunks):
                    pred = jepa.predict(pred_chain[:, -HISTORY:], ctx_act[:, -HISTORY:])
                    z_hat = pred[:, -1:]              # (1,1,192)
                    pred_chain = torch.cat([pred_chain, z_hat], dim=1)
                    # advance action context with the ground-truth action at this future step
                    nxt = act_emb[:, HISTORY + step: HISTORY + step + 1]
                    ctx_act = torch.cat([ctx_act, nxt], dim=1)
                    c = step + 1
                    if c in errs:
                        z_gt = emb[:, HISTORY + step]          # encoded ground truth (1,192)
                        e = torch.norm(z_hat.squeeze(1) - z_gt, dim=-1).item()
                        errs[c].append(e)
    # map chunk steps -> raw frames
    return {c * FRAMESKIP: np.array(v, dtype=np.float32) for c, v in errs.items()}


def encoder_l2_vs_gap(cache_latents, n_traj, kmax, seed=0):
    """D2: mean ||z_t - z_{t+k}|| over k=1..kmax, from encoded latents."""
    rng = np.random.default_rng(seed)
    eps = [e.numpy() if torch.is_tensor(e) else np.asarray(e) for e in cache_latents]
    eps = [e for e in eps if e.shape[0] > kmax + 1]
    sel = rng.choice(len(eps), size=min(n_traj, len(eps)), replace=False)
    res = {k: [] for k in range(1, kmax + 1)}
    for i in sel:
        z = eps[i]
        T = z.shape[0]
        for k in range(1, kmax + 1):
            d = np.linalg.norm(z[:T - k] - z[k:], axis=1)
            res[k].append(d.mean())
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in res.items()}


def latent_norms(cache_latents, jepa, env_family, device, n_pool=20000, seed=0):
    """D3: ||z|| over dataset states + the 5 rendered task goals."""
    rng = np.random.default_rng(seed)
    flat = np.concatenate(
        [e.numpy() if torch.is_tensor(e) else np.asarray(e) for e in cache_latents], axis=0)
    if flat.shape[0] > n_pool:
        flat = flat[rng.choice(flat.shape[0], n_pool, replace=False)]
    dataset_norms = np.linalg.norm(flat, axis=1)

    goal_norms = []
    try:
        import gymnasium as gym
        import ogbench  # noqa: F401
        from envs.jepa_loader import encode_pixels_to_latent
        for t in (1, 2, 3, 4, 5):
            env = gym.make(f"visual-{env_family}-singletask-task{t}-v0", width=224, height=224)
            _, info = env.reset(seed=0, options=dict(render_goal=True))
            g = info.get("goal", info.get("target"))
            z = encode_pixels_to_latent(jepa, g, device)
            goal_norms.append(float(np.linalg.norm(z)))
            env.close()
    except Exception as e:
        print(f"[D3] goal rendering skipped ({type(e).__name__}: {str(e)[:80]})")
    return dataset_norms, np.array(goal_norms, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jepa', required=True)
    ap.add_argument('--hdf5', required=True, help='pixel HDF5 (needed for D1 drift)')
    ap.add_argument('--cache', required=True)
    ap.add_argument('--env_family', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--n_traj', type=int, default=200)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.makedirs(args.out_dir, exist_ok=True)

    from envs.jepa_loader import load_jepa
    print(f"[diag] loading WM {args.jepa}")
    jepa = load_jepa(args.jepa, device=args.device, img_size=224)
    h5 = args.hdf5 if args.hdf5.endswith('.h5') else args.hdf5 + '.h5'
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    cache_latents = cache['all_latents'] if isinstance(cache, dict) else cache

    # D1 predictor drift (chunk steps 1,2,5,10 -> raw frames 5,10,25,50)
    print("[D1] predictor drift vs horizon ...")
    drift = predictor_drift(jepa, h5, args.device, args.n_traj, horizons_chunks=[1, 2, 5, 10])
    # D2 encoder geometry
    print("[D2] encoder L2 vs frame gap ...")
    geo = encoder_l2_vs_gap(cache_latents, args.n_traj, kmax=50)
    # D3 latent norms
    print("[D3] latent-norm histogram ...")
    ds_norms, goal_norms = latent_norms(cache_latents, jepa, args.env_family, args.device)

    # ---- summary.txt
    lines = [f"env_family: {args.env_family}", f"wm: {args.jepa}", f"done_threshold: {DONE_THRESHOLD}", ""]
    lines.append("== D1: predictor drift  ||z_hat_{t+d} - z_{t+d}||_2  (d = raw frames) ==")
    for d in sorted(drift):
        a = drift[d]
        lines.append(f"  d={d:3d} frames  mean={a.mean():.4f}  std={a.std():.4f}  n={len(a)}")
    lines.append("")
    lines.append("== D2: encoder L2 vs frame gap  ||z_t - z_{t+k}|| ==")
    for k in (1, 2, 5, 10, 25, 50):
        if k in geo:
            lines.append(f"  k={k:3d}  mean={geo[k][0]:.4f}  std={geo[k][1]:.4f}")
    lines.append("")
    lines.append("== D3: latent norms ||z|| ==")
    lines.append(f"  dataset: mean={ds_norms.mean():.4f} std={ds_norms.std():.4f} "
                 f"p5={np.percentile(ds_norms,5):.4f} p95={np.percentile(ds_norms,95):.4f}")
    if goal_norms.size:
        lines.append(f"  goals:   {np.round(goal_norms,3).tolist()}  mean={goal_norms.mean():.4f}")
    summary = "\n".join(lines)
    print("\n" + summary + "\n")
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as fh:
        fh.write(summary + "\n")

    np.savez(os.path.join(args.out_dir, "diagnostics.npz"),
             drift_d=np.array(sorted(drift)),
             drift_mean=np.array([drift[d].mean() for d in sorted(drift)]),
             drift_std=np.array([drift[d].std() for d in sorted(drift)]),
             geo_k=np.array(sorted(geo)),
             geo_mean=np.array([geo[k][0] for k in sorted(geo)]),
             dataset_norms=ds_norms, goal_norms=goal_norms)

    # ---- figures
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        ds = sorted(drift)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(ds, [drift[d].mean() for d in ds], yerr=[drift[d].std() for d in ds],
                    marker='o', capsize=3, label='predictor drift')
        ax.axhline(DONE_THRESHOLD, ls='--', color='r', label=f'done-threshold $\\delta_d$={DONE_THRESHOLD}')
        ax.set_xlabel('prediction horizon $\\Delta$ (raw frames)')
        ax.set_ylabel('mean $\\ell_2$ drift')
        ax.set_title(f'Predictor drift vs horizon ({args.env_family})')
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "predictor_drift.png"), dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(ds_norms, bins=80, density=True, alpha=0.6, label=f'dataset (μ={ds_norms.mean():.2f})')
        for g in goal_norms:
            ax.axvline(g, color='C3', alpha=0.7)
        if goal_norms.size:
            ax.axvline(goal_norms[0], color='C3', alpha=0.7, label='task goals')
        ax.set_xlabel('$||z||$'); ax.set_ylabel('density')
        ax.set_title(f'Latent-norm concentration ({args.env_family})')
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "goal_norms_ood.png"), dpi=120)
        plt.close(fig)
        print(f"[diag] figures + summary written to {args.out_dir}")
    except ImportError:
        print("[diag] matplotlib unavailable — arrays saved, skipping figures")


if __name__ == '__main__':
    main()
