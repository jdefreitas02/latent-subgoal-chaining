"""Extended WM diagnostics for the thesis report — OOD ability, action
conditioning, goal-latent separation, and reward landscape. Env-parameterized so
the SAME script runs on the original cube-single WM and the new scene/puzzle WMs,
giving directly comparable figures.

Diagnostics (all written to --out_dir):

  A. OOD rollout manifold + action conditioning
     Roll the predictor f_psi forward H chunked steps from real context, under
     three action regimes: TRUE actions, SHUFFLED actions (wrong dynamics), ZERO
     actions. For each horizon record (i) drift vs encoded ground truth, (ii)
     dataset-NN L2 distance of the endpoint (does it stay on the manifold?),
     (iii) endpoint ||z||. If ZERO/SHUFFLED drift ~ TRUE drift, the WM ignores
     actions (MPC would be meaningless). If rollout endpoints leave the dataset-
     NN / norm band, the WM extrapolates off-manifold.
       -> ood_action_conditioning.png, ood_manifold.png

  B. Goal-latent separation map
     PCA-2D of dataset latents + the 5 rendered task goals, plus goal-vs-pool L2.
       -> goal_separation_pca.png

  C. Reward landscape
     Distribution of the dense reward r = -||z' - g|| over dataset next-states
     per task, and the fraction within done_threshold (sparse-reward density).
       -> reward_landscape.png

Needs: WM ckpt, latent cache (ground-truth latents), and a HDF5 with the
'action' column (state-only HDF5 is fine — pixels are NOT required; the predictor
rolls in latent space and act_emb = action_encoder(action)).
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
ACT_CHUNK = 25  # frameskip * action_dim


def load_actions_by_episode(h5_path):
    """Return list of per-episode action arrays (T_frames, 5) and ep metadata."""
    with h5py.File(h5_path, 'r') as f:
        actions = f['action'][...].astype(np.float32)
        ep_off = f['ep_offset'][...].astype(np.int64)
        ep_len = f['ep_len'][...].astype(np.int64)
    eps = [actions[o:o + l] for o, l in zip(ep_off, ep_len)]
    return eps


def chunk_action(raw_acts, t_frame):
    """25-D chunked action = 5 raw frames x 5-D starting at t_frame."""
    a = raw_acts[t_frame:t_frame + FRAMESKIP].reshape(-1)
    if a.shape[0] < ACT_CHUNK:
        a = np.concatenate([a, np.zeros(ACT_CHUNK - a.shape[0], np.float32)])
    return a


def rollout_diag(jepa, cache_latents, act_eps, device, n_anchors, horizons, refs, rng):
    """A: roll predictor under true/shuffled/zero actions; record drift, NN-dist, norm."""
    max_h = max(horizons)
    # per regime -> per horizon -> list of (drift, nn, norm)
    regimes = ['true', 'shuffled', 'zero']
    out = {r: {h: {'drift': [], 'nn': [], 'norm': []} for h in horizons} for r in regimes}

    # latents stored per raw frame; chunk granularity = stride FRAMESKIP
    usable = [i for i, z in enumerate(cache_latents)
              if z.shape[0] > (HISTORY + max_h) * FRAMESKIP + 1
              and act_eps[i].shape[0] > (HISTORY + max_h) * FRAMESKIP + 1]
    refs_t = torch.from_numpy(refs).to(device)
    refs_sq = (refs_t * refs_t).sum(-1)

    def nn_dist(z):  # z: (192,) torch
        d2 = (z * z).sum() + refs_sq - 2.0 * (refs_t @ z)
        return d2.clamp(min=0).min().sqrt().item()

    n_done = 0
    tries = 0
    while n_done < n_anchors and tries < n_anchors * 5:
        tries += 1
        ei = int(rng.choice(usable))
        z = cache_latents[ei]
        z = z.numpy() if torch.is_tensor(z) else np.asarray(z)
        raw = act_eps[ei]
        span = (HISTORY + max_h) * FRAMESKIP
        t0 = int(rng.integers(0, z.shape[0] - span - 1))

        # context latents at chunk stride, ground-truth future latents
        ctx_idx = [t0 + j * FRAMESKIP for j in range(HISTORY)]
        ctx_z = torch.from_numpy(np.stack([z[i] for i in ctx_idx])).float().to(device).unsqueeze(0)
        gt = {h: torch.from_numpy(z[t0 + (HISTORY + h - 1) * FRAMESKIP]).float().to(device)
              for h in horizons}

        # action chunks for context + future (true)
        a_idx = [t0 + j * FRAMESKIP for j in range(HISTORY + max_h)]
        true_chunks = np.stack([chunk_action(raw, ti) for ti in a_idx])  # (HISTORY+max_h, 25)

        for regime in regimes:
            if regime == 'true':
                chunks = true_chunks.copy()
            elif regime == 'zero':
                chunks = np.zeros_like(true_chunks)
            else:  # shuffled: scramble the FUTURE action order
                chunks = true_chunks.copy()
                fut = chunks[HISTORY:].copy()
                rng.shuffle(fut)
                chunks[HISTORY:] = fut
            ca = torch.from_numpy(chunks).float().to(device).unsqueeze(0)  # (1, N, 25)
            with torch.no_grad():
                act_emb_all = jepa.action_encoder(ca)  # (1, N, 192)
                chain = ctx_z.clone()
                ctx_act = act_emb_all[:, :HISTORY].clone()
                for step in range(max_h):
                    pred = jepa.predict(chain[:, -HISTORY:], ctx_act[:, -HISTORY:])
                    zh = pred[:, -1:]
                    chain = torch.cat([chain, zh], 1)
                    nxt = act_emb_all[:, HISTORY + step: HISTORY + step + 1]
                    ctx_act = torch.cat([ctx_act, nxt], 1)
                    h = step + 1
                    if h in horizons:
                        ze = zh.squeeze(0).squeeze(0)
                        out[regime][h]['drift'].append(torch.norm(ze - gt[h]).item())
                        out[regime][h]['nn'].append(nn_dist(ze))
                        out[regime][h]['norm'].append(torch.norm(ze).item())
        n_done += 1
    return out, n_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jepa', required=True)
    ap.add_argument('--hdf5', required=True, help='HDF5 with action column (state-only OK)')
    ap.add_argument('--cache', required=True)
    ap.add_argument('--env_family', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--n_anchors', type=int, default=300)
    ap.add_argument('--done_threshold', type=float, default=2.0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    from envs.jepa_loader import load_jepa, encode_pixels_to_latent
    print(f"[ood] loading WM {args.jepa}")
    jepa = load_jepa(args.jepa, device=args.device, img_size=224)

    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    cache_latents = cache['all_latents'] if isinstance(cache, dict) else cache
    h5 = args.hdf5 if args.hdf5.endswith('.h5') else args.hdf5 + '.h5'
    act_eps = load_actions_by_episode(h5)

    flat = np.concatenate([z.numpy() if torch.is_tensor(z) else np.asarray(z)
                           for z in cache_latents], axis=0).astype(np.float32)
    refs = flat[rng.choice(flat.shape[0], min(20000, flat.shape[0]), replace=False)]

    # ---- A: rollout / action conditioning / manifold
    horizons = [1, 2, 5, 10]   # chunk steps -> raw frames 5,10,25,50
    print("[A] rollout manifold + action conditioning ...")
    roll, n = rollout_diag(jepa, cache_latents, act_eps, args.device,
                           args.n_anchors, horizons, refs, rng)

    # ---- B: goal latents
    print("[B] goal separation ...")
    goals = []
    try:
        import gymnasium as gym
        import ogbench  # noqa
        for t in (1, 2, 3, 4, 5):
            env = gym.make(f"visual-{args.env_family}-singletask-task{t}-v0", width=224, height=224)
            _, info = env.reset(seed=0, options=dict(render_goal=True))
            g = info.get("goal", info.get("target"))
            goals.append(encode_pixels_to_latent(jepa, g, args.device))
            env.close()
        goals = np.stack(goals).astype(np.float32)
    except Exception as e:
        print(f"  goal render skipped: {type(e).__name__}: {str(e)[:80]}")
        goals = np.zeros((0, LATENT_DIM), np.float32)

    # ---- C: reward landscape (dense reward -||z'-g|| per task)
    print("[C] reward landscape ...")
    pool = flat[rng.choice(flat.shape[0], min(20000, flat.shape[0]), replace=False)]
    reward_by_task, frac_within = {}, {}
    for ti in range(goals.shape[0]):
        d = np.linalg.norm(pool - goals[ti][None], axis=1)
        reward_by_task[ti + 1] = -d
        frac_within[ti + 1] = float((d < args.done_threshold).mean())

    # ---- summary
    L = [f"env_family: {args.env_family}", f"wm: {args.jepa}",
         f"n_anchors: {n}  done_threshold: {args.done_threshold}", ""]
    L.append("== A. drift (mean L2) by action regime and horizon (raw frames) ==")
    L.append(f"   {'Δframes':>8} {'true':>8} {'shuffled':>9} {'zero':>8}")
    for h in horizons:
        d = h * FRAMESKIP
        tr = np.mean(roll['true'][h]['drift']); sh = np.mean(roll['shuffled'][h]['drift'])
        ze = np.mean(roll['zero'][h]['drift'])
        L.append(f"   {d:>8} {tr:>8.3f} {sh:>9.3f} {ze:>8.3f}")
    L.append("   (shuffled/zero >> true  => WM is action-conditioned, good for MPC)")
    L.append("")
    L.append("== A. rollout endpoint dataset-NN L2 (manifold; true actions) ==")
    for h in horizons:
        nn = np.mean(roll['true'][h]['nn'])
        L.append(f"   Δ={h*FRAMESKIP:>3} frames  NN={nn:.3f}")
    L.append("")
    L.append("== A. rollout endpoint ||z|| vs dataset band ==")
    L.append(f"   dataset ||z||: mean={np.linalg.norm(refs,axis=1).mean():.3f} "
             f"std={np.linalg.norm(refs,axis=1).std():.3f}")
    for h in horizons:
        nm = np.mean(roll['true'][h]['norm'])
        L.append(f"   Δ={h*FRAMESKIP:>3} frames  rollout ||z||={nm:.3f}")
    L.append("")
    if goals.shape[0]:
        gg = np.linalg.norm(goals[:, None] - goals[None], axis=-1)
        iu = np.triu_indices(goals.shape[0], 1)
        L.append("== B. goal-vs-goal L2 ==")
        L.append(f"   min={gg[iu].min():.3f} mean={gg[iu].mean():.3f} max={gg[iu].max():.3f}")
        L.append("== C. sparse-reward false-positive frac (per task) ==")
        L.append("   " + "  ".join(f"t{k}:{frac_within[k]:.4f}" for k in frac_within))
    summary = "\n".join(L)
    print("\n" + summary + "\n")
    with open(os.path.join(args.out_dir, "ood_summary.txt"), "w") as fh:
        fh.write(summary + "\n")

    np.savez(os.path.join(args.out_dir, "ood_diagnostics.npz"),
             horizons_frames=np.array([h * FRAMESKIP for h in horizons]),
             **{f"drift_{r}": np.array([np.mean(roll[r][h]['drift']) for h in horizons]) for r in roll},
             nn_true=np.array([np.mean(roll['true'][h]['nn']) for h in horizons]),
             norm_true=np.array([np.mean(roll['true'][h]['norm']) for h in horizons]),
             dataset_norm_mean=float(np.linalg.norm(refs, axis=1).mean()),
             goals=goals)

    # ---- figures
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        xs = [h * FRAMESKIP for h in horizons]

        # A1 action conditioning
        fig, ax = plt.subplots(figsize=(6, 4))
        for r, c in [('true', 'C0'), ('shuffled', 'C1'), ('zero', 'C2')]:
            ax.plot(xs, [np.mean(roll[r][h]['drift']) for h in horizons], 'o-', color=c, label=r)
        ax.axhline(args.done_threshold, ls='--', color='r', label=f'δ_d={args.done_threshold}')
        ax.set_xlabel('horizon Δ (raw frames)'); ax.set_ylabel('mean L2 drift')
        ax.set_title(f'Action conditioning ({args.env_family})'); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, 'ood_action_conditioning.png'), dpi=120)
        plt.close(fig)

        # A2 manifold: NN-distance (true vs zero) at largest horizon + norm band
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        Hh = horizons[-1]
        axes[0].hist(roll['true'][Hh]['nn'], bins=40, alpha=0.6, density=True,
                     label=f'true (μ={np.mean(roll["true"][Hh]["nn"]):.2f})')
        axes[0].hist(roll['zero'][Hh]['nn'], bins=40, alpha=0.6, density=True,
                     label=f'zero-act (μ={np.mean(roll["zero"][Hh]["nn"]):.2f})')
        axes[0].set_xlabel('endpoint dataset-NN L2'); axes[0].set_ylabel('density')
        axes[0].set_title(f'Manifold proximity @Δ={Hh*FRAMESKIP}f'); axes[0].legend()
        dn = np.linalg.norm(refs, axis=1)
        axes[1].hist(dn, bins=60, alpha=0.5, density=True, label=f'dataset (μ={dn.mean():.2f})')
        axes[1].hist(roll['true'][Hh]['norm'], bins=40, alpha=0.6, density=True, label='rollout endpoints')
        axes[1].set_xlabel('||z||'); axes[1].set_title('SIGReg norm band'); axes[1].legend()
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, 'ood_manifold.png'), dpi=120)
        plt.close(fig)

        # B goal separation PCA
        if goals.shape[0]:
            from sklearn.decomposition import PCA
            sub = pool[rng.choice(pool.shape[0], min(4000, pool.shape[0]), replace=False)]
            pca = PCA(n_components=2).fit(sub)
            ps, pg = pca.transform(sub), pca.transform(goals)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(ps[:, 0], ps[:, 1], s=4, alpha=0.2, color='gray', label='dataset')
            for i in range(goals.shape[0]):
                ax.scatter(pg[i, 0], pg[i, 1], s=180, marker='*', label=f'task {i+1}')
            ax.set_title(f'Goal-latent separation PCA ({args.env_family})')
            ax.legend(markerscale=1.0, fontsize=8); fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, 'goal_separation_pca.png'), dpi=120)
            plt.close(fig)

            # C reward landscape
            fig, ax = plt.subplots(figsize=(6, 4))
            for k in reward_by_task:
                ax.hist(reward_by_task[k], bins=60, alpha=0.4, density=True, label=f'task{k}')
            ax.axvline(-args.done_threshold, ls='--', color='r', label=f'-δ_d')
            ax.set_xlabel('dense reward  -||z\'-g||'); ax.set_ylabel('density')
            ax.set_title(f'Reward landscape ({args.env_family})'); ax.legend(fontsize=8)
            fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, 'reward_landscape.png'), dpi=120)
            plt.close(fig)
        print(f"[ood] figures + summary written to {args.out_dir}")
    except ImportError as e:
        print(f"[ood] plotting skipped: {e}")


if __name__ == '__main__':
    main()
