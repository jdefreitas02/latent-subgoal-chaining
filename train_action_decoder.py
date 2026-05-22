"""
train_action_decoder.py
Pretrains a frozen goal-conditioned action-chunk decoder
D_theta : R^5 x R^192 x R^192 -> R^25.

Used by train_hiql_wgsp.py so that the LL policy can output a single 5-D
env action while still feeding the world model a 25-D action chunk
(frameskip=5 stacked actions). Dataset target is the StandardScaler-
normalised 25-D chunk; input is (chunk_first_action, latent_z_t, latent_z_goal).

Goal latents are sampled with HER random-future relabeling (same strategy as
the HIQL dataset): for step t, goal index g ~ Uniform(t+1, T) from same episode.

Output:
  {save_dir}/action_decoder.pth     state_dict
  {save_dir}/decoder_meta.pt        {'in_dim': 5, 'out_dim': 25, 'latent_dim': 192,
                                     'goal_dim': 192, 'hidden_dims': [...], ...}
  {save_dir}/decoder_loss.csv       (epoch, train_mse, val_mse)
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing as sk_pre

import stable_worldmodel as swm


# =============================================================================
# Network
# =============================================================================

class ActionChunkDecoder(nn.Module):
    """D_theta(a_first, z, z_goal) -> a_chunk.

    a_first ∈ R^5    (StandardScaler-normalised, first env action of the chunk)
    z       ∈ R^192  (WM-step latent at the start of the chunk)
    z_goal  ∈ R^goal_dim  (raw WM latent of the task goal; 0 = no goal)
    Output  ∈ R^25   (StandardScaler-normalised, full 5×5 stacked chunk).

    Designed to be small so it can sit inside the WGSP rollout inner loop
    without dominating the WM forward.  goal_dim=0 recovers the original
    state-only decoder (backward compatible).
    """

    def __init__(self, in_dim=5, out_dim=25, latent_dim=192,
                 hidden_dims=(256, 256), goal_dim=192):
        super().__init__()
        self.goal_dim = goal_dim
        in_d = in_dim + latent_dim + goal_dim
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.GELU()]
            in_d = h
        layers.append(nn.Linear(in_d, out_dim))
        self.net = nn.Sequential(*layers)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
        # Final layer small so initial output is near zero.
        nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)
        nn.init.constant_(self.net[-1].bias, 0.0)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.latent_dim = latent_dim

    def forward(self, a_first, z, z_goal=None):
        parts = [a_first, z]
        if self.goal_dim > 0:
            if z_goal is None:
                raise ValueError("ActionChunkDecoder: goal_dim>0 but z_goal=None")
            parts.append(z_goal)
        return self.net(torch.cat(parts, dim=-1))


# =============================================================================
# Data extraction (mirrors RealOfflineCache slicing)
# =============================================================================

def build_decoder_dataset(all_latents, all_actions, action_scaler,
                          frameskip=5, device='cpu', goal_conditioned=True):
    """Returns (a_first[N,5], z[N,192], z_goal[N,192], a_chunk[N,25]) on device.

    Mirrors the WM-step slicing used by RealOfflineCache so the decoder
    sees the exact same (z_t, action chunk) pairs as the LL policy.

    When goal_conditioned=True (default), z_goal is sampled with HER
    random-future relabeling: for WM-step t, goal index g ~ Uniform(t+1, n_wm)
    from the same episode (matching the HIQL dataset strategy).
    """
    raw_adim = all_actions[0].shape[-1] if all_actions else 5
    wm_adim  = frameskip * raw_adim

    a_first_list, z_list, z_goal_list, a_chunk_list = [], [], [], []
    skipped = 0

    for ep_z_raw, ep_a_raw in zip(all_latents, all_actions):
        ep_z = ep_z_raw.float()
        T    = ep_z.shape[0]
        n_wm = (T - 1) // frameskip
        if n_wm <= 1:   # need at least 2 WM steps for a future goal
            skipped += 1
            continue

        step_idx = torch.arange(0, (n_wm + 1) * frameskip, frameskip)[:n_wm + 1]
        ep_z_wm  = ep_z[step_idx]                                # [n_wm+1, 192]

        n_raw  = n_wm * frameskip
        scaled = action_scaler.transform(
            ep_a_raw[:n_raw].numpy().astype(np.float32))
        chunk  = torch.tensor(
            scaled.reshape(n_wm, wm_adim), dtype=torch.float32)  # [n_wm, 25]

        a_first = chunk[:, :raw_adim]                            # [n_wm, 5]

        if goal_conditioned:
            # HER random-future relabeling: for step t sample g ~ Uniform(t+1, n_wm)
            # ep_z_wm has n_wm+1 entries (indices 0..n_wm); WM steps are 0..n_wm-1.
            goal_indices = torch.randint(1, n_wm + 1, (n_wm,))  # always > step idx
            for t in range(n_wm):
                min_g = t + 1
                goal_indices[t] = torch.randint(min_g, n_wm + 1, (1,)).item()
            z_goal_ep = ep_z_wm[goal_indices]                    # [n_wm, 192]
        else:
            z_goal_ep = ep_z_wm[-1:].expand(n_wm, -1)           # last frame

        z_list.append(ep_z_wm[:-1])
        a_first_list.append(a_first)
        z_goal_list.append(z_goal_ep)
        a_chunk_list.append(chunk)

    if skipped:
        print(f"  build_decoder_dataset: skipped {skipped} short episodes")

    z       = torch.cat(z_list, dim=0).to(device)
    a_first = torch.cat(a_first_list, dim=0).to(device)
    z_goal  = torch.cat(z_goal_list, dim=0).to(device)
    a_chunk = torch.cat(a_chunk_list, dim=0).to(device)
    print(f"  build_decoder_dataset: {z.shape[0]:,} (z, a_first, z_goal, a_chunk) tuples")
    return a_first, z, z_goal, a_chunk


# =============================================================================
# Training
# =============================================================================

def train_decoder(decoder, a_first, z, z_goal, a_chunk, save_dir,
                  epochs=20, batch_size=4096, lr=3e-4, val_frac=0.05,
                  device='cuda'):
    os.makedirs(save_dir, exist_ok=True)

    n = a_first.shape[0]
    perm = torch.randperm(n, device=device)
    n_val = max(1, int(n * val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    opt = torch.optim.Adam(decoder.parameters(), lr=lr)
    csv_path = os.path.join(save_dir, 'decoder_loss.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_mse', 'val_mse'])

    goal_cond = decoder.goal_dim > 0
    best_val = float('inf')
    for epoch in range(epochs):
        decoder.train()
        perm_e = tr_idx[torch.randperm(tr_idx.shape[0], device=device)]
        tr_losses = []
        for s in range(0, perm_e.shape[0], batch_size):
            idx = perm_e[s:s + batch_size]
            pred = decoder(a_first[idx], z[idx],
                           z_goal[idx] if goal_cond else None)
            loss = ((pred - a_chunk[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            opt.step()
            tr_losses.append(loss.item())

        decoder.eval()
        with torch.no_grad():
            v_pred = decoder(a_first[val_idx], z[val_idx],
                             z_goal[val_idx] if goal_cond else None)
            v_mse = ((v_pred - a_chunk[val_idx]) ** 2).mean().item()

        tr_mse = float(np.mean(tr_losses))
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, tr_mse, v_mse])
        print(f"  Epoch {epoch:02d} | train_mse {tr_mse:.5f} | val_mse {v_mse:.5f}",
              flush=True)

        if v_mse < best_val:
            best_val = v_mse
            torch.save(decoder.state_dict(),
                       os.path.join(save_dir, 'action_decoder.pth'))

    meta = {
        'in_dim': decoder.in_dim,
        'out_dim': decoder.out_dim,
        'latent_dim': decoder.latent_dim,
        'goal_dim': decoder.goal_dim,
        'frameskip': decoder.out_dim // decoder.in_dim,
        'val_mse_best': best_val,
        'epochs': epochs,
    }
    torch.save(meta, os.path.join(save_dir, 'decoder_meta.pt'))
    print(f"  Best val_mse = {best_val:.5f}; checkpoint at "
          f"{os.path.join(save_dir, 'action_decoder.pth')}")
    return best_val


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_path', type=str, default=None,
                        help='Pre-computed latent cache (lewm_*_latents_cache.pt).')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='HDF5 dataset, used only for fitting the StandardScaler.')
    parser.add_argument('--save_dir', type=str,
                        default='./checkpoints_action_decoder')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--frameskip', type=int, default=5)
    parser.add_argument('--hidden', type=int, nargs='+', default=[256, 256])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    STABLEWM_HOME = os.environ.get(
        'STABLEWM_HOME', os.path.join(os.path.expanduser('~'), 'stable_wm_data'))
    cache_path = (args.cache_path or
                  os.path.join(STABLEWM_HOME, 'lewm_224_latents_cache.pt'))
    data_path  = (args.dataset_path or
                  os.path.join(STABLEWM_HOME, 'ogbench', 'cube_single_expert'))

    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    print(f"Loading cache: {cache_path}")
    cache = torch.load(cache_path, map_location='cpu')
    all_latents = cache['all_latents']
    all_actions = cache.get('all_actions', [])
    if not all_actions:
        raise RuntimeError("Cache has no 'all_actions'; rebuild with --save_actions.")

    print(f"Fitting StandardScaler on dataset actions: {data_path}")
    _ds_stem = data_path.rstrip('/')
    if _ds_stem.endswith('.h5'):
        _ds_stem = _ds_stem[:-3]
    ds = swm.data.HDF5Dataset(_ds_stem, keys_to_cache=['action'],
                              cache_dir=os.path.dirname(_ds_stem))
    a_raw = ds.get_col_data('action')
    a_raw = a_raw[~np.isnan(a_raw).any(axis=1)]
    scaler = sk_pre.StandardScaler()
    scaler.fit(a_raw)
    print(f"  Scaler fit on {len(a_raw):,} action frames")

    a_first, z, z_goal, a_chunk = build_decoder_dataset(
        all_latents, all_actions, scaler,
        frameskip=args.frameskip, device=device, goal_conditioned=True)

    goal_dim = z_goal.shape[-1]  # 192
    decoder = ActionChunkDecoder(
        in_dim=a_first.shape[-1],
        out_dim=a_chunk.shape[-1],
        latent_dim=z.shape[-1],
        hidden_dims=tuple(args.hidden),
        goal_dim=goal_dim,
    ).to(device)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder: {n_params:,} params  "
          f"(in={a_first.shape[-1]}, out={a_chunk.shape[-1]}, "
          f"latent={z.shape[-1]}, goal_dim={goal_dim}, hidden={args.hidden})")

    train_decoder(decoder, a_first, z, z_goal, a_chunk, args.save_dir,
                  epochs=args.epochs, batch_size=args.batch_size,
                  lr=args.lr, device=device)
