"""
train_flow_action_decoder.py
Row 13: conditional flow-matching action-chunk decoder D_θ.

Replaces the MSE ActionChunkDecoder with a flow model that learns
p(a_chunk | z, a_first) via conditional flow matching (Q-chunking style).
This is motivated by the chunk-variance audit (A3) which found intra-cluster
std/global std ≈ 0.51-0.63 — above the 0.30 trigger threshold — indicating
real multimodality that a deterministic MSE decoder collapses.

Architecture (adapts LLBCFlow from train_hiql_flow.py):
  u_θ(z, a_first, x_t, t) → velocity   [192+5+25+1 → hidden → 25]
  Trained with conditional flow matching:
    x0 ~ N(0,I),  x1 = a_chunk,  t ~ U(0,1)
    x_t = (1-t)·x0 + t·x1,  vel = x1 - x0
    L_cfm = ‖u_θ(z, a_first, x_t, t) - vel‖²

Outputs (same interface as train_action_decoder.py):
  {save_dir}/flow_decoder.pth          state_dict for FlowChunkDecoder
  {save_dir}/flow_decoder_meta.pt      metadata + val_nll
  {save_dir}/flow_decoder_loss.csv     (epoch, train_cfm, val_cfm)

Usage:
  python latent_hindsight_rl/train_flow_action_decoder.py \\
      --cache_path $STABLEWM_HOME/lewm_224_latents_cache.pt \\
      --dataset_path $STABLEWM_HOME/ogbench/cube_single_expert \\
      --save_dir checkpoints_flow_decoder \\
      --epochs 50
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing as sk_pre

_repo_root = os.path.abspath(os.path.dirname(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import stable_worldmodel as swm
from train_action_decoder import build_decoder_dataset


# =============================================================================
# Flow network
# =============================================================================

class FlowChunkDecoder(nn.Module):
    """Conditional flow decoder u_θ(z, a_first, x_t, t) → velocity.

    Conditioning: z ∈ R^192,  a_first ∈ R^5
    Noisy sample: x_t ∈ R^25,  t ∈ [0,1]
    Output: velocity ∈ R^25

    No LayerNorm (matches FQL/flow convention).  4 hidden layers (512 each).

    Euler sampling:
      x_0 ~ N(0,I)
      for step in range(flow_steps):
          t = step / flow_steps  (scalar → [B,1])
          x_{step+1} = x_step + (1/flow_steps) * u_θ(z, a_first, x_step, t)
      a_chunk = x_flow_steps
    """

    def __init__(self, latent_dim=192, a_first_dim=5, action_dim=25,
                 hidden_dims=(512, 512, 512, 512)):
        super().__init__()
        in_dim = latent_dim + a_first_dim + action_dim + 1
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.GELU()]
            d = h
        layers.append(nn.Linear(d, action_dim))
        self.net = nn.Sequential(*layers)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.latent_dim  = latent_dim
        self.a_first_dim = a_first_dim
        self.action_dim  = action_dim

    def forward(self, z, a_first, x_t, t):
        """
        z       [B, 192]
        a_first [B, 5]
        x_t     [B, 25]
        t       [B, 1]  float in [0,1]
        """
        return self.net(torch.cat([z, a_first, x_t, t], dim=-1))

    @torch.no_grad()
    def sample(self, z, a_first, flow_steps=10):
        """Draw a sample by Euler integration.

        Returns a_chunk [B, action_dim].
        """
        B = z.shape[0]
        x = torch.randn(B, self.action_dim, device=z.device)
        dt = 1.0 / flow_steps
        for step in range(flow_steps):
            t = torch.full((B, 1), step * dt, device=z.device)
            vel = self.forward(z, a_first, x, t)
            x = x + dt * vel
        return x


# =============================================================================
# Training
# =============================================================================

def cfm_loss(flow, z, a_first, a_chunk):
    """Conditional flow matching loss on a batch.

    L = ‖u_θ(z, a_first, x_t, t) - (x1 - x0)‖²
    """
    B = a_chunk.shape[0]
    x0 = torch.randn_like(a_chunk)
    x1 = a_chunk
    t  = torch.rand(B, 1, device=a_chunk.device)
    x_t = (1 - t) * x0 + t * x1
    vel_target = x1 - x0
    vel_pred   = flow(z, a_first, x_t, t)
    return (vel_pred - vel_target).pow(2).mean()


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    STABLEWM_HOME = os.environ.get(
        'STABLEWM_HOME', os.path.join(os.path.expanduser('~'), 'stable_wm_data'))
    cache_path = (args.cache_path or
                  os.path.join(STABLEWM_HOME, 'lewm_224_latents_cache.pt'))
    data_path  = (args.dataset_path or
                  os.path.join(STABLEWM_HOME, 'ogbench', 'cube_single_expert'))
    save_dir   = args.save_dir or 'checkpoints_flow_decoder'
    os.makedirs(save_dir, exist_ok=True)

    print(f'Loading cache: {cache_path}')
    cache = torch.load(cache_path, map_location='cpu')
    all_latents = cache['all_latents']
    all_actions = cache.get('all_actions', [])
    if not all_actions:
        raise RuntimeError("Cache has no 'all_actions'.")

    print(f'Fitting StandardScaler from {data_path}')
    ds = swm.data.HDF5Dataset(data_path, keys_to_cache=['action'],
                              cache_dir=os.path.dirname(data_path))
    a_raw = ds.get_col_data('action')
    a_raw = a_raw[~np.isnan(a_raw).any(axis=1)]
    scaler = sk_pre.StandardScaler()
    scaler.fit(a_raw)
    print(f'  Scaler fit on {len(a_raw):,} action frames')

    a_first, z, a_chunk = build_decoder_dataset(
        all_latents, all_actions, scaler,
        frameskip=args.frameskip, device=device)
    N = a_first.shape[0]
    print(f'  Total tuples: {N:,}')

    # Train/val split (90/10)
    perm    = torch.randperm(N, device=device)
    n_val   = max(1, N // 10)
    val_idx = perm[:n_val]
    tr_idx  = perm[n_val:]

    a_first_tr, z_tr, a_chunk_tr = a_first[tr_idx], z[tr_idx], a_chunk[tr_idx]
    a_first_va, z_va, a_chunk_va = a_first[val_idx], z[val_idx], a_chunk[val_idx]
    print(f'  Train: {len(tr_idx):,}  Val: {len(val_idx):,}')

    flow = FlowChunkDecoder(
        latent_dim=args.latent_dim, a_first_dim=5, action_dim=25,
        hidden_dims=tuple(args.hidden_dims),
    ).to(device)
    print(f'  FlowChunkDecoder params: {sum(p.numel() for p in flow.parameters()):,}')

    opt = torch.optim.Adam(flow.parameters(), lr=args.lr)

    csv_path = os.path.join(save_dir, 'flow_decoder_loss.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_cfm', 'val_cfm'])

    best_val = float('inf')
    n_tr = len(tr_idx)
    for epoch in range(1, args.epochs + 1):
        flow.train()
        perm_ep = torch.randperm(n_tr, device=device)
        sum_loss = 0.0
        n_batches = 0
        for s in range(0, n_tr, args.batch_size):
            e   = min(s + args.batch_size, n_tr)
            idx = perm_ep[s:e]
            loss = cfm_loss(flow, z_tr[idx], a_first_tr[idx], a_chunk_tr[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            opt.step()
            sum_loss += loss.item()
            n_batches += 1

        # Validation
        flow.eval()
        with torch.no_grad():
            val_loss = 0.0
            n_vb = 0
            for s in range(0, len(val_idx), args.batch_size):
                e = min(s + args.batch_size, len(val_idx))
                vl = cfm_loss(flow, z_va[s:e], a_first_va[s:e], a_chunk_va[s:e])
                val_loss += vl.item()
                n_vb += 1
            val_loss /= max(n_vb, 1)

        tr_loss = sum_loss / max(n_batches, 1)
        print(f'Epoch {epoch:03d}/{args.epochs}  '
              f'train_cfm={tr_loss:.5f}  val_cfm={val_loss:.5f}', flush=True)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, tr_loss, val_loss])

        if val_loss < best_val:
            best_val = val_loss
            torch.save(flow.state_dict(), os.path.join(save_dir, 'flow_decoder.pth'))

    print(f'\nBest val_cfm = {best_val:.5f}')
    meta = {
        'latent_dim':   args.latent_dim,
        'a_first_dim':  5,
        'action_dim':   25,
        'hidden_dims':  args.hidden_dims,
        'frameskip':    args.frameskip,
        'val_cfm':      best_val,
        'epochs':       args.epochs,
        'flow_steps':   args.flow_steps,
    }
    torch.save(meta, os.path.join(save_dir, 'flow_decoder_meta.pt'))
    print(f'Saved flow_decoder.pth and flow_decoder_meta.pt to {save_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train flow-matching chunk decoder (row 13).')
    parser.add_argument('--cache_path',    type=str, default=None)
    parser.add_argument('--dataset_path',  type=str, default=None)
    parser.add_argument('--save_dir',      type=str, default=None)

    parser.add_argument('--latent_dim',    type=int,   default=192)
    parser.add_argument('--hidden_dims',   type=int,   nargs='+',
                        default=[512, 512, 512, 512])
    parser.add_argument('--frameskip',     type=int,   default=5)

    parser.add_argument('--epochs',        type=int,   default=50)
    parser.add_argument('--batch_size',    type=int,   default=512)
    parser.add_argument('--lr',            type=float, default=3e-4)
    parser.add_argument('--flow_steps',    type=int,   default=10,
                        help='Euler integration steps at sample time.')
    parser.add_argument('--seed',          type=int,   default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
