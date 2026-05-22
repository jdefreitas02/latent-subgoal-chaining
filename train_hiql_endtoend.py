"""
train_hiql_endtoend.py

End-to-end HIQL with a trainable ImpalaSmall CNN encoder on raw 224×224 pixels.
Directly comparable to train_hiql_baseline.py but replaces the frozen 192D LeWM
encoder with a trainable CNN trained jointly with value/actor losses.

Architecture:
  pixels [B, 3, H, W] → ImpalaSmall → 256D → goal_rep / value / actors

ImpalaSmall mirrors ogbench's impala_small but adds AdaptiveAvgPool2d((4,4)) so
it handles any input resolution (84×84 or 224×224) and always produces 512→256D.
At 64×64 this is exactly ogbench's encoder; at 224×224 the pooling compresses
the larger spatial map to the same 4×4 footprint.

All HIQL hyperparameters match ogbench cube-single-play defaults exactly.
LATENT_DIM=256 (encoder output dim, matching ogbench).
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# =============================================================================
# 1. CNN Encoder
# =============================================================================

class ImpalaSmall(nn.Module):
    """impala_small from ogbench, adapted for any input resolution.

    Adds AdaptiveAvgPool2d((4,4)) after the conv stack so the spatial map
    is always 4×4 regardless of input resolution. Produces 512→out_dim embedding,
    matching ogbench's 256D encoder output when out_dim=256.

    Weight init: orthogonal (standard for RL CNNs).
    """

    def __init__(self, out_dim=256):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(32 * 4 * 4, out_dim)
        self.out_dim = out_dim
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """x: [B, 3, H, W] float32, ImageNet-normalized. Returns [B, out_dim]."""
        return F.relu(self.fc(self.convs(x).flatten(1)))


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


def normalize_imgs(imgs_uint8, device):
    """[B, 3, H, W] uint8 → [B, 3, H, W] float32, ImageNet-normalized."""
    x = imgs_uint8.to(device, dtype=torch.float32) / 255.0
    mean = _IMAGENET_MEAN.to(device).view(1, 3, 1, 1)
    std  = _IMAGENET_STD.to(device).view(1, 3, 1, 1)
    return (x - mean) / std


# =============================================================================
# 2. HIQL Networks (identical to train_hiql_baseline.py, LATENT_DIM=256)
# =============================================================================

def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


class LengthNormalize(nn.Module):
    def forward(self, x):
        return x / (x.norm(dim=-1, keepdim=True) + 1e-8) * (x.shape[-1] ** 0.5)


class GoalRep(nn.Module):
    """phi([z; g]): MLP + LengthNormalize. Identical to train_hiql_baseline.py."""

    def __init__(self, latent_dim=256, rep_dim=10,
                 hidden_dims=(512, 512, 512), layer_norm=True):
        super().__init__()
        in_dim = latent_dim * 2
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            if layer_norm:
                layers.append(nn.LayerNorm(h))
            in_dim = h
        layers.append(nn.Linear(in_dim, rep_dim))
        self.mlp = nn.Sequential(*layers)
        self.normalize = LengthNormalize()
        self.mlp.apply(_init_weights)

    def forward(self, z, g):
        return self.normalize(self.mlp(torch.cat([z, g], dim=-1)))


class GaussianActor(nn.Module):
    """ogbench GCActor with const_std=True. Identical to train_hiql_baseline.py."""

    def __init__(self, input_dim, output_dim,
                 hidden_dims=(512, 512, 512), layer_norm=True):
        super().__init__()
        in_dim = input_dim
        backbone = []
        for h in hidden_dims:
            backbone.append(nn.Linear(in_dim, h))
            backbone.append(nn.GELU())
            if layer_norm:
                backbone.append(nn.LayerNorm(h))
            in_dim = h
        self.backbone = nn.Sequential(*backbone)
        self.mean_head = nn.Linear(in_dim, output_dim)
        self.backbone.apply(_init_weights)
        nn.init.uniform_(self.mean_head.weight, -1e-3, 1e-3)
        nn.init.constant_(self.mean_head.bias, 0.0)

    def _mean(self, inp):
        return self.mean_head(self.backbone(inp))

    def log_prob(self, inp, target):
        mean = self._mean(inp)
        return Normal(mean, torch.ones_like(mean)).log_prob(target).sum(dim=-1)

    def mode(self, inp):
        return self._mean(inp)


class TwinValue(nn.Module):
    """Ensembled V(z, phi). Identical to train_hiql_baseline.py."""

    def __init__(self, latent_dim=256, rep_dim=10,
                 hidden_dims=(512, 512, 512), layer_norm=True):
        super().__init__()
        in_dim = latent_dim + rep_dim

        def _make_v():
            layers, d = [], in_dim
            for h in hidden_dims:
                layers.append(nn.Linear(d, h))
                layers.append(nn.GELU())
                if layer_norm:
                    layers.append(nn.LayerNorm(h))
                d = h
            layers.append(nn.Linear(d, 1))
            net = nn.Sequential(*layers)
            net.apply(_init_weights)
            return net

        self.v1 = _make_v()
        self.v2 = _make_v()

    def forward(self, z, phi):
        x = torch.cat([z, phi], dim=-1)
        return self.v1(x).squeeze(-1), self.v2(x).squeeze(-1)


# =============================================================================
# 3. Image Offline Cache
# =============================================================================

class ImageOfflineCache:
    """Offline cache for end-to-end training: stores raw pixel images in uint8
    CPU RAM, encodes on-the-fly during training with the CNN encoder.

    Mirrors RealOfflineCache episode structure and HGCDataset sampling exactly.
    Images are stored at cache_img_size×cache_img_size to manage memory:
      - 84×84:  ~4GB for 200k frames (default, manageable on 32GB RAM)
      - 224×224: ~30GB for 200k frames (needs 64GB+ RAM; use on HPC)

    sample_batch() returns 6 sets of [B, 3, H, W] uint8 pixel tensors plus
    actions/rewards/masks. The encoder is called once per step on all 6*B images.
    """

    def __init__(self, dataset, all_actions, action_scaler,
                 cache_img_size, device='cuda'):
        self.device = device

        n_eps = len(dataset.lengths)
        a_list, img_list = [], []
        ep_offsets, ep_n_list = [], []
        ep_ids, t_within = [], []
        lat_offset = flat_offset = skipped = 0

        print(f"  Loading {n_eps} episodes at {cache_img_size}×{cache_img_size} …")
        t0 = time.time()
        BATCH = 16  # episodes per load_chunk call

        for i in range(0, n_eps, BATCH):
            end_i = min(i + BATCH, n_eps)
            ep_idx = np.arange(i, end_i)
            ep_lens = dataset.lengths[ep_idx]
            starts = np.zeros(len(ep_idx), dtype=int)
            chunks = dataset.load_chunk(ep_idx, starts, ep_lens)

            for j, chunk in enumerate(chunks):
                ep_pix = chunk['pixels']         # [T, 3, H, W] uint8
                ep_a   = chunk.get('action',
                                   all_actions[i + j] if all_actions else None)
                if ep_a is None:
                    raise RuntimeError("No action data in chunk and all_actions not provided.")
                ep_a = ep_a.float()              # [T, 5]

                T = ep_pix.shape[0]
                n = T - 1
                if n <= 0:
                    skipped += 1
                    continue

                # Resize if caching at a smaller size than native
                native_h = ep_pix.shape[-2]
                if native_h != cache_img_size:
                    ep_pix_f = ep_pix.float() / 255.0  # [T, 3, H, W]
                    ep_pix_r = F.interpolate(
                        ep_pix_f, size=(cache_img_size, cache_img_size),
                        mode='bilinear', align_corners=False,
                    )
                    ep_pix = (ep_pix_r * 255).byte()   # [T, 3, cs, cs] uint8

                scaled = action_scaler.transform(ep_a[:n].numpy().astype(np.float32))
                a_list.append(torch.tensor(scaled, dtype=torch.float32))

                ep_id = len(ep_offsets)
                img_list.append(ep_pix.cpu())
                ep_offsets.append(lat_offset)
                ep_n_list.append(n)
                ep_ids.extend([ep_id] * n)
                t_within.extend(range(n))
                lat_offset += T
                flat_offset += n

            if (i // BATCH) % 10 == 0:
                print(f"    {end_i}/{n_eps} episodes  ({time.time()-t0:.0f}s)", flush=True)

        if skipped:
            print(f"  Skipped {skipped} short episodes.")

        # Concatenate into flat tensors
        self.images_flat = torch.cat(img_list, dim=0)  # [sum T_i, 3, cs, cs] uint8 on CPU
        self.a_flat = torch.cat(a_list, dim=0).to(device)
        self.ep_offsets = torch.tensor(ep_offsets, dtype=torch.long, device=device)
        self.ep_n       = torch.tensor(ep_n_list,  dtype=torch.long, device=device)
        self.ep_ids     = torch.tensor(ep_ids,     dtype=torch.long, device=device)
        self.t_within   = torch.tensor(t_within,   dtype=torch.long, device=device)
        self.total = flat_offset
        self.n_eps = len(ep_offsets)

        mem_gb = self.images_flat.numel() / 1e9
        print(f"  ImageOfflineCache: {self.n_eps} eps, {self.total:,} transitions, "
              f"image cache {mem_gb:.1f} GB RAM", flush=True)

    def _imgs_at(self, ep_ids_t, ts_t):
        """Gather images for a batch of (ep_id, t) pairs.
        Returns [B, 3, H, W] uint8 on device."""
        flat_idx = (self.ep_offsets[ep_ids_t] + ts_t).cpu()
        return self.images_flat[flat_idx].to(self.device)

    def sample_batch(self, batch_size, subgoal_steps,
                     p_curgoal=0.2, p_trajgoal=0.5, p_randomgoal=0.3,
                     discount=0.99):
        """Mirrors HGCDataset.sample() exactly (same as RealOfflineCache)."""
        idxs = torch.randint(0, self.total, (batch_size,), device=self.device)
        ep_id = self.ep_ids[idxs]
        t = self.t_within[idxs]
        n = self.ep_n[ep_id]

        # Value goals: HER mix with geometric offset
        roll_type = torch.rand(batch_size, device=self.device)
        geom_off  = (torch.distributions.Geometric(probs=1.0 - discount)
                     .sample((batch_size,)).long().to(self.device) + 1)
        traj_t = (t + geom_off).clamp(max=n)

        rand_idxs = torch.randint(0, self.total, (batch_size,), device=self.device)
        rand_ep = self.ep_ids[rand_idxs]
        rand_t  = self.t_within[rand_idxs]

        is_cur  = roll_type < p_curgoal
        is_traj = (roll_type >= p_curgoal) & (roll_type < p_curgoal + p_trajgoal)

        value_ep = torch.where(is_cur | is_traj, ep_id, rand_ep)
        value_t  = torch.where(is_cur, t, torch.where(is_traj, traj_t, rand_t))

        successes = ((value_ep == ep_id) & (value_t == t)).float()
        rewards   = successes - 1.0
        masks     = 1.0 - successes

        low_t = (t + subgoal_steps).clamp(max=n)

        d      = torch.rand(batch_size, device=self.device)
        high_t = torch.round(
            torch.minimum(t + 1, n).float() * d + n.float() * (1.0 - d)
        ).long()
        high_target_t = (t + subgoal_steps).clamp(max=high_t)

        return dict(
            obs_imgs  = self._imgs_at(ep_id,    t),            # [B, 3, H, W]
            nobs_imgs = self._imgs_at(ep_id,    t + 1),
            vg_imgs   = self._imgs_at(value_ep, value_t),
            llg_imgs  = self._imgs_at(ep_id,    low_t),
            hlg_imgs  = self._imgs_at(ep_id,    high_t),
            hlt_imgs  = self._imgs_at(ep_id,    high_target_t),
            actions   = self.a_flat[idxs],
            rewards   = rewards,
            masks     = masks,
        )


# =============================================================================
# 4. IQL expectile loss
# =============================================================================

def _expectile_loss(adv, diff, expectile):
    weight = torch.where(adv >= 0,
                         torch.full_like(adv, expectile),
                         torch.full_like(adv, 1.0 - expectile))
    return (weight * diff.pow(2)).mean()


# =============================================================================
# 5. Training Loop
# =============================================================================

def train_loop(
    encoder, goal_rep, ll_actor, hl_actor, value_net, value_target,
    optimizer,
    img_cache,
    total_steps=500_000,
    subgoal_steps=10,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    expectile=0.7,
    alpha_low=3.0,
    alpha_high=3.0,
    save_dir='./checkpoints_hiql_endtoend',
    device='cuda',
    log_interval=100,
    save_interval=10_000,
):
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*65}", flush=True)
    print(f"  HIQL end-to-end (trainable ImpalaSmall encoder)", flush=True)
    print(f"  encoder_out_dim  : {encoder.out_dim}", flush=True)
    print(f"  total_steps      : {total_steps:,}", flush=True)
    print(f"  subgoal_steps    : {subgoal_steps}", flush=True)
    print(f"  rep_dim          : {goal_rep.mlp[-1].out_features}", flush=True)
    print(f"  batch_size       : {batch_size}", flush=True)
    print(f"  save_dir         : {save_dir}", flush=True)
    print(f"{'='*65}\n", flush=True)

    csv_path = os.path.join(save_dir, 'training_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['step', 'value_loss', 'll_actor_loss', 'hl_actor_loss', 'elapsed_s'])

    t0 = time.time()
    t_interval = time.time()
    sum_v = sum_ll = sum_hl = 0.0
    log_n = 0

    for step in range(total_steps):
        batch = img_cache.sample_batch(batch_size, subgoal_steps, discount=gamma)

        actions = batch['actions']
        rewards = batch['rewards']
        masks   = batch['masks']

        # ----- Encode all image sets -----
        # obs, vg, llg: need gradients (used in loss forward passes outside no_grad)
        # nobs, hlg, hlt: only used in no_grad advantage/target blocks
        #
        # Encode obs + vg + llg together (3*B images in one pass, grads enabled)
        obs_vg_llg_norm = normalize_imgs(
            torch.cat([batch['obs_imgs'], batch['vg_imgs'], batch['llg_imgs']], 0),
            device)
        enc_grad = encoder(obs_vg_llg_norm)     # [3*B, 256] with grad
        z_obs, z_vg, z_llg = enc_grad.split(batch_size, dim=0)

        # Encode nobs + hlg + hlt without gradients (target computations only)
        with torch.no_grad():
            nobs_hlg_hlt_norm = normalize_imgs(
                torch.cat([batch['nobs_imgs'], batch['hlg_imgs'], batch['hlt_imgs']], 0),
                device)
            enc_ng = encoder(nobs_hlg_hlt_norm)  # [3*B, 256] no grad
            z_nobs, z_hlg, z_hlt = enc_ng.split(batch_size, dim=0)

        # ----- Value loss -----
        with torch.no_grad():
            phi_next_t  = goal_rep(z_nobs, z_vg)
            v1n_t, v2n_t = value_target(z_nobs, phi_next_t)
            vn_min_t     = torch.min(v1n_t, v2n_t)
            q            = rewards + gamma * masks * vn_min_t

            phi_t_t     = goal_rep(z_obs, z_vg)
            v1_tt, v2_tt = value_target(z_obs, phi_t_t)
            v_tt         = (v1_tt + v2_tt) / 2
            adv          = q - v_tt

            q1 = rewards + gamma * masks * v1n_t
            q2 = rewards + gamma * masks * v2n_t

        phi_cur  = goal_rep(z_obs, z_vg)        # grad → encoder via z_obs, z_vg
        v1, v2   = value_net(z_obs, phi_cur)
        v_loss   = (_expectile_loss(adv, q1 - v1, expectile) +
                    _expectile_loss(adv, q2 - v2, expectile))

        # ----- LL actor loss (low_actor_rep_grad=True) -----
        with torch.no_grad():
            phi_ll_a  = goal_rep(z_obs, z_llg)
            phi_ll_b  = goal_rep(z_nobs, z_llg)
            v1c, v2c  = value_net(z_obs,  phi_ll_a)
            v1n, v2n  = value_net(z_nobs, phi_ll_b)
            v_curr    = (v1c + v2c) / 2
            v_next    = (v1n + v2n) / 2
            adv_ll    = v_next - v_curr
            w_ll      = (alpha_low * adv_ll).exp().clamp(max=100.0)

        phi_ll_grad = goal_rep(z_obs, z_llg)    # grad → encoder via z_obs, z_llg
        ll_inp      = torch.cat([z_obs, phi_ll_grad], dim=-1)
        log_p_ll    = ll_actor.log_prob(ll_inp, actions)
        ll_loss     = -(w_ll * log_p_ll).mean()

        # ----- HL actor loss -----
        with torch.no_grad():
            phi_hl_a  = goal_rep(z_obs,  z_hlg)
            phi_hl_b  = goal_rep(z_hlt,  z_hlg)
            v1t, v2t  = value_net(z_obs,  phi_hl_a)
            v1tk, v2tk= value_net(z_hlt,  phi_hl_b)
            v_t       = (v1t + v2t) / 2
            v_tk      = (v1tk + v2tk) / 2
            adv_hl    = v_tk - v_t
            w_hl      = (alpha_high * adv_hl).exp().clamp(max=100.0)
            target_rep = goal_rep(z_obs, z_hlt)

        hl_inp   = torch.cat([z_obs, z_hlg], dim=-1)
        log_p_hl = hl_actor.log_prob(hl_inp, target_rep)
        hl_loss  = -(w_hl * log_p_hl).mean()

        # ----- Combined backward -----
        total = v_loss + ll_loss + hl_loss
        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        # ----- EMA target update -----
        with torch.no_grad():
            for tp, p in zip(value_target.parameters(), value_net.parameters()):
                tp.data.mul_(1.0 - tau).add_(p.data * tau)

        sum_v  += v_loss.item()
        sum_ll += ll_loss.item()
        sum_hl += hl_loss.item()
        log_n  += 1

        if step % log_interval == 0:
            d   = max(log_n, 1)
            now = time.time()
            interval  = now - t_interval
            sps       = log_interval / max(interval, 1e-6)
            remaining = (total_steps - step) / max(sps, 1e-6)
            pct       = 100.0 * step / total_steps
            print(
                f"Step {step:06d}/{total_steps:,} ({pct:4.1f}%) | "
                f"V: {sum_v/d:.4f} | LL: {sum_ll/d:.4f} | HL: {sum_hl/d:.4f} | "
                f"{sps:.1f} sps | ETA: {remaining/60:.0f}m | "
                f"elapsed: {(now-t0)/60:.0f}m",
                flush=True,
            )
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([step, sum_v/d, sum_ll/d, sum_hl/d, interval])
            sum_v = sum_ll = sum_hl = 0.0
            log_n = 0
            t_interval = now

        if step > 0 and (step % save_interval == 0 or step == total_steps - 1):
            torch.save(encoder.state_dict(),   os.path.join(save_dir, 'encoder.pth'))
            torch.save(goal_rep.state_dict(),  os.path.join(save_dir, 'goal_rep.pth'))
            torch.save(ll_actor.state_dict(),  os.path.join(save_dir, 'll_actor.pth'))
            torch.save(hl_actor.state_dict(),  os.path.join(save_dir, 'hl_actor.pth'))
            torch.save(value_net.state_dict(), os.path.join(save_dir, 'value_net.pth'))
            print(f"  → checkpoint saved at step {step}", flush=True)


# =============================================================================
# 6. Main
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='HIQL end-to-end with trainable ImpalaSmall encoder on 224×224 pixels')

    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to HDF5 dataset (no .h5 extension)')
    parser.add_argument('--save_dir',     type=str, default=None)

    # Image / encoder
    parser.add_argument('--cache_img_size', type=int, default=224,
                        help='Resize images to this size for CPU RAM storage. '
                             '224=native (needs ~30GB RAM for 200k frames). '
                             '84=memory-efficient (~4GB, comparable to ogbench 64×64).')
    parser.add_argument('--encoder_out_dim', type=int, default=256,
                        help='ImpalaSmall output dim (256 matches ogbench).')

    # HIQL hyperparameters — match ogbench cube-single-play defaults
    parser.add_argument('--total_steps',   type=int,   default=500_000)
    parser.add_argument('--subgoal_steps', type=int,   default=10)
    parser.add_argument('--rep_dim',       type=int,   default=10)
    parser.add_argument('--batch_size',    type=int,   default=256)
    parser.add_argument('--alpha_low',     type=float, default=3.0)
    parser.add_argument('--alpha_high',    type=float, default=3.0)
    parser.add_argument('--gamma',         type=float, default=0.99)
    parser.add_argument('--expectile',     type=float, default=0.7)
    parser.add_argument('--lr',            type=float, default=3e-4)
    parser.add_argument('--tau',           type=float, default=0.005)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    parent_dir = os.path.abspath(os.path.dirname(__file__))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    import stable_worldmodel as swm
    from sklearn import preprocessing as sk_pre

    STABLEWM_HOME = os.environ.get(
        'STABLEWM_HOME', os.path.join(os.path.expanduser('~'), 'stable_wm_data'))

    data_path = args.dataset_path or os.path.join(
        STABLEWM_HOME, 'ogbench', 'visual-cube-single-play-v0_224')
    save_dir = args.save_dir or (
        f'./checkpoints_hiql_endtoend_k{args.subgoal_steps}_rep{args.rep_dim}')

    # Action scaler (from dataset)
    print('Fitting action scaler ...')
    _ds_act = swm.data.HDF5Dataset(
        data_path, keys_to_cache=['action'],
        cache_dir=os.path.dirname(data_path))
    action_raw = _ds_act.get_col_data('action')
    action_raw = action_raw[~np.isnan(action_raw).any(axis=1)]
    action_scaler = sk_pre.StandardScaler()
    action_scaler.fit(action_raw)
    print(f'  Scaler fit on {len(action_raw):,} action frames')

    # Load images from HDF5 dataset
    print(f'\nLoading dataset from {data_path} ...')
    dataset = swm.data.HDF5Dataset(data_path)
    print(f'  {len(dataset.lengths)} episodes, '
          f'total frames: {dataset.lengths.sum():,}')
    mem_est = dataset.lengths.sum() * 3 * args.cache_img_size**2 / 1e9
    print(f'  Estimated image cache size: {mem_est:.1f} GB '
          f'(at {args.cache_img_size}×{args.cache_img_size} uint8)')

    img_cache = ImageOfflineCache(
        dataset=dataset,
        all_actions=None,
        action_scaler=action_scaler,
        cache_img_size=args.cache_img_size,
        device=device,
    )

    LATENT_DIM  = args.encoder_out_dim
    ACTION_DIM  = 5
    REP_DIM     = args.rep_dim
    HIDDEN_DIMS = (512, 512, 512)

    encoder = ImpalaSmall(out_dim=LATENT_DIM).to(device)

    goal_rep = GoalRep(
        latent_dim=LATENT_DIM, rep_dim=REP_DIM,
        hidden_dims=HIDDEN_DIMS, layer_norm=True,
    ).to(device)

    ll_actor = GaussianActor(
        input_dim=LATENT_DIM + REP_DIM, output_dim=ACTION_DIM,
        hidden_dims=HIDDEN_DIMS, layer_norm=True,
    ).to(device)

    hl_actor = GaussianActor(
        input_dim=LATENT_DIM * 2, output_dim=REP_DIM,
        hidden_dims=HIDDEN_DIMS, layer_norm=True,
    ).to(device)

    value_net = TwinValue(
        latent_dim=LATENT_DIM, rep_dim=REP_DIM,
        hidden_dims=HIDDEN_DIMS, layer_norm=True,
    ).to(device)
    value_target = TwinValue(
        latent_dim=LATENT_DIM, rep_dim=REP_DIM,
        hidden_dims=HIDDEN_DIMS, layer_norm=True,
    ).to(device)
    value_target.load_state_dict(value_net.state_dict())
    for p in value_target.parameters():
        p.requires_grad = False

    enc_params = sum(p.numel() for p in encoder.parameters())
    print(
        f'\nNetworks (ImpalaSmall encoder, {LATENT_DIM}D, rep_dim={REP_DIM}):\n'
        f'  Encoder:  {enc_params:>10,} params\n'
        f'  GoalRep:  {sum(p.numel() for p in goal_rep.parameters()):>10,} params\n'
        f'  LL Actor: {sum(p.numel() for p in ll_actor.parameters()):>10,} params\n'
        f'  HL Actor: {sum(p.numel() for p in hl_actor.parameters()):>10,} params\n'
        f'  Value:    {sum(p.numel() for p in value_net.parameters()):>10,} params'
    )

    # Single Adam over all trainable params (encoder included)
    params = (list(encoder.parameters())
              + list(goal_rep.parameters())
              + list(ll_actor.parameters())
              + list(hl_actor.parameters())
              + list(value_net.parameters()))
    optimizer = torch.optim.Adam(params, lr=args.lr)

    train_loop(
        encoder=encoder, goal_rep=goal_rep, ll_actor=ll_actor, hl_actor=hl_actor,
        value_net=value_net, value_target=value_target,
        optimizer=optimizer,
        img_cache=img_cache,
        total_steps=args.total_steps,
        subgoal_steps=args.subgoal_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        expectile=args.expectile,
        alpha_low=args.alpha_low,
        alpha_high=args.alpha_high,
        save_dir=save_dir,
        device=device,
    )
