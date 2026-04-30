"""
train_hiql_baseline.py
Faithful PyTorch port of ~/ogbench/impls/agents/hiql.py for visual-cube-single-play-v0.

Identical to ogbench HIQL except the visual encoder is the frozen LeWM ViT-tiny
(192D latents from a pre-computed cache, instead of a trainable impala_small CNN).
Everything else mirrors ogbench:
  - Native 5D actions, one transition per env step (no frameskip, no chunking)
  - subgoal_steps in raw env steps (cube-single default = 10)
  - batch_size=256, lr=3e-4, 500k steps
  - Single batch per step shared by value/LL/HL (HGCDataset.sample equivalent)
  - Single total_loss.backward() so gradients on goal_rep sum from value + LL
  - low_actor_rep_grad=True (LL grads flow into goal_rep, as in pixel HIQL)
  - HL goal sampling: uniform future state (actor_p_randomgoal=0)
  - Value goal sampling: 20%/50%/30% cur/traj/random with geometric traj offsets
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
# 1. Network Architectures (mirror ogbench utils/networks.py)
# =============================================================================

def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


class LengthNormalize(nn.Module):
    def forward(self, x):
        return x / (x.norm(dim=-1, keepdim=True) + 1e-8) * (x.shape[-1] ** 0.5)


class GoalRep(nn.Module):
    """phi([z; g]): MLP(value_hidden_dims + [rep_dim], activate_final=False) + LengthNormalize."""

    def __init__(self, latent_dim=192, rep_dim=10,
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
    """ogbench GCActor with const_std=True, state_dependent_std=False, tanh_squash=False."""

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
        # const_std=True: std = 1
        return Normal(mean, torch.ones_like(mean)).log_prob(target).sum(dim=-1)

    def mode(self, inp):
        return self._mean(inp)


class TwinValue(nn.Module):
    """Ensembled V(z, phi) — two independent MLPs, returns (v1, v2)."""

    def __init__(self, latent_dim=192, rep_dim=10,
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


class LatentAdapter(nn.Module):
    """Trainable 2-layer MLP that projects frozen 192D LeWM latents into a
    task-relevant space, trained end-to-end with the HIQL value/actor losses.

    Maps 192D → out_dim (default 256, matching ogbench impala_small output).
    When out_dim=256 the downstream network widths match ogbench exactly.
    """

    def __init__(self, in_dim=192, out_dim=256, layer_norm=True):
        super().__init__()
        layers = [nn.Linear(in_dim, out_dim), nn.GELU()]
        if layer_norm:
            layers.append(nn.LayerNorm(out_dim))
        layers += [nn.Linear(out_dim, out_dim), nn.GELU()]
        if layer_norm:
            layers.append(nn.LayerNorm(out_dim))
        self.mlp = nn.Sequential(*layers)
        self.mlp.apply(_init_weights)

    def forward(self, x):
        return self.mlp(x)


# =============================================================================
# 2. Data Pipeline (mirrors ogbench HGCDataset.sample exactly)
# =============================================================================

class RealOfflineCache:
    """Sampler from pre-computed LeWM latents. One transition = one env step.

    Mirrors ogbench HGCDataset.sample: a single batch contains observations,
    next_observations, actions, rewards, masks, value_goals, low_actor_goals,
    high_actor_goals, and high_actor_targets — all indexed off the same idxs.
    """

    def __init__(self, all_latents, all_actions, action_scaler, device='cuda'):
        self.device = device

        a_list = []
        ep_offsets, ep_n_list = [], []
        ep_ids, t_within = [], []
        all_lat = []

        flat_offset = 0   # flat transition counter (excludes terminal states)
        lat_offset = 0    # flat latent counter (includes terminal states)
        skipped = 0

        for ep_z, ep_a in zip(all_latents, all_actions):
            ep_z = ep_z.float()
            T = ep_z.shape[0]
            n = T - 1   # number of transitions in this episode
            if n <= 0:
                skipped += 1
                continue

            scaled = action_scaler.transform(ep_a[:n].numpy().astype(np.float32))
            a_list.append(torch.tensor(scaled, dtype=torch.float32))

            all_lat.append(ep_z)
            ep_id = len(ep_offsets)
            ep_offsets.append(lat_offset)
            ep_n_list.append(n)
            ep_ids.extend([ep_id] * n)
            t_within.extend(range(n))

            lat_offset += T
            flat_offset += n

        if skipped:
            print(f"  RealOfflineCache: skipped {skipped} short episodes")

        self.total = flat_offset
        self.n_eps = len(ep_offsets)
        print(f"  RealOfflineCache: {self.n_eps} episodes, {self.total:,} transitions")

        self.latents_flat = torch.cat(all_lat, dim=0).to(device)              # [sum T_i, 192]
        self.a_flat = torch.cat(a_list, dim=0).to(device)                     # [sum n_i, 5]
        self.ep_offsets = torch.tensor(ep_offsets, dtype=torch.long, device=device)  # [n_eps]
        self.ep_n = torch.tensor(ep_n_list, dtype=torch.long, device=device)         # [n_eps]
        self.ep_ids = torch.tensor(ep_ids, dtype=torch.long, device=device)          # [total]
        self.t_within = torch.tensor(t_within, dtype=torch.long, device=device)      # [total]

    def _z_at(self, ep_id, t):
        return self.latents_flat[self.ep_offsets[ep_id] + t]

    def sample_batch(self, batch_size, subgoal_steps,
                     p_curgoal=0.2, p_trajgoal=0.5, p_randomgoal=0.3,
                     discount=0.99):
        """Mirrors HGCDataset.sample() for cube-single-play config:
            value_geom_sample=True, actor_p_randomgoal=0.0, actor_geom_sample=False.
        """
        idxs = torch.randint(0, self.total, (batch_size,), device=self.device)
        ep_id = self.ep_ids[idxs]
        t = self.t_within[idxs]
        n = self.ep_n[ep_id]   # last valid t = n (terminal latent index within ep)

        # ---- Value goals: HER mix ----
        roll_type = torch.rand(batch_size, device=self.device)
        # Geometric offsets ≥ 1 (np.random.geometric starts at 1, torch starts at 0)
        geom_off = (torch.distributions.Geometric(probs=1.0 - discount)
                    .sample((batch_size,)).long().to(self.device) + 1)
        traj_t = (t + geom_off).clamp(max=n)

        rand_idxs = torch.randint(0, self.total, (batch_size,), device=self.device)
        rand_ep = self.ep_ids[rand_idxs]
        rand_t = self.t_within[rand_idxs]

        is_cur = roll_type < p_curgoal
        is_traj = (roll_type >= p_curgoal) & (roll_type < p_curgoal + p_trajgoal)

        value_ep = torch.where(is_cur | is_traj, ep_id, rand_ep)
        value_t = torch.where(is_cur, t,
                  torch.where(is_traj, traj_t, rand_t))

        # ogbench: successes = (idxs == value_goal_idxs). Same flat idx ⇒ same ep AND same t.
        successes = ((value_ep == ep_id) & (value_t == t)).float()
        rewards = successes - 1.0          # gc_negative=True
        masks = 1.0 - successes

        # ---- Low-level actor goal: idxs + subgoal_steps clamped to final ----
        low_t = (t + subgoal_steps).clamp(max=n)

        # ---- High-level actor goal & target ----
        # Uniform sampling (actor_geom_sample=False, actor_p_randomgoal=0):
        #   high_traj_goal = round(min(t+1, n)*d + n*(1-d)),  d ~ U[0,1)
        #   high_target    = min(t + subgoal_steps, high_goal)
        d = torch.rand(batch_size, device=self.device)
        high_t = torch.round(
            torch.minimum(t + 1, n).float() * d + n.float() * (1.0 - d)
        ).long()
        high_target_t = (t + subgoal_steps).clamp(max=high_t)

        # ---- Fetch latents ----
        z = self._z_at(ep_id, t)
        z_next = self._z_at(ep_id, t + 1)
        z_value_goal = self._z_at(value_ep, value_t)
        z_low_goal = self._z_at(ep_id, low_t)
        z_high_goal = self._z_at(ep_id, high_t)
        z_high_target = self._z_at(ep_id, high_target_t)
        a = self.a_flat[idxs]

        return dict(
            observations=z,
            next_observations=z_next,
            actions=a,
            rewards=rewards,
            masks=masks,
            value_goals=z_value_goal,
            low_actor_goals=z_low_goal,
            high_actor_goals=z_high_goal,
            high_actor_targets=z_high_target,
        )


# =============================================================================
# 3. IQL expectile loss (mirrors ogbench expectile_loss)
# =============================================================================

def _expectile_loss(adv, diff, expectile):
    weight = torch.where(adv >= 0,
                         torch.full_like(adv, expectile),
                         torch.full_like(adv, 1.0 - expectile))
    return (weight * diff.pow(2)).mean()


# =============================================================================
# 4. Training Loop (mirrors HIQLAgent.update / total_loss)
# =============================================================================

def train_loop(
    goal_rep, ll_actor, hl_actor, value_net, value_target,
    optimizer,
    real_cache,
    total_steps=500_000,
    subgoal_steps=10,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    expectile=0.7,
    alpha_low=3.0,
    alpha_high=3.0,
    save_dir='./checkpoints_hiql_baseline',
    device='cuda',
    log_interval=100,
    save_interval=10_000,
    latent_adapter=None,
):
    os.makedirs(save_dir, exist_ok=True)

    adapt = (lambda z: latent_adapter(z)) if latent_adapter is not None else (lambda z: z)

    print(f"\n{'='*65}", flush=True)
    print(f"  HIQL baseline (faithful ogbench port)", flush=True)
    print(f"  total_steps   : {total_steps:,}", flush=True)
    print(f"  subgoal_steps : {subgoal_steps}  (raw env steps)", flush=True)
    print(f"  rep_dim       : {goal_rep.mlp[-1].out_features}", flush=True)
    print(f"  batch_size    : {batch_size}", flush=True)
    print(f"  latent_adapter: {'enabled' if latent_adapter is not None else 'disabled'}", flush=True)
    print(f"  save_dir      : {save_dir}", flush=True)
    print(f"{'='*65}\n", flush=True)

    csv_path = os.path.join(save_dir, 'training_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow([
            'step', 'value_loss', 'll_actor_loss', 'hl_actor_loss', 'elapsed_s'])

    t0 = time.time()
    t_interval = time.time()
    sum_v = sum_ll = sum_hl = 0.0
    log_n = 0

    for step in range(total_steps):
        batch = real_cache.sample_batch(batch_size, subgoal_steps, discount=gamma)

        obs      = adapt(batch['observations'])
        nobs     = adapt(batch['next_observations'])
        actions  = batch['actions']
        rewards  = batch['rewards']
        masks    = batch['masks']
        v_goals  = adapt(batch['value_goals'])
        ll_goals = adapt(batch['low_actor_goals'])
        hl_goals = adapt(batch['high_actor_goals'])
        hl_targets = adapt(batch['high_actor_targets'])

        # ----- Value loss (mirrors HIQLAgent.value_loss) -----
        with torch.no_grad():
            phi_next_t = goal_rep(nobs, v_goals)
            v1n_t, v2n_t = value_target(nobs, phi_next_t)
            vn_min_t = torch.min(v1n_t, v2n_t)
            q = rewards + gamma * masks * vn_min_t

            phi_t_t = goal_rep(obs, v_goals)
            v1_tt, v2_tt = value_target(obs, phi_t_t)
            v_tt = (v1_tt + v2_tt) / 2
            adv = q - v_tt

            q1 = rewards + gamma * masks * v1n_t
            q2 = rewards + gamma * masks * v2n_t

        phi_cur = goal_rep(obs, v_goals)            # gradients flow into goal_rep
        v1, v2 = value_net(obs, phi_cur)
        v_loss = (_expectile_loss(adv, q1 - v1, expectile) +
                  _expectile_loss(adv, q2 - v2, expectile))

        # ----- LL actor loss (mirrors low_actor_loss, low_actor_rep_grad=True) -----
        with torch.no_grad():
            phi_ll_a = goal_rep(obs, ll_goals)
            phi_ll_b = goal_rep(nobs, ll_goals)
            v1c, v2c = value_net(obs, phi_ll_a)
            v1n, v2n = value_net(nobs, phi_ll_b)
            v_curr = (v1c + v2c) / 2
            v_next = (v1n + v2n) / 2
            adv_ll = v_next - v_curr
            w_ll = (alpha_low * adv_ll).exp().clamp(max=100.0)

        # low_actor_rep_grad=True → goal_rep gets gradients from LL loss
        phi_ll_grad = goal_rep(obs, ll_goals)
        ll_inp = torch.cat([obs, phi_ll_grad], dim=-1)
        log_p_ll = ll_actor.log_prob(ll_inp, actions)
        ll_loss = -(w_ll * log_p_ll).mean()

        # ----- HL actor loss (mirrors high_actor_loss) -----
        with torch.no_grad():
            phi_hl_a = goal_rep(obs, hl_goals)
            phi_hl_b = goal_rep(hl_targets, hl_goals)
            v1t, v2t = value_net(obs, phi_hl_a)
            v1tk, v2tk = value_net(hl_targets, phi_hl_b)
            v_t = (v1t + v2t) / 2
            v_tk = (v1tk + v2tk) / 2
            adv_hl = v_tk - v_t
            w_hl = (alpha_high * adv_hl).exp().clamp(max=100.0)
            target_rep = goal_rep(obs, hl_targets)   # stop_grad in ogbench

        hl_inp = torch.cat([obs, hl_goals], dim=-1)
        log_p_hl = hl_actor.log_prob(hl_inp, target_rep)
        hl_loss = -(w_hl * log_p_hl).mean()

        # ----- Combined backward (mirrors total_loss + apply_loss_fn) -----
        total = v_loss + ll_loss + hl_loss
        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        # ----- EMA target update -----
        with torch.no_grad():
            for tp, p in zip(value_target.parameters(), value_net.parameters()):
                tp.data.mul_(1.0 - tau).add_(p.data * tau)

        sum_v += v_loss.item()
        sum_ll += ll_loss.item()
        sum_hl += hl_loss.item()
        log_n += 1

        if step % log_interval == 0:
            d = max(log_n, 1)
            now = time.time()
            interval = now - t_interval
            sps = log_interval / max(interval, 1e-6)
            remaining = (total_steps - step) / max(sps, 1e-6)
            pct = 100.0 * step / total_steps
            total_elapsed = now - t0
            print(
                f"Step {step:06d}/{total_steps:,} ({pct:4.1f}%) | "
                f"V: {sum_v/d:.4f} | "
                f"LL: {sum_ll/d:.4f} | HL: {sum_hl/d:.4f} | "
                f"{sps:.1f} sps | ETA: {remaining/60:.0f}m | "
                f"elapsed: {total_elapsed/60:.0f}m",
                flush=True,
            )
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    step, sum_v/d, sum_ll/d, sum_hl/d, interval])
            sum_v = sum_ll = sum_hl = 0.0
            log_n = 0
            t_interval = now

        if step > 0 and (step % save_interval == 0 or step == total_steps - 1):
            torch.save(goal_rep.state_dict(),  os.path.join(save_dir, 'goal_rep.pth'))
            torch.save(ll_actor.state_dict(),  os.path.join(save_dir, 'll_actor.pth'))
            torch.save(hl_actor.state_dict(),  os.path.join(save_dir, 'hl_actor.pth'))
            torch.save(value_net.state_dict(), os.path.join(save_dir, 'value_net.pth'))
            if latent_adapter is not None:
                torch.save(latent_adapter.state_dict(),
                           os.path.join(save_dir, 'adapter.pth'))
            print(f"  → checkpoint saved at step {step}", flush=True)


# =============================================================================
# 5. Main
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='HIQL baseline on LeWM 192D latents (faithful ogbench port)')

    parser.add_argument('--cache_path',   type=str, default=None)
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--save_dir',     type=str, default=None)

    # Defaults match ogbench cube-single-play config exactly.
    parser.add_argument('--img_size',       type=int,   default=224)
    parser.add_argument('--total_steps',    type=int,   default=500_000)
    parser.add_argument('--subgoal_steps',  type=int,   default=10,
                        help='Raw env steps (ogbench cube-single default = 10).')
    parser.add_argument('--rep_dim',        type=int,   default=10)
    parser.add_argument('--batch_size',     type=int,   default=256)
    parser.add_argument('--alpha_low',      type=float, default=3.0)
    parser.add_argument('--alpha_high',     type=float, default=3.0)
    parser.add_argument('--gamma',          type=float, default=0.99)
    parser.add_argument('--expectile',      type=float, default=0.7)
    parser.add_argument('--lr',             type=float, default=3e-4)
    parser.add_argument('--tau',            type=float, default=0.005)
    parser.add_argument('--adapter_dim',    type=int,   default=0,
                        help='Output dim of trainable MLP adapter on top of frozen 192D latents. '
                             '0=disabled (frozen encoder as-is). '
                             '256=recommended (matches ogbench impala_small output dim).')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    STABLEWM_HOME = os.environ.get(
        'STABLEWM_HOME', os.path.join(os.path.expanduser('~'), 'stable_wm_data'))

    if args.img_size == 224:
        _default_cache   = os.path.join(STABLEWM_HOME, 'ogbench', 'lewm_224_latents_cache.pt')
        _default_dataset = os.path.join(STABLEWM_HOME, 'ogbench', 'visual-cube-single-play-v0_224')
    else:
        _default_cache   = os.path.join(STABLEWM_HOME, 'ogbench', 'lewm_64_latents_cache.pt')
        _default_dataset = os.path.join(STABLEWM_HOME, 'ogbench', 'visual-cube-single-play-v0')

    data_path  = args.dataset_path or _default_dataset
    cache_path = args.cache_path   or _default_cache
    save_dir   = args.save_dir     or (
        f'./checkpoints_hiql_baseline_k{args.subgoal_steps}_rep{args.rep_dim}')

    print(f'Loading cache from {cache_path} ...')
    cache_data  = torch.load(cache_path, map_location='cpu')
    all_latents = cache_data['all_latents']
    all_actions = cache_data.get('all_actions', [])

    if not all_actions:
        raise RuntimeError(
            "Cache has no 'all_actions'. Rebuild the cache with analyse_lewm_224.py.")

    import stable_worldmodel as swm
    from sklearn import preprocessing as sk_pre
    print('Fitting StandardScaler on dataset actions ...')
    _ds = swm.data.HDF5Dataset(
        data_path, keys_to_cache=['action'],
        cache_dir=os.path.dirname(data_path))
    action_raw = _ds.get_col_data('action')
    action_raw = action_raw[~np.isnan(action_raw).any(axis=1)]
    action_scaler = sk_pre.StandardScaler()
    action_scaler.fit(action_raw)
    print(f'  Scaler fit on {len(action_raw):,} action frames')

    real_cache = RealOfflineCache(
        all_latents=all_latents,
        all_actions=all_actions,
        action_scaler=action_scaler,
        device=device,
    )

    RAW_LATENT_DIM = 192
    ACTION_DIM     = 5
    REP_DIM        = args.rep_dim
    HIDDEN_DIMS    = (512, 512, 512)

    # Trainable adapter (optional): projects 192D frozen latents → LATENT_DIM
    # so all downstream networks can learn task-relevant representations.
    latent_adapter = None
    LATENT_DIM = RAW_LATENT_DIM
    if args.adapter_dim > 0:
        latent_adapter = LatentAdapter(
            in_dim=RAW_LATENT_DIM, out_dim=args.adapter_dim).to(device)
        LATENT_DIM = args.adapter_dim
        print(f'LatentAdapter: 192 → {LATENT_DIM}D '
              f'({sum(p.numel() for p in latent_adapter.parameters()):,} params)')

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

    print(
        f'Networks (LeWM 192D encoder → {LATENT_DIM}D, rep_dim={REP_DIM}):\n'
        f'  GoalRep:  {sum(p.numel() for p in goal_rep.parameters()):>10,} params\n'
        f'  LL Actor: {sum(p.numel() for p in ll_actor.parameters()):>10,} params\n'
        f'  HL Actor: {sum(p.numel() for p in hl_actor.parameters()):>10,} params\n'
        f'  Value:    {sum(p.numel() for p in value_net.parameters()):>10,} params'
    )

    # Single Adam over all trainable params (mirrors ogbench's single optax.adam).
    params = (list(goal_rep.parameters())
              + list(ll_actor.parameters())
              + list(hl_actor.parameters())
              + list(value_net.parameters()))
    if latent_adapter is not None:
        params = list(latent_adapter.parameters()) + params
    optimizer = torch.optim.Adam(params, lr=args.lr)

    train_loop(
        goal_rep=goal_rep, ll_actor=ll_actor, hl_actor=hl_actor,
        value_net=value_net, value_target=value_target,
        optimizer=optimizer,
        real_cache=real_cache,
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
        latent_adapter=latent_adapter,
    )
