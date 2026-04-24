"""
train_hiql_baseline.py
Ablation baseline: pure offline HIQL on LeWM 192D latents.

This is a faithful PyTorch port of ~/ogbench/impls/agents/hiql.py, with the
only architectural change being that the encoder is the frozen LeWM ViT-tiny
(192D representations instead of 256D).  There is NO world-model imagination —
this is a pure offline IQL+AWR baseline operating on the same latent cache used
by train_hiql_lewm.py.

Architecture (mirrors ogbench HIQL exactly):
  - GoalRep phi([z; g])  : MLP → rep_dim → LengthNormalize
  - TwinValue V(z, phi)  : shared across HL + LL (IQL expectile)
  - LL GaussianActor     : pi^l(a | z, phi([z; w]))  → 25D WM action
  - HL GaussianActor     : pi^h(phi | z, g)          → rep_dim subgoal rep
  - TwinValue target     : EMA copy for Bellman backup

Loss schedule (identical to ogbench hiql.py):
  1. Value update  : expectile regression toward Q_target = r + γ·V_target(z', phi)
  2. LL actor AWR  : exp(α_l · (V(z', phi) - V(z, phi))) · log π^l(a | z, phi)
  3. HL actor AWR  : exp(α_h · (V(z_tk, g) - V(z_t, g))) · log π^h(phi_tk | z_t, g)

Differences from ogbench reference:
  - Input dim : 192 (LeWM) instead of 256 (impala_small)
  - Framework  : PyTorch instead of JAX/Flax
  - Data       : pre-encoded latent cache (same as train_hiql_lewm.py)
  - Sampling   : same RealOfflineCache with WM-step frameskip=5
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
# 1. Network Architectures
# =============================================================================

def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


class LengthNormalize(nn.Module):
    """Length-normalize to unit sphere, then rescale to sqrt(dim).

    Mirrors ogbench's LengthNormalize: x / ||x|| * sqrt(dim).
    """
    def forward(self, x):
        return x / (x.norm(dim=-1, keepdim=True) + 1e-8) * (x.shape[-1] ** 0.5)


class GoalRep(nn.Module):
    """phi([z; g]) — state-dependent subgoal representation.

    Mirrors ogbench HIQL goal_rep_def:
        MLP(hidden_dims + [rep_dim], activate_final=False) → LengthNormalize

    Input : concat of z (192D) and g (192D) → 384D
    Output: length-normalised rep_dim vector
    """

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
        x = torch.cat([z, g], dim=-1)
        return self.normalize(self.mlp(x))


class GaussianActor(nn.Module):
    """Goal-conditioned Gaussian actor.

    Used for:
      - LL: input_dim = latent_dim + rep_dim,  output_dim = action_dim, tanh_squash=False
      - HL: input_dim = latent_dim * 2,         output_dim = rep_dim,    tanh_squash=False

    Mirrors ogbench GCActor (const_std=True, no tanh squash, clip at inference).
    log_std clamped to [LOG_STD_MIN, LOG_STD_MAX].
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX =  2.0

    def __init__(self, input_dim, output_dim,
                 hidden_dims=(512, 512, 512), layer_norm=True,
                 tanh_squash=False, action_scale=1.0):
        super().__init__()
        self.tanh_squash = tanh_squash
        self.action_scale = action_scale

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
        # const_std=True: fixed zero log_std (std=1), matching ogbench GCActor const_std=True
        # which uses jnp.zeros_like(means) — NOT a learnable parameter.

        self.backbone.apply(_init_weights)
        nn.init.uniform_(self.mean_head.weight, -1e-3, 1e-3)
        nn.init.constant_(self.mean_head.bias,   0.0)

    def _forward(self, inp):
        h = self.backbone(inp)
        mean    = self.mean_head(h)
        log_std = torch.zeros_like(mean)
        return mean, log_std

    def sample(self, inp):
        """Reparameterised sample. Returns (action, log_prob, mode)."""
        mean, log_std = self._forward(inp)
        std  = log_std.exp()
        dist = Normal(mean, std)
        x    = dist.rsample()

        if self.tanh_squash:
            y      = torch.tanh(x)
            action = y * self.action_scale
            lp     = dist.log_prob(x) - torch.log(self.action_scale * (1.0 - y.pow(2)) + 1e-6)
            log_prob = lp.sum(dim=-1)
            mode = torch.tanh(mean) * self.action_scale
        else:
            action   = x
            log_prob = dist.log_prob(x).sum(dim=-1)
            mode     = mean

        return action, log_prob, mode

    def log_prob(self, inp, target):
        """Log probability of target under the actor's distribution."""
        mean, log_std = self._forward(inp)
        std  = log_std.exp()
        dist = Normal(mean, std)

        if self.tanh_squash:
            t_norm = (target / self.action_scale).clamp(-1 + 1e-6, 1 - 1e-6)
            x  = torch.atanh(t_norm)
            lp = dist.log_prob(x) - torch.log(self.action_scale * (1.0 - t_norm.pow(2)) + 1e-6)
            return lp.sum(dim=-1)
        else:
            return dist.log_prob(target).sum(dim=-1)


class TwinValue(nn.Module):
    """Twin V-network: V(z, phi) → (scalar, scalar).

    input_dim = latent_dim + rep_dim  (state + goal representation)
    Mirrors ogbench GCValue with ensemble=True.
    """

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
            nn.init.uniform_(net[-1].weight, -1e-3, 1e-3)
            nn.init.constant_(net[-1].bias,   0.0)
            return net

        self.v1 = _make_v()
        self.v2 = _make_v()

    def forward(self, z, phi):
        x = torch.cat([z, phi], dim=-1)
        return self.v1(x).squeeze(-1), self.v2(x).squeeze(-1)


# =============================================================================
# 2. Data Pipeline  (identical to train_hiql_lewm.py)
# =============================================================================

class RealOfflineCache:
    """GPU-native sampler for HIQL from pre-computed LeWM latents.

    Operates at WM-step granularity (frameskip=5 raw frames per WM step).
    Actions are StandardScaler-normalised and stacked to 25D (5 × 5D).

    sample_value_batch(batch_size, k):
        (z_t, a, r, mask, z_next, z_value_goal)
        Goal is HER-relabeled (mirrors ogbench HGCDataset):
          20% current state (r=0, mask=0)
          50% future traj state (r=0 if same step, else r=-1, mask=1-success)
          30% random state (r=-1, mask=1)
        Sparse binary reward: r = 0 if z_t is the goal step, else -1.

    sample_ll_batch(batch_size, k):
        (z_t, a, z_subgoal)
        z_subgoal = z_{t+k}  (clamped to episode end) — no reward needed

    sample_hl_batch(batch_size, k):
        (z_t, z_{t+k}, g_ultimate)
        g_ultimate = last WM-step latent in episode
    """

    def __init__(self, all_latents, all_actions, action_scaler,
                 device='cuda', frameskip=5):
        self.device    = device
        self.frameskip = frameskip

        raw_adim = all_actions[0].shape[-1] if all_actions else 5
        wm_adim  = frameskip * raw_adim

        z_t_list, a_list, z_next_list = [], [], []
        ep_starts, ep_ends, ep_goals  = [], [], []
        ep_ids_list, t_within_list    = [], []
        all_wm_lat_list               = []
        wm_offsets                    = []

        offset, wm_offset, skipped = 0, 0, 0

        for ep_z_raw, ep_a_raw in zip(all_latents, all_actions):
            ep_z = ep_z_raw.float()
            T    = ep_z.shape[0]
            n_wm = (T - 1) // frameskip
            if n_wm <= 0:
                skipped += 1
                continue

            step_idx = torch.arange(0, (n_wm + 1) * frameskip, frameskip)[:n_wm + 1]
            ep_z_wm  = ep_z[step_idx]

            n_raw  = n_wm * frameskip
            scaled = action_scaler.transform(
                ep_a_raw[:n_raw].numpy().astype(np.float32))
            ep_a_wm = torch.tensor(
                scaled.reshape(n_wm, wm_adim), dtype=torch.float32)

            z_t_list.append(ep_z_wm[:-1])
            a_list.append(ep_a_wm)
            z_next_list.append(ep_z_wm[1:])

            ep_id = len(ep_starts)
            ep_starts.append(offset)
            ep_ends.append(offset + n_wm)
            ep_goals.append(ep_z_wm[-1])

            ep_ids_list.extend([ep_id] * n_wm)
            t_within_list.extend(range(n_wm))

            wm_offsets.append(wm_offset)
            all_wm_lat_list.append(ep_z_wm)
            wm_offset += n_wm + 1

            offset += n_wm

        if skipped:
            print(f"  RealOfflineCache: skipped {skipped} short episodes")

        self.total = offset
        self.n_eps = len(ep_starts)
        print(f"  RealOfflineCache: {self.n_eps} episodes, {self.total:,} WM-step transitions")

        self.z_t_flat    = torch.cat(z_t_list,    dim=0).to(device)
        self.a_flat      = torch.cat(a_list,      dim=0).to(device)
        self.z_next_flat = torch.cat(z_next_list, dim=0).to(device)
        self.ep_starts   = torch.tensor(ep_starts, device=device)
        self.ep_ends     = torch.tensor(ep_ends,   device=device)
        self.ep_goals    = torch.stack(ep_goals).to(device)
        self.ep_ids      = torch.tensor(ep_ids_list,   dtype=torch.long, device=device)
        self.t_within    = torch.tensor(t_within_list, dtype=torch.long, device=device)

        self.wm_latents_flat = torch.cat(all_wm_lat_list, dim=0).to(device)
        self.ep_wm_offsets   = torch.tensor(wm_offsets, dtype=torch.long, device=device)
        self.ep_n_wm         = self.ep_ends - self.ep_starts

    def _kstep_latents(self, idx, k):
        ep_id    = self.ep_ids[idx]
        t        = self.t_within[idx]
        n_wm     = self.ep_n_wm[ep_id]
        target_t = (t + k).clamp(max=n_wm)
        flat_idx = self.ep_wm_offsets[ep_id] + target_t
        return self.wm_latents_flat[flat_idx]

    def sample_value_batch(self, batch_size, subgoal_steps,
                           p_curgoal=0.2, p_trajgoal=0.5, p_randomgoal=0.3,
                           gc_negative=True, discount=0.99):
        """HER goal relabeling with sparse reward (mirrors ogbench HGCDataset).

        Traj goals use geometric sampling (value_geom_sample=True default):
          offsets ~ Geometric(1 - discount), offset ≥ 1 → always a FUTURE state.
        r = 0 if transition step == goal step (success), else -1 (gc_negative=True).
        mask = 1 - success  (Bellman backup terminated at goal).
        """
        idx    = torch.randint(0, self.total, (batch_size,), device=self.device)
        ep_id  = self.ep_ids[idx]
        t      = self.t_within[idx]
        n_wm   = self.ep_n_wm[ep_id]

        # Sample goal steps via HER mixing
        roll = torch.rand(batch_size, device=self.device)

        # Current goal: goal step = t (always success)
        cur_goal_t = t.clone()

        # Future traj goal: geometric offsets ≥ 1 (mirrors ogbench value_geom_sample=True).
        # NumPy's np.random.geometric starts at 1; PyTorch Geometric starts at 0 → add 1.
        geom_offset = (torch.distributions.Geometric(probs=1.0 - discount)
                       .sample((batch_size,)).long().to(self.device) + 1)
        traj_goal_t = (t + geom_offset).clamp(max=n_wm)

        # Random goal: independent flat index from anywhere in dataset
        rand_idx   = torch.randint(0, self.total, (batch_size,), device=self.device)
        rand_ep_id = self.ep_ids[rand_idx]
        rand_t     = self.t_within[rand_idx]
        rand_goal_t = rand_t

        # Pick which goal type each sample uses
        is_cur  = roll < p_curgoal
        is_traj = (roll >= p_curgoal) & (roll < p_curgoal + p_trajgoal)
        # else: random

        # Resolve goal latents
        # For cur/traj: goal is in same episode → index via wm_offsets[ep_id]
        # For random:   goal is from a different episode → never a success
        cur_flat  = self.ep_wm_offsets[ep_id]  + cur_goal_t
        traj_flat = self.ep_wm_offsets[ep_id]  + traj_goal_t
        rand_flat = self.ep_wm_offsets[rand_ep_id] + rand_goal_t

        goal_flat    = torch.where(is_cur,  cur_flat,
                       torch.where(is_traj, traj_flat, rand_flat))
        z_value_goal = self.wm_latents_flat[goal_flat]   # [B, 192]

        # Success: cur goal always succeeds; traj only if offset clamped to t (ep end == t).
        # Geometric offset ≥ 1, so traj_goal_t > t unless n_wm == t (last step of ep).
        # Random goals never succeed.
        goal_step = torch.where(is_cur,  cur_goal_t,
                    torch.where(is_traj, traj_goal_t,
                                torch.full_like(t, -1)))   # -1 never matches t
        success = ((goal_step == t) & (is_cur | is_traj)).float()

        r    = success - (1.0 if gc_negative else 0.0)   # 0 or -1
        mask = 1.0 - success                              # 0 at goal, 1 elsewhere

        z_t    = self.z_t_flat[idx]
        a      = self.a_flat[idx]
        z_next = self.z_next_flat[idx]
        return z_t, a, r, mask, z_next, z_value_goal

    def sample_ll_batch(self, batch_size, subgoal_steps):
        """(z_t, a, z_subgoal) for LL AWR — subgoal is z_{t+k}, no reward needed."""
        idx       = torch.randint(0, self.total, (batch_size,), device=self.device)
        z_t       = self.z_t_flat[idx]
        a         = self.a_flat[idx]
        z_subgoal = self._kstep_latents(idx, subgoal_steps)
        return z_t, a, z_subgoal

    def sample_hl_batch(self, batch_size, subgoal_steps):
        """(z_t, z_target, z_hl_goal) for HL AWR.

        Mirrors ogbench HGCDataset with actor_geom_sample=False, actor_p_randomgoal=0:
          hl_goal_t  = round(d * (t+1) + (1-d) * n_wm)  for d ~ Uniform[0,1)
                       ≈ uniform in {t+1, ..., n_wm}
          target_t   = min(t + k, hl_goal_t)  (clamped to HL goal, not episode end)

        The HL actor predicts phi([z_t; z_target]) given (z_t, z_hl_goal).
        """
        idx   = torch.randint(0, self.total, (batch_size,), device=self.device)
        ep_id = self.ep_ids[idx]
        t     = self.t_within[idx]
        n_wm  = self.ep_n_wm[ep_id]

        # Uniform future goal in [t+1, n_wm] via the ogbench interpolation formula.
        d         = torch.rand(batch_size, device=self.device)
        hl_goal_t = torch.round(
            d * (t + 1).float() + (1.0 - d) * n_wm.float()
        ).long().clamp(min=t + 1, max=n_wm)

        # Subgoal target: min(t + k, hl_goal_t)  — clamped to HL goal, not episode end
        target_t = (t + subgoal_steps).clamp(max=hl_goal_t)

        goal_flat   = self.ep_wm_offsets[ep_id] + hl_goal_t
        target_flat = self.ep_wm_offsets[ep_id] + target_t

        z_t        = self.z_t_flat[idx]
        z_hl_goal  = self.wm_latents_flat[goal_flat]    # high-level ultimate goal
        z_target   = self.wm_latents_flat[target_flat]  # subgoal = min(t+k, hl_goal)
        return z_t, z_target, z_hl_goal


# =============================================================================
# 3. IQL Loss (mirrors ogbench hiql.py expectile_loss)
# =============================================================================

def _expectile(adv, diff, tau):
    """Expectile loss — faithful port of ogbench HIQLAgent.expectile_loss.

    adv:  advantage used to set the weight (from target network — double-DQN
          trick: separates weight computation from gradient computation to
          mitigate overestimation bias).
    diff: TD error that receives gradients (q_i - v_i, current network).
    tau:  expectile level (e.g. 0.7).
    """
    w = torch.where(adv >= 0,
                    tau * torch.ones_like(adv),
                    (1.0 - tau) * torch.ones_like(adv))
    return (w * diff.pow(2)).mean()


# =============================================================================
# 4. Training Loop
# =============================================================================

def train_loop(
    goal_rep,        goal_rep_optimizer,
    ll_actor,        ll_optimizer,
    hl_actor,        hl_optimizer,
    value_net,       value_optimizer,
    value_target,
    real_cache,
    total_steps      = 200_000,
    subgoal_steps    = 25,
    batch_size       = 1024,
    gamma            = 0.99,
    tau              = 0.005,
    expectile        = 0.7,
    alpha_low        = 3.0,
    alpha_high       = 3.0,
    ll_grad_updates  = 1,
    hl_grad_updates  = 1,
    save_dir         = './checkpoints_hiql_baseline',
    device           = 'cuda',
    log_interval     = 100,
    save_interval    = 10_000,
):
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*65}", flush=True)
    print(f"  HIQL baseline (pure offline)", flush=True)
    print(f"  total_steps    : {total_steps:,}", flush=True)
    print(f"  subgoal_steps  : {subgoal_steps}", flush=True)
    print(f"  rep_dim        : {goal_rep.mlp[-1].out_features}", flush=True)
    print(f"  batch_size     : {batch_size}", flush=True)
    print(f"  ll_grad_updates: {ll_grad_updates}  hl_grad_updates: {hl_grad_updates}", flush=True)
    print(f"  save_dir       : {save_dir}", flush=True)
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

        # --------------------------------------------------------
        # A. Value update  (mirrors ogbench HIQLAgent.value_loss)
        # Sparse HER reward: r = 0 if s==g else -1, mask = 1 - success
        # Q_target = r + γ · mask · min V_target(z', phi(z', g))
        # --------------------------------------------------------
        for _ in range(ll_grad_updates):
            z, a, r, mask, z_next, z_vgoal = real_cache.sample_value_batch(
                batch_size, subgoal_steps)

            with torch.no_grad():
                phi_next_t   = goal_rep(z_next, z_vgoal)
                vn1_t, vn2_t = value_target(z_next, phi_next_t)
                v_next_min_t = torch.min(vn1_t, vn2_t)
                q            = r + gamma * mask * v_next_min_t  # for adv weight

                phi_t      = goal_rep(z, z_vgoal)
                v1_t, v2_t = value_target(z, phi_t)
                v_t        = (v1_t + v2_t) / 2
                adv        = q - v_t   # weight from target net (double-DQN trick)

                q1 = r + gamma * mask * vn1_t  # per-ensemble, for loss diff
                q2 = r + gamma * mask * vn2_t

            # Current value (gradients flow through value_net and goal_rep)
            phi_cur = goal_rep(z, z_vgoal)
            v1, v2  = value_net(z, phi_cur)

            v_loss = (_expectile(adv, q1 - v1, expectile) +
                      _expectile(adv, q2 - v2, expectile))

            value_optimizer.zero_grad()
            goal_rep_optimizer.zero_grad()
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(),  1.0)
            torch.nn.utils.clip_grad_norm_(goal_rep.parameters(),   1.0)
            value_optimizer.step()
            goal_rep_optimizer.step()

            sum_v += v_loss.item()

        # EMA target update
        with torch.no_grad():
            for tp, p in zip(value_target.parameters(), value_net.parameters()):
                tp.data.mul_(1.0 - tau).add_(p.data * tau)

        # --------------------------------------------------------
        # B. LL Actor AWR  (mirrors ogbench HIQLAgent.low_actor_loss)
        # Goal = z_{t+k} (fixed subgoal horizon, not HER-relabeled).
        # A_low = mean(V(z', phi(z', g))) - mean(V(z, phi(z, g)))
        # Uses current value_net (no grad), matching ogbench which calls
        # 'value' without params=grad_params (stop-gradient through value).
        # --------------------------------------------------------
        for _ in range(ll_grad_updates):
            idx    = torch.randint(0, real_cache.total, (batch_size,), device=device)
            z      = real_cache.z_t_flat[idx]
            a      = real_cache.a_flat[idx]
            z_next = real_cache.z_next_flat[idx]
            z_sub  = real_cache._kstep_latents(idx, subgoal_steps)

            with torch.no_grad():
                phi      = goal_rep(z,      z_sub)
                phi_next = goal_rep(z_next, z_sub)
                v1c, v2c = value_net(z,      phi)   # current net, no grad
                v1n, v2n = value_net(z_next, phi_next)
                v_curr   = (v1c + v2c) / 2          # mean, not min (ogbench line 64)
                v_next   = (v1n + v2n) / 2          # mean, not min (ogbench line 65)
                adv      = v_next - v_curr
                w        = (alpha_low * adv).exp().clamp(max=100.0)
                phi_inp  = goal_rep(z, z_sub)

            ll_inp   = torch.cat([z, phi_inp], dim=-1)
            log_p    = ll_actor.log_prob(ll_inp, a)
            ll_loss  = -(w * log_p).mean()

            ll_optimizer.zero_grad()
            ll_loss.backward()
            torch.nn.utils.clip_grad_norm_(ll_actor.parameters(), 1.0)
            ll_optimizer.step()
            sum_ll += ll_loss.item()

        # --------------------------------------------------------
        # C. HL Actor AWR  (mirrors ogbench HIQLAgent.high_actor_loss)
        # A_high = mean(V(z_target, phi(z_target, g))) - mean(V(z_t, phi(z_t, g)))
        # target = phi([z_t; z_target])  where z_target = min(t+k, hl_goal)
        # Uses current value_net (no grad), matching ogbench.
        # --------------------------------------------------------
        for _ in range(hl_grad_updates):
            z_t, z_target, z_hl_goal = real_cache.sample_hl_batch(batch_size, subgoal_steps)

            with torch.no_grad():
                phi_t      = goal_rep(z_t,      z_hl_goal)
                phi_target = goal_rep(z_target, z_hl_goal)
                v1t,  v2t  = value_net(z_t,      phi_t)      # current net, no grad
                v1tk, v2tk = value_net(z_target, phi_target)
                v_t   = (v1t  + v2t)  / 2   # mean, not min (ogbench line 103)
                v_tk  = (v1tk + v2tk) / 2   # mean, not min (ogbench line 104)
                adv   = v_tk - v_t
                w     = (alpha_high * adv).exp().clamp(max=100.0)

                # HL regression target: phi([z_t; z_target])  (ogbench line 111-113)
                target_rep = goal_rep(z_t, z_target)   # [B, rep_dim]

            hl_inp  = torch.cat([z_t, z_hl_goal], dim=-1)
            log_p   = hl_actor.log_prob(hl_inp, target_rep)
            hl_loss = -(w * log_p).mean()

            hl_optimizer.zero_grad()
            hl_loss.backward()
            torch.nn.utils.clip_grad_norm_(hl_actor.parameters(), 1.0)
            hl_optimizer.step()
            sum_hl += hl_loss.item()

        log_n += 1

        # --------------------------------------------------------
        # Logging
        # --------------------------------------------------------
        if step % log_interval == 0:
            d        = max(log_n, 1)
            now      = time.time()
            interval = now - t_interval
            sps      = log_interval / max(interval, 1e-6)
            remaining = (total_steps - step) / max(sps, 1e-6)
            pct      = 100.0 * step / total_steps
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

        # --------------------------------------------------------
        # Checkpointing
        # --------------------------------------------------------
        if step > 0 and (step % save_interval == 0 or step == total_steps - 1):
            torch.save(goal_rep.state_dict(),  os.path.join(save_dir, 'goal_rep.pth'))
            torch.save(ll_actor.state_dict(),  os.path.join(save_dir, 'll_actor.pth'))
            torch.save(hl_actor.state_dict(),  os.path.join(save_dir, 'hl_actor.pth'))
            torch.save(value_net.state_dict(), os.path.join(save_dir, 'value_net.pth'))
            print(f"  → checkpoint saved at step {step}", flush=True)


# =============================================================================
# 5. Main
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='HIQL baseline on LeWM 192D latents (no imagination)')

    # Paths
    parser.add_argument('--cache_path',   type=str, default=None,
                        help='Latent cache (.pt) with all_latents + all_actions keys.')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='HDF5 dataset path (no .h5 extension).')
    parser.add_argument('--save_dir',     type=str, default=None,
                        help='Checkpoint output directory (auto-named if omitted).')

    # Algorithm (defaults match ogbench get_config())
    parser.add_argument('--img_size',        type=int,   default=224,
                        help='64 or 224 — selects default cache/dataset paths.')
    parser.add_argument('--total_steps',    type=int,   default=200_000)
    parser.add_argument('--subgoal_steps',  type=int,   default=25,
                        help='k: subgoal horizon in WM steps (ogbench default 25).')
    parser.add_argument('--rep_dim',        type=int,   default=10,
                        help='Goal representation dimension (ogbench default 10).')
    parser.add_argument('--batch_size',     type=int,   default=1024)
    parser.add_argument('--alpha_low',      type=float, default=3.0)
    parser.add_argument('--alpha_high',     type=float, default=3.0)
    parser.add_argument('--gamma',          type=float, default=0.99)
    parser.add_argument('--expectile',      type=float, default=0.7)
    parser.add_argument('--action_scale',   type=float, default=3.0,
                        help='Clamp bound for LL actions (default 3.0).')

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

    # --- Load latent cache ---
    print(f'Loading cache from {cache_path} ...')
    cache_data  = torch.load(cache_path, map_location='cpu')
    all_latents = cache_data['all_latents']
    all_actions = cache_data.get('all_actions', [])

    if not all_actions:
        raise RuntimeError(
            "Cache has no 'all_actions'. Rebuild the cache with analyse_lewm_224.py.")

    # --- Fit action scaler ---
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

    # --- Data pipeline ---
    real_cache = RealOfflineCache(
        all_latents=all_latents,
        all_actions=all_actions,
        action_scaler=action_scaler,
        device=device,
        frameskip=5,
    )

    # --- Networks ---
    LATENT_DIM  = 192   # LeWM ViT-tiny encoder output (vs 256 for impala_small)
    ACTION_DIM  = 25    # 5 frameskip × 5D raw action
    REP_DIM     = args.rep_dim
    HIDDEN_DIMS = (512, 512, 512)

    goal_rep = GoalRep(
        latent_dim=LATENT_DIM, rep_dim=REP_DIM,
        hidden_dims=HIDDEN_DIMS, layer_norm=True,
    ).to(device)

    # LL actor: input = [z (192D) || phi (rep_dim)]
    ll_actor = GaussianActor(
        input_dim=LATENT_DIM + REP_DIM, output_dim=ACTION_DIM,
        hidden_dims=HIDDEN_DIMS, layer_norm=True,
        tanh_squash=False,
    ).to(device)

    # HL actor: input = [z (192D) || g (192D)], output = rep_dim
    hl_actor = GaussianActor(
        input_dim=LATENT_DIM * 2, output_dim=REP_DIM,
        hidden_dims=HIDDEN_DIMS, layer_norm=True,
        tanh_squash=False,
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
        f'Networks (LeWM 192D encoder, rep_dim={REP_DIM}):\n'
        f'  GoalRep:  {sum(p.numel() for p in goal_rep.parameters()):>10,} params\n'
        f'  LL Actor: {sum(p.numel() for p in ll_actor.parameters()):>10,} params\n'
        f'  HL Actor: {sum(p.numel() for p in hl_actor.parameters()):>10,} params\n'
        f'  Value:    {sum(p.numel() for p in value_net.parameters()):>10,} params'
    )

    goal_rep_optimizer = torch.optim.Adam(goal_rep.parameters(),  lr=3e-4)
    ll_optimizer       = torch.optim.Adam(ll_actor.parameters(),  lr=3e-4)
    hl_optimizer       = torch.optim.Adam(hl_actor.parameters(),  lr=3e-4)
    value_optimizer    = torch.optim.Adam(value_net.parameters(), lr=3e-4)

    train_loop(
        goal_rep         = goal_rep,       goal_rep_optimizer = goal_rep_optimizer,
        ll_actor         = ll_actor,       ll_optimizer       = ll_optimizer,
        hl_actor         = hl_actor,       hl_optimizer       = hl_optimizer,
        value_net        = value_net,      value_optimizer    = value_optimizer,
        value_target     = value_target,
        real_cache       = real_cache,
        total_steps      = args.total_steps,
        subgoal_steps    = args.subgoal_steps,
        batch_size       = args.batch_size,
        gamma            = args.gamma,
        tau              = 0.005,
        expectile        = args.expectile,
        alpha_low        = args.alpha_low,
        alpha_high       = args.alpha_high,
        save_dir         = save_dir,
        device           = device,
    )
