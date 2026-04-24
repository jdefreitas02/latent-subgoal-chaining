"""
train_hiql_flow.py
HIQL + LeWM with Flow-matching Low-Level Actor.

Extends train_hiql_lewm.py (HER + 10D goal_rep) by replacing the LL Gaussian
actor with a two-network FQL-style flow policy:

  1. BCFlow u_θ(z, rep, x_t, t) → velocity
       Trained via conditional flow matching on ALL dataset transitions.
       Learns the full multimodal action distribution.

  2. OneStep µ_ψ(z, rep, ε) → action
       Trained by distilling BCFlow, weighted by AWR advantage.
       Gives fast single-pass inference without ODE integration.

LL Actor Loss (per training step):
  ─ BC flow loss ─────────────────────────────────────────────────────────
    x0 ~ N(0,I),  x1 = a_data,  t ~ U(0,1)
    x_t = (1-t)·x0 + t·x1,     vel = x1 - x0
    L_bc = ‖u_θ(z, rep, x_t, t) − vel‖²

  ─ Advantage-weighted distillation ──────────────────────────────────────
    ε ~ N(0,I)
    a_flow = Euler(u_θ, z, rep, ε, flow_steps)       [no_grad through u_θ]
    adv    = V(z_next, rep_next) − V(z, rep_curr)     [from current V]
    w      = exp(α · adv).clamp(max=100)
    L_dist = mean(w · ‖µ_ψ(z, rep, ε) − a_flow‖²)

  ─ Total ─────────────────────────────────────────────────────────────────
    L_ll   = L_bc + L_dist

    adv weights the distillation in the same role as FQL's −Q(s, µ_ψ)
    term, adapted for HIQL's V-function (which has no action argument).
    BCFlow is updated only by L_bc (Euler integration is no_grad).
    OneStep is updated only by L_dist.

No log_prob is needed anywhere → fully compatible with the multimodal
25D action chunks from WM frameskip = 5.

HL Actor (GaussianActor) and all other components are identical to
train_hiql_lewm.py (HER + GoalRep + dense LeWM reward for LL value).

References:
  FQL:  fql/agents/fql.py  (FQLAgent.actor_loss)
  HIQL: ogbench/impls/agents/hiql.py (HIQLAgent)
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

import stable_worldmodel as swm
from hydra import initialize, compose

from latent_env import LatentEnv


# =============================================================================
# 1. Network Architectures
# =============================================================================

def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


class GoalRep(nn.Module):
    """State-dependent subgoal representation φ([z; z_goal]) → rep_dim.

    Output length-normalised onto sphere of radius √rep_dim.
    Trained jointly with the value function.
    """

    def __init__(self, latent_dim=192, rep_dim=10,
                 hidden_dims=(512, 512, 512), layer_norm=True):
        super().__init__()
        in_dim = latent_dim * 2
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h)]
            if layer_norm:
                layers += [nn.LayerNorm(h)]
            layers += [nn.GELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, rep_dim))
        self.net = nn.Sequential(*layers)
        self.net.apply(_init_weights)
        self.rep_dim = rep_dim

    def forward(self, z, z_goal):
        x = torch.cat([z, z_goal], dim=-1)
        rep = self.net(x)
        rep = rep / (rep.norm(dim=-1, keepdim=True) + 1e-8) * (self.rep_dim ** 0.5)
        return rep


class LLBCFlow(nn.Module):
    """Behavior-cloning conditional flow network for the LL actor.

    Matches FQL's ActorVectorField called with (obs, x_t, t):
      input = concat([z (latent_dim), rep (rep_dim), x_t (action_dim), t (1)])
      output = velocity (action_dim)

    No LayerNorm (FQL default actor_layer_norm=False). 4 hidden layers.
    """

    def __init__(self, latent_dim=192, rep_dim=10, action_dim=25,
                 hidden_dims=(512, 512, 512, 512)):
        super().__init__()
        in_dim = latent_dim + rep_dim + action_dim + 1   # +1 for time
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.GELU()]
            d = h
        layers.append(nn.Linear(d, action_dim))
        self.net = nn.Sequential(*layers)
        self.net.apply(_init_weights)

    def forward(self, z, rep, x_t, t):
        """t: [B, 1] float in [0, 1]."""
        x = torch.cat([z, rep, x_t, t], dim=-1)
        return self.net(x)


class LLOneStep(nn.Module):
    """One-step distilled LL policy µ_ψ(z, rep, ε) → action.

    Matches FQL's ActorVectorField called without time:
      input = concat([z (latent_dim), rep (rep_dim), ε (action_dim)])
      output = action (action_dim)

    No LayerNorm (matching FQL actor_layer_norm=False). 4 hidden layers.
    Actions clipped to ±action_scale at inference — no tanh squash needed
    since the flow operates in the StandardScaler-normalised action space.
    """

    def __init__(self, latent_dim=192, rep_dim=10, action_dim=25,
                 hidden_dims=(512, 512, 512, 512)):
        super().__init__()
        self.action_dim  = action_dim
        in_dim = latent_dim + rep_dim + action_dim
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.GELU()]
            d = h
        layers.append(nn.Linear(d, action_dim))
        self.net = nn.Sequential(*layers)
        self.net.apply(_init_weights)

    def forward(self, z, rep, noise):
        x = torch.cat([z, rep, noise], dim=-1)
        return self.net(x)

    def sample(self, z, rep, deterministic=False):
        """Return (action, None, action_det) matching GaussianActor signature.

        deterministic=True uses ε=0 (mode of the flow distribution).
        """
        if deterministic:
            noise = torch.zeros(z.shape[0], self.action_dim, device=z.device)
        else:
            noise = torch.randn(z.shape[0], self.action_dim, device=z.device)
        a = self.forward(z, rep, noise)
        return a, None, a


class GaussianActor(nn.Module):
    """Goal-conditioned Gaussian actor (used for HL only).

    HL: state_dim=192, goal_dim=192, output_dim=rep_dim, tanh_squash=False.
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX =  2.0

    def __init__(self, state_dim=192, goal_dim=192, output_dim=10,
                 hidden_dims=(512, 512, 512),
                 tanh_squash=False, action_scale=1.0):
        super().__init__()
        self.tanh_squash = tanh_squash
        self.action_scale = action_scale

        in_dim = state_dim + goal_dim
        backbone = []
        for h in hidden_dims:
            backbone += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU()]
            in_dim = h
        self.backbone = nn.Sequential(*backbone)
        self.mean_head = nn.Linear(in_dim, output_dim)
        self.log_stds  = nn.Parameter(torch.zeros(output_dim))

        self.backbone.apply(_init_weights)
        nn.init.uniform_(self.mean_head.weight, -1e-3, 1e-3)
        nn.init.constant_(self.mean_head.bias,   0.0)

    def forward(self, state, goal):
        h = self.backbone(torch.cat([state, goal], dim=-1))
        mean    = self.mean_head(h)
        log_std = self.log_stds.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX).expand_as(mean)
        return mean, log_std

    def sample(self, state, goal):
        mean, log_std = self.forward(state, goal)
        dist = Normal(mean, log_std.exp())
        x    = dist.rsample()
        if self.tanh_squash:
            y             = torch.tanh(x)
            action        = y * self.action_scale
            lp = dist.log_prob(x) - torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
            log_prob      = lp.sum(dim=-1)
            deterministic = torch.tanh(mean) * self.action_scale
        else:
            action        = x
            log_prob      = dist.log_prob(x).sum(dim=-1)
            deterministic = mean
        return action, log_prob, deterministic

    def log_prob(self, state, goal, target):
        mean, log_std = self.forward(state, goal)
        dist = Normal(mean, log_std.exp())
        if self.tanh_squash:
            t_norm = (target / self.action_scale).clamp(-1 + 1e-6, 1 - 1e-6)
            x  = torch.atanh(t_norm)
            lp = dist.log_prob(x) - torch.log(self.action_scale * (1 - t_norm.pow(2)) + 1e-6)
            return lp.sum(dim=-1)
        else:
            return dist.log_prob(target).sum(dim=-1)


class TwinValue(nn.Module):
    """Twin V-network: V(z, rep) → (scalar, scalar)."""

    def __init__(self, latent_dim=192, rep_dim=10, hidden_dims=(512, 512, 512)):
        super().__init__()
        in_dim = latent_dim + rep_dim

        def _make_v():
            layers, d = [], in_dim
            for h in hidden_dims:
                layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()]
                d = h
            layers.append(nn.Linear(d, 1))
            net = nn.Sequential(*layers)
            net.apply(_init_weights)
            nn.init.uniform_(net[-1].weight, -1e-3, 1e-3)
            nn.init.constant_(net[-1].bias,   0.0)
            return net

        self.v1 = _make_v()
        self.v2 = _make_v()

    def forward(self, state, rep):
        x = torch.cat([state, rep], dim=-1)
        return self.v1(x).squeeze(-1), self.v2(x).squeeze(-1)


# =============================================================================
# 2. Flow Utilities
# =============================================================================

@torch.no_grad()
def euler_integrate(bc_flow, z, rep, noise, flow_steps, action_scale):
    """Integrate BC flow via Euler method (no gradients, matching FQL.compute_flow_actions).

    FQL Euler:
        for i in range(flow_steps):
            t   = i / flow_steps
            vel = u_θ(obs, x, t)
            x   = x + vel / flow_steps
        actions = clip(x, -1, 1)

    Args:
        bc_flow:     LLBCFlow network (frozen during call via no_grad context).
        z:           [B, latent_dim] current state latent.
        rep:         [B, rep_dim] goal representation.
        noise:       [B, action_dim] initial sample x_0 ~ N(0, I).
        flow_steps:  number of Euler steps (FQL default: 10).
        action_scale: clip bound.

    Returns:
        [B, action_dim] integrated action clipped to ±action_scale.
    """
    x = noise.clone()
    B = x.shape[0]
    for i in range(flow_steps):
        t   = torch.full((B, 1), i / flow_steps, device=x.device, dtype=x.dtype)
        vel = bc_flow(z, rep, x, t)
        x   = x + vel / flow_steps
    return x.clamp(-action_scale, action_scale)


# =============================================================================
# 3. Data Pipeline  (identical to train_hiql_lewm.py)
# =============================================================================

class RealOfflineCache:
    """GPU-native sampler for HIQL from pre-computed encoder-space latents.

    sample_ll_batch:       (z_t, a, r, z_next, z_subgoal) — dense LeWM reward
    sample_hl_batch:       HER uniform-future (z_t, z_target, g_ultimate)
    sample_value_her_batch: (z_t, z_next, g_her) — HER for auxiliary V update
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
        all_wm_lat_list, wm_offsets   = [], []

        offset = wm_offset = skipped = 0

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
            scaled = action_scaler.transform(ep_a_raw[:n_raw].numpy().astype(np.float32))
            ep_a_wm = torch.tensor(scaled.reshape(n_wm, wm_adim), dtype=torch.float32)

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
            offset    += n_wm

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
        self.ep_n_wm = self.ep_ends - self.ep_starts

    def _kstep_latents(self, idx, k):
        ep_id    = self.ep_ids[idx]
        t        = self.t_within[idx]
        n_wm     = self.ep_n_wm[ep_id]
        target_t = (t + k).clamp(max=n_wm)
        return self.wm_latents_flat[self.ep_wm_offsets[ep_id] + target_t]

    def _her_future_idxs(self, idx):
        """Uniform HER goal index in [t+1, n_wm], matching HGCDataset."""
        ep_id = self.ep_ids[idx]
        t     = self.t_within[idx]
        n_wm  = self.ep_n_wm[ep_id]
        t1    = torch.minimum(t + 1, n_wm).float()
        d     = torch.rand_like(t1)
        g_idx = torch.round(t1 * d + n_wm.float() * (1.0 - d)).long().clamp(max=n_wm)
        return g_idx, ep_id, t, n_wm

    def sample_ll_batch(self, batch_size, subgoal_steps):
        idx       = torch.randint(0, self.total, (batch_size,), device=self.device)
        z_t       = self.z_t_flat[idx]
        a         = self.a_flat[idx]
        z_next    = self.z_next_flat[idx]
        z_subgoal = self._kstep_latents(idx, subgoal_steps)
        r         = -torch.norm(z_next - z_subgoal, p=2, dim=-1)
        return z_t, a, r, z_next, z_subgoal

    def sample_hl_batch(self, batch_size, subgoal_steps):
        idx = torch.randint(0, self.total, (batch_size,), device=self.device)
        z_t = self.z_t_flat[idx]
        g_idx, ep_id, t, _ = self._her_future_idxs(idx)
        target_idx = torch.minimum(t + subgoal_steps, g_idx)
        g_ult   = self.wm_latents_flat[self.ep_wm_offsets[ep_id] + g_idx]
        z_target = self.wm_latents_flat[self.ep_wm_offsets[ep_id] + target_idx]
        return z_t, z_target, g_ult

    def sample_value_her_batch(self, batch_size):
        idx    = torch.randint(0, self.total, (batch_size,), device=self.device)
        z_t    = self.z_t_flat[idx]
        z_next = self.z_next_flat[idx]
        g_idx, ep_id, _, _ = self._her_future_idxs(idx)
        g_her  = self.wm_latents_flat[self.ep_wm_offsets[ep_id] + g_idx]
        return z_t, z_next, g_her


class SyntheticWMBuffer:
    """FIFO circular buffer — stores 192D z_subgoal for re-computing rep on the fly."""

    def __init__(self, capacity=500_000, latent_dim=192, action_dim=25, device='cuda'):
        self.capacity = capacity
        self.device   = device
        self.ptr  = 0
        self.size = 0
        self.z_t       = torch.zeros(capacity, latent_dim, device=device)
        self.a         = torch.zeros(capacity, action_dim,  device=device)
        self.r         = torch.zeros(capacity,              device=device)
        self.z_next    = torch.zeros(capacity, latent_dim, device=device)
        self.z_subgoal = torch.zeros(capacity, latent_dim, device=device)

    def push(self, z_t, a, r, z_next, z_subgoal):
        B = z_t.shape[0]; end = self.ptr + B
        if end <= self.capacity:
            sl = slice(self.ptr, end)
            self.z_t[sl] = z_t; self.a[sl] = a; self.r[sl] = r
            self.z_next[sl] = z_next; self.z_subgoal[sl] = z_subgoal
        else:
            ov = end - self.capacity; vd = B - ov
            self.z_t[self.ptr:] = z_t[:vd];    self.z_t[:ov] = z_t[vd:]
            self.a[self.ptr:]   = a[:vd];       self.a[:ov]   = a[vd:]
            self.r[self.ptr:]   = r[:vd];       self.r[:ov]   = r[vd:]
            self.z_next[self.ptr:] = z_next[:vd]; self.z_next[:ov] = z_next[vd:]
            self.z_subgoal[self.ptr:] = z_subgoal[:vd]; self.z_subgoal[:ov] = z_subgoal[vd:]
        self.ptr  = end % self.capacity
        self.size = min(self.size + B, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.z_t[idx], self.a[idx], self.r[idx], self.z_next[idx], self.z_subgoal[idx]


# =============================================================================
# 4. Imagination Engine
# =============================================================================

@torch.no_grad()
def run_imagination(wm_model, ll_bc_flow, ll_onestep, goal_rep,
                    real_cache, syn_buffer, k, batch_size, action_scale, device):
    """k-step WM rollouts anchored on real (z_t, z_subgoal) pairs.

    LL uses the one-step distilled policy for fast sampling.
    Dense reward against the 192D anchor subgoal preserves the LeWM advantage.
    """
    idx       = torch.randint(0, real_cache.total, (batch_size,), device=device)
    z_curr    = real_cache.z_t_flat[idx].clone()
    z_subgoal = real_cache._kstep_latents(idx, k)

    for _ in range(k):
        rep_sub = goal_rep(z_curr, z_subgoal)
        a, _, _ = ll_onestep.sample(z_curr, rep_sub, deterministic=False)
        a = a.clamp(-action_scale, action_scale)

        z_state_wm = z_curr.unsqueeze(1)
        a_wm       = a.unsqueeze(1)
        act_emb    = wm_model.action_encoder(a_wm)
        z_next     = wm_model.predict(z_state_wm, act_emb)[:, -1, :]

        r = -torch.norm(z_next - z_subgoal, p=2, dim=-1)
        syn_buffer.push(z_curr, a, r, z_next, z_subgoal)
        z_curr = z_next


# =============================================================================
# 5. IQL Loss Functions (value update identical to train_hiql_lewm.py)
# =============================================================================

def _expectile(adv, diff, tau):
    w = torch.where(adv >= 0,
                    tau * torch.ones_like(adv),
                    (1.0 - tau) * torch.ones_like(adv))
    return (w * diff.pow(2)).mean()


def _mixed_batch(real_cache, syn_buffer, batch_size, subgoal_steps, syn_ratio, device):
    n_real = int(batch_size * (1.0 - syn_ratio))
    n_syn  = batch_size - n_real
    zr, ar, rr, znr, zsr = real_cache.sample_ll_batch(n_real, subgoal_steps)
    zs, as_, rs, zns, zss = syn_buffer.sample(n_syn)
    return (torch.cat([zr, zs]), torch.cat([ar, as_]),
            torch.cat([rr, rs]), torch.cat([znr, zns]),
            torch.cat([zsr, zss]))


def _value_step(value_net, value_target, goal_rep, goal_rep_target,
                z, z_next, z_goal, r, gamma, expectile, value_optimizer):
    """Joint V + φ value update (identical to train_hiql_lewm.py)."""
    with torch.no_grad():
        rep_curr_t = goal_rep_target(z,      z_goal)
        rep_next_t = goal_rep_target(z_next, z_goal)
        vn1_t, vn2_t = value_target(z_next, rep_next_t)
        q     = r + gamma * torch.min(vn1_t, vn2_t)
        v1_t, v2_t = value_target(z, rep_curr_t)
        adv   = q - (v1_t + v2_t) / 2
        q1    = r + gamma * vn1_t
        q2    = r + gamma * vn2_t

    rep_curr = goal_rep(z, z_goal)
    v1, v2   = value_net(z, rep_curr)
    v_loss   = (_expectile(adv, q1 - v1, expectile) +
                _expectile(adv, q2 - v2, expectile))

    value_optimizer.zero_grad()
    v_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(value_net.parameters()) + list(goal_rep.parameters()), 1.0)
    value_optimizer.step()
    return v_loss.item()


# =============================================================================
# 6. Training Loop
# =============================================================================

def train_loop(
    wm_model,
    hl_actor,         hl_optimizer,
    ll_bc_flow,       ll_onestep,    ll_flow_optimizer,
    value_net,        value_optimizer,
    value_target,
    goal_rep,         goal_rep_target,
    real_cache,
    syn_buffer,
    total_steps       = 200_000,
    warmup_fraction   = 0.2,
    subgoal_steps     = 8,
    imagination_k     = 8,
    syn_ratio         = 0.5,
    batch_size        = 256,
    gamma             = 0.99,
    tau               = 0.005,
    expectile         = 0.7,
    alpha_low         = 3.0,
    alpha_high        = 3.0,
    action_scale      = 3.0,
    rep_dim           = 10,
    flow_steps        = 10,
    ll_grad_updates   = 1,
    hl_grad_updates   = 1,
    imagination_batches = 4,
    save_dir          = './checkpoints_hiql_flow',
    device            = 'cuda',
    log_interval      = 100,
    save_interval     = 1000,
):
    os.makedirs(save_dir, exist_ok=True)
    warmup_steps = int(total_steps * warmup_fraction)

    print(f"\n{'='*65}", flush=True)
    print(f"  HIQL+LeWM+Flow training (HER + {rep_dim}D goal_rep + FQL LL)", flush=True)
    print(f"  total_steps    : {total_steps:,}", flush=True)
    print(f"  warmup_steps   : {warmup_steps:,}", flush=True)
    print(f"  subgoal_steps  : {subgoal_steps}  flow_steps: {flow_steps}", flush=True)
    print(f"  imagination_k  : {imagination_k}  syn_ratio: {syn_ratio}", flush=True)
    print(f"  batch_size     : {batch_size}  rep_dim: {rep_dim}", flush=True)
    print(f"  save_dir       : {save_dir}", flush=True)
    print(f"{'='*65}\n", flush=True)

    csv_path = os.path.join(save_dir, 'training_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow([
            'step', 'phase',
            'value_loss', 'll_bc_loss', 'll_distill_loss', 'hl_actor_loss',
            'syn_buf_size', 'elapsed_s',
        ])

    t0 = time.time()
    t_interval = time.time()
    sum_v = sum_bc = sum_dist = sum_hl = 0.0
    log_n = 0
    prev_phase = 1

    for step in range(total_steps):
        phase = 1 if step < warmup_steps else 2

        if phase != prev_phase:
            print(f"\n>>> Phase 2 begins at step {step:,} — imagination engine ON\n",
                  flush=True)
            prev_phase = phase

        # --------------------------------------------------------
        # Phase 2: imagination
        # --------------------------------------------------------
        if phase == 2:
            for _ in range(imagination_batches):
                run_imagination(
                    wm_model, ll_bc_flow, ll_onestep, goal_rep,
                    real_cache, syn_buffer,
                    k=imagination_k, batch_size=batch_size,
                    action_scale=action_scale, device=device,
                )

        use_mixed = (phase == 2 and syn_buffer.size >= batch_size)

        # --------------------------------------------------------
        # Update A: IQL Value (V + φ) on LL subgoal goals
        # --------------------------------------------------------
        for _ in range(ll_grad_updates):
            if use_mixed:
                z, a, r, z_next, z_sub = _mixed_batch(
                    real_cache, syn_buffer, batch_size, subgoal_steps, syn_ratio, device)
            else:
                z, a, r, z_next, z_sub = real_cache.sample_ll_batch(batch_size, subgoal_steps)

            sum_v += _value_step(
                value_net, value_target, goal_rep, goal_rep_target,
                z, z_next, z_sub, r, gamma, expectile, value_optimizer,
            )

        # Auxiliary value update on HER goals (covers HL query distribution)
        z_hl, z_next_hl, g_her = real_cache.sample_value_her_batch(batch_size)
        r_hl = -torch.norm(z_next_hl - g_her, p=2, dim=-1)
        sum_v += _value_step(
            value_net, value_target, goal_rep, goal_rep_target,
            z_hl, z_next_hl, g_her, r_hl, gamma, expectile, value_optimizer,
        )

        # Soft-update targets
        with torch.no_grad():
            for tp, p in zip(value_target.parameters(), value_net.parameters()):
                tp.data.mul_(1.0 - tau).add_(p.data * tau)
            for tp, p in zip(goal_rep_target.parameters(), goal_rep.parameters()):
                tp.data.mul_(1.0 - tau).add_(p.data * tau)

        # --------------------------------------------------------
        # Update B: Flow Low-Level Actor
        #
        # BCFlow  ← L_bc   (pure BC, no advantage)
        # OneStep ← L_dist (advantage-weighted distillation)
        #
        # adv = V(z_next, rep_next) - V(z, rep_curr)   [stop_grad]
        # w   = exp(alpha_low * adv).clamp(max=100)
        #
        # L_bc   = ||u_θ(z, rep, x_t, t) - vel||²
        # L_dist = mean(w * ||µ_ψ(z, rep, ε) - a_flow||²)
        #   where a_flow = Euler(u_θ, z, rep, ε)       [stop_grad through u_θ]
        # --------------------------------------------------------
        for _ in range(ll_grad_updates):
            if use_mixed:
                z, a, r, z_next, z_sub = _mixed_batch(
                    real_cache, syn_buffer, batch_size, subgoal_steps, syn_ratio, device)
            else:
                z, a, r, z_next, z_sub = real_cache.sample_ll_batch(batch_size, subgoal_steps)

            B = z.shape[0]

            with torch.no_grad():
                rep_curr = goal_rep(z,      z_sub)   # [B, rep_dim]
                rep_next = goal_rep(z_next, z_sub)
                v1c, v2c = value_net(z,      rep_curr)
                v1n, v2n = value_net(z_next, rep_next)
                adv = (v1n + v2n) / 2 - (v1c + v2c) / 2      # [B]
                w   = (alpha_low * adv).exp().clamp(max=100.0) # [B]

                # Euler integration target for distillation
                # (same noise ε used for both — matches FQL.actor_loss)
                eps    = torch.randn(B, a.shape[-1], device=device)
                a_flow = euler_integrate(ll_bc_flow, z, rep_curr, eps,
                                         flow_steps, action_scale)  # [B, 25]

            # ── BC flow loss ──────────────────────────────────────────────
            # Linear interpolation in data space (not tanh space): the
            # StandardScaler normalisation already makes actions ~ N(0,1),
            # so the flow operates in a well-conditioned space without squash.
            x_0 = torch.randn_like(a)                              # [B, 25]
            x_1 = a                                                # [B, 25] data actions
            t   = torch.rand(B, 1, device=device)                  # [B, 1]
            x_t = (1.0 - t) * x_0 + t * x_1
            vel = x_1 - x_0

            pred    = ll_bc_flow(z, rep_curr, x_t, t)              # [B, 25]
            bc_loss = ((pred - vel) ** 2).mean()

            # ── Distillation loss ─────────────────────────────────────────
            # Gradients flow into ll_onestep only; a_flow is no_grad.
            a_onestep = ll_onestep(z, rep_curr, eps)               # [B, 25]
            # Match FQL: distill loss uses unclamped one-step vs clamped flow target
            distill_loss = (w.unsqueeze(-1) * (a_onestep - a_flow) ** 2).mean()

            ll_loss = bc_loss + distill_loss

            ll_flow_optimizer.zero_grad()
            ll_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(ll_bc_flow.parameters()) + list(ll_onestep.parameters()), 1.0)
            ll_flow_optimizer.step()

            sum_bc   += bc_loss.item()
            sum_dist += distill_loss.item()

        # --------------------------------------------------------
        # Update C: High-Level Actor AWR (identical to train_hiql_lewm.py)
        # target = φ(z_t, z_target).detach()  ← 10D log-prob target
        # HL actor input: (z_t [192D], g_ult [192D]) → 10D mean+std
        # --------------------------------------------------------
        for _ in range(hl_grad_updates):
            z_t, z_tk, g_ult = real_cache.sample_hl_batch(batch_size, subgoal_steps)

            with torch.no_grad():
                rep_t   = goal_rep(z_t,  g_ult)
                rep_tk  = goal_rep(z_tk, g_ult)
                v1t, v2t   = value_net(z_t,  rep_t)
                v1tk, v2tk = value_net(z_tk, rep_tk)
                adv  = (v1tk + v2tk) / 2 - (v1t + v2t) / 2
                w    = (alpha_high * adv).exp().clamp(max=100.0)
                target_rep = goal_rep(z_t, z_tk)                   # 10D target

            log_p   = hl_actor.log_prob(z_t, g_ult, target_rep)
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
            d_ll = max(log_n * ll_grad_updates, 1)
            d_hl = max(log_n * hl_grad_updates, 1)
            now  = time.time()
            sps  = log_interval / max(now - t_interval, 1e-6)
            eta  = (total_steps - step) / max(sps, 1e-6)
            print(
                f"Step {step:06d}/{total_steps:,} ({100.*step/total_steps:4.1f}%) | "
                f"Ph{phase} | V: {sum_v/d_ll:.4f} | "
                f"BC: {sum_bc/d_ll:.4f} | Dist: {sum_dist/d_ll:.4f} | "
                f"HL: {sum_hl/d_hl:.4f} | SynBuf: {syn_buffer.size:,} | "
                f"{sps:.1f} sps | ETA: {eta/60:.0f}m | "
                f"elapsed: {(now-t0)/60:.0f}m",
                flush=True,
            )
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    step, phase,
                    sum_v/d_ll, sum_bc/d_ll, sum_dist/d_ll, sum_hl/d_hl,
                    syn_buffer.size, now - t_interval,
                ])
            sum_v = sum_bc = sum_dist = sum_hl = 0.0
            log_n = 0
            t_interval = now

        # --------------------------------------------------------
        # Checkpointing
        # --------------------------------------------------------
        if step > 0 and (step % save_interval == 0 or step == total_steps - 1):
            torch.save(ll_bc_flow.state_dict(),  os.path.join(save_dir, 'll_bc_flow.pth'))
            torch.save(ll_onestep.state_dict(),  os.path.join(save_dir, 'll_onestep.pth'))
            torch.save(hl_actor.state_dict(),    os.path.join(save_dir, 'hl_actor.pth'))
            torch.save(value_net.state_dict(),   os.path.join(save_dir, 'value_net.pth'))
            torch.save(goal_rep.state_dict(),    os.path.join(save_dir, 'goal_rep.pth'))
            print(f"  → checkpoint saved at step {step}", flush=True)


# =============================================================================
# 7. Inference Helper
# =============================================================================

def select_action(wm_model, hl_actor, ll_onestep, goal_rep,
                  obs_pixels, g_pixels, subgoal_steps,
                  step_counter, current_subgoal_rep,
                  device, img_transform, rep_dim=10):
    """Encode → HL every k steps → LL (one-step policy) every step.

    HL outputs rep_dim-D mean, length-normalised onto sphere of radius √rep_dim.
    LL uses the one-step distilled policy with ε=0 (deterministic mode).

    Returns: action [25], new_subgoal_rep [rep_dim], z_curr [192]
    """
    with torch.no_grad():
        obs_f = img_transform(obs_pixels.to(device)).unsqueeze(0)
        g_f   = img_transform(g_pixels.to(device)).unsqueeze(0)
        z_curr = wm_model.encode({'pixels': obs_f})['emb'].squeeze(0).squeeze(0)
        g_ult  = wm_model.encode({'pixels': g_f})['emb'].squeeze(0).squeeze(0)

        if current_subgoal_rep is None or step_counter % subgoal_steps == 0:
            _, _, new_rep = hl_actor.sample(z_curr.unsqueeze(0), g_ult.unsqueeze(0))
            new_rep = new_rep.squeeze(0)
            new_rep = new_rep / (new_rep.norm() + 1e-8) * (rep_dim ** 0.5)
        else:
            new_rep = current_subgoal_rep

        _, _, action = ll_onestep.sample(
            z_curr.unsqueeze(0), new_rep.unsqueeze(0), deterministic=True)
        action = action.squeeze(0)

    return action, new_rep, z_curr


# =============================================================================
# 8. JEPA Loader (identical to train_hiql_lewm.py)
# =============================================================================

def _load_jepa_from_ckpt(ckpt_path, device, img_size=224, patch_size=14):
    if img_size == 224:
        model = swm.policy.AutoCostModel(ckpt_path)
        print(f"  Loaded 224×224 JEPA via AutoCostModel from {ckpt_path}")
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    import stable_pretraining as spt
    from jepa import JEPA
    from module import ARPredictor, Embedder
    from module import MLP as WM_MLP

    encoder = spt.backbone.utils.vit_hf(
        "tiny", patch_size=patch_size, image_size=img_size,
        pretrained=False, use_mask_token=False,
    )
    predictor = ARPredictor(
        num_frames=3, input_dim=192, hidden_dim=192, output_dim=192,
        depth=6, heads=16, mlp_dim=2048, dim_head=64,
        dropout=0.1, emb_dropout=0.0,
    )
    action_encoder = Embedder(input_dim=25, emb_dim=192)
    projector  = WM_MLP(input_dim=192, output_dim=192, hidden_dim=2048,
                        norm_fn=torch.nn.BatchNorm1d)
    pred_proj  = WM_MLP(input_dim=192, output_dim=192, hidden_dim=2048,
                        norm_fn=torch.nn.BatchNorm1d)
    model = JEPA(encoder=encoder, predictor=predictor,
                 action_encoder=action_encoder,
                 projector=projector, pred_proj=pred_proj)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'state_dict' in ckpt:
        raw_sd = {k[len('model.'):]: v
                  for k, v in ckpt['state_dict'].items() if k.startswith('model.')}
        epoch, gstep = ckpt.get('epoch', '?'), ckpt.get('global_step', '?')
    else:
        raw_sd = dict(ckpt)
        epoch, gstep = '?', '?'
    model.load_state_dict(raw_sd, strict=True)
    print(f"  Loaded 64×64 JEPA: {ckpt_path}  (epoch {epoch}, step {gstep})")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# =============================================================================
# 9. Main
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HIQL + LeWM + Flow LL Training')

    parser.add_argument('--ckpt_path',    type=str, default=None)
    parser.add_argument('--cache_path',   type=str, default=None)
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--save_dir',     type=str, default=None)

    parser.add_argument('--total_steps',      type=int,   default=200_000)
    parser.add_argument('--warmup_fraction',  type=float, default=0.2)

    parser.add_argument('--subgoal_steps',  type=int,   default=8)
    parser.add_argument('--imagination_k',  type=int,   default=None)
    parser.add_argument('--syn_ratio',      type=float, default=0.5)
    parser.add_argument('--batch_size',     type=int,   default=256)
    parser.add_argument('--alpha_low',      type=float, default=3.0)
    parser.add_argument('--alpha_high',     type=float, default=3.0)
    parser.add_argument('--gamma',          type=float, default=0.99)
    parser.add_argument('--expectile',      type=float, default=0.7)
    parser.add_argument('--action_scale',   type=float, default=3.0)
    parser.add_argument('--rep_dim',        type=int,   default=10)
    parser.add_argument('--flow_steps',     type=int,   default=10,
                        help='Euler integration steps for BCFlow target (FQL default: 10).')

    parser.add_argument('--img_size',   type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=14)

    args = parser.parse_args()
    if args.imagination_k is None:
        args.imagination_k = args.subgoal_steps

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    STABLEWM_HOME = os.environ.get(
        'STABLEWM_HOME', os.path.join(os.path.expanduser('~'), 'stable_wm_data'))
    data_path  = args.dataset_path or os.path.join(STABLEWM_HOME, 'ogbench', 'cube_single_expert')
    _default_ckpt = ('lejepa' if args.img_size == 224 else 'lewm_ogbench_weights.ckpt')
    ckpt_path  = args.ckpt_path  or os.path.join(STABLEWM_HOME, 'cube', _default_ckpt)
    cache_path = args.cache_path or os.path.join(STABLEWM_HOME, 'lewm_224_latents_cache.pt')
    save_dir   = args.save_dir   or (
        f'./checkpoints_hiql_flow_lewm_k{args.subgoal_steps}'
        f'_sr{args.syn_ratio}_wf{args.warmup_fraction}'
        f'_rep{args.rep_dim}_fs{args.flow_steps}')

    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # --- Load frozen WM ---
    if args.ckpt_path is not None:
        wm_model = _load_jepa_from_ckpt(ckpt_path, device, args.img_size, args.patch_size)
    else:
        with initialize(version_base=None, config_path='../config'):
            cfg = compose(config_name='eval/cube', overrides=['+policy=cube/lejepa'])
        wm_model = swm.policy.AutoCostModel(cfg.policy).to(device).eval()
    for p in wm_model.parameters():
        p.requires_grad = False

    # --- Load latent cache ---
    print(f'Loading cache from {cache_path} ...')
    cache_data  = torch.load(cache_path, map_location='cpu')
    all_latents = cache_data['all_latents']
    all_actions = cache_data.get('all_actions', [])
    if not all_actions:
        raise RuntimeError("Cache has no 'all_actions'. Rebuild with --save_actions.")

    from sklearn import preprocessing as sk_pre
    print('Fitting StandardScaler ...')
    _ds = swm.data.HDF5Dataset(data_path, keys_to_cache=['action'],
                               cache_dir=os.path.dirname(data_path))
    action_raw = _ds.get_col_data('action')
    action_raw = action_raw[~np.isnan(action_raw).any(axis=1)]
    action_scaler = sk_pre.StandardScaler()
    action_scaler.fit(action_raw)
    print(f'  Scaler fit on {len(action_raw):,} frames')

    real_cache = RealOfflineCache(
        all_latents=all_latents, all_actions=all_actions,
        action_scaler=action_scaler, device=device, frameskip=5)
    syn_buffer = SyntheticWMBuffer(capacity=500_000, latent_dim=192, action_dim=25, device=device)

    LATENT_DIM  = 192
    ACTION_DIM  = 25
    REP_DIM     = args.rep_dim
    HIDDEN      = (512, 512, 512)
    FLOW_HIDDEN = (512, 512, 512, 512)   # 4 layers, matching FQL default

    goal_rep = GoalRep(LATENT_DIM, REP_DIM, HIDDEN, layer_norm=True).to(device)
    goal_rep_target = GoalRep(LATENT_DIM, REP_DIM, HIDDEN, layer_norm=True).to(device)
    goal_rep_target.load_state_dict(goal_rep.state_dict())
    for p in goal_rep_target.parameters():
        p.requires_grad = False

    # HL actor: (z 192, g 192) → 10D rep (Gaussian AWR, unchanged from v2)
    hl_actor = GaussianActor(
        state_dim=LATENT_DIM, goal_dim=LATENT_DIM, output_dim=REP_DIM,
        hidden_dims=HIDDEN, tanh_squash=False,
    ).to(device)

    # LL BC flow: (z 192, rep 10, x_t 25, t 1) → velocity 25
    ll_bc_flow = LLBCFlow(
        latent_dim=LATENT_DIM, rep_dim=REP_DIM, action_dim=ACTION_DIM,
        hidden_dims=FLOW_HIDDEN,
    ).to(device)

    # LL one-step policy: (z 192, rep 10, ε 25) → action 25
    ll_onestep = LLOneStep(
        latent_dim=LATENT_DIM, rep_dim=REP_DIM, action_dim=ACTION_DIM,
        hidden_dims=FLOW_HIDDEN,
    ).to(device)

    value_net    = TwinValue(LATENT_DIM, REP_DIM, HIDDEN).to(device)
    value_target = TwinValue(LATENT_DIM, REP_DIM, HIDDEN).to(device)
    value_target.load_state_dict(value_net.state_dict())
    for p in value_target.parameters():
        p.requires_grad = False

    print(
        f'Networks (LATENT={LATENT_DIM}, REP={REP_DIM}, FLOW_STEPS={args.flow_steps}):\n'
        f'  GoalRep φ:    {sum(p.numel() for p in goal_rep.parameters()):>10,} params\n'
        f'  HL Actor:     {sum(p.numel() for p in hl_actor.parameters()):>10,} params\n'
        f'  LL BCFlow:    {sum(p.numel() for p in ll_bc_flow.parameters()):>10,} params\n'
        f'  LL OneStep:   {sum(p.numel() for p in ll_onestep.parameters()):>10,} params\n'
        f'  Value:        {sum(p.numel() for p in value_net.parameters()):>10,} params'
    )

    hl_optimizer       = torch.optim.Adam(hl_actor.parameters(), lr=3e-4)
    # BCFlow + OneStep trained jointly with one optimizer (matches FQL single optim)
    ll_flow_optimizer  = torch.optim.Adam(
        list(ll_bc_flow.parameters()) + list(ll_onestep.parameters()), lr=3e-4)
    # V + φ trained jointly
    value_optimizer    = torch.optim.Adam(
        list(value_net.parameters()) + list(goal_rep.parameters()), lr=3e-4)

    train_loop(
        wm_model         = wm_model,
        hl_actor         = hl_actor,          hl_optimizer       = hl_optimizer,
        ll_bc_flow       = ll_bc_flow,
        ll_onestep       = ll_onestep,         ll_flow_optimizer  = ll_flow_optimizer,
        value_net        = value_net,          value_optimizer    = value_optimizer,
        value_target     = value_target,
        goal_rep         = goal_rep,           goal_rep_target    = goal_rep_target,
        real_cache       = real_cache,
        syn_buffer       = syn_buffer,
        total_steps      = args.total_steps,
        warmup_fraction  = args.warmup_fraction,
        subgoal_steps    = args.subgoal_steps,
        imagination_k    = args.imagination_k,
        syn_ratio        = args.syn_ratio,
        batch_size       = args.batch_size,
        gamma            = args.gamma,
        tau              = 0.005,
        expectile        = args.expectile,
        alpha_low        = args.alpha_low,
        alpha_high       = args.alpha_high,
        action_scale     = args.action_scale,
        rep_dim          = REP_DIM,
        flow_steps       = args.flow_steps,
        ll_grad_updates  = 1,
        hl_grad_updates  = 1,
        imagination_batches = 4,
        save_dir         = save_dir,
        device           = device,
    )
