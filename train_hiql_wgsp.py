"""
train_hiql_wgsp.py
WGSP: World-Grounded Subgoal Planning for offline hierarchical RL.

Replaces the (in-)famous Phase-2 imagination-engine in train_hiql_lewm.py
with a principled use of the frozen LeWM:

  * V is trained on REAL data only (no synthetic transitions).
  * HL is trained by AWR over imagined endpoint scores (HL-WGSP).
  * LL is distilled on the same rollouts using a within-rep advantage
    (WGSP-distil), keeping it competent on the rep distribution HL is
    converging toward.
  * LL outputs 5-D env actions; a frozen state-conditioned decoder
    D_θ : R^5 × R^192 -> R^25 inflates them to a 25-D chunk that the
    WM can consume. This restores AWR statistical efficiency.

All four design choices are individually toggleable via CLI flags so the
ablation matrix in thesis/wgsp.tex can be reproduced from a single
training entry point.
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

from train_action_decoder import ActionChunkDecoder


class FlowDecoderWrapper(nn.Module):
    """Drop-in replacement for ActionChunkDecoder when --decoder_type flow.

    Wraps FlowChunkDecoder so it exposes the same call signature:
        forward(a_first, z) → chunk_25d
    as ActionChunkDecoder, making it transparent to _ll_action.
    """

    def __init__(self, flow_model, flow_steps=10):
        super().__init__()
        self.flow  = flow_model
        self.flow_steps = flow_steps

    def forward(self, a_first, z):
        return self.flow.sample(z, a_first, flow_steps=self.flow_steps)


# =============================================================================
# 1. Network Architectures (verbatim copies of train_hiql_lewm.py)
# =============================================================================

def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


class GoalRep(nn.Module):
    """φ([z; z_goal]) -> rep, length-normalised onto sphere of radius √rep_dim."""

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
        self.net = nn.Sequential(*layers)
        self.net.apply(_init_weights)
        self.rep_dim = rep_dim

    def forward(self, z, z_goal):
        x = torch.cat([z, z_goal], dim=-1)
        rep = self.net(x)
        return rep / (rep.norm(dim=-1, keepdim=True) + 1e-8) * (self.rep_dim ** 0.5)


class GaussianActor(nn.Module):
    """Goal-conditioned Gaussian actor.

    HL: state_dim=192, goal_dim=192,        output_dim=rep_dim, tanh_squash=False, const_std=True
    LL: state_dim=192, goal_dim=rep_dim,    output_dim=5 or 25, tanh_squash=True,  const_std=False

    const_std=True fixes std=1 (matching ogbench const_std=True / baseline GaussianActor),
    preventing variance collapse in the HL that kills WGSP candidate diversity.
    """

    # LOG_STD_MIN raised from -5 to -2 (std floor 0.0067 → 0.135) to prevent
    # extreme overconfidence in the LL actor under prolonged AWR + WGSP-distil
    # pressure.  At -5 the per-dim log_p can grow beyond 4, making the distil
    # loss `−w·log_p` runaway negative; at -2 it's bounded near 1, keeping the
    # gradient signal healthy.  Doesn't affect HL (which uses const_std=True).
    LOG_STD_MIN = -2.0
    LOG_STD_MAX =  2.0

    def __init__(self, state_dim=192, goal_dim=192, output_dim=25,
                 hidden_dims=(512, 512, 512),
                 tanh_squash=False, action_scale=1.0, const_std=False):
        super().__init__()
        self.tanh_squash  = tanh_squash
        self.action_scale = action_scale
        self.const_std    = const_std

        in_dim = state_dim + goal_dim
        backbone = []
        for h in hidden_dims:
            backbone += [nn.Linear(in_dim, h), nn.GELU(), nn.LayerNorm(h)]
            in_dim = h
        self.backbone  = nn.Sequential(*backbone)
        self.mean_head = nn.Linear(in_dim, output_dim)
        if const_std:
            # Fixed std=1; register as buffer so state_dict round-trips cleanly.
            self.register_buffer('log_stds', torch.zeros(output_dim))
        else:
            self.log_stds = nn.Parameter(torch.zeros(output_dim))

        self.backbone.apply(_init_weights)
        nn.init.uniform_(self.mean_head.weight, -1e-3, 1e-3)
        nn.init.constant_(self.mean_head.bias,   0.0)

    def _trunk(self, state, goal):
        return self.backbone(torch.cat([state, goal], dim=-1))

    def forward(self, state, goal):
        h = self._trunk(state, goal)
        mean = self.mean_head(h)
        if self.const_std:
            log_std = torch.zeros_like(mean)   # std=1
        else:
            log_std = self.log_stds.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX).expand_as(mean)
        return mean, log_std

    def sample(self, state, goal):
        mean, log_std = self.forward(state, goal)
        std  = log_std.exp()
        dist = Normal(mean, std)
        x    = dist.rsample()
        if self.tanh_squash:
            y      = torch.tanh(x)
            action = y * self.action_scale
            lp = dist.log_prob(x)
            lp -= torch.log(self.action_scale * (1.0 - y.pow(2)) + 1e-6)
            log_prob = lp.sum(dim=-1)
            deterministic = torch.tanh(mean) * self.action_scale
        else:
            action        = x
            log_prob      = dist.log_prob(x).sum(dim=-1)
            deterministic = mean
        return action, log_prob, deterministic

    def log_prob(self, state, goal, target):
        mean, log_std = self.forward(state, goal)
        std  = log_std.exp()
        dist = Normal(mean, std)
        if self.tanh_squash:
            t_norm = (target / self.action_scale).clamp(-1 + 1e-6, 1 - 1e-6)
            x  = torch.atanh(t_norm)
            lp = dist.log_prob(x)
            lp -= torch.log(self.action_scale * (1.0 - t_norm.pow(2)) + 1e-6)
            return lp.sum(dim=-1)
        return dist.log_prob(target).sum(dim=-1)


class EnsembleValue(nn.Module):
    """V(z, rep) -> [B, n_heads].

    n_heads=2 reproduces the original TwinValue (IQL twin) behavior. With
    n_heads>2 the extra heads provide a disagreement signal usable as a
    MOPO-style uncertainty penalty in WGSP scoring.
    """

    def __init__(self, latent_dim=192, rep_dim=10,
                 hidden_dims=(512, 512, 512), n_heads=2):
        super().__init__()
        self.n_heads = n_heads
        in_dim = latent_dim + rep_dim

        def _make_v():
            layers, d = [], in_dim
            for h in hidden_dims:
                layers += [nn.Linear(d, h), nn.GELU(), nn.LayerNorm(h)]
                d = h
            layers.append(nn.Linear(d, 1))
            net = nn.Sequential(*layers)
            net.apply(_init_weights)
            nn.init.uniform_(net[-1].weight, -1e-3, 1e-3)
            nn.init.constant_(net[-1].bias,   0.0)
            return net

        self.heads = nn.ModuleList([_make_v() for _ in range(n_heads)])

    def forward(self, state, rep):
        x = torch.cat([state, rep], dim=-1)
        return torch.stack([h(x).squeeze(-1) for h in self.heads], dim=-1)

    def mean_v(self, state, rep):
        return self.forward(state, rep).mean(dim=-1)

    def std_v(self, state, rep):
        # Sample std (unbiased) — note for n_heads=2 this is just |v1-v2|/sqrt(2)
        return self.forward(state, rep).std(dim=-1, unbiased=True)


# Back-compat alias so existing checkpoints with key prefixes 'v1.', 'v2.'
# still load — see _load_legacy_twin_value below.
TwinValue = EnsembleValue


class LatentAdapter(nn.Module):
    """Trainable MLP on top of frozen 192D LeWM latents.

    Trained end-to-end with all HIQL/WGSP losses on the policy/value path.
    The WM path (rollouts, geometric term) continues to use raw 192D latents.
    """

    def __init__(self, in_dim=192, out_dim=256, hidden_dim=None, depth=2,
                 layer_norm=True):
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        dims = [in_dim] + [hidden_dim] * (depth - 1) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.GELU())
            if layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
        self.mlp = nn.Sequential(*layers)
        self.mlp.apply(_init_weights)

    def forward(self, x):
        return self.mlp(x)


def _load_legacy_twin_value(value_net, state_dict):
    """Map a legacy TwinValue state_dict ('v1.', 'v2.' prefixes) onto
    EnsembleValue's 'heads.0.', 'heads.1.' prefixes. Returns the remapped
    state_dict so caller can call value_net.load_state_dict(...)."""
    if any(k.startswith('heads.') for k in state_dict.keys()):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        if k.startswith('v1.'):
            out['heads.0.' + k[3:]] = v
        elif k.startswith('v2.'):
            out['heads.1.' + k[3:]] = v
        else:
            out[k] = v
    return out


# =============================================================================
# 2. Data Pipeline (verbatim copy of RealOfflineCache)
# =============================================================================

class RealOfflineCache:
    """GPU-native HIQL sampler from pre-computed encoder-space latents.

    Operates at WM-step granularity (frameskip=5 raw frames per WM step).
    Actions are StandardScaler-normalised and stacked to 25D (5 × 5D).

    Sampling methods:
      sample_ll_batch(B, k)         -> (z_t, a, r, z_next, z_subgoal)
      sample_hl_batch(B, k)         -> (z_t, z_target, g_ultimate)        (HER)
      sample_value_her_batch(B)     -> (z_t, z_next, g_her)               (HER for V coverage)
      sample_anchor_batch(B)        -> z_t  (used as WGSP rollout anchor)
    """

    def __init__(self, all_latents, all_actions, action_scaler,
                 device='cuda', frameskip=5):
        self.device   = device
        self.frameskip = frameskip

        raw_adim = all_actions[0].shape[-1] if all_actions else 5
        wm_adim  = frameskip * raw_adim

        z_t_list, a_list, z_next_list = [], [], []
        a_first_list                  = []
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
            ep_a_first = ep_a_wm[:, :raw_adim]

            z_t_list.append(ep_z_wm[:-1])
            a_list.append(ep_a_wm)
            a_first_list.append(ep_a_first)
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
        self.raw_adim = raw_adim
        self.wm_adim  = wm_adim
        print(f"  RealOfflineCache: {self.n_eps} episodes, {self.total:,} WM-step transitions")

        self.z_t_flat       = torch.cat(z_t_list,    dim=0).to(device)
        self.a_flat         = torch.cat(a_list,      dim=0).to(device)   # 25-D chunk
        self.a_first_flat   = torch.cat(a_first_list, dim=0).to(device)  # 5-D first
        self.z_next_flat    = torch.cat(z_next_list, dim=0).to(device)
        self.ep_starts      = torch.tensor(ep_starts, device=device)
        self.ep_ends        = torch.tensor(ep_ends,   device=device)
        self.ep_goals       = torch.stack(ep_goals).to(device)
        self.ep_ids         = torch.tensor(ep_ids_list,  dtype=torch.long, device=device)
        self.t_within       = torch.tensor(t_within_list, dtype=torch.long, device=device)
        self.wm_latents_flat = torch.cat(all_wm_lat_list, dim=0).to(device)
        self.ep_wm_offsets  = torch.tensor(wm_offsets, dtype=torch.long, device=device)
        self.ep_n_wm        = self.ep_ends - self.ep_starts

    def _kstep_latents(self, idx, k):
        ep_id    = self.ep_ids[idx]
        t        = self.t_within[idx]
        n_wm     = self.ep_n_wm[ep_id]
        target_t = (t + k).clamp(max=n_wm)
        flat_idx = self.ep_wm_offsets[ep_id] + target_t
        return self.wm_latents_flat[flat_idx]

    def _her_future_idxs(self, idx):
        ep_id = self.ep_ids[idx]
        t     = self.t_within[idx]
        n_wm  = self.ep_n_wm[ep_id]
        t1     = torch.minimum(t + 1, n_wm).float()
        n_wm_f = n_wm.float()
        d      = torch.rand_like(t1)
        g_idx  = torch.round(t1 * d + n_wm_f * (1.0 - d)).long().clamp(max=n_wm)
        return g_idx, ep_id, t, n_wm

    def sample_ll_batch(self, B, k, use_5d=False):
        idx        = torch.randint(0, self.total, (B,), device=self.device)
        z_t        = self.z_t_flat[idx]
        a          = self.a_first_flat[idx] if use_5d else self.a_flat[idx]
        z_next     = self.z_next_flat[idx]
        z_subgoal  = self._kstep_latents(idx, k)
        r          = -torch.norm(z_next - z_subgoal, p=2, dim=-1)
        return z_t, a, r, z_next, z_subgoal

    def sample_hl_batch(self, B, k):
        idx = torch.randint(0, self.total, (B,), device=self.device)
        z_t = self.z_t_flat[idx]
        g_idx, ep_id, t, _ = self._her_future_idxs(idx)
        target_idx = torch.minimum(t + k, g_idx)
        g_flat   = self.ep_wm_offsets[ep_id] + g_idx
        tgt_flat = self.ep_wm_offsets[ep_id] + target_idx
        return z_t, self.wm_latents_flat[tgt_flat], self.wm_latents_flat[g_flat]

    def sample_value_her_batch(self, B,
                               p_curgoal=0.2, p_trajgoal=0.5):
        """Sparse-HER value batch matching train_hiql_baseline.

        Mixed goal sampling: 20% current state (boundary V(z,z)=0),
        50% HER trajectory future (same episode), 30% random (other episode).

        Returns (z_t, z_next, z_goal, success).  success=1 when value_goal IS
        the current state z_t (the baseline's success convention).  Caller
        derives r = success - 1, mask = 1 - success.
        """
        idx    = torch.randint(0, self.total, (B,), device=self.device)
        z_t    = self.z_t_flat[idx]
        z_next = self.z_next_flat[idx]
        ep_id  = self.ep_ids[idx]
        t      = self.t_within[idx]
        n_wm   = self.ep_n_wm[ep_id]

        # Sample roll type per batch element
        roll = torch.rand(B, device=self.device)
        is_cur  = roll < p_curgoal
        is_traj = (roll >= p_curgoal) & (roll < p_curgoal + p_trajgoal)

        # Trajectory goal: HER future state from same episode (uniform in [t+1, n_wm])
        d      = torch.rand(B, device=self.device)
        t1     = torch.minimum(t + 1, n_wm).float()
        n_wm_f = n_wm.float()
        traj_t = torch.round(t1 * d + n_wm_f * (1.0 - d)).long().clamp(max=n_wm)

        # Random goal: any state from any episode
        rand_idx = torch.randint(0, self.total, (B,), device=self.device)
        rand_ep  = self.ep_ids[rand_idx]
        rand_t   = self.t_within[rand_idx]

        value_ep = torch.where(is_cur | is_traj, ep_id, rand_ep)
        value_t  = torch.where(is_cur, t,
                   torch.where(is_traj, traj_t, rand_t))

        # success = (value_goal IS current state)
        success = ((value_ep == ep_id) & (value_t == t)).float()

        g_flat = self.ep_wm_offsets[value_ep] + value_t
        z_goal = self.wm_latents_flat[g_flat]
        return z_t, z_next, z_goal, success

    def sample_anchor_with_goal(self, B):
        """Returns (z_t, g_her) — the (anchor, goal) pair WGSP rolls out from."""
        idx = torch.randint(0, self.total, (B,), device=self.device)
        z_t = self.z_t_flat[idx]
        g_idx, ep_id, _, _ = self._her_future_idxs(idx)
        g_flat = self.ep_wm_offsets[ep_id] + g_idx
        return z_t, self.wm_latents_flat[g_flat]


# =============================================================================
# 3. IQL Loss Helpers (real-data-only V update — no synthetic mixing)
# =============================================================================

def _expectile(adv, diff, tau):
    w = torch.where(adv >= 0,
                    tau * torch.ones_like(adv),
                    (1.0 - tau) * torch.ones_like(adv))
    return (w * diff.pow(2)).mean()


def _value_loss(value_net, value_target, goal_rep, goal_rep_target,
                z, z_next, z_goal, r, gamma, expectile,
                adapter=None, mask=None):
    """IQL expectile loss over an arbitrary-size value ensemble.

    Mirrors the baseline pattern: adapter is applied uniformly to all
    observations and goals at the input boundary; V and GoalRep then
    operate purely in adapter (policy) space.

    mask: optional [B] tensor; 0 at goal (terminates bootstrap), 1 otherwise.
          For sparse HER: mask = 1 - success.  Without mask, dense bootstrap.

    Returns the loss tensor (graph attached). Caller is responsible for
    backward / optimizer.step in a combined update.
    """
    adapt = (lambda x: adapter(x)) if adapter is not None else (lambda x: x)
    za_t  = adapt(z)
    za_n  = adapt(z_next)
    za_g  = adapt(z_goal)
    if mask is None:
        mask = torch.ones_like(r)
    with torch.no_grad():
        rep_curr_t = goal_rep_target(za_t, za_g)
        rep_next_t = goal_rep_target(za_n, za_g)
        v_next_t = value_target(za_n, rep_next_t)              # [B, n_heads]
        v_next_min_t = v_next_t.min(dim=-1).values
        q_min        = r + gamma * mask * v_next_min_t         # [B]

        v_t = value_target(za_t, rep_curr_t).mean(dim=-1)      # [B]
        adv = q_min - v_t                                      # [B]

        q_per_head = r.unsqueeze(-1) + gamma * mask.unsqueeze(-1) * v_next_t  # [B, n_heads]

    rep_curr = goal_rep(za_t, za_g)
    v_pred   = value_net(za_t, rep_curr)                       # [B, n_heads]
    diff     = q_per_head - v_pred                             # [B, n_heads]

    v_loss = sum(_expectile(adv, diff[:, h], expectile)
                 for h in range(diff.shape[-1]))
    return v_loss


# =============================================================================
# 4. WGSP Core (rollout + HL update + LL distil)
# =============================================================================

def _wm_predict(wm_model, z, chunk_25d):
    """Single-step WM forward. z: [B, 192], chunk_25d: [B, 25] -> [B, 192]."""
    z_in = z.unsqueeze(1)                                    # [B, 1, 192]
    a_in = chunk_25d.unsqueeze(1)                            # [B, 1, 25]
    act_emb = wm_model.action_encoder(a_in)
    return wm_model.predict(z_in, act_emb)[:, -1, :]


def _ll_action(ll_actor, z, rep, decoder, action_scale,
               use_decoder, deterministic=False, z_raw=None, g_raw=None):
    """LL forward → (action_log_prob_input, chunk_25d_for_WM, raw_action_for_distil).

    If use_decoder: ll_actor outputs 5-D, decoder inflates to 25-D.
    Else:           ll_actor outputs 25-D directly.

    g_raw: raw 192D goal latent forwarded to goal-conditioned decoder.

    Returns the *sampled* (or mean) action plus the 25-D chunk to feed the WM.
    """
    if deterministic:
        mean, log_std = ll_actor(z, rep)
        std = log_std.exp()
        dist = Normal(mean, std)
        x = mean
    else:
        mean, log_std = ll_actor(z, rep)
        std = log_std.exp()
        dist = Normal(mean, std)
        x = dist.rsample()

    if ll_actor.tanh_squash:
        a = torch.tanh(x) * action_scale
    else:
        a = x

    if use_decoder:
        z_in = z_raw if z_raw is not None else z
        goal_arg = g_raw if (getattr(decoder, 'goal_dim', 0) > 0) else None
        chunk_25d = decoder(a, z_in, goal_arg)
    else:
        chunk_25d = a
    # log_prob of the pre-tanh sample x (matches ll_actor.log_prob convention)
    return a, chunk_25d, x, dist


def wgsp_rollout(wm_model, ll_actor, decoder,
                 z_anchor, g_ult, rep_candidates,
                 k, action_scale, use_decoder, adapter=None):
    """Run one stochastic LL+WM rollout per (anchor, rep) pair.

    Args:
      z_anchor       [B, 192]  raw WM latents
      g_ult          [B, 192]  raw WM goal latent (forwarded to goal-conditioned decoder)
      rep_candidates [B, rep_dim]
      adapter        LatentAdapter or None — applied to z before LL; WM uses raw z

    Returns:
      traj_z      [B, k+1, 192]  raw WM latents (for geometric term and WM stepping)
      traj_a_pre  [B, k, adim]   pre-tanh samples (for log_prob target)
      traj_a_post [B, k, adim]
    """
    adapt = (lambda x: adapter(x)) if adapter is not None else (lambda x: x)
    B  = z_anchor.shape[0]
    z  = z_anchor                      # raw WM space throughout
    zs, as_pre, as_post = [z], [], []
    for _ in range(k):
        a_post, chunk_25d, x_pre, _ = _ll_action(
            ll_actor, adapt(z), rep_candidates, decoder, action_scale, use_decoder,
            deterministic=False, z_raw=z, g_raw=g_ult)
        z = _wm_predict(wm_model, z, chunk_25d)   # WM needs raw z
        zs.append(z)
        as_pre.append(x_pre)
        as_post.append(a_post)
    traj_z      = torch.stack(zs, dim=1)           # raw [B, k+1, 192]
    traj_a_pre  = torch.stack(as_pre, dim=1)
    traj_a_post = torch.stack(as_post, dim=1)
    return traj_z, traj_a_pre, traj_a_post


@torch.no_grad()
def _score_endpoints(z_k, g_ult, value_net, goal_rep, beta_geom,
                     use_geometric_term, use_v_in_J=True,
                     lambda_mopo=0.0, return_diag=False, adapter=None):
    """Score per-rollout endpoint.

    J = V_mean(z_k, φ(z_k, g)) − β·‖z_k − g‖₂ − λ_mopo·V_std(z_k, φ(z_k, g)).

    V/φ queries use adapter(z_k) and adapter(g_ult) so that the scoring
    network sees inputs in the same space as it was trained on (uniform
    adapter pattern, mirrors baseline).  Geometric residual stays in raw
    192D WM space (SIGReg isotropy is in 192D).
    """
    adapt = (lambda x: adapter(x)) if adapter is not None else (lambda x: x)
    score = torch.zeros(z_k.shape[0], device=z_k.device)
    v_mean = None
    v_std = None
    if use_v_in_J or lambda_mopo > 0 or return_diag:
        z_k_pol = adapt(z_k)
        g_pol   = adapt(g_ult)
        rep = goal_rep(z_k_pol, g_pol)
        v_all = value_net(z_k_pol, rep)              # [B, n_heads]
        v_mean = v_all.mean(dim=-1)
        v_std = v_all.std(dim=-1, unbiased=True) if v_all.shape[-1] > 1 \
                else torch.zeros_like(v_mean)
    if use_v_in_J:
        score = score + v_mean
    if use_geometric_term and beta_geom > 0:
        score = score - beta_geom * torch.norm(z_k - g_ult, p=2, dim=-1)
    if lambda_mopo > 0:
        score = score - lambda_mopo * v_std
    if return_diag:
        return score, {'v_mean': v_mean, 'v_std': v_std}
    return score


def hl_wgsp_step(z_t, g_ult,
                 hl_actor, ll_actor, decoder, wm_model,
                 goal_rep, value_net,
                 N, M, k, beta_geom, alpha_H, alpha_L, action_scale,
                 use_decoder, use_geometric_term, use_v_in_J,
                 lambda_anchor=0.0,
                 z_target=None, ll_grad_only=False, return_rollouts=False,
                 lambda_mopo=0.0, return_v_diag=False,
                 ll_score_mode='goal', adapter=None,
                 value_target=None, goal_rep_target=None):
    """One HL-WGSP gradient step. Optionally returns rollouts for distillation.

    Adapter pattern matches train_hiql_baseline: applied uniformly to every
    192D WM observation/goal at the input boundary; HL, V and GoalRep then
    operate purely in adapter (policy) space.  Inside wgsp_rollout the
    adapter is also forwarded so the LL actor sees the same adapted state.

    ll_score_mode:
      'goal'      : LL within-rep advantage uses J^(i,m) (goal-based, default).
      'rep_reach' : LL within-rep advantage uses J_LL^(i,m) =
                    -||φ(z_t, z_k^(i,m)) - rep^(i)||_2  (faithfulness to the
                    HL-requested rep).  Decouples LL credit assignment from
                    the ultimate goal — addresses the "overshoot warping"
                    pathology where goal-based J trains LL to overshoot reps
                    in the direction of g.  Row 14 of the ablation.

    z_t       [B, 192]   raw WM latent (adapter applied inside this fn)
    g_ult     [B, 192]   raw WM latent
    z_target  [B, 192] or None — used only if lambda_anchor>0 (HIQL HL anchor)
    """
    adapt = (lambda x: adapter(x)) if adapter is not None else (lambda x: x)
    B, dz = z_t.shape

    # ---- 1. Sample N candidate reps from π^H  (HL operates in adapter space)
    # Detach adapter outputs so the WGSP backward only updates HL actor's own
    # weights.  The adapter is trained by V and LL losses; letting WGSP also
    # modify it causes the HL actor's input to shift, producing a non-tangential
    # radial component in mean and blowing up ‖mean‖.
    z_t_pol = adapt(z_t).detach()                             # [B, P]
    g_pol   = adapt(g_ult).detach()                           # [B, P]
    mean, log_std = hl_actor(z_t_pol, g_pol)
    std = log_std.exp()
    rep_dim = mean.shape[-1]

    # Normalized mean direction on the sphere.  The HL actor outputs a raw
    # vector mean ∈ R^rep_dim; we project to the sphere so that:
    #   (a) exploration diversity is always σ/‖μ_sphere‖ ≈ 1/√d, constant,
    #   (b) the AWR gradient is purely tangential (‖mean‖ conserved).
    mu_sphere   = mean / (mean.norm(dim=-1, keepdim=True).clamp_min(1e-8)) \
                  * (rep_dim ** 0.5)                          # [B, rep]  on sphere
    mu_sphere_e = mu_sphere.unsqueeze(1).expand(-1, N, -1)   # [B, N, rep]

    # Sample N candidates around the detached normalized mean so that:
    #   • diversity ∝ σ=1 relative to a fixed sphere radius — never collapses
    #   • rep_cand_pre carries NO gradient from mean (sampling is via detached
    #     centre + independent randn), removing the spurious radial gradient
    #     that caused ‖mean‖ to blow up in the pre-normalisation formulation.
    rep_cand_pre = mu_sphere_e.detach() + torch.randn_like(mu_sphere_e)  # [B, N, rep]
    rep_cand = rep_cand_pre / (rep_cand_pre.norm(dim=-1, keepdim=True) + 1e-8) \
               * (rep_dim ** 0.5)                            # [B, N, rep]  on sphere

    # Spherical AWR log_p: squared distance between fixed rep targets and the
    # live (gradient-carrying) normalised mean.  Gradient w.r.t. mean flows
    # ONLY through mu_sphere_e (tangential Jacobian) — no radial component.
    log_p_rep   = -0.5 * (rep_cand.detach() - mu_sphere_e).pow(2).sum(-1)  # [B, N]

    # ---- 2. Roll out N×M trajectories  (WM stepping uses raw 192D)
    BN  = B * N
    BNM = B * N * M
    z_anchor_rep = z_t.unsqueeze(1).expand(-1, N, -1).reshape(BN, dz)
    g_rep        = g_ult.unsqueeze(1).expand(-1, N, -1).reshape(BN, dz)
    rep_flat     = rep_cand.reshape(BN, rep_dim)

    z_anchor_NM = z_anchor_rep.unsqueeze(1).expand(-1, M, -1).reshape(BNM, dz)
    g_NM        = g_rep.unsqueeze(1).expand(-1, M, -1).reshape(BNM, dz)
    rep_NM      = rep_flat.unsqueeze(1).expand(-1, M, -1).reshape(BNM, rep_dim)

    with torch.no_grad():
        traj_z, traj_a_pre, _ = wgsp_rollout(
            wm_model, ll_actor, decoder,
            z_anchor_NM, g_NM, rep_NM,
            k=k, action_scale=action_scale, use_decoder=use_decoder,
            adapter=adapter)
        # traj_z [BNM, k+1, 192] raw WM space, traj_a_pre [BNM, k, adim]
        z_k = traj_z[:, -1, :]                                # [BNM, 192] raw
        # Use EMA target nets for J scoring so HL doesn't chase a moving live V.
        # value_target/goal_rep_target lag live by ~200-step half-life (τ=0.005);
        # this is the standard actor-critic pattern (the actor's value estimate
        # comes from the target net, e.g. DDPG/TD3/SAC).  Live nets fall back if
        # targets aren't passed (training-time scaffolding for legacy callers).
        v_for_J  = value_target if value_target is not None else value_net
        gr_for_J = goal_rep_target if goal_rep_target is not None else goal_rep
        score_out = _score_endpoints(
            z_k, g_NM, v_for_J, gr_for_J,
            beta_geom, use_geometric_term, use_v_in_J,
            lambda_mopo=lambda_mopo, return_diag=return_v_diag,
            adapter=adapter)
        if return_v_diag:
            J_flat, v_diag = score_out
        else:
            J_flat = score_out
            v_diag = None
        J = J_flat.reshape(B, N, M)

        # Per-rep score: mean over M.  Note: the cross-N baseline subtraction
        # below acts as the advantage; explicitly subtracting V(z_t) would be
        # mathematically a no-op since it's constant in (i, m) and cancels in
        # A_hl = J_per_rep - J_baseline.
        J_per_rep = J.mean(dim=2)                             # [B, N]
        J_baseline = J_per_rep.mean(dim=1, keepdim=True)
        A_hl = J_per_rep - J_baseline
        # Normalise to unit std so alpha_H acts as a fixed temperature regardless
        # of J scale. Without this, large geometric-term variance (J_std ~162)
        # makes alpha_H * A_hl std ~486 → near-argmax, and MOPO corrections are
        # too small to change rankings.
        A_hl = A_hl / (A_hl.std(dim=1, keepdim=True).clamp_min(1e-6))
        w_hl = torch.softmax(alpha_H * A_hl, dim=1)           # [B, N]

        # ============================================================
        # Per-component diagnostics — verify each piece of WGSP works
        # ============================================================
        if v_diag is None:
            v_diag = {}
        # --- HL actor: is it producing meaningful diverse candidates? ---
        # mu_norm: ‖μ‖ of the HL output. Starts ~0, should grow as HL learns.
        v_diag['mu_norm'] = mean.norm(dim=-1)                 # [B]
        # rep_cand_pair_dist: mean pairwise L2 distance among the N reps on
        # the sphere. Sphere of radius √rep_dim has max diameter 2·√rep_dim.
        # If reps are perfectly diverse the expected pairwise dist is √(2·rep_dim).
        # If reps collapse to a point this → 0.
        rc_a = rep_cand.unsqueeze(2)                          # [B, N, 1, rep]
        rc_b = rep_cand.unsqueeze(1)                          # [B, 1, N, rep]
        pair_dist = (rc_a - rc_b).norm(dim=-1)                # [B, N, N]
        # exclude diagonal: sum off-diagonal / (N*(N-1))
        pair_sum = pair_dist.sum(dim=(1, 2))                  # [B]
        v_diag['rep_cand_pair_dist'] = pair_sum / (N * (N - 1))  # [B]
        # mu_target_cos: cosine between HL mean μ and the on-data HIQL target
        # rep φ(z_t, z_target). After phase 1 should be ≈ 1 (HL converged to
        # HIQL HL target). Drops below 1 in phase 2 if WGSP has shifted μ
        # to a non-HIQL optimum — useful "is WGSP changing what HL proposes" signal.
        if z_target is not None:
            z_tg_pol_dg = adapt(z_target)
            target_rep_dg = goal_rep(z_t_pol, z_tg_pol_dg)    # [B, rep] on sphere
            mu_norm_safe   = mean.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            tgt_norm_safe  = target_rep_dg.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            cos = (mean / mu_norm_safe * target_rep_dg / tgt_norm_safe).sum(dim=-1)  # [B]
            v_diag['mu_target_cos'] = cos                     # [B]

        # --- LL + WM rollout: do candidates → distinguishable trajectories? ---
        # z_k_std_NM: spread of rollout endpoints (per-anchor, across N·M).
        v_diag['z_k_std_NM'] = z_k.reshape(B, N * M, dz).std(dim=1).mean(dim=-1)  # [B]
        # z_k_disp: mean L2 displacement of the rollout (anchor → endpoint).
        # Tells us if the rollout "goes somewhere" rather than staying put.
        v_diag['z_k_disp']   = (z_k - z_anchor_NM).norm(dim=-1).reshape(B, N * M).mean(dim=1)
        # closer_frac: fraction of rollouts whose endpoint is closer to g_ult
        # than the anchor was.  Should be > 0.5 if the LL+WM are actually goal-seeking.
        d_anchor = (z_anchor_NM - g_NM).norm(dim=-1)          # [BNM]
        d_endpt  = (z_k - g_NM).norm(dim=-1)                  # [BNM]
        v_diag['closer_frac'] = (d_endpt < d_anchor).float().reshape(B, N * M).mean(dim=1)

        # --- V scoring: does V_mean differentiate endpoints? ---
        # v_mean_range_N: (max - min) of v_mean across N candidates per anchor.
        # If V is flat at endpoints, this → 0 even if z_k differ.
        if v_diag.get('v_mean') is not None:
            vm = v_diag['v_mean'].reshape(B, N, M).mean(dim=2)  # [B, N]
            v_diag['v_mean_range_N'] = vm.max(dim=1).values - vm.min(dim=1).values  # [B]

        # --- WGSP signal sharpness ---
        # J_per_rep_std_N: cross-N variance of the WGSP score (the actual signal).
        v_diag['J_per_rep_std_N'] = J_per_rep.std(dim=1)      # [B]
        v_diag['J_per_rep_mean']  = J_per_rep.mean(dim=1)     # [B]
        # w_hl_max + w_hl_entropy: distinguish uniform-w from sharp-w.
        v_diag['w_hl_max']     = w_hl.max(dim=1).values       # [B]
        v_diag['w_hl_entropy'] = -(w_hl * (w_hl + 1e-12).log()).sum(dim=1)   # [B]; uniform=log N≈2.08, sharp→0

        # Per-rollout within-rep advantage (for LL distil)
        if ll_score_mode == 'rep_reach':
            # J_LL^(i,m) = -||φ(z_t, z_k^(i,m)) - rep^(i)||_2 (adapter-space φ)
            z_t_NM = z_t.unsqueeze(1).unsqueeze(1).expand(
                -1, N, M, -1).reshape(BNM, dz)
            phi_achieved = goal_rep(adapt(z_t_NM), adapt(z_k))  # [BNM, rep_dim]
            J_ll_flat = -torch.norm(
                phi_achieved - rep_NM, p=2, dim=-1)           # [BNM]
            J_ll = J_ll_flat.reshape(B, N, M)
        else:
            J_ll = J
        J_ll_rep_mean = J_ll.mean(dim=2, keepdim=True)        # [B, N, 1]
        A_ll = J_ll - J_ll_rep_mean                           # [B, N, M]
        A_ll = A_ll / (A_ll.std(dim=2, keepdim=True).clamp_min(1e-6))
        u_ll = torch.softmax(alpha_L * A_ll, dim=2)           # [B, N, M]

    # ---- 3. HL loss = -Σ w·log π^H(rep)
    if not ll_grad_only:
        hl_loss_wgsp = -(w_hl * log_p_rep).sum(dim=1).mean()
        hl_loss = hl_loss_wgsp

        if lambda_anchor > 0 and z_target is not None:
            with torch.no_grad():
                z_tg_pol = adapt(z_target)
                target_rep = goal_rep(z_t_pol, z_tg_pol)
            log_p_anchor = hl_actor.log_prob(z_t_pol, g_pol, target_rep)
            with torch.no_grad():
                rep_t  = goal_rep(z_t_pol,  g_pol)
                rep_tk = goal_rep(z_tg_pol, g_pol)
                v_t  = value_net(z_t_pol,  rep_t).mean(dim=-1)
                v_tk = value_net(z_tg_pol, rep_tk).mean(dim=-1)
                adv_anchor = v_tk - v_t
                w_anchor   = (alpha_H * adv_anchor).exp().clamp(max=100.0)
            hl_loss = hl_loss + lambda_anchor * (-(w_anchor * log_p_anchor).mean())
    else:
        hl_loss = None

    if return_rollouts:
        # Detach so the LL-distil consumer doesn't share the HL autograd graph.
        out = {
            'traj_z':      traj_z.detach(),
            'traj_a_pre':  traj_a_pre.detach(),
            'rep_NM':      rep_NM.detach(),
            'w_hl':        w_hl.detach(),
            'u_ll':        u_ll.detach(),
            'B': B, 'N': N, 'M': M, 'k': k,
        }
        if v_diag is not None:
            out['v_diag'] = {k: v.detach() for k, v in v_diag.items()}
        return hl_loss, out
    if return_v_diag and v_diag is not None:
        return hl_loss, {'v_diag': {k: v.detach()
                                     for k, v in v_diag.items()}}
    return hl_loss, None


def ll_distil_loss(rollouts, ll_actor, lambda_distil, adapter=None):
    """LL distillation loss using within-rep advantages from HL-WGSP.

    Returns the loss tensor (graph attached) scaled by lambda_distil, or None
    if the distil branch is inactive. Caller does the combined backward.
    """
    if rollouts is None or lambda_distil <= 0:
        return None

    adapt = (lambda x: adapter(x)) if adapter is not None else (lambda x: x)
    B, N, M, k = rollouts['B'], rollouts['N'], rollouts['M'], rollouts['k']
    BNM = B * N * M

    traj_z     = rollouts['traj_z']        # [BNM, k+1, 192] raw WM latents
    traj_a_pre = rollouts['traj_a_pre']    # [BNM, k, adim] — pre-tanh samples
    rep_NM     = rollouts['rep_NM']        # [BNM, rep_dim]
    w_hl       = rollouts['w_hl']          # [B, N]
    u_ll       = rollouts['u_ll']          # [B, N, M]

    z_steps  = traj_z[:, :-1, :]           # [BNM, k, 192] raw
    adim     = traj_a_pre.shape[-1]
    rep_dim  = rep_NM.shape[-1]

    z_flat   = adapt(z_steps.reshape(BNM * k, 192))   # adapted for LL
    rep_flat = rep_NM.unsqueeze(1).expand(-1, k, -1).reshape(BNM * k, rep_dim)
    a_flat   = traj_a_pre.reshape(BNM * k, adim)

    # Compute log-prob in pre-tanh sample coordinates (matches the rsample path)
    mean, log_std = ll_actor(z_flat, rep_flat)
    std = log_std.exp()
    dist = Normal(mean, std)
    log_p = dist.log_prob(a_flat).sum(dim=-1)                  # [BNM*k]

    # If LL is tanh-squashed, subtract the log|det Jacobian| of the tanh squash
    # so we recover log π^L(a_post). We follow the same convention as
    # GaussianActor.sample(): lp -= log(action_scale * (1 - y^2) + 1e-6)
    if ll_actor.tanh_squash:
        y = torch.tanh(a_flat)
        lp_jac = torch.log(ll_actor.action_scale * (1.0 - y.pow(2)) + 1e-6).sum(dim=-1)
        log_p = log_p - lp_jac

    log_p = log_p.reshape(BNM, k).sum(dim=-1)                  # [BNM] — sum over τ
    log_p = log_p.reshape(B, N, M)                             # [B, N, M]

    # Combined two-level weighting w^(i) · u^(i,m), then mean over batch.
    weight = (w_hl.unsqueeze(-1) * u_ll)                       # [B, N, M]
    loss = -(weight * log_p).sum(dim=(1, 2)).mean()

    return lambda_distil * loss


# =============================================================================
# 5. Standard HIQL LL/HL updates (used in real-data branches and as anchor)
# =============================================================================

def ll_hiql_loss(real_cache, ll_actor, value_net, goal_rep,
                 batch_size, k, alpha_L, use_5d, mode='awr',
                 adapter=None):
    """LL HIQL loss on real data. Returns the loss tensor for combined backward.

    Adapter applied uniformly to z, z_next, z_sub at the input boundary
    (mirrors baseline).  GoalRep, V, and LL actor all consume adapter-space
    inputs so the network sees the same representation everywhere.

    mode='awr' : advantage-weighted regression (standard HIQL LL).
    mode='bc'  : pure behavioural cloning — uniform weights, no V queries.
                 Tests the WGSP HL + BC LL variant (ablation of AWR in LL).
    """
    adapt = (lambda x: adapter(x)) if adapter is not None else (lambda x: x)
    z, a, _, z_next, z_sub = real_cache.sample_ll_batch(batch_size, k, use_5d=use_5d)

    za_t = adapt(z)
    za_n = adapt(z_next)
    za_s = adapt(z_sub)

    with torch.no_grad():
        if mode == 'awr':
            rep_curr_adv = goal_rep(za_t, za_s)
            rep_next_adv = goal_rep(za_n, za_s)
            v_curr = value_net(za_t, rep_curr_adv).mean(dim=-1)
            v_next = value_net(za_n, rep_next_adv).mean(dim=-1)
            adv    = v_next - v_curr
            w      = (alpha_L * adv).exp().clamp(max=100.0)
        else:
            w = torch.ones(z.shape[0], device=z.device)

    # rep_curr WITH gradients (low_actor_rep_grad=True ⇒ goal_rep is also
    # trained by the LL loss, matching ogbench/baseline).
    rep_curr = goal_rep(za_t, za_s)
    log_p   = ll_actor.log_prob(za_t, rep_curr, a)
    return -(w * log_p).mean()


def hl_hiql_loss(real_cache, hl_actor, value_net, goal_rep,
                 batch_size, k, alpha_H, adapter=None):
    """HL HIQL loss on real data. Returns the loss tensor for combined backward.

    Adapter applied uniformly to all 192D inputs at the boundary.  HL, V,
    and GoalRep all operate in adapter (policy) space (mirrors baseline).
    """
    adapt = (lambda x: adapter(x)) if adapter is not None else (lambda x: x)
    z_t, z_tk, g_ult = real_cache.sample_hl_batch(batch_size, k)

    za_t  = adapt(z_t)
    za_tk = adapt(z_tk)
    za_g  = adapt(g_ult)

    with torch.no_grad():
        rep_t   = goal_rep(za_t,  za_g)
        rep_tk  = goal_rep(za_tk, za_g)
        v_t  = value_net(za_t,  rep_t).mean(dim=-1)
        v_tk = value_net(za_tk, rep_tk).mean(dim=-1)
        adv  = v_tk - v_t
        w    = (alpha_H * adv).exp().clamp(max=100.0)
        target_rep = goal_rep(za_t, za_tk)

    log_p   = hl_actor.log_prob(za_t, za_g, target_rep)
    return -(w * log_p).mean()


# =============================================================================
# 6. Training Loop (with all WGSP toggles)
# =============================================================================

def train_loop(
    wm_model,
    hl_actor,
    ll_actor,
    value_net,
    value_target,
    goal_rep,        goal_rep_target,
    real_cache,
    decoder,
    optimizer,
    trainable_params,
    use_wgsp, use_distil, use_decoder, use_geometric_term, use_v_in_J,
    total_steps, warmup_fraction,
    subgoal_steps, k_plan,
    N, M,
    batch_size,
    gamma, tau, expectile,
    alpha_H, alpha_L,
    alpha_H_wgsp, alpha_L_distil,
    beta_geom, lambda_anchor, lambda_distil_max,
    action_scale, rep_dim,
    save_dir, device,
    log_interval=100, save_interval=1000,
    lambda_mopo=0.0, audit_v_disagreement=True,
    ll_score_mode='goal',
    ll_mode='awr',
    adapter=None,
    hl_wgsp_coeff=10.0,
    lambda_hiql_reg=0.1,
):
    os.makedirs(save_dir, exist_ok=True)
    warmup_steps = int(total_steps * warmup_fraction)
    use_5d = use_decoder

    print(f"\n{'='*65}", flush=True)
    print(f"  WGSP training", flush=True)
    print(f"  use_wgsp           : {use_wgsp}", flush=True)
    print(f"  use_distil         : {use_distil}", flush=True)
    print(f"  use_decoder (LL 5D): {use_decoder}", flush=True)
    print(f"  use_geometric_term : {use_geometric_term}  (β={beta_geom})", flush=True)
    print(f"  use_v_in_J         : {use_v_in_J}", flush=True)
    print(f"  N candidates       : {N}", flush=True)
    print(f"  M rollouts         : {M}", flush=True)
    print(f"  k_plan             : {k_plan}", flush=True)
    print(f"  subgoal_steps      : {subgoal_steps}", flush=True)
    print(f"  warmup_steps       : {warmup_steps:,}  (wf={warmup_fraction})", flush=True)
    print(f"  total_steps        : {total_steps:,}", flush=True)
    print(f"  λ_anchor           : {lambda_anchor}", flush=True)
    print(f"  λ_distil (max)     : {lambda_distil_max}", flush=True)
    print(f"  λ_mopo             : {lambda_mopo}", flush=True)
    print(f"  hl_wgsp_coeff      : {hl_wgsp_coeff}", flush=True)
    print(f"  λ_hiql_reg         : {lambda_hiql_reg}  (phase-2 HIQL scale; 0.05 ≈ 50/50 gradient split with WGSP)", flush=True)
    print(f"  V heads            : {value_net.n_heads}", flush=True)
    print(f"  ll_score_mode      : {ll_score_mode}", flush=True)
    print(f"  ll_mode (LL step)  : {ll_mode}  (bc=pure BC, no advantage)", flush=True)
    if adapter is not None:
        in_d  = adapter.mlp[0].in_features
        out_d = next(l.out_features for l in reversed(list(adapter.mlp)) if hasattr(l, 'out_features'))
        n_p   = sum(p.numel() for p in adapter.parameters())
        print(f"  adapter (uniform)  : {in_d}D → {out_d}D  ({n_p:,} params)", flush=True)
    else:
        print(f"  adapter            : disabled (raw 192D)", flush=True)
    print(f"{'='*65}\n", flush=True)

    csv_path = os.path.join(save_dir, 'training_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow([
            'step', 'phase', 'value_loss',
            'll_hiql_loss', 'll_distil_loss',
            'hl_hiql_loss', 'hl_wgsp_loss',
            'lambda_distil', 'elapsed_s',
            # V-head disagreement
            'v_std_endpoint', 'v_std_data',
            # HL actor candidate quality
            'mu_norm', 'mu_target_cos', 'rep_cand_pair_dist',
            # LL+WM rollout health
            'z_k_std_NM', 'z_k_disp', 'closer_frac',
            # V scoring informativeness
            'v_mean_range_N', 'J_std_N',
            # WGSP signal sharpness
            'w_hl_max', 'w_hl_entropy',
        ])

    t0 = time.time()
    t_int = time.time()
    sum_v = sum_ll_hiql = sum_ll_distil = 0.0
    sum_hl_hiql = sum_hl_wgsp = 0.0
    sum_v_std_ep = sum_v_std_dat = 0.0
    # WGSP per-component diagnostics
    sum_mu_norm = sum_rep_pair = sum_mu_tgt_cos = 0.0
    sum_z_k_disp = sum_closer = sum_z_k_std = 0.0
    sum_v_mean_range = 0.0
    sum_J_std_N = sum_w_hl_max = sum_w_hl_ent = 0.0
    audit_log_n = 0
    diag_log_n = 0
    log_n = 0
    prev_phase = 1

    for step in range(total_steps):
        phase = 1 if step < warmup_steps else 2
        if phase != prev_phase:
            print(f"\n>>> Phase 2 begins at step {step:,} — WGSP active\n", flush=True)
            prev_phase = phase

        # ---- Value loss (sparse HER only — matches train_hiql_baseline) ----
        # We use mixed cur/traj/random goal sampling and r = success - 1, mask =
        # 1 - success.  The previous dense reward r = -‖z_next - g‖ was unsound
        # because L2 in JEPA latent space is a weak goal signal (validation
        # showed Spearman ρ(t, dist) = -0.193).  Sparse HER + masking gives
        # V(z, g) ≈ -E[time-to-g], bounded in [-1/(1-γ), 0], goal-directional.
        # The previous "v_loss_main" (which used z_sub = z[t+k] as a fixed
        # k-step goal) is dropped: with sparse rewards z_next is never z_sub
        # (k>1), so success is always 0 and mask is always 1, making the
        # bootstrap a uniform decay that loses goal information.
        z_hl, z_next_hl, g_her, success_hl = real_cache.sample_value_her_batch(
            batch_size)
        r_hl    = success_hl - 1.0      # 0 at goal, -1 elsewhere
        mask_hl = 1.0 - success_hl      # 0 at goal (no bootstrap), 1 elsewhere
        v_loss = _value_loss(
            value_net, value_target, goal_rep, goal_rep_target,
            z_hl, z_next_hl, g_her, r_hl, gamma, expectile, adapter=adapter,
            mask=mask_hl)
        sum_v += v_loss.item()

        # ---- LL HIQL (always on; the on-data anchor)
        ll_loss = ll_hiql_loss(real_cache, ll_actor, value_net, goal_rep,
                               batch_size, subgoal_steps, alpha_L,
                               use_5d=use_5d, mode=ll_mode, adapter=adapter)
        sum_ll_hiql += ll_loss.item()

        # ---- HL: WGSP if enabled and Phase 2; else HIQL HL
        hl_loss = None
        hl_hiql_reg = None   # HIQL HL regulariser, active in WGSP phase 2
        ll_distil = None

        if use_wgsp and phase == 2:
            z_t, z_target, g_ult = real_cache.sample_hl_batch(batch_size, subgoal_steps)
            # Pass z_target through unconditionally so the diagnostic
            # mu_target_cos can be computed even when lambda_anchor=0.
            # The anchor-loss branch inside hl_wgsp_step gates on lambda_anchor.
            hl_loss, rollouts = hl_wgsp_step(
                z_t=z_t, g_ult=g_ult,
                hl_actor=hl_actor, ll_actor=ll_actor, decoder=decoder,
                wm_model=wm_model, goal_rep=goal_rep, value_net=value_net,
                N=N, M=M, k=k_plan,
                beta_geom=beta_geom,
                alpha_H=alpha_H_wgsp, alpha_L=alpha_L_distil,
                action_scale=action_scale,
                use_decoder=use_decoder, use_geometric_term=use_geometric_term,
                use_v_in_J=use_v_in_J,
                lambda_anchor=lambda_anchor, z_target=z_target,
                return_rollouts=use_distil,
                lambda_mopo=lambda_mopo,
                return_v_diag=audit_v_disagreement,
                ll_score_mode=ll_score_mode,
                adapter=adapter,
                value_target=value_target, goal_rep_target=goal_rep_target,
            )
            sum_hl_wgsp += hl_loss.item()

            # HIQL HL regulariser: keeps ‖mean‖ near on-data subgoal radius.
            # Phase 1 HIQL was the sole restoring force on the actor mean; without
            # it Adam's accumulated momentum drives ‖mean‖ to diverge immediately.
            hl_hiql_reg = hl_hiql_loss(real_cache, hl_actor, value_net, goal_rep,
                                        batch_size, subgoal_steps, alpha_H,
                                        adapter=adapter)
            sum_hl_hiql += hl_hiql_reg.item()

            if audit_v_disagreement and rollouts is not None:
                vd = rollouts.get('v_diag')
                if vd is not None and vd.get('v_std') is not None:
                    sum_v_std_ep += vd['v_std'].mean().item()
                    # Compare against disagreement on a dataset batch (adapter space).
                    # Use target nets here to match _score_endpoints (apples-to-apples
                    # endpoint vs data std comparison).
                    with torch.no_grad():
                        z_dat, _, _, _, z_sub_dat = real_cache.sample_ll_batch(
                            batch_size, subgoal_steps, use_5d=False)
                        _adapt = (lambda x: adapter(x)) if adapter is not None else (lambda x: x)
                        rep_dat = goal_rep_target(_adapt(z_dat), _adapt(z_sub_dat))
                        v_dat_all = value_target(_adapt(z_dat), rep_dat)
                        v_std_dat = (v_dat_all.std(dim=-1, unbiased=True)
                                     if v_dat_all.shape[-1] > 1
                                     else torch.zeros(batch_size, device=device))
                    sum_v_std_dat += v_std_dat.mean().item()
                    audit_log_n += 1
                # Per-component WGSP diagnostics
                if vd is not None and vd.get('J_per_rep_std_N') is not None:
                    # HL actor candidate quality
                    sum_mu_norm   += vd['mu_norm'].mean().item()
                    sum_rep_pair  += vd['rep_cand_pair_dist'].mean().item()
                    if vd.get('mu_target_cos') is not None:
                        sum_mu_tgt_cos += vd['mu_target_cos'].mean().item()
                    # LL+WM rollout health
                    sum_z_k_std   += vd['z_k_std_NM'].mean().item()
                    sum_z_k_disp  += vd['z_k_disp'].mean().item()
                    sum_closer    += vd['closer_frac'].mean().item()
                    # V scoring informativeness
                    if vd.get('v_mean_range_N') is not None:
                        sum_v_mean_range += vd['v_mean_range_N'].mean().item()
                    # WGSP signal sharpness
                    sum_J_std_N   += vd['J_per_rep_std_N'].mean().item()
                    sum_w_hl_max  += vd['w_hl_max'].mean().item()
                    sum_w_hl_ent  += vd['w_hl_entropy'].mean().item()
                    diag_log_n    += 1

            # ---- LL distil (only when WGSP active and toggle on)
            if use_distil:
                # Linear ramp of λ_distil over Phase 2
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                lambda_distil_now = lambda_distil_max * min(1.0, progress * 2)
                ll_distil = ll_distil_loss(rollouts, ll_actor,
                                           lambda_distil_now, adapter=adapter)
                if ll_distil is not None:
                    sum_ll_distil += ll_distil.item()
        else:
            hl_loss = hl_hiql_loss(real_cache, hl_actor, value_net, goal_rep,
                                   batch_size, subgoal_steps, alpha_H,
                                   adapter=adapter)
            sum_hl_hiql += hl_loss.item()

        # ---- Combined backward + single optimizer step (mirrors baseline) ----
        total_loss = v_loss + ll_loss
        if hl_loss is not None:
            # In WGSP phase 2 the raw hl_wgsp_loss (~14) is ~40x smaller than
            # V+LL (~610), so clip_grad_norm(1.0) effectively zeros the HL
            # actor's gradient.  Scale it up to roughly match HL_h magnitude
            # from phase 1 so the HL actor gets a comparable update.
            # w_hl is inside torch.no_grad(), so scaling only affects HL actor
            # (and adapter inputs to it) — V and LL gradients are unchanged.
            if use_wgsp and phase == 2:
                total_loss = total_loss + hl_wgsp_coeff * hl_loss
            else:
                total_loss = total_loss + hl_loss
        if hl_hiql_reg is not None:
            total_loss = total_loss + lambda_hiql_reg * hl_hiql_reg
        if ll_distil is not None:
            total_loss = total_loss + ll_distil

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        # Soft-update target nets (after the parameter update so targets EMA the new weights)
        with torch.no_grad():
            for tp, p in zip(value_target.parameters(), value_net.parameters()):
                tp.data.mul_(1.0 - tau).add_(p.data * tau)
            for tp, p in zip(goal_rep_target.parameters(), goal_rep.parameters()):
                tp.data.mul_(1.0 - tau).add_(p.data * tau)

        log_n += 1

        # ---- Logging
        if step % log_interval == 0:
            d = max(log_n, 1)
            now = time.time()
            sps = log_interval / max(now - t_int, 1e-6)
            remaining = (total_steps - step) / max(sps, 1e-6)
            pct = 100.0 * step / total_steps
            elapsed = now - t0
            ld_now = (lambda_distil_max *
                      min(1.0, max(0.0, (step - warmup_steps) /
                                       max(total_steps - warmup_steps, 1)) * 2)
                      if use_distil and use_wgsp else 0.0)
            ad = max(audit_log_n, 1)
            dd = max(diag_log_n, 1)
            v_std_ep_avg   = sum_v_std_ep   / ad
            v_std_dat_avg  = sum_v_std_dat  / ad
            v_ratio        = v_std_ep_avg / max(v_std_dat_avg, 1e-8)
            mu_norm_avg    = sum_mu_norm    / dd
            rep_pair_avg   = sum_rep_pair   / dd
            mu_tgt_cos_avg = sum_mu_tgt_cos / dd
            z_k_std_avg    = sum_z_k_std    / dd
            z_k_disp_avg   = sum_z_k_disp   / dd
            closer_avg     = sum_closer     / dd
            v_range_avg    = sum_v_mean_range / dd
            J_std_avg      = sum_J_std_N    / dd
            w_max_avg      = sum_w_hl_max   / dd
            w_ent_avg      = sum_w_hl_ent   / dd
            print(
                f"Step {step:06d}/{total_steps:,} ({pct:4.1f}%) | Ph{phase} | "
                f"V {sum_v/(2*d):.4f} | "
                f"LL_h {sum_ll_hiql/d:.4f} LL_d {sum_ll_distil/d:.4f} | "
                f"HL_h {sum_hl_hiql/d:.4f} HL_w {sum_hl_wgsp/d:.4f} | "
                f"λd {ld_now:.3f} | "
                f"Vstd ep/dat {v_std_ep_avg:.3f}/{v_std_dat_avg:.3f}({v_ratio:.1f}x) | "
                f"μ‖{mu_norm_avg:.2f} μ·tgt {mu_tgt_cos_avg:+.2f} repΔ {rep_pair_avg:.2f} | "
                f"zσ_NM {z_k_std_avg:.2f} zΔ {z_k_disp_avg:.2f} closer {closer_avg:.2f} | "
                f"Vrng_N {v_range_avg:.2f} Jσ_N {J_std_avg:.3f} | "
                f"wmax {w_max_avg:.3f} wH {w_ent_avg:.3f} | "
                f"{sps:.1f} sps | ETA {remaining/60:.0f}m | el {elapsed/60:.0f}m",
                flush=True,
            )
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    step, phase, sum_v/(2*d),
                    sum_ll_hiql/d, sum_ll_distil/d,
                    sum_hl_hiql/d, sum_hl_wgsp/d,
                    ld_now, now - t_int,
                    v_std_ep_avg, v_std_dat_avg,
                    mu_norm_avg, mu_tgt_cos_avg, rep_pair_avg,
                    z_k_std_avg, z_k_disp_avg, closer_avg,
                    v_range_avg, J_std_avg,
                    w_max_avg, w_ent_avg,
                ])
            sum_v = sum_ll_hiql = sum_ll_distil = 0.0
            sum_hl_hiql = sum_hl_wgsp = 0.0
            sum_v_std_ep = sum_v_std_dat = 0.0
            sum_mu_norm = sum_rep_pair = sum_mu_tgt_cos = 0.0
            sum_z_k_disp = sum_closer = sum_z_k_std = 0.0
            sum_v_mean_range = 0.0
            sum_J_std_N = sum_w_hl_max = sum_w_hl_ent = 0.0
            audit_log_n = 0
            diag_log_n = 0
            log_n = 0
            t_int = now

        # ---- Checkpoint
        if step > 0 and (step % save_interval == 0 or step == total_steps - 1):
            torch.save(ll_actor.state_dict(),  os.path.join(save_dir, 'll_actor.pth'))
            torch.save(hl_actor.state_dict(),  os.path.join(save_dir, 'hl_actor.pth'))
            torch.save(value_net.state_dict(), os.path.join(save_dir, 'value_net.pth'))
            torch.save(goal_rep.state_dict(),  os.path.join(save_dir, 'goal_rep.pth'))
            if adapter is not None:
                torch.save(adapter.state_dict(), os.path.join(save_dir, 'adapter.pth'))
            import json as _json
            with open(os.path.join(save_dir, 'metadata.json'), 'w') as _mf:
                _json.dump({'encode_mode': 'cls_projected'}, _mf)
            print(f"  → checkpoint saved at step {step}", flush=True)


# =============================================================================
# 7. Inference helper (LL outputs 5-D when use_decoder=True)
# =============================================================================

def select_action(wm_model, hl_actor, ll_actor, goal_rep,
                  obs_pixels, g_pixels, subgoal_steps,
                  step_counter, current_subgoal_rep,
                  device, img_transform, rep_dim=10, use_decoder=False):
    with torch.no_grad():
        obs_f = img_transform(obs_pixels.to(device)).unsqueeze(0)
        g_f   = img_transform(g_pixels.to(device)).unsqueeze(0)
        z_curr = wm_model.encode({'pixels': obs_f})['emb'].squeeze(0).squeeze(0)
        g_ult  = wm_model.encode({'pixels': g_f})['emb'].squeeze(0).squeeze(0)

        if current_subgoal_rep is None or step_counter % subgoal_steps == 0:
            _, _, new_rep = hl_actor.sample(
                z_curr.unsqueeze(0), g_ult.unsqueeze(0))
            new_rep = new_rep.squeeze(0)
            new_rep = new_rep / (new_rep.norm() + 1e-8) * (rep_dim ** 0.5)
        else:
            new_rep = current_subgoal_rep

        _, _, action = ll_actor.sample(
            z_curr.unsqueeze(0), new_rep.unsqueeze(0))
        action = action.squeeze(0)

    return action, new_rep, z_curr


# =============================================================================
# 8. JEPA loader (verbatim from train_hiql_lewm.py)
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
    predictor = ARPredictor(num_frames=3, input_dim=192, hidden_dim=192,
                            output_dim=192, depth=6, heads=16, mlp_dim=2048,
                            dim_head=64, dropout=0.1, emb_dropout=0.0)
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
                  for k, v in ckpt['state_dict'].items()
                  if k.startswith('model.')}
    else:
        raw_sd = dict(ckpt)
    model.load_state_dict(raw_sd, strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# =============================================================================
# 9. Main
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WGSP training')

    parser.add_argument('--ckpt_path',    type=str, default=None)
    parser.add_argument('--cache_path',   type=str, default=None)
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--save_dir',     type=str, default=None)
    parser.add_argument('--decoder_ckpt', type=str, default=None,
                        help='Path to action_decoder.pth or flow_decoder.pth '
                             '(required if --use_decoder).')
    parser.add_argument('--decoder_type', type=str, default='mse',
                        choices=['mse', 'flow'],
                        help='"mse" → ActionChunkDecoder (row 1); '
                             '"flow" → FlowDecoderWrapper (row 13).')

    parser.add_argument('--total_steps',     type=int,   default=200_000)
    parser.add_argument('--warmup_fraction', type=float, default=0.2)
    parser.add_argument('--subgoal_steps',   type=int,   default=8)
    parser.add_argument('--k_plan',          type=int,   default=None,
                        help='WGSP rollout horizon (defaults to subgoal_steps).')
    parser.add_argument('--N',               type=int,   default=8,
                        help='Number of candidate reps per anchor.')
    parser.add_argument('--M',               type=int,   default=4,
                        help='Stochastic rollouts per (anchor, rep).')
    parser.add_argument('--batch_size',      type=int,   default=256)

    parser.add_argument('--alpha_H',         type=float, default=3.0)
    parser.add_argument('--alpha_L',         type=float, default=3.0)
    parser.add_argument('--alpha_H_wgsp',    type=float, default=3.0)
    parser.add_argument('--alpha_L_distil',  type=float, default=3.0)
    parser.add_argument('--beta_geom',       type=float, default=0.1)
    parser.add_argument('--lambda_anchor',   type=float, default=0.0)
    parser.add_argument('--lambda_distil',   type=float, default=1.0,
                        help='Max λ_distil after warmup ramp.')

    parser.add_argument('--gamma',         type=float, default=0.99)
    parser.add_argument('--expectile',     type=float, default=0.7)
    parser.add_argument('--action_scale',  type=float, default=3.0)
    parser.add_argument('--rep_dim',       type=int,   default=10)

    # ---- Master toggles for the ablation matrix ----
    def _bool(s):
        return str(s).lower() in ('1', 'true', 'yes', 'y', 't')
    parser.add_argument('--use_wgsp',           type=_bool, default=True)
    parser.add_argument('--use_distil',         type=_bool, default=True)
    parser.add_argument('--use_decoder',        type=_bool, default=True)
    parser.add_argument('--use_geometric_term', type=_bool, default=True)
    parser.add_argument('--use_v_in_J',         type=_bool, default=True,
                        help='Set False for the geometric-only WGSP ablation.')

    parser.add_argument('--img_size',   type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=14)
    parser.add_argument('--seed',       type=int, default=0)

    # ---- Row-11 / V-audit flags ----
    parser.add_argument('--n_value_heads', type=int, default=2,
                        help='Number of V heads in EnsembleValue (2 = legacy TwinValue).')
    parser.add_argument('--use_mopo',     type=_bool, default=False,
                        help='Subtract λ_mopo·V_std from WGSP endpoint scores (row 11).')
    parser.add_argument('--lambda_mopo',  type=float, default=1.0,
                        help='Scale for MOPO-style V-disagreement penalty.')
    parser.add_argument('--ll_score_mode', type=str, default='goal',
                        choices=['goal', 'rep_reach'],
                        help='LL within-rep advantage source. "goal" uses '
                             'J^(i,m) (default, headline). "rep_reach" uses '
                             '-||φ(z_t, z_k) - rep||_2 (row 14, decouples '
                             'LL credit from ultimate goal).')
    parser.add_argument('--ll_mode', type=str, default='awr',
                        choices=['awr', 'bc'],
                        help='LL real-data update. "awr" (default) uses IQL '
                             'advantage-weighted regression. "bc" uses pure '
                             'behavioural cloning (no V queries, uniform weights). '
                             'Tests WGSP HL + BC LL vs WGSP HL + AWR LL.')

    # ---- Latent adapter (uniform front-end for V/GoalRep/HL/LL; mirrors baseline) ----
    parser.add_argument('--use_adapter', type=_bool, default=True,
                        help='Add a trainable MLP adapter on top of frozen 192D '
                             'LeWM latents (default: True). Applied uniformly to '
                             'all observations and goals — V, GoalRep, HL, LL all '
                             'operate in the resulting policy space. Set False '
                             'for the raw-latent ablation.')
    parser.add_argument('--adapter_dim',        type=int,   default=256,
                        help='Output dim of the latent adapter (default 256).')
    parser.add_argument('--adapter_hidden_dim', type=int,   default=None,
                        help='Hidden dim of adapter MLP (default = adapter_dim).')
    parser.add_argument('--adapter_depth',      type=int,   default=2,
                        help='Number of linear layers in adapter (default 2).')
    parser.add_argument('--hl_wgsp_coeff',      type=float, default=10.0,
                        help='Scale factor applied to hl_wgsp_loss before the combined '
                             'backward.  Compensates for the scale gap between '
                             'hl_wgsp (~10) and V+LL (~610).')
    parser.add_argument('--lambda_hiql_reg',    type=float, default=0.1,
                        help='Scale factor for the HIQL HL regulariser that runs '
                             'alongside WGSP in phase 2.  1.0 = full phase-1 strength '
                             '(~680, 96%% of actor gradient, actor frozen). '
                             '0.1 gives ~30%% WGSP gradient share while keeping '
                             'HIQL 2.3x stronger than WGSP — enough restoring force '
                             'to prevent mu-norm drift without freezing the actor.')

    args = parser.parse_args()

    if args.k_plan is None:
        args.k_plan = args.subgoal_steps

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    STABLEWM_HOME = os.environ.get(
        'STABLEWM_HOME', os.path.join(os.path.expanduser('~'), 'stable_wm_data'))
    data_path  = args.dataset_path or os.path.join(STABLEWM_HOME, 'ogbench', 'cube_single_expert')
    _default_ckpt = ('lejepa' if args.img_size == 224 else 'lewm_ogbench_weights.ckpt')
    ckpt_path  = args.ckpt_path or os.path.join(STABLEWM_HOME, 'cube', _default_ckpt)
    cache_path = args.cache_path or os.path.join(STABLEWM_HOME, 'lewm_224_latents_cache.pt')
    _mopo_tag    = f'_mopo{args.lambda_mopo}' if args.use_mopo else ''
    _heads_tag   = f'_h{args.n_value_heads}' if args.n_value_heads != 2 else ''
    _llmode_tag  = f'_ll{args.ll_score_mode}' if args.ll_score_mode != 'goal' else ''
    _llupd_tag   = '_llbc' if args.ll_mode == 'bc' else ''
    _adapter_tag = f'_adapter{args.adapter_dim}' if args.use_adapter else '_noadapter'
    save_dir   = args.save_dir or (
        f'./checkpoints_hiql_wgsp_k{args.subgoal_steps}_N{args.N}_M{args.M}'
        f'_b{args.beta_geom}_la{args.lambda_anchor}_ld{args.lambda_distil}'
        f'_dec{int(args.use_decoder)}_geo{int(args.use_geometric_term)}'
        f'_v{int(args.use_v_in_J)}_wgsp{int(args.use_wgsp)}'
        f'_dis{int(args.use_distil)}_rep{args.rep_dim}'
        f'{_heads_tag}{_mopo_tag}{_llmode_tag}{_llupd_tag}{_adapter_tag}_s{args.seed}')

    parent_dir = os.path.abspath(os.path.dirname(__file__))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    if args.ckpt_path is not None:
        wm_model = _load_jepa_from_ckpt(ckpt_path, device, args.img_size, args.patch_size)
    else:
        with initialize(version_base=None, config_path='../config'):
            cfg = compose(config_name='eval/cube', overrides=['+policy=cube/lejepa'])
        wm_model = swm.policy.AutoCostModel(cfg.policy).to(device).eval()
    for p in wm_model.parameters():
        p.requires_grad = False

    print(f'Loading cache from {cache_path} ...')
    cache_data  = torch.load(cache_path, map_location='cpu')
    all_latents = cache_data['all_latents']
    all_actions = cache_data.get('all_actions', [])
    if not all_actions:
        raise RuntimeError("Cache has no 'all_actions'.")

    from sklearn import preprocessing as sk_pre
    print('Fitting StandardScaler on dataset actions ...')
    _dp_stem = data_path.rstrip('/')
    if _dp_stem.endswith('.h5'):
        _dp_stem = _dp_stem[:-3]
    _ds_for_scaler = swm.data.HDF5Dataset(
        _dp_stem, keys_to_cache=['action'],
        cache_dir=os.path.dirname(_dp_stem))
    action_raw = _ds_for_scaler.get_col_data('action')
    action_raw = action_raw[~np.isnan(action_raw).any(axis=1)]
    action_scaler = sk_pre.StandardScaler()
    action_scaler.fit(action_raw)
    print(f'  Scaler fit on {len(action_raw):,} action frames')

    real_cache = RealOfflineCache(
        all_latents=all_latents, all_actions=all_actions,
        action_scaler=action_scaler, device=device, frameskip=5,
    )

    LATENT_DIM  = 192                  # frozen WM / cache / geometric term
    REP_DIM     = args.rep_dim
    HIDDEN_DIMS = (512, 512, 512)
    LL_OUT_DIM  = 5 if args.use_decoder else 25

    # Trainable adapter: maps LATENT_DIM → POLICY_DIM uniformly for ALL networks
    # (V, GoalRep, HL, LL).  Mirrors train_hiql_baseline: adapter is a single
    # front-end transformation applied at the input boundary; everything
    # downstream operates in policy space.  WM rollouts and the geometric
    # residual still use raw LATENT_DIM (the WM is frozen 192D).
    adapter = None
    POLICY_DIM = LATENT_DIM
    if args.use_adapter:
        adapter = LatentAdapter(
            in_dim=LATENT_DIM, out_dim=args.adapter_dim,
            hidden_dim=args.adapter_hidden_dim, depth=args.adapter_depth,
        ).to(device)
        POLICY_DIM = args.adapter_dim
        n_adapter = sum(p.numel() for p in adapter.parameters())
        print(f'LatentAdapter: {LATENT_DIM}D → {POLICY_DIM}D  ({n_adapter:,} params, applied uniformly)')

    goal_rep = GoalRep(latent_dim=POLICY_DIM, rep_dim=REP_DIM,
                       hidden_dims=HIDDEN_DIMS, layer_norm=True).to(device)
    goal_rep_target = GoalRep(latent_dim=POLICY_DIM, rep_dim=REP_DIM,
                              hidden_dims=HIDDEN_DIMS, layer_norm=True).to(device)
    goal_rep_target.load_state_dict(goal_rep.state_dict())
    for p in goal_rep_target.parameters():
        p.requires_grad = False

    ll_actor = GaussianActor(
        state_dim=POLICY_DIM, goal_dim=REP_DIM, output_dim=LL_OUT_DIM,
        hidden_dims=HIDDEN_DIMS, tanh_squash=True, action_scale=args.action_scale,
    ).to(device)

    hl_actor = GaussianActor(
        state_dim=POLICY_DIM, goal_dim=POLICY_DIM, output_dim=REP_DIM,
        hidden_dims=HIDDEN_DIMS, tanh_squash=False, const_std=True,
    ).to(device)

    value_net = EnsembleValue(latent_dim=POLICY_DIM, rep_dim=REP_DIM,
                              hidden_dims=HIDDEN_DIMS,
                              n_heads=args.n_value_heads).to(device)
    value_target = EnsembleValue(latent_dim=POLICY_DIM, rep_dim=REP_DIM,
                                 hidden_dims=HIDDEN_DIMS,
                                 n_heads=args.n_value_heads).to(device)
    value_target.load_state_dict(value_net.state_dict())
    for p in value_target.parameters():
        p.requires_grad = False

    # Decoder (frozen). Only required if use_decoder=True (since the WM
    # consumes 25-D actions, factored through the decoder there too).
    # --decoder_type mse  : ActionChunkDecoder (MSE, ~50k params)
    # --decoder_type flow : FlowDecoderWrapper (flow-matching, row 13)
    decoder = None
    if args.use_decoder:
        if args.decoder_ckpt is None:
            raise ValueError("--decoder_ckpt is required when --use_decoder=True. "
                             "Train one with train_action_decoder.py "
                             "(MSE) or train_flow_action_decoder.py (flow) first.")
        if args.decoder_type == 'flow':
            from train_flow_action_decoder import FlowChunkDecoder
            meta_path = os.path.join(
                os.path.dirname(args.decoder_ckpt), 'flow_decoder_meta.pt')
            flow_steps = 10
            if os.path.exists(meta_path):
                meta = torch.load(meta_path, map_location='cpu')
                flow_steps = meta.get('flow_steps', 10)
            _flow = FlowChunkDecoder(
                latent_dim=LATENT_DIM, a_first_dim=5, action_dim=25,
                hidden_dims=(512, 512, 512, 512),
            ).to(device)
            _flow.load_state_dict(torch.load(args.decoder_ckpt, map_location=device))
            _flow.eval()
            for p in _flow.parameters():
                p.requires_grad = False
            decoder = FlowDecoderWrapper(_flow, flow_steps=flow_steps).to(device)
            print(f"  Loaded frozen FlowChunkDecoder from {args.decoder_ckpt} "
                  f"(flow_steps={flow_steps})")
        else:
            # Auto-detect goal_dim from first-layer input width:
            #   old (state-only): net.0.weight [256, 197]  → goal_dim = 0
            #   new (goal-cond):  net.0.weight [256, 389]  → goal_dim = 192
            _dec_sd = torch.load(args.decoder_ckpt, map_location='cpu')
            _goal_dim = max(0, _dec_sd['net.0.weight'].shape[1] - 5 - LATENT_DIM)
            decoder = ActionChunkDecoder(in_dim=5, out_dim=25,
                                         latent_dim=LATENT_DIM,
                                         hidden_dims=(256, 256),
                                         goal_dim=_goal_dim).to(device)
            decoder.load_state_dict(_dec_sd)
            decoder.eval()
            for p in decoder.parameters():
                p.requires_grad = False
            print(f"  Loaded frozen ActionChunkDecoder from {args.decoder_ckpt} "
                  f"(goal_dim={_goal_dim})")

    print(
        f'Networks (LATENT_DIM={LATENT_DIM}, POLICY_DIM={POLICY_DIM}, '
        f'REP_DIM={REP_DIM}, LL_OUT_DIM={LL_OUT_DIM}):\n'
        f'  GoalRep φ:  {sum(p.numel() for p in goal_rep.parameters()):>10,} params\n'
        f'  LL Actor:   {sum(p.numel() for p in ll_actor.parameters()):>10,} params\n'
        f'  HL Actor:   {sum(p.numel() for p in hl_actor.parameters()):>10,} params\n'
        f'  Value:      {sum(p.numel() for p in value_net.parameters()):>10,} params'
    )

    # Single combined optimizer over all trainable params (mirrors the
    # ogbench/baseline pattern: one Adam state, one combined backward, one step).
    # Previously goal_rep was duplicated across ll_/value_optimizer with two
    # independent Adam moment buffers, and adapter only saw the value loss
    # (LL/HL gradients were accumulated then wiped without ever being applied).
    adapter_params  = list(adapter.parameters()) if adapter is not None else []
    trainable_params = (
        adapter_params
        + list(goal_rep.parameters())
        + list(value_net.parameters())
        + list(ll_actor.parameters())
        + list(hl_actor.parameters())
    )
    optimizer = torch.optim.Adam(trainable_params, lr=3e-4)

    train_loop(
        wm_model=wm_model,
        hl_actor=hl_actor,
        ll_actor=ll_actor,
        value_net=value_net,
        value_target=value_target,
        goal_rep=goal_rep, goal_rep_target=goal_rep_target,
        real_cache=real_cache, decoder=decoder,
        optimizer=optimizer,
        trainable_params=trainable_params,
        use_wgsp=args.use_wgsp, use_distil=args.use_distil,
        use_decoder=args.use_decoder,
        use_geometric_term=args.use_geometric_term,
        use_v_in_J=args.use_v_in_J,
        total_steps=args.total_steps, warmup_fraction=args.warmup_fraction,
        subgoal_steps=args.subgoal_steps, k_plan=args.k_plan,
        N=args.N, M=args.M, batch_size=args.batch_size,
        gamma=args.gamma, tau=0.005, expectile=args.expectile,
        alpha_H=args.alpha_H, alpha_L=args.alpha_L,
        alpha_H_wgsp=args.alpha_H_wgsp, alpha_L_distil=args.alpha_L_distil,
        beta_geom=args.beta_geom, lambda_anchor=args.lambda_anchor,
        lambda_distil_max=args.lambda_distil,
        action_scale=args.action_scale, rep_dim=REP_DIM,
        save_dir=save_dir, device=device,
        lambda_mopo=args.lambda_mopo if args.use_mopo else 0.0,
        audit_v_disagreement=True,
        ll_score_mode=args.ll_score_mode,
        ll_mode=args.ll_mode,
        adapter=adapter,
        hl_wgsp_coeff=args.hl_wgsp_coeff,
        lambda_hiql_reg=args.lambda_hiql_reg,
    )
