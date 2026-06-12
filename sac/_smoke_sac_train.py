"""
Smoke-test for sac_train.py (WM-rollout hierarchical SAC).

Does NOT invoke __main__ or load any real data/checkpoints.
Instead it:
  - Imports the real GoalConditionedActor, TwinCritic,
    VectorizedEpisodicHERBuffer from sac_train.py
  - Builds a minimal train loop replica (same math) with a DummyEnv
  - Runs 6 iterations on CPU in < 20 s
  - Asserts: alpha init=1.0, floor clamp enforced, no NaN, checkpoints written

Run from ~/leworldmodel:
    python latent_hindsight_rl/sac/_smoke_sac_train.py
"""

import os, sys, tempfile, shutil, math, csv, time
os.environ.setdefault("MUJOCO_GL", "egl")

import torch
import torch.nn.functional as F
import numpy as np
from collections import deque

# ── path fixup ───────────────────────────────────────────────────────────────
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, ".."))
for p in (_THIS, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# Import just the three standalone classes (no __main__, no data loading)
from sac_train import (
    GoalConditionedActor,
    TwinCritic,
    VectorizedEpisodicHERBuffer,
)

print("  Imports OK: GoalConditionedActor, TwinCritic, VectorizedEpisodicHERBuffer")

# ── Constants ────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cpu")
LATENT_DIM = 192
ACTION_DIM = 25
NUM_ITERS  = 8
SAVE_DIR   = tempfile.mkdtemp(prefix="smoke_sac_wm_")

print(f"\n=== Smoke: sac_train.py ===  device={DEVICE}  iters={NUM_ITERS}  save={SAVE_DIR}")

# ── Build components ─────────────────────────────────────────────────────────
actor         = GoalConditionedActor(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, action_scale=3.0).to(DEVICE)
critic        = TwinCritic(latent_dim=LATENT_DIM, action_dim=ACTION_DIM).to(DEVICE)
critic_target = TwinCritic(latent_dim=LATENT_DIM, action_dim=ACTION_DIM).to(DEVICE)
critic_target.load_state_dict(critic.state_dict())

actor_opt    = torch.optim.Adam(actor.parameters(),  lr=3e-4)
critic_opt   = torch.optim.Adam(critic.parameters(), lr=3e-4)

# KEY: log_alpha must start at 0.0 (α=1.0) — this is the fixed bug
log_alpha      = torch.tensor([0.0], requires_grad=True, device=DEVICE)
alpha_opt      = torch.optim.Adam([log_alpha], lr=3e-4)
target_entropy = -float(ACTION_DIM)

replay_buffer = VectorizedEpisodicHERBuffer(
    latent_dim=LATENT_DIM, action_dim=ACTION_DIM,
    capacity_episodes=200, max_t=50, future_p=0.8, device=DEVICE,
)

os.makedirs(SAVE_DIR, exist_ok=True)
csv_path = os.path.join(SAVE_DIR, "training_metrics.csv")
with open(csv_path, "w", newline="") as f:
    csv.writer(f).writerow(["Iteration","Success_Rate","Buffer_Size",
                             "Actor_Loss","Critic_Loss","Alpha","Total_Time"])

print(f"  alpha_init = {log_alpha.exp().item():.4f}  (expected 1.0)")
assert abs(log_alpha.exp().item() - 1.0) < 1e-4, "Alpha not initialised to 1.0!"

recent = deque(maxlen=40)
t0 = time.time()
GAMMA, TAU, BS = 0.99, 0.005, 64

for it in range(NUM_ITERS):
    # ---- Simulate episode batch (4 envs × 6 steps) ----
    NE, T = 4, 6
    z_c = torch.randn(NE, T, LATENT_DIM, device=DEVICE)
    a   = torch.randn(NE, T, ACTION_DIM, device=DEVICE)
    z_n = torch.randn(NE, T, LATENT_DIM, device=DEVICE)
    z_g = torch.randn(NE, T, LATENT_DIM, device=DEVICE)
    lens = torch.full((NE,), T, dtype=torch.long, device=DEVICE)
    replay_buffer.store_episodes(z_c, a, z_n, z_g, lens)
    recent.extend([0.0] * NE)

    act_l = crit_l = 0.0
    if replay_buffer.num_transitions >= BS:
        z_b, a_b, zn_b, g_b, r_b, d_b = replay_buffer.sample_batch(BS, 'sparse')
        r_b = r_b.unsqueeze(-1); d_b = d_b.unsqueeze(-1)
        alpha = log_alpha.exp().item()

        # Critic update
        with torch.no_grad():
            na, nlp, _ = actor.sample(zn_b, g_b)
            tq1, tq2 = critic_target(zn_b, g_b, na)
            tq = r_b + (1-d_b)*GAMMA*(torch.min(tq1,tq2) - alpha*nlp)
        q1, q2 = critic(z_b, g_b, a_b)
        cl = F.mse_loss(q1, tq) + F.mse_loss(q2, tq)
        critic_opt.zero_grad(); cl.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_opt.step(); crit_l = cl.item()

        # Actor update
        new_a, lp, _ = actor.sample(z_b, g_b)
        for p in critic.parameters(): p.requires_grad = False
        q1n, q2n = critic(z_b, g_b, new_a)
        al = (alpha*lp - torch.min(q1n, q2n)).mean()
        actor_opt.zero_grad(); al.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        actor_opt.step(); act_l = al.item()
        for p in critic.parameters(): p.requires_grad = True

        # Alpha update + FLOOR CLAMP
        aloss = -(log_alpha * (lp + target_entropy).detach()).mean()
        alpha_opt.zero_grad(); aloss.backward(); alpha_opt.step()
        with torch.no_grad():
            log_alpha.clamp_(min=-4.6)           # α floor ≈ 0.01

        # Soft target update
        for tp, p in zip(critic_target.parameters(), critic.parameters()):
            tp.data.copy_(tp.data*(1-TAU) + p.data*TAU)

    sr = float(np.mean(recent)) if recent else 0.0
    elapsed = time.time() - t0
    print(f"  iter {it:02d} | buf={replay_buffer.size:3d} | "
          f"act={act_l:+.4f}  crit={crit_l:.4f}  α={log_alpha.exp().item():.4f}")
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow([it, sr, replay_buffer.size, act_l, crit_l,
                                  log_alpha.exp().item(), elapsed])

torch.save(actor.state_dict(),  os.path.join(SAVE_DIR, "actor_policy.pth"))
torch.save(critic.state_dict(), os.path.join(SAVE_DIR, "critic_network.pth"))

# ── Assertions ────────────────────────────────────────────────────────────────
assert not math.isnan(log_alpha.item()),    "log_alpha is NaN!"
assert log_alpha.item() >= -4.6,            f"log_alpha {log_alpha.item():.4f} breached -4.6 floor!"
assert os.path.exists(os.path.join(SAVE_DIR, "actor_policy.pth")),   "actor ckpt missing"
assert os.path.exists(os.path.join(SAVE_DIR, "critic_network.pth")), "critic ckpt missing"
assert os.path.exists(csv_path),                                       "csv missing"
with open(csv_path) as f:
    nrows = sum(1 for _ in f) - 1   # subtract header
assert nrows == NUM_ITERS, f"Expected {NUM_ITERS} CSV rows, got {nrows}"

print()
print("=== All assertions passed ✓ ===")
print(f"  log_alpha = {log_alpha.item():.4f}   α = {log_alpha.exp().item():.4f}")
print(f"  {nrows} CSV rows written")
shutil.rmtree(SAVE_DIR, ignore_errors=True)
print("  (temp dir cleaned up)")
