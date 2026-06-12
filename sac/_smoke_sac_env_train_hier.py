"""
Smoke-test for sac_env_train_hier.py (env-rollout hierarchical SAC).

Replaces the JEPA model, OGBench environment, and action scaler with
CPU-friendly dummies.  Tests all real code paths:
  - GoalConditionedActor / TwinCritic forward + backward passes
  - VectorizedEpisodicHERBuffer store + sample
  - collect_env_episode_hier() with a mock env + mock JEPA
  - train_loop_hier() N iterations end-to-end
  - log_alpha floor clamp (never < -4.6)
  - Checkpoint files written correctly

Run from ~/leworldmodel:
    python latent_hindsight_rl/sac/_smoke_sac_env_train_hier.py
"""

import os, sys, tempfile, shutil, math
os.environ.setdefault("MUJOCO_GL", "egl")

import torch
import torch.nn as nn
import numpy as np

# ── path fixup ───────────────────────────────────────────────────────────────
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, ".."))
for p in [_THIS, _ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── import real classes/functions from sac_env_train_hier ───────────────────
import importlib.util
spec = importlib.util.spec_from_file_location(
    "sac_env_train_hier",
    os.path.join(_THIS, "sac_env_train_hier.py"),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

GoalConditionedActor        = mod.GoalConditionedActor
TwinCritic                  = mod.TwinCritic
VectorizedEpisodicHERBuffer = mod.VectorizedEpisodicHERBuffer
collect_env_episode_hier    = mod.collect_env_episode_hier
train_loop_hier             = mod.train_loop_hier


# ── Dummy OGBench env ────────────────────────────────────────────────────────
class DummyEnv:
    """Minimal env that returns 224×224×3 numpy obs and a goal image in info."""
    H, W, C = 224, 224, 3

    def reset(self, options=None):
        obs  = np.random.randint(0, 255, (self.H, self.W, self.C), dtype=np.uint8)
        info = {"goal": np.random.randint(0, 255, (self.H, self.W, self.C), dtype=np.uint8)}
        return obs, info

    def step(self, action):
        obs  = np.random.randint(0, 255, (self.H, self.W, self.C), dtype=np.uint8)
        info = {"success": False}
        return obs, 0.0, False, False, info


# ── Dummy JEPA encoder ───────────────────────────────────────────────────────
class DummyJEPA(nn.Module):
    """Returns a fixed-size latent for any image input."""
    def __init__(self, latent_dim=192):
        super().__init__()
        self.latent_dim = latent_dim
        # Small linear so parameters() works and model.to(device) is valid
        self._dummy = nn.Linear(1, 1)

    def encode(self, batch):
        # batch["pixels"]: [B, T, C, H, W]
        B, T = batch["pixels"].shape[:2]
        return {"emb": torch.randn(B, T, self.latent_dim, device=batch["pixels"].device)}


# ── Dummy action scaler ───────────────────────────────────────────────────────
class DummyScaler:
    """inverse_transform → just return zeros of the right shape."""
    def inverse_transform(self, X):
        return np.zeros_like(X)


# ── Dummy image transform ─────────────────────────────────────────────────────
def dummy_transform(obs_hwc):
    """Returns a [C, H, W] float32 tensor without any real normalisation."""
    t = torch.from_numpy(obs_hwc.copy()).permute(2, 0, 1).float() / 255.0
    return t


# ── Monkeypatch encode_obs to use our dummy transform ────────────────────────
# The real encode_obs calls transform(obs_hwc).unsqueeze(0).unsqueeze(0)
# which expects a Compose that returns [C,H,W]. Our dummy_transform does that.
import types as _types

def _encode_obs_dummy(obs_hwc, model, transform, device):
    t = transform(obs_hwc).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,C,H,W]
    return model.encode({"pixels": t})["emb"][:, -1].squeeze(0)  # [192]

mod.encode_obs = _encode_obs_dummy


# ── Setup ─────────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cpu")
LATENT_DIM = 192
ACTION_DIM = 25
NUM_ITERS  = 4     # 4 iters × 5 tasks = 20 episodes total
SAVE_DIR   = tempfile.mkdtemp(prefix="smoke_sac_env_hier_")

print("=== Smoke test: sac_env_train_hier.py ===")
print(f"  device={DEVICE}  iters={NUM_ITERS}  save={SAVE_DIR}")

env           = DummyEnv()
jepa_model    = DummyJEPA(LATENT_DIM).to(DEVICE)
transform     = dummy_transform
action_scaler = DummyScaler()

actor         = GoalConditionedActor(latent_dim=LATENT_DIM, action_dim=ACTION_DIM,
                                     action_scale=3.0).to(DEVICE)
critic        = TwinCritic(latent_dim=LATENT_DIM, action_dim=ACTION_DIM).to(DEVICE)
critic_target = TwinCritic(latent_dim=LATENT_DIM, action_dim=ACTION_DIM).to(DEVICE)
critic_target.load_state_dict(critic.state_dict())

actor_opt     = torch.optim.Adam(actor.parameters(),  lr=3e-4)
critic_opt    = torch.optim.Adam(critic.parameters(), lr=3e-4)

log_alpha      = torch.tensor([0.0], requires_grad=True, device=DEVICE)
alpha_opt      = torch.optim.Adam([log_alpha], lr=3e-4)
target_entropy = -float(ACTION_DIM)

replay_buffer = VectorizedEpisodicHERBuffer(
    latent_dim=LATENT_DIM, action_dim=ACTION_DIM,
    capacity_episodes=200, max_t=50, future_p=0.8, device=DEVICE,
)

# ── Quick collect_env_episode_hier smoke ──────────────────────────────────────
print("\n[1/2] Testing collect_env_episode_hier (task_id=1, T_max=3)...")
z_curr, actions, z_next, z_goal, ep_len, success = collect_env_episode_hier(
    env=env, actor=actor, jepa_model=jepa_model,
    transform=transform, action_scaler=action_scaler,
    task_id=1, T_max=3, device=DEVICE,
)
assert z_curr.shape  == (ep_len, LATENT_DIM), f"z_curr shape {z_curr.shape}"
assert actions.shape == (ep_len, ACTION_DIM), f"actions shape {actions.shape}"
assert z_next.shape  == (ep_len, LATENT_DIM), f"z_next shape {z_next.shape}"
assert z_goal.shape  == (LATENT_DIM,),         f"z_goal shape {z_goal.shape}"
print(f"  ✓ ep_len={ep_len}  success={success}  shapes OK")

# ── Full train_loop_hier ──────────────────────────────────────────────────────
print(f"\n[2/2] Running train_loop_hier for {NUM_ITERS} iterations...")
train_loop_hier(
    env=env,
    actor=actor, critic=critic, critic_target=critic_target,
    actor_optimizer=actor_opt,
    critic_optimizer=critic_opt,
    alpha_optimizer=alpha_opt,
    log_alpha=log_alpha,
    target_entropy=target_entropy,
    replay_buffer=replay_buffer,
    jepa_model=jepa_model,
    transform=transform,
    action_scaler=action_scaler,
    num_iterations=NUM_ITERS,
    gamma=0.99, tau=0.005,
    T_max=3,           # short episodes for speed
    num_task_ids=5,
    reward_mode='sparse',
    bc_model=None, bc_alpha=0.0,
    save_dir=SAVE_DIR,
)

# ── Assertions ────────────────────────────────────────────────────────────────
assert not math.isnan(log_alpha.item()),           "log_alpha is NaN!"
assert log_alpha.item() >= -4.6,                   f"log_alpha {log_alpha.item()} breached -4.6 floor!"
assert os.path.exists(os.path.join(SAVE_DIR, "actor_policy.pth")),   "actor ckpt missing"
assert os.path.exists(os.path.join(SAVE_DIR, "critic_network.pth")), "critic ckpt missing"
assert os.path.exists(os.path.join(SAVE_DIR, "training_metrics.csv")),"csv missing"
# Note: training_config.json is written by __main__, not train_loop_hier — not checked here

# verify CSV has data rows
with open(os.path.join(SAVE_DIR, "training_metrics.csv")) as f:
    rows = f.readlines()
assert len(rows) >= 2, f"CSV has only {len(rows)} lines (header + 0 data rows?)"

print()
print("=== All assertions passed ✓ ===")
print(f"  log_alpha={log_alpha.item():.4f}  (α={log_alpha.exp().item():.4f})")
print(f"  CSV rows: {len(rows)-1} data rows")
shutil.rmtree(SAVE_DIR, ignore_errors=True)
print("  (temp dir cleaned up)")
