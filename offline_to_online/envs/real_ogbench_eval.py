"""Authoritative evaluator: runs the trained qc agent against the real
swm/OGBCube-v0 env for each of the 5 OGBench cube-single tasks.

For B2/E (agent operates on 192-D JEPA latents): we encode pixel observations
through the frozen JEPA model at every step and feed the resulting latent to
the agent. For B1 (raw pixels): pass the wrapped env directly to the standard
qc evaluate() instead -- this module is for the JEPA-encoded path.

Two dispatch modes for the 25-D output:
  - 'single' (B2): policy is action_chunking=True, horizon_length=5. The 25-D
    output reshapes to 5 separate 5-D actions and is dispatched through
    qc's action_queue (5 env.step calls per chunk).
  - 'chunk25' (E): policy is action_chunking=False, horizon_length=1. The 25-D
    output is *also* dispatched as 5 separate 5-D actions to the real env
    (open-loop within the chunk), because there is no other way to consume a
    chunk on a real env that steps one action at a time. The agent does not
    re-encode pixels until the chunk completes.
"""

from collections import defaultdict

import jax
import numpy as np
import torch
from tqdm import trange

from envs.jepa_loader import make_img_transform


LATENT_DIM = 192
ACTION_DIM_REAL = 5  # per-step OGBench cube action dim


def _encode(jepa_model, pixels_hwc_uint8, device, img_transform):
    img = torch.from_numpy(pixels_hwc_uint8).permute(2, 0, 1).contiguous()
    img = img_transform(img).to(device)
    info = {"pixels": img.unsqueeze(0).unsqueeze(0)}
    with torch.no_grad():
        info = jepa_model.encode(info)
    return info["emb"].squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)


def evaluate_real_ogbench(
    agent,
    real_env,
    jepa_model,
    device="cuda",
    task_ids=(1, 2, 3, 4, 5),
    num_episodes_per_task=50,
    max_steps=200,
    action_dispatch="single",  # 'single' for B1/B2, 'chunk25' for E
    pass_task_id_on_reset=True,
    obs_augment=None,
):
    """Return dict of per-task and aggregate success rates.

    Args:
        real_env: Either a single gymnasium env OR a dict {task_id: env}. When
            a dict is passed, the per-task env is selected by task_id and
            `pass_task_id_on_reset` should be False (each env is already bound
            to its task via registration).
        pass_task_id_on_reset: If True, calls reset(options=dict(task_id=...)) — works
            for swm/OGBCube-v0 which reads task_id from options. If False, just calls
            reset() — works for OGBench singletask envs where task_id is baked in via
            registration.
        obs_augment: Optional callable obs_augment(z, task_id) -> obs_aug applied
            before passing the latent to agent.sample_actions. Used by goal-conditioned
            agents to concatenate the task goal latent: obs_aug = concat([z, g_task]).
            Default None preserves single-task behavior.
    """
    img_transform = make_img_transform()
    rng = jax.random.PRNGKey(np.random.randint(0, 2**31 - 1))
    actor_fn = agent.sample_actions

    per_task_success = {}
    all_successes = []
    envs_dict = isinstance(real_env, dict)

    for task_id in task_ids:
        env_t = real_env[task_id] if envs_dict else real_env
        successes = []
        for ep in trange(num_episodes_per_task, desc=f"task {task_id}"):
            if pass_task_id_on_reset:
                obs_pix, info = env_t.reset(options=dict(task_id=task_id))
            else:
                obs_pix, info = env_t.reset()
            done = False
            step = 0
            action_queue = []
            success_seen = 0.0

            while not done and step < max_steps:
                if len(action_queue) == 0:
                    z = _encode(jepa_model, obs_pix, device, img_transform)
                    if obs_augment is not None:
                        z = obs_augment(z, task_id)
                    rng, key = jax.random.split(rng)
                    action_out = np.array(actor_fn(observations=z, rng=key)).reshape(-1)
                    # Whether single or chunk25, action_out is 25-D = 5 actions of 5 dims.
                    # Both paths reshape and dispatch one-by-one to the real env.
                    chunk_5x5 = action_out.reshape(-1, ACTION_DIM_REAL)
                    for a in chunk_5x5:
                        action_queue.append(np.clip(a.astype(np.float32), -1.0, 1.0))

                a = action_queue.pop(0)
                obs_pix, _r, terminated, truncated, info = env_t.step(a)
                done = terminated or truncated
                step += 1
                success_seen = max(success_seen, float(info.get("success", 0.0)))

            # Final-step success (matches OGBench's evaluation.py convention)
            final_success = float(info.get("success", success_seen))
            successes.append(final_success)

        sr = float(np.mean(successes)) if successes else 0.0
        per_task_success[f"task_{task_id}/success_rate"] = sr
        all_successes.extend(successes)
        print(f"  task {task_id}: {sr*100:5.1f}% ({int(sum(successes))}/{len(successes)})")

    overall = {"overall/success_rate": float(np.mean(all_successes)) if all_successes else 0.0}
    out = {**per_task_success, **overall}
    print(f"  OVERALL: {out['overall/success_rate']*100:5.1f}%")
    return out
