"""
Patch cube_single_play_v0.h5 to add fields required by eval_actor.py.
All fields are derived from data already in the file — nothing is downloaded.

Run once:
  python latent_hindsight_rl/patch_play_hdf5.py
"""
import os
import numpy as np
import h5py

stablewm_home = os.environ.get("STABLEWM_HOME", os.path.join(os.path.expanduser("~"), "stable_wm_data"))
path = os.path.join(stablewm_home, "ogbench", "cube_single_play_v0.h5")
print(f"Patching: {path}")

with h5py.File(path, "a") as f:
    ep_len = f["ep_len"][:]   # (1000,)
    qpos   = f["qpos"][:]     # (1001000, 21)

    if "ep_idx" not in f:
        ep_idx = np.repeat(np.arange(len(ep_len), dtype=np.int32), ep_len)
        f.create_dataset("ep_idx", data=ep_idx)
        print(f"  + ep_idx {ep_idx.shape}")
    else:
        print("  ep_idx already exists")

    if "step_idx" not in f:
        step_idx = np.concatenate([np.arange(l, dtype=np.int64) for l in ep_len])
        f.create_dataset("step_idx", data=step_idx)
        print(f"  + step_idx {step_idx.shape}")
    else:
        print("  step_idx already exists")

    if "privileged_block_0_pos" not in f:
        # qpos[14:17] = cube xyz (verified: 1.000 correlation with expert privileged_block_0_pos)
        f.create_dataset("privileged_block_0_pos", data=qpos[:, 14:17].astype(np.float32))
        print(f"  + privileged_block_0_pos (from qpos[14:17])")
    else:
        print("  privileged_block_0_pos already exists")

    if "privileged_block_0_quat" not in f:
        f.create_dataset("privileged_block_0_quat", data=qpos[:, 17:21].astype(np.float32))
        print(f"  + privileged_block_0_quat (from qpos[17:21])")
    else:
        print("  privileged_block_0_quat already exists")

print("Done.")
