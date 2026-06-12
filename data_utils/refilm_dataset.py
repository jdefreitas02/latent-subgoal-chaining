import os
import gc
import h5py
import hdf5plugin 
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import mujoco
import ogbench 

def refilm_dataset(original_path, new_path, env_name, target_res=(224, 224)):
    print(f"Loading environment: {env_name} at {target_res}...")
    env = gym.make(env_name, width=target_res[0], height=target_res[1])
    env.reset()
    
    # Check if existing file is valid before attempting to resume
    if os.path.exists(new_path):
        try:
            with h5py.File(new_path, 'r') as test_f:
                _ = test_f.attrs.get('frames_rendered', 0)
                pixel_keys = [k for k in test_f.keys() if 'pixel' in k.lower()]
                if pixel_keys:
                    _ = test_f[pixel_keys[0]]  # force open the dataset
            mode = 'a'
        except Exception as e:
            print(f"Existing file is corrupt ({e}), restarting from scratch...")
            os.remove(new_path)
            mode = 'w'
    else:
        mode = 'w'

    with h5py.File(original_path, 'r') as old_f, h5py.File(new_path, mode) as new_f:
        
        pixel_keys = [k for k in old_f.keys() if 'pixel' in k.lower()]
        if not pixel_keys:
            raise ValueError("Could not find a pixel key in the original dataset.")
            
        target_pixel_key = pixel_keys[0]
        num_steps = old_f[target_pixel_key].shape[0]
        
        # 1. First-time setup
        if mode == 'w':
            for key in old_f.keys():
                if 'pixel' not in key.lower():
                    print(f"Copying metadata/state key: {key}")
                    old_f.copy(key, new_f)
            
            print(f"Creating new 224x224 pixel dataset...")
            new_pixels = new_f.create_dataset(
                target_pixel_key, 
                shape=(num_steps, target_res[0], target_res[1], 3),
                dtype=np.uint8,
                chunks=(100, target_res[0], target_res[1], 3),
                compression=hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
            )
            new_f.attrs['frames_rendered'] = 0
        else:
            new_pixels = new_f[target_pixel_key]
        
        start_idx = new_f.attrs.get('frames_rendered', 0)
        
        if start_idx >= num_steps:
            print("Dataset already completely rendered!")
            env.close()
            return
            
        print(f"Re-filming from frame {start_idx} to {num_steps}...")
        
        qpos_data = old_f['qpos'][:]
        qvel_data = old_f['qvel'][:]
        # scene / puzzle envs override set_state(qpos, qvel, button_states); their
        # button colors are part of the visible state, so they must be re-applied.
        button_data = old_f['button_states'][:] if 'button_states' in old_f else None
        if button_data is not None:
            print(f"Dataset carries button_states {button_data.shape} — will pass to set_state.")

        # 2. Optimization: Pre-check environment capabilities outside the loop
        has_set_state = hasattr(env.unwrapped, 'set_state')
        unwrapped_env = env.unwrapped
        
        # 3. Optimization: Buffer for batch writing (speeds up HDF5 I/O massively)
        BATCH_SIZE = 5000
        buffer = np.zeros((BATCH_SIZE, target_res[0], target_res[1], 3), dtype=np.uint8)
        buffer_idx = 0
        
        for i in tqdm(range(start_idx, num_steps), initial=start_idx, total=num_steps):
            
            # Fast physics teleport
            if has_set_state:
                if button_data is not None:
                    # scene/puzzle: set_state(qpos, qvel, button_states)
                    unwrapped_env.set_state(qpos_data[i], qvel_data[i], button_data[i])
                else:
                    unwrapped_env.set_state(qpos_data[i], qvel_data[i])
            else:
                unwrapped_env.data.qpos[:] = qpos_data[i]
                unwrapped_env.data.qvel[:] = qvel_data[i]
                mujoco.mj_forward(unwrapped_env.model, unwrapped_env.data)
            
            # Render to RAM buffer instead of disk
            buffer[buffer_idx] = env.render()
            buffer_idx += 1
            
            # Batch write to disk every BATCH_SIZE frames
            if buffer_idx == BATCH_SIZE:
                # Write the whole chunk at once
                new_pixels[i - BATCH_SIZE + 1 : i + 1] = buffer
                buffer_idx = 0
                
                # Save progress and clear memory
                new_f.attrs['frames_rendered'] = i + 1
                new_f.flush()
                gc.collect()
                
        # 4. Handle any remaining frames at the end
        if buffer_idx > 0:
            new_pixels[num_steps - buffer_idx : num_steps] = buffer[:buffer_idx]
            new_f.attrs['frames_rendered'] = num_steps
            new_f.flush()

    print("Re-filming complete! File saved to:", new_path)
    env.close()

if __name__ == "__main__":
    import argparse
    base_dir = os.path.expandvars(
        os.path.join(os.environ.get("STABLEWM_HOME", "$HOME/stable_wm_data"), "ogbench")
    )
    parser = argparse.ArgumentParser(description="Re-render an OGBench 64x64 HDF5 to higher res via MuJoCo replay")
    parser.add_argument("--input",  default=os.path.join(base_dir, "visual-cube-single-play-v0.h5"),
                        help="Path to 64x64 HDF5 (must contain qpos/qvel, plus button_states for scene/puzzle)")
    parser.add_argument("--output", default=os.path.join(base_dir, "visual-cube-single-play-v0_224.h5"),
                        help="Path for output high-res HDF5")
    parser.add_argument("--env_name", default="visual-cube-single-v0",
                        help="Gym env id used to re-render (e.g. visual-scene-v0, visual-puzzle-3x3-v0)")
    parser.add_argument("--res", type=int, default=224, help="Target square resolution")
    args = parser.parse_args()

    refilm_dataset(
        original_path=args.input,
        new_path=args.output,
        env_name=args.env_name,
        target_res=(args.res, args.res),
    )