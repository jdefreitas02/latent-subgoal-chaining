import torch
import numpy as np
import os
import sys
import time
import torchvision.transforms.v2 as tv_transforms
import stable_worldmodel as swm
from hydra import initialize, compose

def precompute():
    print("--- Starting Full Dataset Latent Pre-computation ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    stablewm_home = os.environ.get("STABLEWM_HOME", os.path.join(os.path.expanduser("~"), "stable_wm_data"))
    data_path = os.path.join(stablewm_home, "ogbench", "cube_single_expert")
    
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path: sys.path.insert(0, parent_dir)
        
    with initialize(version_base=None, config_path="../config"): 
        cfg = compose(config_name="eval/cube", overrides=["+policy=cube/lejepa"])
    
    dataset = swm.data.HDF5Dataset(data_path)
    model = swm.policy.AutoCostModel(cfg.policy).to(device)
    model.eval()
    
    # --- THE MASSIVE BUG FIX: IMAGE NORMALIZATION ---
    # We MUST transform the raw uint8 [0, 255] pixels into normalized floats
    # exactly the way the World Model was trained!
    img_transform = tv_transforms.Compose([
        tv_transforms.ToDtype(torch.float32, scale=True),
        tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    num_episodes = len(dataset.lengths)
    batch_size = 10 
    
    all_latents = []
    total_frames_processed = 0
    
    t0 = time.time()
    
    with torch.no_grad():
        for i in range(0, num_episodes, batch_size):
            end_idx = min(i + batch_size, num_episodes)
            ep_indices = np.arange(i, end_idx)
            ep_lens = dataset.lengths[ep_indices]
            
            starts = np.zeros(len(ep_indices), dtype=int)
            ends = ep_lens
            
            chunks = dataset.load_chunk(ep_indices, starts, ends)
            
            for chunk in chunks:
                # 1. Get raw pixels: shape [ep_len, 3, 224, 224]
                raw_pixels = chunk['pixels'].to(device)
                
                # 2. APPLY THE MISSING IMAGENET TRANSFORM!
                clean_pixels = img_transform(raw_pixels)
                
                # 3. Add batch dimension: [1, ep_len, 3, 224, 224]
                pixels_5d = clean_pixels.unsqueeze(0)
                
                # 4. Encode the perfectly clean images
                z_ep = model.encode({'pixels': pixels_5d})['emb'].squeeze(0)
                
                all_latents.append(z_ep.cpu())
                total_frames_processed += len(z_ep)
            
            elapsed = time.time() - t0
            print(f"Processed {end_idx}/{num_episodes} episodes | "
                  f"Total Frames: {total_frames_processed:,} | "
                  f"Time: {elapsed:.2f}s")

    save_path = os.path.join(ephemeral, "stable_wm_data", "cube_all_latents_cache.pt")
    
    print("\nSaving CLEAN latent cache to disk...")
    torch.save({
        'all_latents': all_latents,          
        'total_frames': total_frames_processed
    }, save_path)
    
    print(f"SUCCESS! Saved cache to {save_path}")

if __name__ == "__main__":
    precompute()