#!/usr/bin/env python3
import os
import torch
import types
import numpy as np
from transformers import logging as hf_logging
import argparse  # Add this import

# suppress HF INFO logs
hf_logging.set_verbosity_error()
 
# === USER PATHS ===
DATASET_PATH   = "/ceph/home/student.aau.dk/wb68dm/Isaac-GR00T_RL/Data_results_0001"
CHECKPOINT_DIR = "/ceph/home/student.aau.dk/wb68dm/reward_model_checkpoints/checkpoint-500"
OUTPUT_PREFIX  = "/ceph/home/student.aau.dk/wb68dm/Isaac-GR00T_RL/reward_model_inference_results"
BASE_MODEL     = "nvidia/GR00T-N1-2B"
EMB_TAG        = "new_embodiment"


# === Add command line arguments ===
parser = argparse.ArgumentParser(description='Run inference with reward model')
parser.add_argument('--max_samples', type=int, default=None, 
                    help='Maximum number of samples to process (default: all)')
args = parser.parse_args()


# === IMPORTS ===
from gr00t.model.gr00t_n1       import GR00T_N1
from gr00t.model.reward_head     import RewardHeadConfig
from gr00t.model.reward_model    import GR00TReward
from gr00t.data.reward_dataset   import RewardDataset
from gr00t.data.dataset          import ModalityConfig
from gr00t.data.transform.base   import ComposedModalityTransform
from gr00t.data.transform        import (
    VideoToTensor, VideoCrop, VideoResize, VideoColorJitter, VideoToNumpy
)
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.concat         import ConcatTransform
from gr00t.data.transform.transform_reward import RewardTransform
from gr00t.model.transforms             import GR00TTransform
 
def build_transforms():
    """Rebuild the exact same transforms you used for training."""
    video_mod  = ModalityConfig(delta_indices=[0], modality_keys=["video.ego_view"])
    state_mod  = ModalityConfig(delta_indices=[0], modality_keys=["state.acceleration"])
    action_mod = ModalityConfig(delta_indices=[0], modality_keys=["action.wheel_commands"])
    reward_mod = ModalityConfig(delta_indices=[0], modality_keys=["next.reward"])
    lang_mod   = ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.validity"])
    modality_configs = {
        "video":    video_mod,
        "state":    state_mod,
        "action":   action_mod,
        "next":     reward_mod,
        "language": lang_mod,
    }
 
    transforms = ComposedModalityTransform(transforms=[
        VideoToTensor(apply_to=video_mod.modality_keys, backend="torchvision"),
        VideoCrop  (apply_to=video_mod.modality_keys, scale=0.95, backend="torchvision"),
        VideoResize(apply_to=video_mod.modality_keys,
                    height=224, width=224,
                    interpolation="linear", backend="torchvision"),
        VideoColorJitter(apply_to=video_mod.modality_keys,
                         brightness=0.3, contrast=0.4,
                         saturation=0.5, hue=0.08,
                         backend="torchvision"),
        VideoToNumpy(apply_to=video_mod.modality_keys),
 
        StateActionToTensor(apply_to=state_mod.modality_keys),
        StateActionTransform(apply_to=state_mod.modality_keys,
                             normalization_modes={"state.acceleration":"min_max"}),
 
        StateActionToTensor(apply_to=action_mod.modality_keys),
        StateActionTransform(apply_to=action_mod.modality_keys,
                             normalization_modes={"action.wheel_commands":"min_max"}),
 
        RewardTransform(apply_to=reward_mod.modality_keys,
                        normalization_mode="min_max"),
 
        ConcatTransform(
            video_concat_order=video_mod.modality_keys,
            state_concat_order=state_mod.modality_keys,
            action_concat_order=action_mod.modality_keys
        ),
 
        GR00TTransform(
            state_horizon = len(state_mod.delta_indices),
            action_horizon = len(action_mod.delta_indices),
            max_state_dim  = 64,
            max_action_dim = 2
        ),
    ])
    return transforms, modality_configs
 
def load_model(checkpoint_dir, base_name, device):
    """Instantiate GR00TReward, load your checkpoint, strip prefixes, cast to bfloat16."""
    # 1) create backbone+head in bfloat16
    base = GR00T_N1.from_pretrained(
        pretrained_model_name_or_path=base_name,
        tune_llm=False, tune_visual=True, tune_projector=True,
        torch_dtype=torch.bfloat16
    )
    
    # Add LoRA configuration matching your training settings
    from gr00t.utils.peft import get_lora_model
    lora_rank = 8
    lora_alpha = 16
    lora_dropout = 0.10
    
    base = get_lora_model(
        base,
        rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    
    head_cfg = RewardHeadConfig(hidden_dim=512, dropout=0.1, reward_dim=1, tune_projector=True)
    model = GR00TReward(gr00t_model=base, reward_head_config=head_cfg)
    
    # Rest of the function remains the same
    # ...
 
    # 2) locate the actual weight file
    if os.path.isdir(checkpoint_dir):
        for fn in ("model.safetensors","pytorch_model.bin","pytorch_model.pt"):
            p = os.path.join(checkpoint_dir, fn)
            if os.path.isfile(p):
                checkpoint_file = p
                break
        else:
            raise FileNotFoundError(f"No weights found under {checkpoint_dir}")
    else:
        checkpoint_file = checkpoint_dir
 
    # 3) load state dict (safetensors or torch.load)
    if checkpoint_file.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file as safe_load
        except ImportError:
            raise ImportError("pip install safetensors to load .safetensors checkpoints")
        state = safe_load(checkpoint_file, device="cpu")
    else:
        try:
            state = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(checkpoint_file, map_location="cpu")
 
    # 4) strip any `model.` prefix (from RewardModelWrapper)
    stripped = {k.removeprefix("model."):v for k,v in state.items()}
 
    # 5) load into model (non-strict) and warn on mismatches
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if missing:
        print("⚠️ Missing keys:", missing)
    if unexpected:
        print("⚠️ Unexpected keys:", unexpected)
 
    # 6) cast to bfloat16 and fix compute_dtype so prepare_input does the right cast
    model.to(device)
    model = model.to(torch.bfloat16)
    model.compute_dtype        = "bfloat16"
    model.config.compute_dtype = "bfloat16"
    model.eval()
    return model
 
def run_inference():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transforms, mods = build_transforms()
     
        # build dataset
        ds = RewardDataset(
            dataset_path     = DATASET_PATH,
            modality_configs = mods,
            embodiment_tag   = EMB_TAG,
            video_backend    = "torchvision_av",
            transforms       = transforms,
        )
        
        # Use command line arguments to limit dataset size
        total_samples = len(ds)
        if args.max_samples is not None:
            max_samples = min(args.max_samples, total_samples)
        else:
            max_samples = total_samples
            
        # Use sample_step to skip samples
        max_samples = args.max_samples  # Process first 100 timesteps in sequence
        indices = list(range(max_samples))  # Keep sequential order
            
        print(f"Dataset loaded with {total_samples} total samples")
        
        model = load_model(CHECKPOINT_DIR, BASE_MODEL, device)
        print("Model loaded successfully")
     
        # --- patch prepare_input to collapse that extra time-dim on video ---
        orig_prep = model.prepare_input
        def patched_prepare(self, inputs):
            b_inputs, r_inputs = orig_prep(inputs)
            # video comes in as [B, T, C, H, W], but SigLIP wants [B,C,H,W]
            pv = b_inputs.get("pixel_values", None)
            if pv is not None and pv.ndim == 5 and pv.shape[1] == 1:
                b_inputs["pixel_values"] = pv.squeeze(1)
            return b_inputs, r_inputs
        model.prepare_input = types.MethodType(patched_prepare, model)
        
        # Add tqdm for progress tracking
        from tqdm import tqdm
     
        all_preds, all_tgts = [], []
        print("Starting inference...")
        for idx in tqdm(indices, desc="Processing samples"):
            item = ds[idx]
            # Fix: Properly handle different data types when converting to tensors
            for k, v in item.items():
                if isinstance(v, np.ndarray):
                    # Convert numpy arrays to tensors
                    t = torch.from_numpy(v)
                    if torch.is_floating_point(t):
                        t = t.to(device=device, dtype=torch.bfloat16)
                    else:
                        t = t.to(device=device)
                elif isinstance(v, (int, float)):
                    # Handle primitive numeric types
                    if isinstance(v, float):
                        t = torch.tensor(v, device=device, dtype=torch.bfloat16)
                    else:
                        t = torch.tensor(v, device=device)
                elif isinstance(v, torch.Tensor):
                    # Already a tensor, just move to device and convert dtype if needed
                    if torch.is_floating_point(v):
                        t = v.to(device=device, dtype=torch.bfloat16)
                    else:
                        t = v.to(device=device)
                else:
                    # Keep other types as they are (e.g., strings)
                    t = v
                item[k] = t
     
            out = model.get_reward(item)
            pred = out["reward_pred"].squeeze().cpu().float().item()
            tgt  = item["target_reward"].squeeze().cpu().float().item()
     
            all_preds.append(pred)
            all_tgts.append(tgt)
     
        all_preds = np.array(all_preds, dtype=np.float32)
        all_tgts  = np.array(all_tgts,  dtype=np.float32)
     
        mse  = np.mean((all_preds-all_tgts)**2)
        mae  = np.mean(np.abs(all_preds-all_tgts))
        corr = np.corrcoef(all_preds, all_tgts)[0,1]
     
        print(f"\nInference results:\n  MSE: {mse:.6f}\n  MAE: {mae:.6f}\n  Corr: {corr:.4f}")
        np.savez(OUTPUT_PREFIX, preds=all_preds, targets=all_tgts)
        print(f"Saved → {OUTPUT_PREFIX}.npz")
    except Exception as e:
        import traceback
        print(f"Error during inference: {str(e)}")
        traceback.print_exc()
 
if __name__ == "__main__":
    run_inference()