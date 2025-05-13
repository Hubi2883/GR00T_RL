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
CHECKPOINT_DIR = "/ceph/home/student.aau.dk/wb68dm/reward_model_checkpoints/checkpoint-1500"
OUTPUT_PREFIX  = "/ceph/home/student.aau.dk/wb68dm/Isaac-GR00T_RL/reward_model_inference_results"
BASE_MODEL     = "nvidia/GR00T-N1-2B"
EMB_TAG        = "new_embodiment"


# === Add command line arguments ===
parser = argparse.ArgumentParser(description='Run inference with reward model')
parser.add_argument('--max_samples', type=int, default=1000, 
                    help='Maximum number of samples to process (default: all)')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Path to a specific checkpoint (overrides CHECKPOINT_DIR)')
parser.add_argument('--prediction_only', action='store_true', 
                    help='Run in prediction-only mode without requiring target values')
args = parser.parse_args()

if args.checkpoint:
    CHECKPOINT_DIR = args.checkpoint
    print(f"Using checkpoint from command line: {CHECKPOINT_DIR}")


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
    # Match exactly what's in the training script
    video_mod  = ModalityConfig(delta_indices=[0], modality_keys=["video.ego_view"])
    state_mod  = ModalityConfig(delta_indices=[0], modality_keys=["state.acceleration"])
    action_mod = ModalityConfig(delta_indices=[0], modality_keys=["action.wheel_commands"])
    reward_mod = ModalityConfig(delta_indices=list(range(16)), modality_keys=["next.reward"])
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
                        normalization_mode="none"),
 
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
    print(f"Loading model from checkpoint: {checkpoint_dir}")
    print(f"Using base model: {base_name}")
    
    # 1) create backbone+head in bfloat16 - MATCH EXACTLY WHAT'S IN TRAINING SCRIPT
    base = GR00T_N1.from_pretrained(
        pretrained_model_name_or_path=base_name,
        tune_llm=False, tune_visual=True, tune_projector=True,
        torch_dtype=torch.bfloat16
    )
    
    # Add LoRA configuration matching your training settings
    from gr00t.utils.peft import get_lora_model
    # Updated to match the training script exactly
    lora_rank = 16
    lora_alpha = 8
    lora_dropout = 0.10
    
    print(f"Applying LoRA with rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")
    base = get_lora_model(
        base,
        rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    
    # Updated to match the RewardHeadConfig in training
    head_cfg = RewardHeadConfig(
        hidden_dim=2048,
        dropout=0.10,
        reward_dim=1,
        reward_horizon=16,
        tune_projector=True
    )
    
    print(f"Created reward head with config: {head_cfg}")
    model = GR00TReward(gr00t_model=base, reward_head_config=head_cfg)
    
    # 2) locate the actual weight file
    if os.path.isdir(checkpoint_dir):
        # Look for checkpoint files with broader patterns
        possible_files = [
            "model.safetensors", "pytorch_model.bin", "pytorch_model.pt", 
            "model.bin", "model.pt", "final_reward_model.pt"
        ]
        for fn in possible_files:
            p = os.path.join(checkpoint_dir, fn)
            if os.path.isfile(p):
                checkpoint_file = p
                print(f"Found checkpoint file: {checkpoint_file}")
                break
        else:
            # Also check if the directory itself is a checkpoint file
            if os.path.isfile(checkpoint_dir):
                checkpoint_file = checkpoint_dir
                print(f"Using directory as checkpoint file: {checkpoint_file}")
            else:
                raise FileNotFoundError(f"No weights found under {checkpoint_dir}")
    else:
        checkpoint_file = checkpoint_dir
        print(f"Using checkpoint file directly: {checkpoint_file}")
 
    # 3) load state dict (safetensors or torch.load)
    print(f"Loading weights from: {checkpoint_file}")
    if checkpoint_file.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file as safe_load
            print("Loading with safetensors")
            state = safe_load(checkpoint_file, device="cpu")
        except ImportError:
            raise ImportError("pip install safetensors to load .safetensors checkpoints")
    else:
        try:
            print("Loading with torch.load")
            state = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(checkpoint_file, map_location="cpu")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise
 
    # 4) strip any `model.` prefix (from RewardModelWrapper) and handle other prefixes
    print("Processing state dict keys...")
    
    # Try different prefix patterns that might exist in the state dict
    patterns = ["model.", "reward_model.", "module.model.", "module."]
    
    # First, check if any of these patterns exist
    found_patterns = set()
    for key in list(state.keys())[:10]:  # Check first few keys
        for pattern in patterns:
            if key.startswith(pattern):
                found_patterns.add(pattern)
    
    # If we found patterns, strip them
    stripped = {}
    if found_patterns:
        print(f"Found prefixes to strip: {found_patterns}")
        for k, v in state.items():
            new_key = k
            for pattern in found_patterns:
                if new_key.startswith(pattern):
                    new_key = new_key.removeprefix(pattern)
            stripped[new_key] = v
    else:
        # If no patterns found, try the original logic
        print("No prefixes found, using original keys")
        stripped = {k.removeprefix("model."):v for k, v in state.items()}
 
    # 5) load into model (non-strict) and warn on mismatches
    print("Loading state dict into model...")
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if missing:
        print("⚠️ Missing keys:", missing[:10], f"...and {len(missing)-10} more" if len(missing) > 10 else "")
    if unexpected:
        print("⚠️ Unexpected keys:", unexpected[:10], f"...and {len(unexpected)-10} more" if len(unexpected) > 10 else "")
 
    # 6) cast to bfloat16 and fix compute_dtype so prepare_input does the right cast
    print(f"Moving model to device: {device}")
    model.to(device)
    model = model.to(torch.bfloat16)
    model.compute_dtype        = "bfloat16"
    model.config.compute_dtype = "bfloat16"
    model.eval()
    
    print("Model loaded successfully")
    return model
 
def run_inference():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        transforms, mods = build_transforms()
        print("Transforms built successfully")
     
        # build dataset
        print(f"Loading dataset from: {DATASET_PATH}")
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
        indices = list(range(max_samples))  # Keep sequential order
            
        print(f"Dataset loaded with {total_samples} total samples, using {max_samples} for inference")
        
        model = load_model(CHECKPOINT_DIR, BASE_MODEL, device)
     
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
        
        # Debugging: Check what keys are actually available in the dataset
        sample_keys = list(ds[0].keys())
        print(f"Available keys in dataset: {sample_keys}")
        
        # Print more info about the target modality configuration
        print(f"Reward modality config: {mods['next']}")
        
        # Determine if there's any key that might contain reward information
        potential_reward_keys = [k for k in sample_keys if 'reward' in k.lower()]
        if potential_reward_keys:
            print(f"Potential reward keys found: {potential_reward_keys}")
        
        with torch.no_grad():  # Add no_grad context for inference
            for idx in tqdm(indices, desc="Processing samples"):
                try:
                    item = ds[idx]
                    # Fix: Properly handle different data types when converting to tensors
                    inputs = {}
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
                        inputs[k] = t
         
                    # Get prediction from model
                    if hasattr(model, "get_reward"):
                        out = model.get_reward(inputs)
                        pred_tensor = out["reward_pred"]
                    else:
                        out = model(inputs)
                        pred_tensor = out.get("reward_pred", out.get("logits", None))
                    
                    if pred_tensor is None:
                        print(f"Warning: No prediction found in model output for sample {idx}")
                        continue
                        
                    # IMPORTANT: Apply sigmoid like in training script
                    pred_tensor = torch.sigmoid(pred_tensor.cpu().float())
                    
                    # SIMPLIFIED: Handle 16-element tensor by taking mean
                    pred_scalar = pred_tensor.mean().item()
                    
                    # Try multiple possible keys for the target reward
                    tgt_scalar = None
                    
                    # List of possible keys for reward, in order of priority
                    all_possible_tgt_keys = [
                        "next.reward", "reward", "target_reward", 
                        "next.reward.0", "next.reward.1", # Check array index variants
                    ] + potential_reward_keys  # Add any keys with 'reward' in the name
                    
                    # Look for any key that might contain the target reward
                    for key in all_possible_tgt_keys:
                        if key in inputs:
                            try:
                                tgt_tensor = inputs[key].cpu().float()
                                # Target may also be a 16-element tensor, so take mean
                                tgt_scalar = tgt_tensor.mean().item()
                                # If we found a valid target, break out of the loop
                                break
                            except Exception as e:
                                print(f"Error processing target key '{key}': {e}")
                    
                    # If we're in prediction-only mode, we don't need targets
                    if tgt_scalar is None and not args.prediction_only:
                        # First sample - print available keys for debugging
                        if idx == 0:
                            print(f"Warning: No target reward found. Available keys: {list(inputs.keys())}")
                            # For the first sample that fails, print the actual keys in the dataset
                            print(f"Dataset item keys: {list(item.keys())}")
                        else:
                            print(f"Warning: No target reward found for sample {idx}")
                        
                        # In prediction-only mode, still include the prediction
                        if args.prediction_only:
                            all_preds.append(pred_scalar)
                            all_tgts.append(float('nan'))  # Use NaN for missing targets
                        continue
                    
                    # Store the prediction (and target if available)
                    all_preds.append(pred_scalar)
                    if tgt_scalar is not None:
                        all_tgts.append(tgt_scalar)
                    else:
                        # In prediction-only mode, use NaN for missing targets
                        all_tgts.append(float('nan'))
                        
                except Exception as e:
                    print(f"Error processing sample {idx}: {e}")
                    # Print tensor shape for debugging
                    if 'pred_tensor' in locals():
                        print(f"Prediction tensor shape: {pred_tensor.shape}")
     
        if not all_preds:
            print("No predictions were made. Check your data and model compatibility.")
            return
            
        all_preds = np.array(all_preds, dtype=np.float32)
        
        # Save with checkpoint name in the output file
        checkpoint_name = os.path.basename(CHECKPOINT_DIR)
        output_file = f"{OUTPUT_PREFIX}_{checkpoint_name}"
        
        # If we have targets, compute and display metrics
        if args.prediction_only or all(np.isnan(t) for t in all_tgts):
            print(f"\nPrediction-only mode: {len(all_preds)} predictions made")
            np.savez(output_file, preds=all_preds)
            print(f"Saved predictions → {output_file}.npz")
        else:
            all_tgts = np.array(all_tgts, dtype=np.float32)
            # Compute metrics only on non-NaN values
            valid_indices = ~np.isnan(all_tgts)
            valid_preds = all_preds[valid_indices]
            valid_tgts = all_tgts[valid_indices]
            
            if len(valid_tgts) > 0:
                mse  = np.mean((valid_preds-valid_tgts)**2)
                mae  = np.mean(np.abs(valid_preds-valid_tgts))
                corr = np.corrcoef(valid_preds, valid_tgts)[0,1] if len(valid_preds) > 1 else float('nan')
                
                # Add binary classification metrics
                binary_preds = (valid_preds > 0.5).astype(np.float32)
                binary_tgts = (valid_tgts > 0.5).astype(np.float32)
                accuracy = np.mean(binary_preds == binary_tgts)
                
                print(f"\nInference results:")
                print(f"  Number of samples processed: {len(valid_preds)}")
                print(f"  MSE: {mse:.6f}")
                print(f"  MAE: {mae:.6f}")
                print(f"  Corr: {corr:.4f}")
                print(f"  Binary accuracy: {accuracy:.4f}")
                
                np.savez(output_file, preds=all_preds, targets=all_tgts)
                print(f"Saved → {output_file}.npz")
                
                # Plotting only if we have valid targets
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 6))
                    plt.scatter(valid_tgts, valid_preds, alpha=0.5)
                    plt.plot([min(valid_tgts), max(valid_tgts)], [min(valid_tgts), max(valid_tgts)], 'r--')
                    plt.xlabel('True Values')
                    plt.ylabel('Predictions')
                    plt.title(f'Predictions vs True Values (Corr: {corr:.4f})')
                    plt.savefig(f"{output_file}_plot.png")
                    print(f"Saved plot → {output_file}_plot.png")
                except Exception as e:
                    print(f"Could not generate plot: {e}")
            else:
                print("No valid targets found. Saving predictions only.")
                np.savez(output_file, preds=all_preds)
                
    except Exception as e:
        import traceback
        print(f"Error during inference: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    run_inference()