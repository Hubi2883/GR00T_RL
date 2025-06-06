"""Reward‑model training script for GR00T‑N1 that includes LoRA adaptation.
Fully expanded (no placeholders removed) so you can diff it line‑by‑line with your original.
"""

import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainingArguments

from gr00t.model.gr00t_n1 import GR00T_N1
from gr00t.model.reward_model import GR00TReward
from gr00t.model.reward_head import RewardHeadConfig
from gr00t.utils.peft import get_lora_model
from gr00t.experiment.runner import TrainRunner

from gr00t.data.dataset import ModalityConfig
from gr00t.data.reward_dataset import RewardDataset
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform import (
    VideoToTensor,
    VideoCrop,
    VideoResize,
    VideoColorJitter,
    VideoToNumpy,
)
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.transform_reward import RewardTransform
from gr00t.model.transforms import GR00TTransform

###############################################################################
# 0. Paths, device, and LoRA hyper‑parameters
###############################################################################

dataset_path = "/ceph/home/student.aau.dk/wb68dm/Isaac-GR00T_RL/Data_results_0001"
output_dir   = "/ceph/home/student.aau.dk/wb68dm/reward_model_checkpoints"
base_model_path = "nvidia/GR00T-N1-2B"  # base 2‑B parameter checkpoint
embodiment_tag  = "new_embodiment"

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------- LoRA params ----------
lora_rank    = 8   # 0 disables LoRA
lora_alpha   = 16
lora_dropout = 0.10
###############################################################################
# 1. Dataset definitions
###############################################################################

video_modality = ModalityConfig(delta_indices=[0], modality_keys=["video.ego_view"])
state_modality = ModalityConfig(delta_indices=[0], modality_keys=["state.acceleration"])
action_modality = ModalityConfig(delta_indices=[0], modality_keys=["action.wheel_commands"])
reward_modality = ModalityConfig(delta_indices=list(range(16)), modality_keys=["next.reward"])
language_modality = ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.validity"])

modality_configs = {
    "video": video_modality,
    "state": state_modality,
    "action": action_modality,
    "next": reward_modality,
    "language": language_modality,
}

# ----------- transforms ----------
transforms = ComposedModalityTransform(
    transforms=[
        # --- video ---
        VideoToTensor(apply_to=video_modality.modality_keys, backend="torchvision"),
        VideoCrop(apply_to=video_modality.modality_keys, scale=0.95, backend="torchvision"),
        VideoResize(apply_to=video_modality.modality_keys, height=224, width=224, interpolation="linear", backend="torchvision"),
        VideoColorJitter(apply_to=video_modality.modality_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08, backend="torchvision"),
        VideoToNumpy(apply_to=video_modality.modality_keys),

        # --- state ---
        StateActionToTensor(apply_to=state_modality.modality_keys),
        StateActionTransform(apply_to=state_modality.modality_keys, normalization_modes={"state.acceleration": "min_max"}),

        # --- action ---
        StateActionToTensor(apply_to=action_modality.modality_keys),
        StateActionTransform(apply_to=action_modality.modality_keys, normalization_modes={"action.wheel_commands": "min_max"}),

        # --- reward ---
        RewardTransform(apply_to=reward_modality.modality_keys, normalization_mode="min_max"),

        # --- concat + GR00T formatting ---
        ConcatTransform(
            video_concat_order=video_modality.modality_keys,
            state_concat_order=state_modality.modality_keys,
            action_concat_order=action_modality.modality_keys,
        ),
        GR00TTransform(
            state_horizon=len(state_modality.delta_indices),
            action_horizon=len(action_modality.delta_indices),
            max_state_dim=64,
            max_action_dim=2,
        ),
    ]
)

print(f"Loading reward dataset from: {dataset_path}")
train_dataset = RewardDataset(
    dataset_path=dataset_path,
    modality_configs=modality_configs,
    embodiment_tag=embodiment_tag,
    video_backend="torchvision_av",
    transforms=transforms,
)

###############################################################################
# 2. Model + LoRA
###############################################################################

print(f"Loading base GR00T‑N1 model from: {base_model_path}")
base_model: GR00T_N1 = GR00T_N1.from_pretrained(
    pretrained_model_name_or_path=base_model_path,
    tune_llm=False,
    tune_visual=True,
    tune_projector=True,
    torch_dtype=torch.bfloat16,
)

if lora_rank > 0:
    print(f"Applying LoRA ➜ rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")
    base_model = get_lora_model(
        base_model,
        rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
else:
    print("LoRA disabled (rank 0)")

# ----- reward head ------
reward_head_cfg = RewardHeadConfig(
    hidden_dim=2048,
    dropout=0.10,
    reward_dim=1,
    reward_horizon=16,
    tune_projector=True,
)

reward_model: GR00TReward = GR00TReward(base_model, reward_head_cfg)
reward_model.compute_dtype       = "bfloat16"
reward_model.config.compute_dtype = "bfloat16"
reward_model.set_trainable_parameters(
    tune_visual=True,
    tune_llm=False,
    tune_projector=True,
)

###############################################################################
# 3. Custom input handling (same logic as your original)
###############################################################################

def fixed_prepare_input(self, inputs):
    """Convert tensors to bfloat16 on the correct device and split reward/others."""
    backbone_inputs, reward_inputs = {}, {}
    compute_dtype = torch.bfloat16
    dev = next(self.parameters()).device

    for k, v in inputs.items():
        if k == "target_reward":
            # reward target stays separate
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                reward_inputs[k] = v.to(device=dev, dtype=compute_dtype)
            else:
                reward_inputs[k] = v.to(device=dev) if isinstance(v, torch.Tensor) else v
        else:
            # backbone inputs
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                backbone_inputs[k] = v.to(device=dev, dtype=compute_dtype)
            else:
                backbone_inputs[k] = v.to(device=dev) if isinstance(v, torch.Tensor) else v
    return backbone_inputs, reward_inputs

reward_model.prepare_input = types.MethodType(fixed_prepare_input, reward_model)

###############################################################################
# 4. Wrapper that always returns a dict with "loss"
###############################################################################

class RewardModelWrapper(nn.Module):
    def __init__(self, mdl: GR00TReward, use_focal_loss=True, focal_gamma=2.0, focal_alpha=0.35):
        super().__init__()
        self.model = mdl
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    def forward(self, inputs):
        outputs = self.model(inputs)
        # Case A: model already returns {"loss": ...}
        if isinstance(outputs, dict) and "loss" in outputs:
            return {"loss": outputs["loss"]}
        
        # Case B: compute loss between prediction and supplied target
        if isinstance(outputs, dict) and "reward_pred" in outputs and "target_reward" in inputs:
            pred, tgt = outputs["reward_pred"], inputs["target_reward"]
            
            # Ensure shapes match correctly for horizon prediction
            if len(pred.shape) == 3 and (len(tgt.shape) < 3 or tgt.shape[1] == 1):
                # If prediction is [batch_size, 16, 1] and target is [batch_size] or [batch_size, 1]
                if len(tgt.shape) == 1:
                    tgt = tgt.unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]
                elif len(tgt.shape) == 2:
                    tgt = tgt.unsqueeze(-1)  # [batch, 1, 1]
                
                # Expand target to all timesteps
                tgt = tgt.expand(-1, pred.shape[1], -1)  # [batch, 16, 1]
            
            # Ensure same device and dtype
            tgt = tgt.to(device=pred.device, dtype=pred.dtype)
            
            # Use focal loss if enabled
            if self.use_focal_loss:
                loss = focal_loss(pred, tgt, gamma=self.focal_gamma, alpha=self.focal_alpha)
            else:
                loss = F.mse_loss(pred, tgt)
                
            return {"loss": loss}
            
        # Case C: tensor loss directly
        if isinstance(outputs, torch.Tensor):
            return {"loss": outputs}
            
        # Fallback: zero-loss (shouldn't happen)
        return {"loss": torch.tensor(0.0, device=next(self.parameters()).device, requires_grad=True)}

    # Allow Trainer.save_model to work
    def save_pretrained(self, out_dir: str, state_dict=None):
        os.makedirs(out_dir, exist_ok=True)
        sd = state_dict or self.state_dict()
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(out_dir, state_dict=sd)
        else:
            torch.save(sd, os.path.join(out_dir, "pytorch_model.bin"))
            if hasattr(self.model, "config"):
                self.model.config.save_pretrained(out_dir)

reward_model = RewardModelWrapper(
    reward_model,
    use_focal_loss=True,
    focal_gamma=2.0,  # Focus on hard examples (higher = more focus)
    focal_alpha=0.35, # Slightly higher weight for positive class
).to(device)

###############################################################################
# 5. TrainingArguments (identical fields to original script, expanded)
###############################################################################

training_args = TrainingArguments(
    output_dir=output_dir,
    run_name="reward_model_with_lora",
    remove_unused_columns=False,

    # ---- precision ----
    bf16=True,
    tf32=True,

    # ---- batching ----
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,

    # ---- memory optimization ----
    gradient_checkpointing=True,

    # ---- dataloader ----
    dataloader_num_workers=2,
    dataloader_pin_memory=False,
    dataloader_persistent_workers=True,

    # ---- optimizer ----
    optim="adamw_torch",
    adam_beta1=0.95,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    learning_rate=1e-4,
    weight_decay=1e-5,

    # ---- schedule ----
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",

    # ---- logging / ckpt ----
    logging_steps=10,
    max_steps=500,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,

    # ---- misc ----
    evaluation_strategy="no",
    seed=42,
    do_eval=False,
    ddp_find_unused_parameters=False,
    ddp_bucket_cap_mb=100,
    torch_compile_mode=None,
)

###############################################################################
# 6. Train!
###############################################################################

print("\nSetting up experiment runner…")
experiment = TrainRunner(train_dataset=train_dataset, model=reward_model, training_args=training_args)
print("Starting reward‑model training…")
experiment.train()

###############################################################################
# 7. Final checkpoint
###############################################################################

print("\nSaving final model weights…")
os.makedirs(output_dir, exist_ok=True)
final_path = os.path.join(output_dir, "final_reward_model.pt")
torch.save(reward_model.state_dict(), final_path)
print(f"Model saved to → {final_path}")

# Add this function after imports section
def focal_loss(predictions, targets, gamma=2.0, alpha=0.25):
    """
    Focal loss for binary classification.
    
    Args:
        predictions: Raw logits from the model
        targets: Binary target values (0 or 1)
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Class weight parameter (higher = more weight to positive class)
        
    Returns:
        Calculated focal loss
    """
    # Apply sigmoid to get probabilities
    p = torch.sigmoid(predictions)
    
    # Calculate binary cross entropy
    ce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
    
    # Apply the focal term
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_term = (1 - p_t) ** gamma
    
    # Apply alpha weighting
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    
    # Combine all terms
    loss = alpha_t * focal_term * ce_loss
    
    return loss.mean()