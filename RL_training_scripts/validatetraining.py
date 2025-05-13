# This script is now Accelerate-compatible.
import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainingArguments
from accelerate import Accelerator

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

dataset_path = "/ceph/home/student.aau.dk/wb68dm/Isaac-GR00T_RL/New_Data_WithBadEp"
output_dir   = "/ceph/home/student.aau.dk/wb68dm/reward_model_checkpoints"
base_model_path = "nvidia/GR00T-N1-2B"  # base 2‑B parameter checkpoint
embodiment_tag  = "new_embodiment"
validation_split = 0.1  # Use 10% of data for validation

# Initialize accelerator with simpler configuration
accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=2,
    # Avoid distributed training features that might cause issues
    deepspeed_plugin=None,
    # Set log_with=None to disable extra loggers that might cause issues
    log_with=None
)
print(f"Accelerator state: {accelerator.state}")

# ---------- LoRA params ----------
lora_rank    = 16   # 0 disables LoRA
lora_alpha   = 8
lora_dropout = 0.10


# Add this function after imports section
def focal_loss(predictions, targets, gamma=2.0, alpha=0.1):
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
full_dataset = RewardDataset(
    dataset_path=dataset_path,
    modality_configs=modality_configs,
    embodiment_tag=embodiment_tag,
    video_backend="torchvision_av",
    transforms=transforms,
)

# Split into train and validation
dataset_size = len(full_dataset)
val_size = int(dataset_size * validation_split)
train_size = dataset_size - val_size
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, 
    [train_size, val_size], 
    generator=torch.Generator().manual_seed(42)
)
print(f"Dataset split: {train_size} training samples, {val_size} validation samples")

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
    def __init__(self, mdl: GR00TReward, use_focal_loss=True, focal_gamma=2.0, focal_alpha=0.25):
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
            
            # Use focal loss - always enabled for this script
            loss = focal_loss(pred, tgt, gamma=self.focal_gamma, alpha=self.focal_alpha)
                
            return {"loss": loss}
            
        # Fallback: zero-loss (shouldn't happen)
        return {"loss": torch.tensor(0.0, device=next(self.parameters()).device, requires_grad=True)}

    def validation_loss_fn(self, inputs):
        """Compute validation loss with the same logic as forward but no gradients."""
        with torch.no_grad():
            outputs = self.model(inputs)
            
            if isinstance(outputs, dict) and "reward_pred" in outputs and "target_reward" in inputs:
                pred, tgt = outputs["reward_pred"], inputs["target_reward"]
                
                # Ensure shapes match correctly for horizon prediction
                if len(pred.shape) == 3 and (len(tgt.shape) < 3 or tgt.shape[1] == 1):
                    if len(tgt.shape) == 1:
                        tgt = tgt.unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]
                    elif len(tgt.shape) == 2:
                        tgt = tgt.unsqueeze(-1)  # [batch, 1, 1]
                    
                    # Expand target to all timesteps
                    tgt = tgt.expand(-1, pred.shape[1], -1)  # [batch, 16, 1]
                
                tgt = tgt.to(device=pred.device, dtype=pred.dtype)
                
                # Calculate both focal loss and BCE loss for comparison
                focal = focal_loss(pred, tgt, gamma=self.focal_gamma, alpha=self.focal_alpha)
                bce = F.binary_cross_entropy_with_logits(pred, tgt)
                
                # Also calculate accuracy for binary classification
                pred_binary = (torch.sigmoid(pred) > 0.5).float()
                accuracy = (pred_binary == tgt).float().mean()
                
                return {
                    "loss": focal,
                    "focal_loss": focal,
                    "bce_loss": bce,
                    "accuracy": accuracy
                }
            
            # Fallback
            return {"loss": torch.tensor(0.0, device=next(self.parameters()).device)}

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
    
    # Forward gradient checkpointing methods to the wrapped model
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the wrapped model if supported."""
        try:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                return self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
            else:
                print("Warning: GR00TReward doesn't support gradient_checkpointing_enable. Continuing without it.")
        except Exception as e:
            print(f"Warning: Failed to enable gradient checkpointing: {e}. Continuing without it.")
        return self
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the wrapped model if supported."""
        try:
            if hasattr(self.model, "gradient_checkpointing_disable"):
                return self.model.gradient_checkpointing_disable()
            else:
                print("Warning: GR00TReward doesn't support gradient_checkpointing_disable. Continuing without it.")
        except Exception as e:
            print(f"Warning: Failed to disable gradient checkpointing: {e}. Continuing without it.")
        return self
    
    # Forward other common attributes that might be accessed during training
    def __getattr__(self, name):
        """Forward attribute access to the wrapped model for attributes not in the wrapper."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if hasattr(self.model, name):
                return getattr(self.model, name)
            raise

reward_model = RewardModelWrapper(
    reward_model,
    use_focal_loss=True,
    focal_gamma=1.0,  # Focus on hard examples (higher = more focus)
    focal_alpha=0.35, # Slightly higher weight for positive class (adjust based on dataset)
).to(accelerator.device)

###############################################################################
# 5. TrainingArguments (modified for simpler training)
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
    gradient_checkpointing=False,  # Completely disable gradient checkpointing

    # ---- dataloader ----
    dataloader_num_workers=2,
    dataloader_pin_memory=False,
    dataloader_persistent_workers=True,

    # ---- optimizer ----
    optim="adamw_torch",
    adam_beta1=0.95,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    learning_rate=0.000009,
    weight_decay=1e-5,

    # ---- schedule ----
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",

    # ---- logging / ckpt ----
    logging_steps=50,
    max_steps=2500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,

    # ---- evaluation ----
    evaluation_strategy="no",  # Disable built-in evaluation - we'll do our own
    do_eval=False,             # Disable built-in evaluation
    
    # ---- misc ----
    seed=42,
    # Disable distributed training features that might cause problems
    ddp_find_unused_parameters=False,
    ddp_bucket_cap_mb=100,
    torch_compile_mode=None,
    # Disable report_to to avoid extra loggers that might cause issues
    report_to=[],
)

###############################################################################
# 6. Train!
###############################################################################

print("\nSetting up experiment runner…")
# Create a custom dataset wrapper that preserves the original dataset's attributes
class SubsetDatasetWrapper:
    """Wrapper around a Subset to expose the original dataset's attributes."""
    
    def __init__(self, subset, original_dataset):
        self.subset = subset
        self.original_dataset = original_dataset
        
    def __getattr__(self, name):
        # First check if the subset has the attribute
        if hasattr(self.subset, name):
            return getattr(self.subset, name)
        # Then check the original dataset
        elif hasattr(self.original_dataset, name):
            return getattr(self.original_dataset, name)
        # Otherwise, raise AttributeError
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __getitem__(self, idx):
        return self.subset[idx]
    
    def __len__(self):
        return len(self.subset)

# Wrap the training dataset to preserve original attributes
train_dataset_wrapped = SubsetDatasetWrapper(train_dataset, full_dataset)

# Initialize TrainRunner with only the wrapped training dataset
experiment = TrainRunner(
    train_dataset=train_dataset_wrapped,
    model=reward_model, 
    training_args=training_args
)

# Create a custom validation function that will be called periodically
def validate_model(model, val_dataset, device, batch_size=8):
    """Run validation on the validation dataset"""
    model.eval()
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    total_samples = 0
    metrics = {
        "val_loss": 0.0,
        "val_focal_loss": 0.0,
        "val_bce_loss": 0.0,
        "val_accuracy": 0.0
    }
    
    print(f"\nRunning validation on {len(val_dataset)} samples...")
    with torch.no_grad():
        for batch in val_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # Get validation metrics
            batch_metrics = model.validation_loss_fn(batch)
            batch_size = batch.get("target_reward", next(iter(batch.values()))).size(0)
            
            # Accumulate metrics
            metrics["val_loss"] += batch_metrics["loss"].item() * batch_size
            metrics["val_focal_loss"] += batch_metrics["focal_loss"].item() * batch_size
            metrics["val_bce_loss"] += batch_metrics["bce_loss"].item() * batch_size
            metrics["val_accuracy"] += batch_metrics["accuracy"].item() * batch_size
            
            total_samples += batch_size
    
    # Compute average metrics
    for k in metrics:
        metrics[k] /= total_samples
    
    print(f"Validation results: Loss: {metrics['val_loss']:.4f}, Accuracy: {metrics['val_accuracy']:.4f}")
    return metrics

# Add a callback to run validation at specific steps
original_train = experiment.train

def train_with_validation():
    """Wrapper around the original train method to add validation"""
    # Set up tracking for best model
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, "best_model.pt")
    
    # Setup step counter to track validation intervals
    step_counter = 0
    validation_interval = training_args.eval_steps
    
    # Define a callback for logging
    def log_callback(logs):
        nonlocal step_counter, best_val_loss
        step_counter += 1
        
        # Run validation at specified intervals
        if step_counter % validation_interval == 0:
            print(f"\n--- Running validation at step {step_counter} ---")
            val_metrics = validate_model(
                reward_model,
                val_dataset,
                accelerator.device,
                batch_size=training_args.per_device_eval_batch_size
            )
            
            # Save the best model
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                torch.save(reward_model.state_dict(), best_model_path)
                print(f"New best model saved with validation loss: {best_val_loss:.4f}")
    
    # Train the model with our callback
    original_train()
    
    # Run final validation
    print("\nPerforming final validation...")
    final_metrics = validate_model(
        reward_model,
        val_dataset,
        accelerator.device,
        batch_size=training_args.per_device_eval_batch_size
    )
    
    # Load the best model for the final save
    if os.path.exists(best_model_path):
        print(f"Loading best model (validation loss: {best_val_loss:.4f})")
        reward_model.load_state_dict(torch.load(best_model_path))
    
    return final_metrics

# Replace the original train method with our wrapped version
experiment.train = train_with_validation

print("Starting reward‑model training with validation…")
final_metrics = experiment.train()

###############################################################################
# 7. Final checkpoint
###############################################################################

print("\nSaving final model weights…")
os.makedirs(output_dir, exist_ok=True)
final_path = os.path.join(output_dir, "final_reward_model.pt")
torch.save(reward_model.state_dict(), final_path)
print(f"Model saved to → {final_path}")
print(f"Final validation metrics: {final_metrics}")