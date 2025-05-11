import os
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from accelerate import Accelerator

from gr00t.model.gr00t_n1 import GR00T_N1
from gr00t.model.reward_model import GR00TReward
from gr00t.model.reward_head import RewardHeadConfig
from gr00t.utils.peft import get_lora_model
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
from gr00t.experiment.runner import TrainRunner
from transformers import TrainingArguments

# ------------------- Config -------------------
dataset_path = "/ceph/home/student.aau.dk/wb68dm/Isaac-GR00T_RL/New_Data_WithBadEp"
base_model_path = "nvidia/GR00T-N1-2B"
embodiment_tag  = "new_embodiment"
lora_rank    = 16
lora_alpha   = 32
lora_dropout = 0.10
batch_size   = 8
num_workers  = 2

# ------------------- Accelerate Init -------------------
accelerator = Accelerator()
device = accelerator.device
print(f"Accelerator: {accelerator.state}")

# ------------------- Focal Loss -------------------
def focal_loss(predictions, targets, gamma=2.0, alpha=0.25):
    p = torch.sigmoid(predictions)
    ce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_term = (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * focal_term * ce_loss
    return loss.mean()

# ------------------- Dataset & Transforms -------------------
video_modality = ModalityConfig(delta_indices=[0], modality_keys=["video.ego_view"])
state_modality = ModalityConfig(delta_indices=[0], modality_keys=["state.acceleration"])
action_modality = ModalityConfig(delta_indices=[0], modality_keys=["action.wheel_commands"])
reward_modality = ModalityConfig(delta_indices=[0], modality_keys=["next.reward"])
language_modality = ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.validity"])

modality_configs = {
    "video": video_modality,
    "state": state_modality,
    "action": action_modality,
    "next": reward_modality,
    "language": language_modality,
}

transforms = ComposedModalityTransform(
    transforms=[
        VideoToTensor(apply_to=video_modality.modality_keys, backend="torchvision"),
        VideoCrop(apply_to=video_modality.modality_keys, scale=0.95, backend="torchvision"),
        VideoResize(apply_to=video_modality.modality_keys, height=224, width=224, interpolation="linear", backend="torchvision"),
        VideoColorJitter(apply_to=video_modality.modality_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08, backend="torchvision"),
        VideoToNumpy(apply_to=video_modality.modality_keys),
        StateActionToTensor(apply_to=state_modality.modality_keys),
        StateActionTransform(apply_to=state_modality.modality_keys, normalization_modes={"state.acceleration": "min_max"}),
        StateActionToTensor(apply_to=action_modality.modality_keys),
        StateActionTransform(apply_to=action_modality.modality_keys, normalization_modes={"action.wheel_commands": "min_max"}),
        RewardTransform(apply_to=reward_modality.modality_keys, normalization_mode="min_max"),
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

# ------------------- Model + LoRA -------------------
print(f"Loading base GR00T-N1 model from: {base_model_path}")
base_model = GR00T_N1.from_pretrained(
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

reward_head_cfg = RewardHeadConfig(
    hidden_dim=512,
    dropout=0.10,
    reward_dim=1,
    tune_projector=True,
)
reward_model_core = GR00TReward(base_model, reward_head_cfg)
reward_model_core.compute_dtype = "bfloat16"
reward_model_core.config.compute_dtype = "bfloat16"
reward_model_core.set_trainable_parameters(
    tune_visual=True,
    tune_llm=False,
    tune_projector=True,
)

def fixed_prepare_input(self, inputs):
    backbone_inputs, reward_inputs = {}, {}
    compute_dtype = torch.bfloat16
    dev = next(self.parameters()).device
    for k, v in inputs.items():
        if k == "target_reward":
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                reward_inputs[k] = v.to(device=dev, dtype=compute_dtype)
            else:
                reward_inputs[k] = v.to(device=dev) if isinstance(v, torch.Tensor) else v
        else:
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                backbone_inputs[k] = v.to(device=dev, dtype=compute_dtype)
            else:
                backbone_inputs[k] = v.to(device=dev) if isinstance(v, torch.Tensor) else v
    return backbone_inputs, reward_inputs
reward_model_core.prepare_input = types.MethodType(fixed_prepare_input, reward_model_core)

# ------------------- Model Wrapper -------------------
class RewardModelWrapper(nn.Module):
    def __init__(self, mdl, use_focal_loss=True, focal_gamma=2.0, focal_alpha=0.35):
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
            while len(tgt.shape) > len(pred.shape):
                tgt = tgt.squeeze(-1)
            if len(pred.shape) > len(tgt.shape):
                tgt = tgt.view(-1, 1).expand(-1, pred.shape[1])
                tgt = tgt.to(device=pred.device, dtype=pred.dtype)
                if self.use_focal_loss:
                    loss = focal_loss(
                        pred, tgt,
                        gamma=self.focal_gamma,
                        alpha=self.focal_alpha
                    )
                else:
                    loss = F.mse_loss(pred, tgt)
                return {"loss": loss}
        # Case C: tensor loss directly
        if isinstance(outputs, dict) and "reward_pred" in outputs and "target_reward" in inputs:
            pred, tgt = outputs["reward_pred"], inputs["target_reward"]
            if pred.dim() > tgt.dim():
                tgt = tgt.unsqueeze(-1).expand(-1, pred.shape[-1])
            tgt = tgt.to(device=pred.device, dtype=pred.dtype)
            if self.use_focal_loss:
                loss = focal_loss(pred, tgt, gamma=self.focal_gamma, alpha=self.focal_alpha)
            else:
                loss = F.mse_loss(pred, tgt)
            return {"loss": loss}
        # Fallback: zero-loss (shouldn't happen)
        return {"loss": torch.tensor(0.0, device=next(self.parameters()).device, requires_grad=True)}

reward_model = RewardModelWrapper(
    reward_model_core,
    use_focal_loss=True,
    focal_gamma=2.0,
    focal_alpha=0.35,
).to(device)

# ------------------- DataLoader (use Trainer's DataLoader for exact match) -------------------
# This ensures the batch collation and input shapes are identical to the training script.
dummy_args = TrainingArguments(
    output_dir="/tmp/lr_finder_tmp",
    per_device_train_batch_size=batch_size,
    remove_unused_columns=False,
    dataloader_num_workers=num_workers,
    dataloader_pin_memory=False,
    fp16=False,
    logging_steps=5000,  # silence
)
dummy_runner = TrainRunner(
    train_dataset=train_dataset,
    model=reward_model,
    training_args=dummy_args,
)
train_loader = dummy_runner.trainer.get_train_dataloader()

# Prepare model, optimizer, and dataloader with accelerator
optimizer = torch.optim.AdamW(
    reward_model.parameters(),
    lr=1e-7,  # initial lr, will be changed by LR finder
    betas=(0.95, 0.999),
    eps=1e-8,
    weight_decay=1e-5,
)
reward_model, optimizer, train_loader = accelerator.prepare(reward_model, optimizer, train_loader)

# ------------------- LR Finder Function (Cyclical LR) -------------------
def clr_range_test(model, optimizer, train_loader, device, min_lr=1e-7, max_lr=1, cycle_length=100, num_cycles=3, smooth_f=0.05, diverge_th=5, grad_noise=1e-6, warmup_iters=10, policy="triangular"):
    """
    Runs a cyclical learning rate range test.
    
    Args:
        model: The model to test
        optimizer: The optimizer to use
        train_loader: DataLoader for training data
        device: Device to train on
        min_lr: Minimum learning rate
        max_lr: Maximum learning rate
        cycle_length: Length of each cycle in iterations
        num_cycles: Number of cycles to run
        smooth_f: Smoothing factor for loss
        diverge_th: Divergence threshold (multiplier of best loss)
        grad_noise: Standard deviation for gradient noise (0 to disable)
        warmup_iters: Number of warmup iterations before starting the test
        policy: LR policy, one of ["triangular", "exp", "cosine"]
    
    Returns:
        Tuple of (learning_rates, losses)
    """
    model.train()
    lrs, losses = [], []
    best_loss = float('inf')
    smoothed_loss = 0
    total_iters = cycle_length * num_cycles
    for param_group in optimizer.param_groups:
        param_group['lr'] = min_lr
    loader_iter = iter(train_loader)
    
    # Warmup phase to stabilize the loss before recording values
    print(f"Running {warmup_iters} warmup iterations before starting LR test...")
    for _ in range(warmup_iters):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)
        optimizer.zero_grad()
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        backbone_inputs, reward_inputs = model.module.model.prepare_input(batch)
        inputs = {**backbone_inputs, **reward_inputs}
        outputs = model(inputs)
        loss = outputs if isinstance(outputs, torch.Tensor) else outputs["loss"]
        accelerator.backward(loss)
        optimizer.step()
    
    # Improved divergence detection
    loss_history = []
    divergence_streak = 0
    slope_history = []
    
    for it in range(total_iters):
        # Calculate learning rate based on policy
        cycle_pos = it % cycle_length
        cycle_progress = cycle_pos / cycle_length
        
        if policy == "triangular":
            # Triangular LR schedule
            half_cycle = cycle_length // 2
            if cycle_pos < half_cycle:
                lr = min_lr + (max_lr - min_lr) * (cycle_pos / half_cycle)
            else:
                lr = max_lr - (max_lr - min_lr) * ((cycle_pos - half_cycle) / half_cycle)
        elif policy == "exp":
            # Exponential schedule (slower ramp up, faster ramp down)
            if cycle_progress < 0.7:  # Longer ramp-up (70% of cycle)
                p = cycle_progress / 0.7
                lr = min_lr * (max_lr/min_lr) ** p
            else:  # Faster ramp-down (30% of cycle)
                p = (cycle_progress - 0.7) / 0.3
                decay_factor = 0.1  # Decay to 10% of max
                lr = max_lr * (decay_factor ** p)
        elif policy == "cosine":
            # Cosine schedule
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * (1 - cycle_progress)))
        else:
            # Default to triangular
            half_cycle = cycle_length // 2
            if cycle_pos < half_cycle:
                lr = min_lr + (max_lr - min_lr) * (cycle_pos / half_cycle)
            else:
                lr = max_lr - (max_lr - min_lr) * ((cycle_pos - half_cycle) / half_cycle)
        
        # Apply learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)
            
        optimizer.zero_grad()
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        # Use .module.model.prepare_input for Accelerate/DDP compatibility
        backbone_inputs, reward_inputs = model.module.model.prepare_input(batch)
        inputs = {**backbone_inputs, **reward_inputs}
        outputs = model(inputs)
        loss = outputs if isinstance(outputs, torch.Tensor) else outputs["loss"]
        
        # Check for NaN/Inf
        if not torch.isfinite(loss).all():
            print(f"Stopping early: NaN or Inf detected at LR={lr:.2e}")
            break
            
        accelerator.backward(loss)
        
        # Add gradient noise for stability if enabled
        if grad_noise > 0:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.add_(torch.randn_like(param.grad) * grad_noise)
                    
        optimizer.step()
        lrs.append(lr)
        
        # Robust loss tracking
        loss_val = loss.item()
        loss_history.append(loss_val)
        if it == 0:
            smoothed_loss = loss_val
        else:
            smoothed_loss = smoothed_loss * (1 - smooth_f) + loss_val * smooth_f
        losses.append(smoothed_loss)
        
        # Update best loss
        if smoothed_loss < best_loss:
            best_loss = smoothed_loss
            divergence_streak = 0  # Reset streak when improvement found
        
        # Multiple divergence criteria
        # 1. Simple threshold exceeded
        simple_diverged = smoothed_loss > best_loss * diverge_th
        
        # 2. Consistent increase over multiple iterations
        consistent_increase = False
        if len(loss_history) >= 5:
            window = loss_history[-5:]
            if all(window[i] > window[i-1] for i in range(1, len(window))):
                divergence_streak += 1
                if divergence_streak >= 3:  # Requires 3 consecutive windows of increases
                    consistent_increase = True
            else:
                divergence_streak = max(0, divergence_streak - 1)  # Decay streak counter
        
        # 3. Compute local slope on smoothed loss (if enough points)
        if len(losses) >= 10:
            recent_lrs = np.log(lrs[-10:])
            recent_losses = losses[-10:]
            try:
                slope = np.polyfit(recent_lrs, recent_losses, deg=1)[0]
                slope_history.append(slope)
                rapid_increase = (len(slope_history) >= 3 and 
                                  all(s > 1.0 for s in slope_history[-3:]))
            except:
                rapid_increase = False
        else:
            rapid_increase = False
            
        # Combine divergence criteria
        if simple_diverged or consistent_increase or rapid_increase:
            reason = []
            if simple_diverged: reason.append("threshold exceeded")
            if consistent_increase: reason.append("consistent increases")
            if rapid_increase: reason.append("rapid slope increase")
            print(f"Stopping early: Loss is diverging ({', '.join(reason)}) at LR={lr:.2e}")
            break
            
        if (it + 1) % 20 == 0:
            print(f"  Iter {it+1:3d}/{total_iters} • LR={lr:.2e} • loss={loss_val:.4f} • smoothed={smoothed_loss:.4f}")
    print("Cyclical learning rate range test complete.")
    return np.array(lrs), np.array(losses)

# ------------------- Main LR Finder Logic (Cyclical) -------------------
# Get number of batches in DataLoader
num_batches = len(train_loader)
print(f"train dataloader length: {num_batches}")

# LR Finder configuration
min_lr = 1e-7
max_lr = 1
cycle_policy = "triangular"  # Options: triangular, exp, cosine
confidence_level = "moderate"  # Options: conservative, moderate, aggressive

# Cycle length and number setup
user_cycle_length = None  # set to int to override
user_num_cycles = None    # set to int to override
if user_cycle_length is not None and user_num_cycles is not None:
    cycle_length = user_cycle_length
    num_cycles = user_num_cycles
else:
    cycle_length = num_batches
    num_cycles = 1 if cycle_policy == "triangular" else 3  # More cycles for non-triangular
    print(f"[LRFinder] Auto-setting cycle_length={cycle_length}, num_cycles={num_cycles} to cover the full dataset once.")
    print(f"[LRFinder] Total iterations: {cycle_length * num_cycles}")

# Add robust options
warmup_iters = min(100, int(0.1 * cycle_length))  # 10% of cycle length for warmup
grad_noise = 1e-6 if confidence_level == "conservative" else 0  # Only add noise in conservative mode
diverge_th = 4 if confidence_level == "conservative" else (5 if confidence_level == "moderate" else 8)

print(f"\nRunning cyclical learning rate range test with {cycle_policy} policy...")
print(f"Configuration: min_lr={min_lr}, max_lr={max_lr}, {confidence_level} confidence level")
print(f"Warmup: {warmup_iters} iterations, gradient noise: {grad_noise}")

lrs, losses = clr_range_test(
    reward_model,
    optimizer,
    train_loader,
    device,
    min_lr=min_lr,
    max_lr=max_lr,
    cycle_length=cycle_length,
    num_cycles=num_cycles,
    smooth_f=0.05,
    diverge_th=diverge_th,
    grad_noise=grad_noise,
    warmup_iters=warmup_iters,
    policy=cycle_policy,
)

# Plot LR vs. loss
plt.figure(figsize=(12, 6))
plt.plot(lrs, losses, label='Loss')
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title(f'Cyclical Learning Rate Finder ({cycle_policy.title()} Policy)')
plt.grid(True, alpha=0.3)

# More robust smoothing for analysis
def savgol_or_gaussian_filter(data, window_size=11, poly_order=3):
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(data, window_size, poly_order)
    except (ImportError, ValueError):
        # Fall back to gaussian convolution
        if len(data) < window_size:
            return data  # Not enough data points
        gaussian_kernel = np.exp(-0.5 * ((np.arange(window_size) - window_size // 2) / (window_size / 4)) ** 2)
        gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
        pad_size = window_size // 2
        padded_data = np.pad(data, (pad_size, pad_size), mode='edge')
        smoothed = np.convolve(padded_data, gaussian_kernel, mode='valid')
        return smoothed

# Suggest LR using multiple methods
if len(lrs) > 20:  # Need enough data points
    # Apply more robust smoothing
    smoothed_losses = savgol_or_gaussian_filter(losses, window_size=min(21, len(losses) // 5 * 2 + 1))
    plt.plot(lrs, smoothed_losses, 'r-', alpha=0.6, label='Smoothed Loss')
    
    # Method 1: Steepest gradient (use smoothed loss)
    gradients = np.gradient(smoothed_losses) / np.gradient(np.log10(lrs))
    min_grad_idx = np.argmin(gradients)
    suggested_lr_steep = lrs[min_grad_idx]
    
    # Method 2: Find point with maximum curvature (second derivative)
    try:
        from scipy.signal import savgol_filter
        second_deriv = savgol_filter(smoothed_losses, min(21, len(losses) // 5 * 2 + 1), 2, deriv=2)
        max_curv_idx = np.argmax(np.abs(second_deriv))
        suggested_lr_curv = lrs[max_curv_idx]
        plt.scatter([suggested_lr_curv], [losses[max_curv_idx]], color='purple', s=100, label=f'Max Curvature: {suggested_lr_curv:.2e}')
    except (ImportError, ValueError):
        suggested_lr_curv = suggested_lr_steep  # Fallback
    
    # Method 3: Min loss
    min_loss_idx = np.argmin(smoothed_losses)
    suggested_lr_min = lrs[min_loss_idx]
    
    # Method 4: Loss valley beginning (where loss starts to decrease significantly)
    valley_indices = []
    for i in range(15, len(smoothed_losses) - 5):
        if (smoothed_losses[i] < smoothed_losses[i-10:i].mean() * 0.95) and (gradients[i] < -0.01):
            valley_indices.append(i)
    
    if valley_indices:
        valley_idx = valley_indices[0]  # First significant drop
        suggested_lr_valley = lrs[valley_idx]
        plt.scatter([suggested_lr_valley], [losses[valley_idx]], color='blue', s=100, label=f'Valley Start: {suggested_lr_valley:.2e}')
    else:
        suggested_lr_valley = suggested_lr_steep / 3  # Fallback
        
    # Plot main suggestions
    plt.scatter([suggested_lr_steep], [losses[min_grad_idx]], color='red', s=100, label=f'Steepest Drop: {suggested_lr_steep:.2e}')
    plt.scatter([suggested_lr_min], [losses[min_loss_idx]], color='green', s=100, label=f'Min Loss: {suggested_lr_min:.2e}')
    
    # Conservative / aggressive recommendations
    conservative_lr = min(suggested_lr_steep/10, suggested_lr_min/10, suggested_lr_valley/3)
    moderate_lr = min(suggested_lr_steep/3, suggested_lr_valley)
    aggressive_lr = min(suggested_lr_steep, suggested_lr_min/3)
    
    print(f"\n=== LEARNING RATE RECOMMENDATIONS ===")
    print(f"Conservative LR: {conservative_lr:.2e}")
    print(f"Moderate LR:     {moderate_lr:.2e}")
    print(f"Aggressive LR:   {aggressive_lr:.2e}")
    print(f"===================================")
    print(f"Analytical points:")
    print(f"- Steepest gradient: {suggested_lr_steep:.2e}")
    print(f"- Min loss point:    {suggested_lr_min:.2e}")
    print(f"- Valley beginning:  {suggested_lr_valley:.2e}")
    if 'suggested_lr_curv' in locals():
        print(f"- Max curvature:    {suggested_lr_curv:.2e}")
    
    plt.legend()
else:
    print("Not enough points to suggest a learning rate.")

plt.show()

print("\nDone. No training performed.\n") 