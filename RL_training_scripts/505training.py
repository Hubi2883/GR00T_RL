import os
import torch.nn
import torch.nn.functional as F
import types
from transformers import TrainingArguments
from gr00t.model.gr00t_n1 import GR00T_N1
from gr00t.model.reward_model import GR00TReward, GR00TRewardConfig
from gr00t.model.reward_head import RewardHead, RewardHeadConfig
from gr00t.experiment.runner import TrainRunner
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform import VideoToTensor, VideoCrop, VideoResize, VideoColorJitter, VideoToNumpy
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.transform_reward import RewardTransform
from gr00t.model.transforms import GR00TTransform
from gr00t.data.reward_dataset import RewardDataset
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only use the first GPU
# Set paths for reward model training 
# Use a dataset with reward annotations - different from policy dataset

dataset_path = "/ceph/home/student.aau.dk/wb68dm/Isaac-GR00T_RL/Results_with_505_Reward"
output_dir   = "/ceph/home/student.aau.dk/wb68dm/reward_model_checkpoints"
base_model_path = "/ceph/home/student.aau.dk/wb68dm/Isaac-GR00T_RL/policy_training/checkpointpolicy/checkpoint-2500"  # Base GR00T model, NOT finetuned policy
embodiment_tag = "new_embodiment"

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define modality configurations
video_modality = ModalityConfig(
    delta_indices=[0],
    modality_keys=["video.ego_view"],
)

state_modality = ModalityConfig(
    delta_indices=[0],
    modality_keys=["state.left_wheel", "state.right_wheel", "state.acceleration"],
)

action_modality = ModalityConfig(
    delta_indices=[0],
    modality_keys=["action.wheel_commands"],
)

# Reward modality using next.reward key
reward_modality = ModalityConfig(
    delta_indices=[0],
    modality_keys=["next.reward"],  
)

language_modality = ModalityConfig(
    delta_indices=[0],
    modality_keys=["annotation.human.validity"],
)

modality_configs = {
    "video": video_modality,
    "state": state_modality,
    "action": action_modality,
    "next": reward_modality,
    "language": language_modality,
}

# Define transforms
transforms = ComposedModalityTransform(
    transforms=[
        # Video transforms
        VideoToTensor(apply_to=video_modality.modality_keys, backend="torchvision"),
        VideoCrop(apply_to=video_modality.modality_keys, scale=0.95, backend="torchvision"),
        VideoResize(
            apply_to=video_modality.modality_keys, 
            height=224, 
            width=224, 
            interpolation="linear", 
            backend="torchvision"
        ),
        VideoColorJitter(
            apply_to=video_modality.modality_keys, 
            brightness=0.3, 
            contrast=0.4, 
            saturation=0.5, 
            hue=0.08, 
            backend="torchvision"
        ),
        VideoToNumpy(apply_to=video_modality.modality_keys),
        
        # State transforms
        StateActionToTensor(apply_to=state_modality.modality_keys),
        StateActionTransform(
            apply_to=state_modality.modality_keys,
            normalization_modes={"state.acceleration": "min_max"},
        ),
        
        # Action transforms
        StateActionToTensor(apply_to=action_modality.modality_keys),
        StateActionTransform(
            apply_to=action_modality.modality_keys,
            normalization_modes={"action.wheel_commands": "min_max"},
        ),
        
        # Reward transform - special for reward model
        RewardTransform(
            apply_to=reward_modality.modality_keys,
            normalization_mode="min_max",
        ),
        
        # Concatenate modalities
        ConcatTransform(
            video_concat_order=video_modality.modality_keys,
            state_concat_order=state_modality.modality_keys,
            action_concat_order=action_modality.modality_keys,
        ),
        
        # GR00T-specific transform
        GR00TTransform(
            state_horizon=len(state_modality.delta_indices),
            action_horizon=len(action_modality.delta_indices),
            max_state_dim=64,
            max_action_dim=2,  # Two-wheel robot action dimension
        ),
    ]
)

# Create dataset with reward annotations
print(f"Loading reward dataset from: {dataset_path}")
train_dataset = RewardDataset(
    dataset_path=dataset_path,
    modality_configs=modality_configs,
    embodiment_tag=embodiment_tag,
    video_backend="torchvision_av",
    transforms=transforms,
)

# Add reward distribution counting functionality
def count_reward_distribution(dataset, sample_size=10000, report_interval=1000):
    """Count the distribution of binary reward values in the dataset using random sampling
    
    Args:
        dataset: The dataset to analyze
        sample_size: Number of random samples to check (default: 10000)
        report_interval: How often to print intermediate results (default: every 1000 samples)
    """
    import tqdm
    import random
    import time
    
    reward_counts = {}  # Dynamic dictionary for all observed rewards
    reward_range = {"min": float('inf'), "max": float('-inf')}  # Track range
    total_dataset_size = len(dataset)
    # Limit sample size to dataset size
    sample_size = min(sample_size, total_dataset_size)
    
    # Generate random indices to sample
    random_indices = random.sample(range(total_dataset_size), sample_size)
    
    print(f"\nCounting reward distribution using {sample_size} random samples...")
    start_time = time.time()
    
    for count, i in enumerate(tqdm.tqdm(random_indices)):
        sample = dataset[i]
        if "target_reward" in sample:
            # Extract target reward and convert to scalar
            reward = sample["target_reward"]
            if isinstance(reward, torch.Tensor):
                reward = reward.item()
            
            # Round to 0 or 1 in case of floating point values
            reward = round(reward)
            
            # Count it
            if reward in reward_counts:
                reward_counts[reward] += 1
            else:
                # For unexpected values
                reward_counts[reward] = 1
        
        # Print intermediate results at specified intervals
        if (count + 1) % report_interval == 0:
            # Calculate current percentages
            processed = count + 1
            current_percentages = {k: (v / processed) * 100 for k, v in reward_counts.items()}
            
            # Print intermediate results
            elapsed_time = time.time() - start_time
            print(f"\n--- Intermediate Results ({processed}/{sample_size} samples, {elapsed_time:.2f}s elapsed) ---")
            for reward, count in reward_counts.items():
                print(f"Reward {reward}: {count} samples ({current_percentages[reward]:.2f}%)")
            print("-------------------------------------")
    
    # Calculate final percentages
    samples_processed = sum(reward_counts.values())
    percentages = {k: (v / samples_processed) * 100 for k, v in reward_counts.items()}
    
    # Print final summary
    elapsed_time = time.time() - start_time
    print(f"\n===== Final Reward Distribution ({elapsed_time:.2f}s) =====")
    print(f"Total dataset size: {total_dataset_size}")
    print(f"Processed samples: {samples_processed}")
    for reward, count in reward_counts.items():
        print(f"Reward {reward}: {count} samples ({percentages[reward]:.2f}%)")
    print("==============================\n")
    
    return reward_counts, percentages

# Count reward distribution before training

reward_counts, reward_percentages = count_reward_distribution(
    train_dataset, 
    sample_size=100,  # Analyze 10,000 random samples - adjust as needed
    report_interval=100  # Report progress every 1,000 samples
)

# Load base GR00T model
print(f"Loading base GR00T model from: {base_model_path}")
base_model = GR00T_N1.from_pretrained(
    pretrained_model_name_or_path=base_model_path,
    tune_llm=False,
    tune_visual=False,
    tune_projector=False,
    tune_diffusion_model=False,# freeze the diffusion module (if you’re not using it)
    torch_dtype=torch.bfloat16,
)

# Create reward head config
reward_head_config = RewardHeadConfig(
    hidden_dim=1024*4,
    dropout=0.1,
    reward_dim=1,
    tune_projector=True
)

# Create reward model
reward_model = GR00TReward(base_model, reward_head_config)
print("Reward-head params require grad:",
      any(p.requires_grad for p in reward_model.reward_head.parameters()))


total_params = sum(p.numel() for p in reward_model.parameters())
trainable_params = sum(p.numel() for p in reward_model.parameters() if p.requires_grad)

print(f"Trainable parameters: {trainable_params} / {total_params} "
      f"({100 * trainable_params/total_params:.2f}%)")

# right after building reward_model, before wrapping:
print("Backbone grads? ", any(p.requires_grad for p in reward_model.backbone.parameters()))
print("Head   grads? ", any(p.requires_grad for p in reward_model.reward_head.parameters()))

# Set compute dtype explicitly
reward_model.compute_dtype = "bfloat16"
reward_model.config.compute_dtype = "bfloat16"
    
# Set training parameters
reward_model.set_trainable_parameters(
    tune_visual=False,
    tune_llm=False,
    tune_projector=True
)

# Fix prepare_input method
def fixed_prepare_input(self, inputs):
    """Fixed prepare_input that handles dtype conversion properly"""
    backbone_inputs = {}
    reward_inputs = {}
    
    # Get compute dtype directly

    compute_dtype = torch.bfloat16  # Explicitly set to bfloat16
    device = next(self.parameters()).device
    
    # Process backbone inputs
    for k, v in inputs.items():
        if k != "target_reward":
            if isinstance(v, torch.Tensor):
                if torch.is_floating_point(v):
                    backbone_inputs[k] = v.to(device=device, dtype=compute_dtype)
                else:
                    backbone_inputs[k] = v.to(device=device)
            else:
                backbone_inputs[k] = v
    
    # Add target reward if available
    if "target_reward" in inputs:
        if isinstance(inputs["target_reward"], torch.Tensor):
            if torch.is_floating_point(inputs["target_reward"]):
                reward_inputs["target_reward"] = inputs["target_reward"].to(device=device, dtype=compute_dtype)
            else:
                reward_inputs["target_reward"] = inputs["target_reward"].to(device=device)
        else:
            reward_inputs["target_reward"] = inputs["target_reward"]
    
    return backbone_inputs, reward_inputs

# Apply the fixed prepare_input method
reward_model.prepare_input = types.MethodType(fixed_prepare_input, reward_model)



class PairwiseFocalRewardWrapper(torch.nn.Module):
    def __init__(self, gr00t_reward, alpha=1, gamma=5.0, margin=1.0):
        super().__init__()
        self.model = gr00t_reward
        self.alpha = alpha
        self.gamma = gamma
        self.margin = margin


    def forward(self, inputs):
        target_raw = inputs.pop("target_reward").squeeze(-1)
        out = self.model(inputs)
        logits = out["reward_pred"].squeeze(-1)
        targets = target_raw.to(logits.device)

        B = logits.size(0)
        mse_loss = F.mse_loss(logits, targets)

        # 2. Improved pairwise comparison
        perm = torch.randperm(B, device=logits.device)
        logits_j = logits[perm]
        targets_j = targets[perm]
        
        # Find pairs with different target values
        mask = targets != targets_j
        idxs = mask.nonzero(as_tuple=True)[0]
        
        # If we have valid pairs, compute pairwise ranking loss
        if idxs.numel() > 0:
            x1 = logits[idxs]
            x2 = logits_j[idxs]
            pref = (targets[idxs] > targets_j[idxs]).float()
            y = (pref * 2 - 1).view(-1)  # +1 or -1
            
            # Lower margin often works better
            margin_loss = F.margin_ranking_loss(x1, x2, y, margin=0.4)
            
            # Weighted combination of losses
            loss = 0.5 * mse_loss + 0.5 * margin_loss
        else:
            # Fallback to MSE loss when no pairs have different targets
            # Add small regularization to prevent getting stuck
            l2_reg = 0.001 * sum(p.pow(2.0).sum() for p in self.model.reward_head.parameters())
            loss = mse_loss + l2_reg
        
        # If we have valid pairs, compute pairwise ranking losspred = logits

        return {"loss": loss}

    def save_pretrained(self, *args, **kwargs):
        return self.model.save_pretrained(*args, **kwargs)


reward_model = reward_model.to(torch.bfloat16)


# Apply wrapper ONCE
#reward_model = RewardModelWrapper(reward_model)
reward_model = PairwiseFocalRewardWrapper(reward_model)
total_params = sum(p.numel() for p in reward_model.parameters())
trainable_params = sum(p.numel() for p in reward_model.parameters() if p.requires_grad)

print(f"Trainable parameters: {trainable_params} / {total_params} "
      f"({100 * trainable_params/total_params:.2f}%)")
reward_model = reward_model.to("cuda")  # Important: move to GPU BEFORE DataParallel

for p in reward_model.parameters():
    if p.dtype == torch.bfloat16:          # keep bfloat16 for forward …
        p.data = p.data.float()            # … but store a FP32 master-copy


# Setup training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    run_name="reward_model_training",
    remove_unused_columns=False,
    gradient_checkpointing=False,
    bf16=True,
    fp16=False,                      # just to be explicit
    bf16_full_eval=False,            # eval in FP32 for stability
    tf32=True,
    max_grad_norm=1.0,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=2,
    dataloader_num_workers=2,
    dataloader_pin_memory=False,
    dataloader_persistent_workers=True,
    optim="adamw_torch",
    adam_beta1=0.95,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    learning_rate=1e-5,
    weight_decay=1e-5,
    warmup_ratio=0.0,
    lr_scheduler_type="cosine",
    logging_steps=1,
    max_steps=1000,
    save_strategy="steps",
    save_steps=200,
    evaluation_strategy="no",
    save_total_limit=5,
    seed=42,
    do_eval=False,
    ddp_find_unused_parameters=False,
    ddp_bucket_cap_mb=100,
    torch_compile_mode=None,
)

# Set up experiment runner
print("Setting up reward model experiment runner...")
experiment = TrainRunner(
    train_dataset=train_dataset,
    model=reward_model,
    training_args=training_args,
)

# Start training
print("\nStarting reward model training...")
experiment.train()

# --- after training finishes – KEEP the wrapper for training, but save the inner model
ckpt_dir = os.path.join(output_dir, "reward_model_final")
os.makedirs(ckpt_dir, exist_ok=True)

# 1-line HuggingFace-style directory (preferred)
reward_model.save_pretrained(ckpt_dir)

# OR – single .pt file (if you prefer)
torch.save(reward_model.model.state_dict(),
           os.path.join(output_dir, "full_reward_model.pt"))


