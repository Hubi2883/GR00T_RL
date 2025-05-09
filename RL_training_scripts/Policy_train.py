import os
import torch
import torch.nn.functional as F
import types
from transformers import TrainingArguments
from gr00t.model.gr00t_n1 import GR00T_N1
from gr00t.experiment.runner import TrainRunner
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform import VideoToTensor, VideoCrop, VideoResize, VideoColorJitter, VideoToNumpy
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.model.transforms import GR00TTransform
from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead
from gr00t.model.action_head.cross_attention_dit import TimestepEncoder

# === Patch 2: Fix for Beta distribution sampling ===
original_sample_time = FlowmatchingActionHead.sample_time

def patched_sample_time(self, batch_size, device, dtype):
    # Always use float32 for the sampling part
    alpha = self.beta_dist.concentration1.to(torch.float32)
    beta = self.beta_dist.concentration0.to(torch.float32)
    float32_beta_dist = torch.distributions.Beta(alpha, beta)
    
    # Sample using float32
    sample = float32_beta_dist.sample([batch_size]).to(device)
    
    # Calculate time in float32 using the same formula as the original
    t = (self.config.noise_s - sample) / self.config.noise_s
    
    # Convert to the desired dtype before returning
    return t.to(dtype)

# Apply the patch
FlowmatchingActionHead.sample_time = patched_sample_time

# Set paths
dataset_path = "/ceph/home/student.aau.dk/wb68dm/Isaac-GR00T_RL/New_results_0001"
output_dir = "/ceph/home/student.aau.dk/wb68dm/two_wheel_checkpoints"
model_path = "nvidia/GR00T-N1-2B"
embodiment_tag = "new_embodiment"

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define modality configurations for two-wheel robot
video_modality = ModalityConfig(
    delta_indices=[0],
    modality_keys=["video.ego_view"],
)

state_modality = ModalityConfig(
    delta_indices=[0],
    modality_keys=["state.acceleration"],
)

action_modality = ModalityConfig(
    delta_indices=list(range(16)),
    modality_keys=["action.wheel_commands"],
)

language_modality = ModalityConfig(
    delta_indices=[0],
    modality_keys=["annotation.human.action.task_description"],
)

modality_configs = {
    "video": video_modality,
    "state": state_modality,
    "action": action_modality,
    "language": language_modality,
}

# Define transforms for data preprocessing
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
            max_action_dim=32,
        ),
    ]
)

# Create dataset
print(f"Loading dataset from: {dataset_path}")
train_dataset = LeRobotSingleDataset(
    dataset_path=dataset_path,
    modality_configs=modality_configs,
    embodiment_tag=embodiment_tag,
    video_backend="torchvision_av",  # Changed from torchvision_av to torchvision
    transforms=transforms,
)

# Load model with bfloat16 for Flash Attention compatibility
print(f"Loading model from: {model_path}")
model = GR00T_N1.from_pretrained(
    pretrained_model_name_or_path=model_path,
    tune_llm=False,
    tune_visual=True,
    tune_projector=True,
    tune_diffusion_model=True,
    torch_dtype=torch.bfloat16,  # Using bfloat16 for Flash Attention
)

# Move model to device


# === Patch 4: Fix for DataParallel output format ===
original_forward_class = GR00T_N1.forward

def wrapped_forward(self, inputs):
    # Call the original forward method
    outputs = original_forward_class(self, inputs)
    
    # BatchFeature handling - most common case
    if hasattr(outputs, "keys") and "loss" in outputs.keys():
        loss = outputs["loss"]
        # Return just a simple dictionary with the loss to avoid DataParallel issues
        return {"loss": loss}
    
    # Fallback cases
    elif isinstance(outputs, dict) and "loss" in outputs:
        return {"loss": outputs["loss"]}
    elif isinstance(outputs, torch.Tensor):
        return {"loss": outputs}
# Apply the patch
model.forward = types.MethodType(wrapped_forward, model)

model.compute_dtype = "bfloat16"
model.config.compute_dtype = "bfloat16"
model = model.to(device)

# Setup training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    run_name="two_wheel_finetune",
    remove_unused_columns=False,
    gradient_checkpointing=False,
    bf16=True,
    tf32=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    dataloader_num_workers=2,
    dataloader_pin_memory=False,
    dataloader_persistent_workers=True,
    optim="adamw_torch",
    adam_beta1=0.95,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    learning_rate=1e-4,
    weight_decay=1e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    max_steps=100,
    save_strategy="steps",
    save_steps=500,
    evaluation_strategy="no",
    save_total_limit=3,
    seed=42,
    do_eval=False,
    ddp_find_unused_parameters=False,
    ddp_bucket_cap_mb=100,
    torch_compile_mode=None,
)

# Set up experiment runner
print("Setting up experiment runner...")
experiment = TrainRunner(
    train_dataset=train_dataset,
    model=model,
    training_args=training_args,
)

# Start training
print("\nStarting training...")
experiment.train()