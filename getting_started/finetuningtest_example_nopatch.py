import os
import torch
from gr00t.data.schema import EmbodimentTag
from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform import VideoToTensor, VideoCrop, VideoResize, VideoColorJitter, VideoToNumpy
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.model.transforms import GR00TTransform
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.gr00t_n1 import GR00T_N1
from transformers import TrainingArguments
from gr00t.experiment.runner import TrainRunner

# Define dataset path and embodiment tag
dataset_path = "/ceph/home/student.aau.dk/wb68dm/Isaac-GR00T_RL/New_results_0001"
output_dir = "/ceph/home/student.aau.dk/wb68dm/two_wheel_checkpoints"
embodiment_tag = "new_embodiment"

# Define modality configurations
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

# Define transforms
to_apply_transforms = ComposedModalityTransform(
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
    video_backend="torchvision_av",
    transforms=to_apply_transforms,
)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
BASE_MODEL_PATH = "nvidia/GR00T-N1-2B"
TUNE_LLM = False
TUNE_VISUAL = True
TUNE_PROJECTOR = True
TUNE_DIFFUSION_MODEL = True

print(f"Loading model from: {BASE_MODEL_PATH}")
model = GR00T_N1.from_pretrained(
    pretrained_model_name_or_path=BASE_MODEL_PATH,
    tune_llm=TUNE_LLM,
    tune_visual=TUNE_VISUAL,
    tune_projector=TUNE_PROJECTOR,
    tune_diffusion_model=TUNE_DIFFUSION_MODEL,
    torch_dtype=torch.float32,  # Try with float32 instead of bfloat16
)

# Move model to device
model = model.to(device)

# Setup training arguments
per_device_train_batch_size = 8
max_steps = 2000
dataloader_num_workers = 8

print(f"Setting up training - output will be saved to: {output_dir}")
training_args = TrainingArguments(
    output_dir=output_dir,
    run_name="two_wheel_finetune",
    remove_unused_columns=False,
    gradient_checkpointing=False,
    bf16=False,  # Disable mixed precision training
    tf32=True,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=1,
    dataloader_num_workers=dataloader_num_workers,
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
    max_steps=max_steps,
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
try:
    experiment.train()
except Exception as e:
    print(f"Training error: {e}")
    import traceback
    traceback.print_exc()