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
dataset_path = "/ceph/home/student.aau.dk/wb68dm/Isaac-GR00T_RL/New_results_0002action"  
output_dir = "/ceph/home/student.aau.dk/wb68dm/reward_model_checkpoints"
base_model_path = "nvidia/GR00T-N1-2B"  # Base GR00T model, NOT finetuned policy
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
    modality_keys=["state.acceleration"],
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
    modality_keys=["annotation.human.action.task_description"],
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
            normalization_mode="min_max"
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

# Load base GR00T model
print(f"Loading base GR00T model from: {base_model_path}")
base_model = GR00T_N1.from_pretrained(
    pretrained_model_name_or_path=base_model_path,
    tune_llm=False,
    tune_visual=True,
    tune_projector=True,
    torch_dtype=torch.bfloat16,
)

# Create reward head config
reward_head_config = RewardHeadConfig(
    hidden_dim=512,
    dropout=0.1,
    reward_dim=1,
    tune_projector=True
)

# Create reward model
reward_model = GR00TReward(base_model, reward_head_config)

# Set compute dtype explicitly
reward_model.compute_dtype = "bfloat16"
reward_model.config.compute_dtype = "bfloat16"
    
# Set training parameters
reward_model.set_trainable_parameters(
    tune_visual=True,
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

# Define RewardModelWrapper ONLY ONCE
class RewardModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, inputs):
        # Forward pass through model
        outputs = self.model(inputs)
        
        # BatchFeature handling (similar to Policy_train.py)
        if hasattr(outputs, "keys") and "loss" in outputs.keys():
            loss = outputs["loss"]
            return {"loss": loss}
        
        # Dict handling    
        elif isinstance(outputs, dict):
            # If loss is already in outputs
            if "loss" in outputs:
                return {"loss": outputs["loss"]}
            
            # If reward_pred is in outputs, calculate loss
# If reward_pred is in outputs, calculate loss
        elif "reward_pred" in outputs and "target_reward" in inputs:
            reward_pred = outputs["reward_pred"]
            target = inputs["target_reward"]
            
            # Ensure target has matching shape without printing warnings
            if target.shape != reward_pred.shape:
                # Silently fix shape
                while len(target.shape) > len(reward_pred.shape):
                    target = target.squeeze(-1)
                    
            target = target.to(device=reward_pred.device, dtype=reward_pred.dtype)
            loss = F.mse_loss(reward_pred, target)
            return {"loss": loss}
        
        # If outputs is a tensor, assume it's the loss
        elif isinstance(outputs, torch.Tensor):
            return {"loss": outputs}
            
        # For debugging - print what we received
        print(f"WARNING: No loss found in outputs! Type: {type(outputs)}")
        if isinstance(outputs, dict):
            print(f"Available keys: {outputs.keys()}")
        print(f"Inputs has target_reward: {'target_reward' in inputs}")
            
        # Return a dummy loss for debugging
        device = next(self.parameters()).device  # Get actual device
        return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}
    
    def save_pretrained(self, output_dir, state_dict=None):
        """Save the model to the specified output directory."""
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get state dict if not provided
        if state_dict is None:
            state_dict = self.state_dict()
        
        # If the wrapped model has save_pretrained, use it
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(output_dir, state_dict=state_dict)
        else:
            # Otherwise save the state dict directly
            torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
            
            # Save config if available
            if hasattr(self.model, 'config'):
                self.model.config.save_pretrained(output_dir)


reward_model = reward_model.to(torch.bfloat16)


# Apply wrapper ONCE
reward_model = RewardModelWrapper(reward_model)
reward_model = reward_model.to("cuda")  # Important: move to GPU BEFORE DataParallel

# Setup training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    run_name="reward_model_training",
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
    max_steps=500,
    save_strategy="steps",
    save_steps=100,
    evaluation_strategy="no",
    save_total_limit=3,
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

# Save the final model
print("\nSaving final model...")
model_save_path = os.path.join(output_dir, "final_reward_model.pt")
torch.save(reward_model.state_dict(), model_save_path)
print(f"Model saved to: {model_save_path}")