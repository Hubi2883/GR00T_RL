import torch
import tree
from dataclasses import dataclass, field
from transformers import PreTrainedModel, PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.gr00t_n1 import GR00T_N1, GR00T_N1Config
from gr00t.model.backbone import EagleBackbone
from gr00t.model.reward_head import RewardHead, RewardHeadConfig


# Define constants similar to GR00T_N1
BACKBONE_FEATURE_KEY = "backbone_features"
REWARD_KEY = "reward_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output for reward model"

@dataclass
class GR00TRewardConfig(PretrainedConfig):
    """Configuration for GR00T reward model."""
    model_type = "gr00t_reward"
    backbone_cfg: dict = field(default_factory=dict, metadata={"help": "Backbone configuration."})
    reward_head_cfg: dict = field(default_factory=dict, metadata={"help": "Reward head configuration."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
            
class GR00TReward(PreTrainedModel):
    """GR00T model with a reward head instead of an action head."""
    config_class = GR00TRewardConfig
    
    def __init__(self, gr00t_model, reward_head_config=None):
        super().__init__(gr00t_model.config)
        
        # Use the backbone from gr00t_model
        self.backbone = gr00t_model.backbone
        
        # Create reward head
        if reward_head_config is None:
            reward_head_config = RewardHeadConfig()
        self.reward_head = RewardHead(reward_head_config)
        
        # Copy compute properties
        self.compute_dtype = gr00t_model.compute_dtype
        self.config.compute_dtype = gr00t_model.config.compute_dtype
        
    def validate_data(self, reward_outputs, backbone_outputs, is_training):
        """Validate outputs for correctness"""
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            raise ValueError(error_msg)

        fail_reward_head = (
            not isinstance(reward_outputs, BatchFeature)
            or not ((LOSS_KEY in reward_outputs and is_training) 
                    or REWARD_KEY in reward_outputs)
        )

        if fail_reward_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(reward_outputs, BatchFeature)=}"
            error_msg += f"\n{REWARD_KEY in reward_outputs=}"
            error_msg += f"\n{LOSS_KEY in reward_outputs=}"
            raise ValueError(error_msg)
        
    def forward(self, inputs):
        """Forward pass for training - calculates loss if ground truth rewards available"""
        backbone_inputs, reward_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        reward_outputs = self.reward_head(backbone_outputs, reward_inputs)
        
        # Validate outputs
        self.validate_data(reward_outputs, backbone_outputs, is_training=True)
        
        # Return loss for training compatibility
        if hasattr(reward_outputs, "keys") and LOSS_KEY in reward_outputs.keys():
            return {"loss": reward_outputs[LOSS_KEY]}
        return reward_outputs
    
    def get_reward(self, inputs):
        """Get reward predictions (consistent with get_action in GR00T_N1)"""
        backbone_inputs, reward_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        reward_outputs = self.reward_head(backbone_outputs, reward_inputs)
        
        # Validate outputs
        self.validate_data(reward_outputs, backbone_outputs, is_training=False)
        
        return reward_outputs
    
    # Alias for compatibility
    predict_reward = get_reward
    
    def prepare_input(self, inputs):
        """Prepare inputs for backbone and reward head"""
        backbone_inputs = self.backbone.prepare_input(inputs)
        reward_inputs = {}
        
        # Pass target_reward to the reward head if available
        if "target_reward" in inputs:
            reward_inputs["target_reward"] = inputs["target_reward"]
        
        # Get device from parameters
        device = next(self.parameters()).device

        def to_device_with_dtype(x):
            if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
                return x.to(device, dtype=getattr(torch, self.compute_dtype))
            if isinstance(x, torch.Tensor):
                return x.to(device)
            return x

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        reward_inputs = tree.map_structure(to_device_with_dtype, reward_inputs)
        return backbone_inputs, reward_inputs
    
    def set_trainable_parameters(self, tune_visual=False, tune_llm=False, tune_projector=True):
        """Set which parts of the model are trainable"""
        # Set backbone trainability
        self.backbone.set_trainable_parameters(tune_visual=tune_visual, tune_llm=tune_llm)
        # Set reward head trainability
        self.reward_head.set_trainable_parameters(tune_projector=tune_projector)
        
        # Print training status
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune reward head projector: {tune_projector}")

    @classmethod
    def from_pretrained(cls, gr00t_model, reward_head_config=None):
        """Create a reward model from a pretrained GR00T model"""
        model = cls(gr00t_model, reward_head_config)
        return model