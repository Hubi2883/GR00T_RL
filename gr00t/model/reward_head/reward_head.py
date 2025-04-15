import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature
from dataclasses import dataclass, field
from transformers import PretrainedConfig

BACKBONE_FEATURE_KEY = "backbone_features"
REWARD_KEY = "reward_pred"
LOSS_KEY = "loss"

@dataclass
class RewardHeadConfig(PretrainedConfig):
    """Configuration for the reward head."""
    hidden_dim: int = field(default=512, metadata={"help": "Hidden dimension size"})
    dropout: float = field(default=0.1, metadata={"help": "Dropout probability"})
    reward_dim: int = field(default=1, metadata={"help": "Reward output dimension"})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector"})
    
class RewardHead(nn.Module):
    """
    Reward prediction head for GR00T model.
    Takes backbone features (which already include action information) and outputs predicted rewards.
    """
    config_class = RewardHeadConfig
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._compute_dtype = torch.float32  # Use different attribute name
    
        backbone_dim = 1536  # Standard GR00T backbone 
        
        
        # Simple projector from backbone features to reward
        self.reward_projector = nn.Sequential(
            nn.Linear(backbone_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.reward_dim)
        )
        
        # Set trainable parameters
        self.set_trainable_parameters(config.tune_projector)
        
    def forward(self, backbone_outputs, inputs=None):
        """
        Forward pass for training - calculates loss if ground truth rewards available.
        
        Args:
            backbone_outputs: Dictionary containing backbone features
            inputs: Dictionary containing target_reward
            
        Returns:
            BatchFeature containing reward predictions and optional loss
        """
        features = backbone_outputs[BACKBONE_FEATURE_KEY]
        
        # Extract features from backbone (first token)
        if len(features.shape) == 3:
            features = features[:, 0]  # Use first token features
            
        # Calculate reward prediction
        reward_pred = self.reward_projector(features)
        
        outputs = {REWARD_KEY: reward_pred}
        
        # Calculate loss if target rewards are provided
        if inputs is not None and "target_reward" in inputs:
            target_reward = inputs["target_reward"]
            
            # Ensure shapes match
            if reward_pred.shape != target_reward.shape:
                #print(f"Shape mismatch: reward_pred {reward_pred.shape} vs target {target_reward.shape}")
                # Reshape target to match prediction shape
                target_reward = target_reward.view(reward_pred.shape)
                
            loss = F.mse_loss(reward_pred, target_reward)
            outputs[LOSS_KEY] = loss
            
        return BatchFeature(outputs)
    
    @torch.no_grad()
    def get_reward(self, backbone_outputs, inputs=None):
        """
        Inference method to get rewards from observations.
        
        Args:
            backbone_outputs: Dictionary containing backbone features
            inputs: Optional additional inputs
            
        Returns:
            BatchFeature containing reward predictions
        """
        features = backbone_outputs[BACKBONE_FEATURE_KEY]
        
        # Extract features from backbone (first token)
        if len(features.shape) == 3:
            features = features[:, 0]  # Use first token features
            
        # Calculate reward prediction
        reward_pred = self.reward_projector(features)
        
        return BatchFeature({REWARD_KEY: reward_pred})
    
    def prepare_input(self, inputs):
        """
        Prepare inputs for the reward head.
        
        Args:
            inputs: Dictionary containing model inputs
            
        Returns:
            BatchFeature with processed inputs
        """
        output_dict = {}
        if "target_reward" in inputs:
            output_dict["target_reward"] = inputs["target_reward"]
        return BatchFeature(output_dict)
    
    def set_trainable_parameters(self, tune_projector=True):
        """
        Set which parameters to train.
        
        Args:
            tune_projector: Whether to train the projector parameters
        """
        self.tune_projector = tune_projector
        for param in self.parameters():
            param.requires_grad = tune_projector
            
        print(f"Tune reward head projector: {self.tune_projector}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No reward head trainable parameters found.")
    
    @property
    def device(self):
        """Get device of the model."""
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        """Get dtype of the model."""
        return next(iter(self.parameters())).dtype