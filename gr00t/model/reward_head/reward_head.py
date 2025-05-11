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
    reward_horizon: int = field(default=16, metadata={"help": "Number of reward predictions in sequence"})
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
        self.reward_horizon = config.reward_horizon
    
        backbone_dim = 1536  # Standard GR00T backbone 
        
        # Sequence processing layers for all features (not just first token)
        self.sequence_projector = nn.Sequential(
            nn.Linear(backbone_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Final reward projector to generate reward_horizon predictions
        self.reward_projector = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, self.reward_horizon * config.reward_dim)
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
        features = backbone_outputs[BACKBONE_FEATURE_KEY]  # [batch_size, seq_len, hidden_dim]
        
        # Process the full sequence (or use global features if needed)
        # Option 1: Use mean pooling across sequence
        batch_size = features.shape[0]
        pooled_features = features.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Project features to intermediate space
        hidden_features = self.sequence_projector(pooled_features)  # [batch_size, hidden_dim]
        
        # Generate reward predictions for the horizon
        reward_flat = self.reward_projector(hidden_features)  # [batch_size, horizon*reward_dim]
        
        # Reshape to [batch_size, horizon, reward_dim]
        reward_pred = reward_flat.view(batch_size, self.reward_horizon, self.config.reward_dim)
        
        outputs = {REWARD_KEY: reward_pred}
        
        # Calculate loss if target rewards are provided
        if inputs is not None and "target_reward" in inputs:
            target_reward = inputs["target_reward"]
            
            # Ensure shapes match
            if reward_pred.shape != target_reward.shape:
                # Handle cases where target is [batch_size] but pred is [batch_size, horizon, reward_dim]
                if len(target_reward.shape) == 1:
                    # Expand target to match prediction horizon
                    target_reward = target_reward.unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]
                    target_reward = target_reward.expand(-1, self.reward_horizon, self.config.reward_dim)
                elif len(target_reward.shape) == 2:
                    # Expand target to match prediction horizon
                    target_reward = target_reward.unsqueeze(-1)  # [batch, 1, 1]
                    target_reward = target_reward.expand(-1, self.reward_horizon, self.config.reward_dim)
                elif len(target_reward.shape) == 3 and target_reward.shape[1] == 1:
                    # Already 3D but first dimension is 1, expand to match horizon
                    target_reward = target_reward.expand(-1, self.reward_horizon, -1)
                
            # Convert to target dtype
            target_reward = target_reward.to(device=reward_pred.device, dtype=reward_pred.dtype)
                
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
        # Use the same forward implementation for consistency
        return self.forward(backbone_outputs, inputs)
    
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