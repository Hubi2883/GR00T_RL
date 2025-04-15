import torch
import numpy as np
from typing import Any, ClassVar
from pydantic import Field, PrivateAttr

from gr00t.data.transform.base import ModalityTransform
from gr00t.data.schema import DatasetMetadata


class RewardNormalizer:
    """Normalizer for reward values"""
    
    valid_modes = ["min_max", "mean_std", "scale", "none"]
    
    def __init__(self, mode: str, statistics: dict):
        self.mode = mode
        self.statistics = statistics
        for key, value in self.statistics.items():
            self.statistics[key] = torch.tensor(value)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize reward values"""
        assert isinstance(x, torch.Tensor), f"Expected torch.Tensor, got {type(x)}"
        
        if self.mode == "none":
            return x
            
        elif self.mode == "min_max":
            # Range of min_max is [-1, 1]
            min_val = self.statistics["min"].to(x.dtype)
            max_val = self.statistics["max"].to(x.dtype)
            
            # Handle case where min == max
            mask = min_val != max_val
            normalized = torch.zeros_like(x)
            
            # Normalize: 2 * (x - min) / (max - min) - 1
            normalized[..., mask] = (x[..., mask] - min_val[..., mask]) / (
                max_val[..., mask] - min_val[..., mask]
            )
            normalized[..., mask] = 2 * normalized[..., mask] - 1
            
            # Set values to 0 where min == max
            normalized[..., ~mask] = 0
            return normalized
            
        elif self.mode == "mean_std":
            mean = self.statistics["mean"].to(x.dtype)
            std = self.statistics["std"].to(x.dtype)
            
            # Handle case where std == 0
            mask = std != 0
            normalized = torch.zeros_like(x)
            
            # Normalize: (x - mean) / std
            normalized[..., mask] = (x[..., mask] - mean[..., mask]) / std[..., mask]
            
            # Set values to 0 where std == 0
            normalized[..., ~mask] = 0
            return normalized
            
        elif self.mode == "scale":
            # Scale to [-1, 1] based on absolute max value
            abs_max = self.statistics["abs_max"].to(x.dtype)
            
            # Handle zero case
            mask = abs_max != 0
            normalized = torch.zeros_like(x)
            normalized[..., mask] = x[..., mask] / abs_max[..., mask]
            return normalized
            
        else:
            raise ValueError(f"Invalid normalization mode: {self.mode}")


class RewardTransform(ModalityTransform):
    """
    Transform for handling reward data.
    Extracts reward values from dataset and prepares them for the reward model.
    
    Args:
        apply_to (list[str]): Reward keys to transform (e.g. ["next.reward"])
        normalization_mode (str): How to normalize rewards ("min_max", "mean_std", "scale", "none")
        statistics (dict): Statistics for normalization (if not using dataset statistics)
    """
    
    # Configurable attributes
    apply_to: list[str] = Field(..., description="Reward keys to transform")
    normalization_mode: str = Field(
        default="min_max", 
        description="Normalization mode for rewards"
    )
    statistics: dict = Field(
        default_factory=dict, 
        description="Statistics for normalization"
    )
    
    # Private attributes
    _normalizer: RewardNormalizer = PrivateAttr(default=None)
    
    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Set up normalizer from dataset statistics"""
        dataset_statistics = dataset_metadata.statistics
        
        # Use dataset statistics if no custom statistics provided
        if not self.statistics:
            for key in self.apply_to:
                split_key = key.split(".")
                assert len(split_key) == 2, "Reward keys should have two parts: 'modality.key'"
                modality, reward_key = split_key
                
                if hasattr(dataset_statistics, modality) and reward_key in getattr(dataset_statistics, modality):
                    reward_stats = getattr(dataset_statistics, modality)[reward_key].model_dump()
                    
                    # Create appropriate statistics based on normalization mode
                    if self.normalization_mode == "min_max":
                        self.statistics = {
                            "min": reward_stats.get("min", 0),
                            "max": reward_stats.get("max", 1)
                        }
                    elif self.normalization_mode == "mean_std":
                        self.statistics = {
                            "mean": reward_stats.get("mean", 0),
                            "std": reward_stats.get("std", 1)
                        }
                    elif self.normalization_mode == "scale":
                        # Calculate absolute max from min and max
                        min_val = reward_stats.get("min", -1)
                        max_val = reward_stats.get("max", 1)
                        abs_max = max(abs(min_val), abs(max_val))
                        self.statistics = {"abs_max": abs_max}
                    break  # Only need statistics for one key
        
        # If still no statistics, use defaults
        if not self.statistics:
            if self.normalization_mode == "min_max":
                self.statistics = {"min": -1, "max": 1}
            elif self.normalization_mode == "mean_std":
                self.statistics = {"mean": 0, "std": 1}
            elif self.normalization_mode == "scale":
                self.statistics = {"abs_max": 1}
        
        # Create normalizer
        self._normalizer = RewardNormalizer(self.normalization_mode, self.statistics)
    
    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """Process reward data and prepare for reward model"""
        rewards = []
        
        # Extract rewards from all specified keys
        for key in self.apply_to:
            if key not in data:
                continue
                
            # Convert to tensor if needed
            reward_data = data[key]
            if not isinstance(reward_data, torch.Tensor):
                reward_data = torch.tensor(reward_data, dtype=torch.float32)
                
            # Ensure proper shape
            if len(reward_data.shape) == 0:  # Scalar
                reward_data = reward_data.unsqueeze(0)
                
            rewards.append(reward_data)
        
        # If no rewards found, return data unchanged
        if not rewards:
            return data
            
        # Combine rewards if multiple keys provided
        if len(rewards) > 1:
            reward = torch.stack(rewards, dim=0).mean(dim=0)
        else:
            reward = rewards[0]
            
        # Normalize reward if normalizer is set up
        if self._normalizer is not None:
            reward = self._normalizer.forward(reward)
            
        # Make sure reward has proper shape [batch_size, reward_dim]
        if len(reward.shape) == 1:
            reward = reward.unsqueeze(-1)
            
        # Store processed reward for model
        data["target_reward"] = reward
        
        return data