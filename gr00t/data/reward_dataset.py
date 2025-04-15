import os
import pandas as pd
import torch
import numpy as np
from gr00t.data.dataset import LeRobotSingleDataset

class RewardDataset(LeRobotSingleDataset):
    def get_data_by_modality(self, trajectory_id: int, modality: str, key: str, base_index: int):
        """Support 'next' modality for rewards."""
        if modality == "next":
            # Extract component name (e.g., "reward" from "next.reward")
            component = key.split(".")[-1]
            
            # Make sure trajectory data is loaded
            if self.curr_traj_data is None or self.curr_traj_id != trajectory_id:
                self.curr_traj_data = self.get_trajectory_data(trajectory_id)
                self.curr_traj_id = trajectory_id
            
            # Check if the component exists in the data
            column_name = f"next.{component}"
            if column_name in self.curr_traj_data.columns:
                # Get the value and convert to numpy array
                value = self.curr_traj_data[column_name].iloc[base_index]
                if isinstance(value, (int, float)):
                    return np.array([value], dtype=np.float32)
                return value
            else:
                # Default value for missing data
                print(f"Warning: '{column_name}' not found in trajectory {trajectory_id}")
                return np.array([0.0], dtype=np.float32)
        else:
            # Use parent implementation for other modalities
            return super().get_data_by_modality(trajectory_id, modality, key, base_index)
    def __getitem__(self, idx):
        """Override to map the reward values to the expected format"""
        # Get the original item from parent class
        item = super().__getitem__(idx)
        
        # Check if we have reward data and map it to target_reward
        if "next" in item and "next.reward" in item["next"]:
            # Create a target_reward field at the top level for the model
            item["target_reward"] = item["next"]["next.reward"]
        
        return item