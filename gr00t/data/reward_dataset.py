import os
import json
import pandas as pd
import torch
import numpy as np
from gr00t.data.dataset import LeRobotSingleDataset

class RewardDataset(LeRobotSingleDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load rewards file if it exists
        rewards_path = self.dataset_path / "meta/rewards.jsonl"
        if not rewards_path.exists():
            raise FileNotFoundError(f"Rewards file is required but not found at {rewards_path}")
            
        with open(rewards_path, "r") as f:
            rewards = [json.loads(line) for line in f]
        self._rewards_df = pd.DataFrame(rewards)
        if "annotation.human.validity" in self._rewards_df.columns:
            self._rewards_df = self._rewards_df.set_index("annotation.human.validity")
            print("Successfully loaded rewards file with annotation.human.validity as index")
        else:
            raise ValueError(f"Rewards file doesn't have 'annotation.human.validity' column. Found columns: {self._rewards_df.columns}")

    def get_language(self, trajectory_id: int, key: str, base_index: int) -> list[str]:
        """Override get_language to handle reward feedback properly"""
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        max_length = self.trajectory_lengths[trajectory_index]
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)
        
        # Get the annotations
        indices = []
        subkey = key.replace("annotation.", "")
        annotation_meta = self.lerobot_modality_meta.annotation
        assert annotation_meta is not None, f"Annotation metadata is None for {subkey}"
        assert subkey in annotation_meta, f"Annotation key {subkey} not found in metadata"
        
        subkey_meta = annotation_meta[subkey]
        original_key = subkey_meta.original_key if subkey_meta.original_key else key
        
        for i in range(len(step_indices)):
            indices.append(self.curr_traj_data[original_key][step_indices[i]].item())
        
        # Use rewards data if the key is human.validity
        if subkey == "human.validity" and self._rewards_df is not None:
            try:
                # Now indices contains the values from annotation.human.validity
                # These should match the task_index in rewards.jsonl
                return self._rewards_df.loc[indices]["feedback"].tolist()
            except KeyError as e:
                print(f"Warning: Some indices {indices} not found in rewards file. Error: {e}")
                # Return the numerical values as strings if we can't find the feedback
                return [f"Reward value: {idx}" for idx in indices]
        else:
            # Fall back to tasks for other keys
            return self.tasks.loc[indices]["task"].tolist()

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