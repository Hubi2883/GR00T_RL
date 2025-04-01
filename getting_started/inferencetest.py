#!/usr/bin/env python
import os
import json
import torch
import gr00t
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP

# Set paths
MODEL_PATH = "nvidia/GR00T-N1-2B"
DATASET_PATH = "/ceph/home/student.aau.dk/wb68dm/New_results_0001"
# Use the embodiment tag from the checkpoint metadata ("gr1")
EMBODIMENT_TAG = "gr1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load your TwoWheelDataConfig from DATA_CONFIG_MAP
data_config = DATA_CONFIG_MAP["two_wheel"]  # Ensure this is your new config.
modality_config = data_config.modality_config()
modality_transform = data_config.transform()

# Initialize the GR00T-N1 policy for inference
policy = Gr00tPolicy(
    model_path=MODEL_PATH,
    embodiment_tag=EMBODIMENT_TAG,
    modality_config=modality_config,
    modality_transform=modality_transform,
    device=device,
)
print("Loaded GR00T-N1 model:")
print(policy.model)

# Override checkpoint metadata with your dataset's meta/modality.json
modality_metadata_path = os.path.join(DATASET_PATH, "meta", "modality.json")
with open(modality_metadata_path, "r") as f:
    modality_metadata = json.load(f)

def dict_to_namespace(d):
    return SimpleNamespace(**{k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()})

# Convert the modalities dict to a namespace object so attribute access works.
metadata_obj = SimpleNamespace(modalities=dict_to_namespace(modality_metadata))
policy._modality_transform.set_metadata(metadata_obj)
print("Custom modality metadata set:", metadata_obj.modalities)

# Load dataset
dataset = LeRobotSingleDataset(
    dataset_path=DATASET_PATH,
    modality_configs=modality_config,
    video_backend="decord",
    transforms=None,  # We handle transforms separately through the policy
    embodiment_tag=EMBODIMENT_TAG,
)
print(f"Initialized dataset from {DATASET_PATH} with embodiment tag: {EMBODIMENT_TAG}")

# Get a sample and visualize a video frame
sample = dataset[0]
print("Sample keys:", sample.keys())
video_key = modality_config["video"].modality_keys[0]  # Expecting "video.ego_view"
img = sample[video_key][0]
plt.imshow(img)
plt.title("Inference Sample Video Frame")
plt.axis("off")
plt.show()

# Run inference: predict actions
predicted_action = policy.get_action(sample)
print("Predicted action shapes:")
for key, value in predicted_action.items():
    print(f"{key}: {value.shape}")
