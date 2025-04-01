# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unified Finetuning Script for GR00T-N1 on a New Embodiment (So100)
This script follows these steps:
    Step 1.1: Define modality configs and transforms for the dataset.
    Step 1.2: Load the dataset using LeRobotSingleDataset.
    Step 2.1: Load the base GR00T-N1 model for tuning.
    Step 2.2: Prepare training arguments and run the training loop.
"""

import argparse
import os
import pathlib
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch

# --------------------------
# STEP 1.1: Modality Configs and Transforms
# --------------------------
from gr00t.data.dataset import LeRobotSingleDataset, ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform import (
    VideoToTensor,
    VideoCrop,
    VideoResize,
    VideoColorJitter,
    VideoToNumpy,
)
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.model.transforms import GR00TTransform
from gr00t.experiment.data_config import DATA_CONFIG_MAP

# CHANGED: Use the "so100" data configuration for the low-cost So100 Lerobot arm dataset.
dataset_path = "./demo_data/so100_strawberry_grape"  # Change as needed !!!!
data_config = DATA_CONFIG_MAP["two_wheel"]
modality_configs = data_config.modality_config()
# Optionally, load transforms from the data_config; here we compose our own:
to_apply_transforms = ComposedModalityTransform(
    transforms=[
        # Video transforms
        VideoToTensor(apply_to=modality_configs["video"].modality_keys, backend="torchvision"),
        VideoCrop(apply_to=modality_configs["video"].modality_keys, scale=0.95, backend="torchvision"),
        VideoResize(apply_to=modality_configs["video"].modality_keys, height=224, width=224, interpolation="linear", backend="torchvision"),
        VideoColorJitter(apply_to=modality_configs["video"].modality_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08, backend="torchvision"),
        VideoToNumpy(apply_to=modality_configs["video"].modality_keys),
        # State transforms
        StateActionToTensor(apply_to=modality_configs["state"].modality_keys),
        StateActionTransform(apply_to=modality_configs["state"].modality_keys,
                               normalization_modes={key: "min_max" for key in modality_configs["state"].modality_keys}),
        # Action transforms
        StateActionToTensor(apply_to=modality_configs["action"].modality_keys),
        StateActionTransform(apply_to=modality_configs["action"].modality_keys,
                               normalization_modes={key: "min_max" for key in modality_configs["action"].modality_keys}),
        # Concat transform: concatenates the modalities in a specified order.
        ConcatTransform(
            video_concat_order=modality_configs["video"].modality_keys,
            state_concat_order=modality_configs["state"].modality_keys,
            action_concat_order=modality_configs["action"].modality_keys,
        ),
        # Model-specific transform for GR00T
        GR00TTransform(
            state_horizon=len(data_config.modality_config()["state"].delta_indices),
            action_horizon=len(data_config.modality_config()["action"].delta_indices),
            max_state_dim=64,
            max_action_dim=32,
        ),
    ]
)

# --------------------------
# STEP 1.2: Load the Dataset
# --------------------------
print("Loading dataset from:", dataset_path)
# We load the dataset without transforms first for visualization
train_dataset = LeRobotSingleDataset(
    dataset_path=dataset_path,
    modality_configs=modality_configs,
    embodiment_tag="new_embodiment",  # CHANGED: Using new embodiment tag for So100
    video_backend="torchvision_av",
)

# Visualize a few images from the dataset.
print("Dataset keys:", train_dataset[0].keys())
images_list = []
video_key = list(modality_configs["video"].modality_keys)[0]  # Assume one video modality
for i in range(5):
    sample = train_dataset[i]
    img = sample[video_key][0]
    images_list.append(img)

fig, axs = plt.subplots(1, 5, figsize=(20, 5))
for i, ax in enumerate(axs.flat):
    ax.imshow(images_list[i])
    ax.axis("off")
    ax.set_title(f"Image {i}")
plt.show()

# --------------------------
# STEP 2.1: Load the Base Model
# --------------------------
from gr00t.model.gr00t_n1 import GR00T_N1

BASE_MODEL_PATH = "nvidia/GR00T-N1-2B"  # Pretrained base model
# CHANGED: Specify which parts of the model to tune
TUNE_LLM = False            # Do not tune the language model component
TUNE_VISUAL = True          # Tune the visual encoder
TUNE_PROJECTOR = True       # Tune the action head's projector
TUNE_DIFFUSION_MODEL = True # Tune the diffusion model

# Load the base model for finetuning
model = GR00T_N1.from_pretrained(
    pretrained_model_name_or_path=BASE_MODEL_PATH,
    tune_llm=TUNE_LLM,
    tune_visual=TUNE_VISUAL,
    tune_projector=TUNE_PROJECTOR,
    tune_diffusion_model=TUNE_DIFFUSION_MODEL,
)
# Set the model to use bfloat16 for computation (if desired)
model.compute_dtype = "bfloat16"
model.config.compute_dtype = "bfloat16"
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# --------------------------
# STEP 2.2: Prepare Training Arguments
# --------------------------
from transformers import TrainingArguments

# CHANGED: Adjust hyperparameters as necessary.
output_dir = os.path.expanduser("~/so100-checkpoints")
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    max_steps=2000,  # Adjust as needed
    learning_rate=1e-4,
    weight_decay=1e-5,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="no",
    seed=42,
    do_eval=False,
    dataloader_num_workers=8,
)

# --------------------------
# STEP 2.3: Run the Finetuning Training Loop
# --------------------------
from gr00t.experiment.runner import TrainRunner

# Reload the training dataset with transforms applied.
train_dataset = LeRobotSingleDataset(
    dataset_path=dataset_path,
    modality_configs=modality_configs,
    embodiment_tag="new_embodiment",
    video_backend="torchvision_av",
    transforms=to_apply_transforms,
)

# Initialize the training runner and start training
experiment = TrainRunner(
    train_dataset=train_dataset,
    model=model,
    training_args=training_args,
)
print("Starting finetuning training...")
experiment.train()
