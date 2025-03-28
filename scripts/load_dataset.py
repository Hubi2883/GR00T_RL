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
This script is a replication of the notebook `getting_started/load_dataset.ipynb`
Adapted for a 2-wheel differential-drive robot:
- Observation: absolute acceleration (1 value)
- Action: wheel commands (2 values)
- Video: single camera feed
- Annotation: natural language prompt (if provided)
"""

import argparse
import json
import pathlib
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from gr00t.data.dataset import (
    LE_ROBOT_MODALITY_FILENAME,
    LeRobotSingleDataset,
    ModalityConfig,
)
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.utils.misc import any_describe

import os

# CHANGED: Update the DATA_PATH to point to our 2-wheel robot simulated dataset.
DATA_PATH = os.path.expanduser("/ceph/home/student.aau.dk/wb68dm/Isaac-GR00T/demo_data/2wheel_robot_sim")

def get_modality_keys(dataset_path: pathlib.Path) -> dict[str, list[str]]:
    """
    Get the modality keys from the dataset path.
    Returns a dictionary with modality types as keys and their corresponding modality keys as values.
    This function is kept for debugging purposes but we override the modality configs for our 2-wheel robot.
    """
    modality_path = dataset_path / LE_ROBOT_MODALITY_FILENAME
    with open(modality_path, "r") as f:
        modality_meta = json.load(f)

    modality_dict = {}
    for key in modality_meta.keys():
        modality_dict[key] = []
        for modality in modality_meta[key]:
            modality_dict[key].append(f"{key}.{modality}")
    return modality_dict

def load_dataset(dataset_path: str, embodiment_tag: str, video_backend: str = "decord"):
    # 1. Optionally, get modality keys from the dataset (for debugging)
    dataset_path = pathlib.Path(dataset_path)
    modality_keys_dict = get_modality_keys(dataset_path)
    print("Original modality keys from dataset (for debugging):")
    pprint(modality_keys_dict)
    
    # CHANGED: Override modality configurations for our 2-wheel robot.
    # Our 2-wheel robot dataset uses:
    #   - State: "state.acceleration" (1 value)
    #   - Action: "action.wheel_commands" (2 values)
    #   - Video: "video.camera" (single camera feed)
    #   - Annotation: "annotation.human.action.prompt" (for natural language instructions)
    modality_configs = {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["video.camera"],  # CHANGED: Use our camera modality key
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["state.acceleration"],  # CHANGED: Only the acceleration reading
        ),
        "action": ModalityConfig(
            delta_indices=[0],
            modality_keys=["action.wheel_commands"],  # CHANGED: Two wheel commands for left and right wheels
        ),
    }

    # 3. language modality config (if exists)
    # CHANGED: Use a new key "annotation.human.action.prompt" for natural language prompts.
    modality_configs["language"] = ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.prompt"],
    )

    # 4. Set gr00t embodiment tag. 
    # CHANGED: Here we assume we use the same tag as before; if a custom tag exists for differential drive,
    # you could specify that here. For now, we keep GR1.
    embodiment_tag: EmbodimentTag = EmbodimentTag(embodiment_tag)
    embodiment_tag = EmbodimentTag.GR1  # CHANGED: For our 2-wheel robot, we choose GR1 (or adjust as needed)

    # 5. Load the dataset using our custom modality configs.
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,  # CHANGED: Use our DATA_PATH
        modality_configs=modality_configs,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend,
    )

    print("\n" * 2)
    print("=" * 100)
    print(f"{' 2-Wheel Robot Dataset ':=^100}")
    print("=" * 100)

    # Print the 7th data point
    resp = dataset[7]
    any_describe(resp)
    print(resp.keys())

    print("=" * 50)
    for key, value in resp.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")

    # 6. Plot the first 100 images (sample every 10th image)
    images_list = []
    video_key = modality_configs["video"].modality_keys[0]  # Use the video modality key ("video.camera")
    for i in range(100):
        if i % 10 == 0:
            resp = dataset[i]
            # CHANGED: Access the first frame of the video sequence
            img = resp[video_key][0]
            images_list.append(img)

    fig, axs = plt.subplots(2, 5, figsize=(20, 10))
    for i, ax in enumerate(axs.flat):
        ax.imshow(images_list[i])
        ax.axis("off")
        ax.set_title(f"Image {i}")
    plt.tight_layout()  # adjust subplots to fit the figure area.
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load 2-Wheel Robot Dataset in LeRobot Format")
    parser.add_argument(
        "--data_path",
        type=str,
        default="demo_data/2wheel_robot_sim",  # CHANGED: Default dataset path for 2-wheel robot simulation
        help="Path to the 2-wheel robot dataset",
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        default="gr1",
        help="Embodiment tag (see gr00t.data.embodiment_tags.EmbodimentTag for a full list)",
    )
    parser.add_argument(
        "--video_backend",
        type=str,
        default="decord",
        choices=["decord", "torchvision_av"],
        help="Backend to use for video loading, use torchvision_av for AV encoded videos",
    )
    args = parser.parse_args()
    load_dataset(args.data_path, args.embodiment_tag, args.video_backend)
