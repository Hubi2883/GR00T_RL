

![Project](Screenshot%202025-11-05%20at%2021.50.58.png)

Project modifing GR00T_RL aiming to improve efficiency of finetuning by incorporating reinforcement learning feedback human  (RLHF).

This work proposes a three-stage reinforcement learning from human feed-back (RLHF) approach for robot navigation and concentrates on the second stage, in which a reward function must be inferred when both states and actions are raw sensor observations. The training is split into the fine-tuning of the base model, a train- ing of a reward model and the use of the reward model for optimization of the original policy model. This approach solves the central difficulty that due to a high dimensional and uninterpretable observation space, explicit rewards are unavailable. The main problem has been defined as
a noise existing in reward-unrelated embedding differences. Introduced is a theoretical analysis that identifies
a methodological parameter that can improve convergence challenged by this observation-level noise. Is shown that the RLHF can be applied to create a reward model for complex and not-understood environments.



# Testing and Evaluating GR00T_RL

This document explains how to quickly test that the RL training and reward model pipelines are working, how to run minimal training to produce checkpoints, and how to evaluate your results. It focuses on the scripts in this directory.

- Policy fine-tuning (supervised/imitation): [Policy_train.py](https://github.com/Hubi2883/GR00T_RL/blob/main/RL_training_scripts/Policy_train.py)
- Reward model training (variants):
  - [Rewardmodel_train_new.py](https://github.com/Hubi2883/GR00T_RL/blob/main/RL_training_scripts/Rewardmodel_train_new.py)
  - [505training.py](https://github.com/Hubi2883/GR00T_RL/blob/main/RL_training_scripts/505training.py)
  - [validatetraining.py](https://github.com/Hubi2883/GR00T_RL/blob/main/RL_training_scripts/validatetraining.py) (Accelerate-compatible, with built-in validation)
- Reward model inference/evaluation: [rewardinference.py](https://github.com/Hubi2883/GR00T_RL/blob/main/RL_training_scripts/rewardinference.py)



---

## Prerequisites

- A machine with an NVIDIA GPU and CUDA drivers. The scripts default to mixed precision (bfloat16) which is intended for modern GPUs.
- Python environment with project dependencies:
  - Install the package in editable mode (preferred):
    ```bash
    pip install -e .
    ```
  - or install from the pinned list (may be heavier than needed):
    ```bash
    pip install -r all_packages.txt
    ```
- Access to the base model:
  - The scripts default to the NVIDIA GR00T model name `nvidia/GR00T-N1-2B`. Make sure you can download it with your environment (e.g., via Hugging Face/NGC as configured in your setup).
- Dataset prepared on disk (see “Dataset expectations”).

---

## Dataset expectations

The scripts use the project’s data abstractions:
- Policy training: `LeRobotSingleDataset`
- Reward model training / inference: `RewardDataset`

Across scripts you will see the same modality keys and configuration patterns:

- Video: `"video.ego_view"`
- State: `"state.acceleration"` (some variants also reference `"state.left_wheel"` and `"state.right_wheel"`)
- Action: `"action.wheel_commands"`
- Reward target: `"next.reward"`
- Embodiment tag: `"new_embodiment"`

These are declared via `ModalityConfig` and transformations via `ComposedModalityTransform` and `GR00TTransform`.

Your dataset must contain those fields under the given keys, or you must update the keys in the scripts’ `ModalityConfig` and transforms accordingly.

Tip: All training scripts define these constants near the top:
- `dataset_path` – folder containing your dataset
- `output_dir` – where checkpoints and artifacts are stored
- `base_model_path` or `model_path` – base model (e.g., `"nvidia/GR00T-N1-2B"` or a previous fine-tuned checkpoint)
- `embodiment_tag` – your embodiment label

Update those to match your environment.

---

## Quick smoke test (data + model forward pass)

If you have any trained reward model checkpoint (see “Training the reward model” below), you can quickly verify the full pipeline:

```bash
python RL_training_scripts/rewardinference.py \
  --checkpoint /path/to/your/checkpoint-or/final_reward_model.pt \
  --max_samples 50 \
  --prediction_only
```

What this does:
- Rebuilds the transforms and loads the dataset (check the script header constants for `DATASET_PATH`, `EMB_TAG` and update as needed).
- Loads the GR00T backbone + reward head and your checkpoint.
- Runs inference on a small number of samples and saves predictions to `reward_model_inference_results_<checkpoint>.npz`.
- In `--prediction_only` mode, the script won’t require ground-truth rewards. Remove that flag if your dataset does include `next.reward`.

If you see warnings about missing keys when loading the checkpoint, ensure your inference configuration (LoRA/no-LoRA, head sizes, reward horizon) matches how you trained (see “Match training and inference settings”).

---

## Train the policy (optional)

For completeness, you can fine-tune the policy (supervised) with:

```bash
python RL_training_scripts/Policy_train.py
```

Edit at top:
- `dataset_path`
- `output_dir`
- `model_path` (base model; default `"nvidia/GR00T-N1-2B"`)
- `embodiment_tag`

Notes:
- Uses bfloat16 and video transforms via `torchvision_av`. If `torchvision_av` gives issues, switch the backend to `"torchvision"` (see the script).
- This training does not produce a reward model; it’s for policy finetuning.

---

## Train the reward model (minimal)

Two main variants are provided.

1) Simple reward model training (no LoRA), 500 steps default:
```bash
python RL_training_scripts/Rewardmodel_train_new.py
```

2) Alternative reward training with pairwise ranking component:
```bash
python RL_training_scripts/505training.py
```

Edit at top of each:
- `dataset_path` (folder containing reward-annotated data)
- `output_dir`
- `base_model_path` (`"nvidia/GR00T-N1-2B"` or a policy checkpoint)
- `embodiment_tag`

Outputs:
- `Rewardmodel_train_new.py` saves `final_reward_model.pt` in `output_dir`.
- `505training.py` saves a Hugging Face-style directory at `output_dir/reward_model_final` and a `full_reward_model.pt` file.

Recommended for first-time test: `Rewardmodel_train_new.py` (fewer moving parts). You can also reduce `max_steps` in the script for a quicker smoke test.

---

## Train the reward model with periodic validation (recommended)

Use the Accelerate-compatible script:
```bash
python RL_training_scripts/validatetraining.py
```

What it does:
- Splits your dataset into train/val based on `validation_split`.
- Applies LoRA to the backbone (rank/alpha/dropout are defined near the top).
- Wraps the model so that the Trainer can always consume a `{"loss": ...}` dict.
- Trains and periodically runs custom validation, keeping the best model.

Edit at top:
- Paths: `dataset_path`, `output_dir`, `base_model_path`, `embodiment_tag`
- LoRA hyperparameters: `lora_rank`, `lora_alpha`, `lora_dropout`
- `validation_split` (default 0.1)

Outputs:
- Saves periodic Hugging Face checkpoints in `output_dir`.
- Saves `best_model.pt` (best val loss) and `final_reward_model.pt` at the end.
- Prints validation metrics (focal loss, BCE, accuracy).

Tip: Defaults use `bf16=True`, small batch size, no gradient checkpointing for simplicity. Adjust in `TrainingArguments` if needed.

---

## Evaluate a checkpoint (metrics + artifacts)

Use:
```bash
python RL_training_scripts/rewardinference.py \
  --checkpoint /path/to/checkpoint_dir_or_file \
  --max_samples 1000
```

The inference script:
- Accepts either a directory (it will search common filenames like `model.safetensors`, `pytorch_model.bin`, `final_reward_model.pt`) or a file path.
- Will compute:
  - MSE, MAE, Pearson correlation (if targets exist)
  - Binary accuracy after thresholding predictions at 0.5
- Saves:
  - `reward_model_inference_results_<checkpoint>.npz` (preds, and targets if available)
  - A scatter plot `<output>_plot.png` when targets exist

Flags:
- `--prediction_only` if your dataset does not contain ground-truth rewards (`next.reward`).
- `--max_samples` to limit processed samples for quicker runs.
- `--checkpoint` to override the default path hardcoded at the top.

Before running, update the constants at the top of the script:
- `DATASET_PATH`, `EMB_TAG`, `BASE_MODEL`, and optionally `CHECKPOINT_DIR`, `OUTPUT_PREFIX`.

---

## Match training and inference settings (important)

Several settings must be consistent between the training script you used and `rewardinference.py`:

- Reward transform normalization:
  - Many training scripts use:
    ```python
    RewardTransform(..., normalization_mode="min_max")
    ```
  - Ensure `rewardinference.py` uses the same normalization. In the file, check the `build_transforms()` function and set the same `normalization_mode` as your training script.

- Reward horizon and head size:
  - `validatetraining.py` configures `RewardHeadConfig(reward_horizon=16, hidden_dim=2048, dropout=0.10, reward_dim=1)`.
  - Ensure `rewardinference.py` uses the same `RewardHeadConfig` parameters.

- LoRA usage:
  - `validatetraining.py` applies LoRA (rank 16 by default).
  - `Rewardmodel_train_new.py` and `505training.py` do not apply LoRA by default.
  - `rewardinference.py` currently applies LoRA with rank 16 by default. If your training did not use LoRA, disable it in the inference script (set rank 0 or remove the LoRA wrapping) to avoid missing/unexpected key warnings.

- Video time dimension patch:
  - `rewardinference.py` patches `prepare_input` to squeeze `[B, 1, C, H, W]` to `[B, C, H, W]` for the visual encoder. Keep that patch if your dataset produces a single time step for the video.

---

## Troubleshooting

- Mixed precision (bfloat16) issues:
  - If your hardware doesn’t support bf16, change `bf16=True` to `False` and run in fp32, and consider reducing batch sizes.
  - Some layers may need fp32 master weights; a safety conversion pattern is used in some scripts already.

- Video backend:
  - If you hit errors with `video_backend="torchvision_av"`, try switching to `"torchvision"`.

- Check dataset keys:
  - If you see “No target reward found” in inference, ensure your dataset includes a reward signal. The script looks for `next.reward` first, but will scan any key containing “reward.”

- Accelerate and eval steps:
  - `validatetraining.py` uses a periodic validation wrapper. The defaults are reasonable (eval every ~500 steps, batch size 8). You can adjust by editing `TrainingArguments` in the script.

- Checkpoint loading:
  - On key mismatches, the script prints “Missing keys” / “Unexpected keys” warnings. Reconcile transform settings, head config, and LoRA usage.

---

## Minimal end-to-end checklist

1) Install:
   ```bash
   pip install -e .
   ```

2) Set dataset path and output dir in one reward training script (e.g., `Rewardmodel_train_new.py`), then run:
   ```bash
   python RL_training_scripts/Rewardmodel_train_new.py
   ```
   This will produce `final_reward_model.pt` in your `output_dir`.

3) Evaluate:
   ```bash
   python RL_training_scripts/rewardinference.py \
     --checkpoint /your/output_dir/final_reward_model.pt \
     --max_samples 200
   ```
   Check the printed metrics and the saved `.npz` and plot image.

Optional) Use `validatetraining.py` instead to get periodic validation and `best_model.pt`.

---

## Script index and roles

- Policy training:
  - [Policy_train.py](https://github.com/Hubi2883/GR00T_RL/blob/main/RL_training_scripts/Policy_train.py)

- Reward model training:
  - [Rewardmodel_train_new.py](https://github.com/Hubi2883/GR00T_RL/blob/main/RL_training_scripts/Rewardmodel_train_new.py) – straightforward regression against `next.reward`.
  - [505training.py](https://github.com/Hubi2883/GR00T_RL/blob/main/RL_training_scripts/505training.py) – regression + ranking loss variant.
  - [validatetraining.py](https://github.com/Hubi2883/GR00T_RL/blob/main/RL_training_scripts/validatetraining.py) – LoRA + periodic validation with Accelerate.

- Evaluation / inference:
  - [rewardinference.py](https://github.com/Hubi2883/GR00T_RL/blob/main/RL_training_scripts/rewardinference.py) – loads a checkpoint, runs predictions, computes metrics, and saves artifacts.

With these steps you can confirm that data loading, model construction, training, checkpointing, and evaluation are all working end-to-end.
```
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
```
