import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Import GR00T modules for dataset and policy.
from gr00t.data.dataset import LeRobotSingleDataset, ModalityConfig
from gr00t.model.policy import Gr00tPolicy
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform import (
    VideoToTensor,
    VideoCrop,
    VideoResize,
    VideoColorJitter,
    VideoToNumpy
)
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.model.transforms import GR00TTransform

def parse_intervals(intervals_str):
    """
    Parse a comma-separated list of intervals specified as "start:end" 
    into a list of (start, end) tuples.
    Example: "0:200,500:700" -> [(0,200), (500,700)]
    """
    intervals = []
    for part in intervals_str.split(","):
        try:
            start_str, end_str = part.split(":")
            start, end = int(start_str.strip()), int(end_str.strip())
            intervals.append((start, end))
        except Exception:
            raise argparse.ArgumentTypeError(
                f"Invalid interval format '{part}'. Expected format 'start:end'."
            )
    return intervals

def process_interval(start_idx, end_idx, dataset, policy, output_dir):
    """
    Process a given interval of samples, perform inference, and save a comparison plot.
    """
    gt_actions = []    # Ground-truth actions, shape: (num_samples, 16, num_joints)
    pred_actions = []  # Predicted actions for the first time step

    num_samples = end_idx - start_idx
    print(f"Processing samples from index {start_idx} to {end_idx} (total {num_samples} samples) ...")
    
    for i in range(start_idx, end_idx):
        sample = dataset[i]
        gt_cmd = np.array(sample["action.wheel_commands"])
        gt_actions.append(gt_cmd)

        with torch.no_grad():
            predicted_action = policy.get_action(sample)

        # Extract the predicted wheel commands for the first time step.
        if "action.wheel_commands" in predicted_action:
            if torch.is_tensor(predicted_action["action.wheel_commands"]):
                pred_cmd = predicted_action["action.wheel_commands"][0].detach().cpu().numpy()
            else:
                pred_cmd = predicted_action["action.wheel_commands"][0]
        else:
            print(f"Warning: Key 'action.wheel_commands' not found for sample {i}. Using ground-truth as fallback.")
            pred_cmd = gt_cmd[0]
        pred_actions.append(pred_cmd)

        if (i - start_idx) % 100 == 0:
            print(f"Processed sample {i}")

    gt_actions = np.array(gt_actions)      # Shape: (num_samples, 16, num_joints)
    pred_actions = np.array(pred_actions)    # Shape: (num_samples, num_joints)
    gt_actions_first = gt_actions[:, 0, :]     # Extract first time step

    time_steps = np.arange(num_samples)
    num_joints = gt_actions_first.shape[1]

    plt.figure(figsize=(10, 6))
    for joint_idx in range(num_joints):
        plt.plot(time_steps, gt_actions_first[:, joint_idx], label=f"GT joint {joint_idx}")
        plt.plot(time_steps, pred_actions[:, joint_idx], label=f"Pred joint {joint_idx}", linestyle="--")
    plt.title(f"Ground-Truth vs. Predicted Wheel Commands (First Time Step)\nInterval {start_idx}-{end_idx}")
    plt.xlabel("Sample Index")
    plt.ylabel("Joint Command Value")
    plt.legend()
    plt.tight_layout()

    # Save as a PDF (vector format allows zooming).
    plot_filename = f"wheel_commands_comparison_{start_idx}_{end_idx}.pdf"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot at: {plot_path}")

    return gt_actions, pred_actions

def main():
    parser = argparse.ArgumentParser(description="Inference on GR00T model with multiple intervals")
    parser.add_argument("--intervals", type=parse_intervals, required=True, default="0:200,200:400,1000:1200,10000:10200,15000:15200",
                        help="Comma-separated list of sample intervals in the format start:end (e.g., '0:200,200:400')")
    parser.add_argument("--model_path", type=str,
                        default="/ceph/home/student.aau.dk/xx06av/Groot_Implem/Isaac-GR00T/gr00t/results/checkpoint-7500_3state",
                        help="Path to the model checkpoint")
    parser.add_argument("--dataset_path", type=str,
                        default="/ceph/home/student.aau.dk/xx06av/Groot_Implem/Isaac-GR00T/gr00t/New_results_0002",
                        help="Path to the dataset")
    parser.add_argument("--embodiment_tag", type=str, default="new_embodiment",
                        help="Embodiment tag to use")
    parser.add_argument("--output_dir", type=str,
                        default="/ceph/home/student.aau.dk/xx06av/Groot_Implem/Isaac-GR00T/gr00t/inference_results",
                        help="Directory to store inference results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # Define modality configurations.
    video_modality = ModalityConfig(
        delta_indices=[0],
        modality_keys=["video.ego_view"],
    )
    # Updated: use keys with a "state." prefix and set delta_indices in increasing order: [-2, -1, 0]
    state_modality = ModalityConfig(
        delta_indices=[-2, -1, 0],
        modality_keys=["state.left_wheel", "state.right_wheel", "state.acceleration"],
    )
    action_modality = ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["action.wheel_commands"],
    )
    language_modality = ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    )
    modality_configs = {
        "video": video_modality,
        "state": state_modality,
        "action": action_modality,
        "language": language_modality,
    }

    transforms = ComposedModalityTransform(
        transforms=[
            VideoToTensor(apply_to=video_modality.modality_keys, backend="torchvision"),
            VideoCrop(apply_to=video_modality.modality_keys, scale=0.95, backend="torchvision"),
            VideoResize(
                apply_to=video_modality.modality_keys,
                height=224,
                width=224,
                interpolation="linear",
                backend="torchvision"
            ),
            VideoColorJitter(
                apply_to=video_modality.modality_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
                backend="torchvision"
            ),
            VideoToNumpy(apply_to=video_modality.modality_keys),
            StateActionToTensor(apply_to=state_modality.modality_keys),
            StateActionTransform(
                apply_to=state_modality.modality_keys,
                normalization_modes={
                    "state.left_wheel": "min_max",
                    "state.right_wheel": "min_max",
                    "state.acceleration": "min_max",
                },
            ),
            StateActionToTensor(apply_to=action_modality.modality_keys),
            StateActionTransform(
                apply_to=action_modality.modality_keys,
                normalization_modes={"action.wheel_commands": "min_max"},
            ),
            ConcatTransform(
                video_concat_order=video_modality.modality_keys,
                state_concat_order=state_modality.modality_keys,
                action_concat_order=action_modality.modality_keys,
            ),
            GR00TTransform(
                state_horizon=len(state_modality.delta_indices),
                action_horizon=len(action_modality.delta_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
    )

    # Load the GR00T Policy.
    print("Loading policy from checkpoint:", args.model_path)
    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=args.embodiment_tag,
        modality_config=modality_configs,
        modality_transform=transforms,
        device=device,
    )
    print("Policy model architecture:")
    print(policy.model)

    # Load the Dataset.
    print("Loading dataset from:", args.dataset_path)
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality_configs,
        video_backend="torchvision_av",
        transforms=None,  # Transforms are applied in the policy.
        embodiment_tag=args.embodiment_tag,
    )
    print("Loaded dataset with {} samples.".format(len(dataset)))

    # Process each specified interval.
    for (start_idx, end_idx) in args.intervals:
        process_interval(start_idx, end_idx, dataset, policy, args.output_dir)

if __name__ == "__main__":
    main()
