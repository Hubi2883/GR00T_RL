import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Import GR00T modules for dataset, policy, and transforms.
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

# -------------------------
# Inline modality configuration and transforms.
# -------------------------
# Video modality.
video_modality = ModalityConfig(
    delta_indices=[0],
    modality_keys=["video.ego_view"],
)
 
# State modality: three state components.
state_modality = ModalityConfig(
    delta_indices=[-2, -1, 0],
    modality_keys=["state.left_wheel", "state.right_wheel", "state.acceleration"],
)
 
# Action modality: a sequence of 16 joint commands.
action_modality = ModalityConfig(
    delta_indices=list(range(16)),
    modality_keys=["action.wheel_commands"],
)
 
# Language modality.
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
        # State transforms.
        StateActionToTensor(apply_to=state_modality.modality_keys),
        StateActionTransform(
            apply_to=state_modality.modality_keys,
            normalization_modes={
                "state.left_wheel": "min_max",
                "state.right_wheel": "min_max",
                "state.acceleration": "min_max",
            },
        ),
        # Action transforms.
        StateActionToTensor(apply_to=action_modality.modality_keys),
        StateActionTransform(
            apply_to=action_modality.modality_keys,
            normalization_modes={"action.wheel_commands": "min_max"},
        ),
        # Concatenate modalities.
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

# -------------------------
# Custom iterative evaluation function.
# -------------------------
def calc_mse_for_single_trajectory_iterative(
    policy,
    dataset,
    traj_id: int,
    modality_keys: list,
    steps=300,
    action_horizon=16,
    plot=False,
    plot_dir=None,
):
    """
    This custom evaluation function performs iterative rollout.
    It uses key_map so that evaluation keys (e.g., "left_wheel", "right_wheel")
    are extracted from the 'action.wheel_commands' array.
    
    At each inference point (every action_horizon steps), it performs iterative rollout,
    updating the input with the predicted action for the next step.
    
    The function computes the MSE across time and saves a PDF plot if plot=True.
    """
    state_joints_across_time = []
    gt_action_joints_across_time = []
    pred_action_joints_across_time = []
    
    key_map = {"left_wheel": 0, "right_wheel": 1}
    
    for step_count in range(steps):
        data_point = dataset.get_step_data(traj_id, step_count)
        gt_action = np.array(data_point["action.wheel_commands"])[0]
        concat_gt_action = gt_action[[key_map[k] for k in modality_keys]]
        gt_action_joints_across_time.append(concat_gt_action)
    
        try:
            concat_state = np.concatenate(
                [data_point[f"state.{k}"][0] for k in modality_keys], axis=0
            )
        except KeyError:
            concat_state = None
        if concat_state is not None:
            state_joints_across_time.append(concat_state)
    
        # At steps where inference is triggered, perform iterative rollout.
        if step_count % action_horizon == 0:
            print("Inferencing at step:", step_count)
            # Get a copy of the current data_point as the initial input.
            current_dp = {k: np.copy(v) for k, v in data_point.items()}
            # Ensure all state keys are float64.
            for k in current_dp:
                if k.startswith("state."):
                    current_dp[k] = np.array(current_dp[k]).astype(np.float64)
            # Remove any annotation keys.
            keys_to_remove = [k for k in current_dp if k.startswith("annotation.")]
            for k in keys_to_remove:
                del current_dp[k]
    
            rollout_preds = []
            for j in range(action_horizon):
                action_chunk = policy.get_action(current_dp)
                pred_action = np.array(action_chunk["action.wheel_commands"])[0]
                concat_pred_action = pred_action[[key_map[k] for k in modality_keys]]
                rollout_preds.append(concat_pred_action)
                # Update current_dp's action field with the new prediction (as a NumPy array, float64).
                current_dp["action.wheel_commands"] = np.expand_dims(concat_pred_action.astype(np.float64), axis=0)
            for pred in rollout_preds:
                pred_action_joints_across_time.append(pred)
    
    gt_action_joints_across_time = np.array(gt_action_joints_across_time)
    pred_action_joints_across_time = np.array(pred_action_joints_across_time)[:steps]
    
    assert gt_action_joints_across_time.shape == pred_action_joints_across_time.shape, (
        "Shape mismatch:",
        gt_action_joints_across_time.shape,
        pred_action_joints_across_time.shape,
    )
    
    mse = np.mean((gt_action_joints_across_time - pred_action_joints_across_time) ** 2)
    print("Iterative rollout: Unnormalized Action MSE across single trajectory:", mse)
    
    num_of_joints = gt_action_joints_across_time.shape[1]
    
    if plot:
        fig, axes = plt.subplots(nrows=num_of_joints, ncols=1, figsize=(8, 4 * num_of_joints))
        fig.suptitle(
            f"Trajectory {traj_id} - Evaluation Modalities: {', '.join(modality_keys)}",
            fontsize=16,
            color="blue",
        )
    
        for i, ax in enumerate(axes):
            ax.plot(gt_action_joints_across_time[:, i], label="GT action joints")
            ax.plot(pred_action_joints_across_time[:, i], label="Pred action joints")
            for j in range(0, steps, action_horizon):
                if j == 0:
                    ax.plot(j, gt_action_joints_across_time[j, i], "ro", label="Inference point")
                else:
                    ax.plot(j, gt_action_joints_across_time[j, i], "ro")
            ax.set_title(f"Joint {i}")
            ax.legend()
    
        plt.tight_layout()
        if plot_dir is None:
            plot_dir = "."
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"trajectory_{traj_id}_evaluation.pdf")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        print(f"Saved evaluation plot at: {plot_path}")
    
    return mse

# -------------------------
# Main function.
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Inference on GR00T model using iterative trajectory evaluation (custom calc_mse_for_single_trajectory)"
    )
    parser.add_argument("--model_path", type=str,
                        default="/ceph/home/student.aau.dk/xx06av/Groot_Implem/Isaac-GR00T/gr00t/results/checkpoint-7500_3state",
                        help="Path to the model checkpoint")
    parser.add_argument("--dataset_path", type=str,
                        default="/ceph/home/student.aau.dk/xx06av/Groot_Implem/Isaac-GR00T/gr00t/New_results_0002",
                        help="Path to the dataset")
    parser.add_argument("--embodiment_tag", type=str, default="new_embodiment",
                        help="Embodiment tag to use")
    parser.add_argument("--traj_id", type=int, default=45,
                        help="Trajectory ID for evaluation")
    parser.add_argument("--steps", type=int, default=300,
                        help="Number of steps to evaluate for the trajectory")
    parser.add_argument("--action_horizon", type=int, default=16,
                        help="Action horizon for prediction")
    parser.add_argument("--modality_keys", nargs='+', default=["left_wheel", "right_wheel"],
                        help="List of modality keys to evaluate (will be mapped to indices in wheel_commands)")
    parser.add_argument("--output_dir", type=str,
                        default="/ceph/home/student.aau.dk/xx06av/Groot_Implem/Isaac-GR00T/inference_results",
                        help="Directory to store inference results/plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=args.embodiment_tag,
        modality_config=modality_configs,
        modality_transform=transforms,
        device=device,
    )
    print("Policy model architecture:")
    print(policy.model)
    
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality_configs,
        video_backend="decord",
        video_backend_kwargs=None,
        transforms=None,
        embodiment_tag=args.embodiment_tag,
    )
    print("Loaded dataset with {} samples.".format(len(dataset)))
    
    mse = calc_mse_for_single_trajectory_iterative(
        policy,
        dataset,
        traj_id=args.traj_id,
        modality_keys=args.modality_keys,
        steps=args.steps,
        action_horizon=args.action_horizon,
        plot=True,
        plot_dir=args.output_dir,
    )
    print(f"MSE loss for trajectory {args.traj_id}: {mse}")
    
if __name__ == "__main__":
    main()
