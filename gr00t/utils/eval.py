import matplotlib.pyplot as plt
import numpy as np

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import BasePolicy

# numpy print precision settings: 3, no scientific notation
np.set_printoptions(precision=3, suppress=True)


def download_from_hg(repo_id: str, repo_type: str) -> str:
    """
    Download the model/dataset from the hugging face hub.
    Return the path to the downloaded resource.
    """
    from huggingface_hub import snapshot_download

    repo_path = snapshot_download(repo_id, repo_type=repo_type)
    return repo_path


def calc_mse_for_single_trajectory(
    policy: BasePolicy,
    dataset: LeRobotSingleDataset,
    traj_id: int,
    modality_keys: list,
    steps=300,
    action_horizon=16,
    plot=False,
):
    """
    Custom evaluation function that computes the MSE for a single trajectory.
    
    Instead of expecting action keys like "action.left_wheel", it uses the 
    "action.wheel_commands" array and maps the evaluation modality keys to indices.
    
    For example, if modality_keys is ["left_wheel", "right_wheel"],
    then it extracts index 0 for left_wheel and index 1 for right_wheel.
    
    If plot is True, the function saves a PDF plot.
    """
    state_joints_across_time = []
    gt_action_joints_across_time = []
    pred_action_joints_across_time = []

    # Define mapping from evaluation key to index in the wheel_commands array.
    key_map = {"left_wheel": 0, "right_wheel": 1}

    for step_count in range(steps):
        data_point = dataset.get_step_data(traj_id, step_count)

        # For state, assume the keys in modality_keys do exist as "state.<key>"
        try:
            concat_state = np.concatenate(
                [data_point[f"state.{key}"][0] for key in modality_keys], axis=0
            )
        except KeyError:
            concat_state = None
        if concat_state is not None:
            state_joints_across_time.append(concat_state)

        # For ground truth action, use "action.wheel_commands" and extract the indices.
        full_gt_action = np.array(data_point["action.wheel_commands"])[0]
        concat_gt_action = np.concatenate(
            [np.atleast_1d(full_gt_action[key_map[key]]) for key in modality_keys], axis=0
        )
        gt_action_joints_across_time.append(concat_gt_action)

        # At every action_horizon step, infer prediction.
        if step_count % action_horizon == 0:
            print("Inferencing at step:", step_count)
            action_chunk = policy.get_action(data_point)
            full_pred_action = np.array(action_chunk["action.wheel_commands"])[0]
            for j in range(action_horizon):
                concat_pred_action = np.concatenate(
                    [np.atleast_1d(full_pred_action[key_map[key]]) for key in modality_keys],
                    axis=0,
                )
                pred_action_joints_across_time.append(concat_pred_action)

    # Convert lists to numpy arrays.
    state_joints_across_time = np.array(state_joints_across_time)
    gt_action_joints_across_time = np.array(gt_action_joints_across_time)
    # Only take as many predictions as steps.
    pred_action_joints_across_time = np.array(pred_action_joints_across_time)[:steps]

    # Assert shapes match.
    assert (
        state_joints_across_time.shape
        == gt_action_joints_across_time.shape
        == pred_action_joints_across_time.shape
    ), f"Shape mismatch: {state_joints_across_time.shape}, {gt_action_joints_across_time.shape}, {pred_action_joints_across_time.shape}"

    mse = np.mean((gt_action_joints_across_time - pred_action_joints_across_time) ** 2)
    print("Unnormalized Action MSE across single trajectory:", mse)

    num_of_joints = state_joints_across_time.shape[1]

    if plot:
        fig, axes = plt.subplots(nrows=num_of_joints, ncols=1, figsize=(8, 4 * num_of_joints))
        fig.suptitle(
            f"Trajectory {traj_id} - Modalities: {', '.join(modality_keys)}",
            fontsize=16,
            color="blue",
        )

        for i, ax in enumerate(axes):
            ax.plot(state_joints_across_time[:, i], label="state joints")
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
        plot_path = f"trajectory_{traj_id}_evaluation.pdf"
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        print("Saved evaluation plot at:", plot_path)

    return mse
