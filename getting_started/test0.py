import os
import gr00t
import matplotlib.pyplot as plt

from gr00t.utils.misc import any_describe
from gr00t.data.dataset import LeRobotSingleDataset, ModalityConfig
from gr00t.data.schema import EmbodimentTag

# Set your dataset path (update as needed).
DATA_PATH = "/ceph/home/student.aau.dk/wb68dm/Isaac-GR00T_RL/New_results_0001"

# Update modality configurations to match the parquet file column names.
modality_configs = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["video.ego_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["state.acceleration"],
    ),
    "action": ModalityConfig(
        delta_indices=[0],
        modality_keys=["action.wheel_commands"],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description", "annotation.human.validity"],
    ),
}


# Set the appropriate embodiment tag (adjust if necessary)
embodiment_tag = EmbodimentTag.GR1

def load_and_test_dataset():
    print("Loading dataset... from", DATA_PATH)
    dataset = LeRobotSingleDataset(DATA_PATH, modality_configs, embodiment_tag=embodiment_tag)
    print("Initialized dataset with EmbodimentTag.GR1")

    # Display details for a sample data point.
    print("\n" + "=" * 100)
    print(f"{' Updated Robot Dataset ':=^100}")
    print("=" * 100)
    resp = dataset[7]
    any_describe(resp)
    print("Data keys:", list(resp.keys()))

    # Optionally, display video frames.
    images_list = []
    for i in range(100):
        if i % 10 == 0:
            resp = dataset[i]
            # Assuming video modality is stored as a list/array with frames
            img = resp["video.ego_view"][0]
            images_list.append(img)

    fig, axs = plt.subplots(2, 5, figsize=(20, 10))
    for i, ax in enumerate(axs.flat):
        ax.imshow(images_list[i])
        ax.axis("off")
        ax.set_title(f"Image {i}")
    plt.tight_layout()
    plt.show()
    plt.savefig("/ceph/home/student.aau.dk/wb68dm/Results/test0.png")

if __name__ == "__main__":
    load_and_test_dataset()
