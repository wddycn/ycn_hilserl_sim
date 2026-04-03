from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    # ===== 1. 填你自己的 repo_id =====
    repo_id = "wddycn/hilserl_sim_pick_lift"

    # ===== 2. 本地数据集 root 路径（必须是你之前 create 时用的那个目录）=====
    root = Path(
        "/home/ycn/hilserl_sim/lerobot/src/lerobot/ycn_hilserl_sim/ycn_records_gamepad"
    )

    print(f"[INFO] Loading local dataset from: {root}")

    # ===== 3. 从本地加载数据集 =====
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
    )

    print("[INFO] Dataset loaded successfully.")
    print("[INFO] Start pushing to Hugging Face Hub...")

    # ===== 4. 上传到 Hub =====
    dataset.push_to_hub()

    print("[INFO] Upload finished successfully!")


if __name__ == "__main__":
    main()
