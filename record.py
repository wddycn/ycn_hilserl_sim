import time
from pathlib import Path

import gymnasium as gym
import numpy as np

import gym_hil
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    # ===== 1. 创建 gym_hil 键盘环境 =====
    env = gym.make(
        "gym_hil/PandaPickCubeGamepad-v0",
        render_mode="human",
        image_obs=True,
        use_gripper=True,
        gripper_penalty=-0.02,
    )

    obs, info = env.reset()

    # 取一次观测，推断 feature 的 shape
    front = obs["pixels"]["front"]   # (H, W, C)
    wrist = obs["pixels"]["wrist"]
    state = obs["agent_pos"]         # (18,)

    # ===== 2. 定义 LeRobotDataset 的特征 schema =====
    features = {
        "observation.images.front": {
            "dtype": "video",
            "shape": front.shape,   # (H, W, C)
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": wrist.shape,
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": state.shape,
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": env.action_space.shape,  # (4,)
            "names": None,
        },
        "next.reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        "next.done": {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        },
    }

    # ===== 3. 创建 LeRobotDataset =====
    # 你可以复用原来的 repo_id 和 root
    repo_id = "wddycn/hilserl_sim_pick_lift"
    root = Path("/home/ycn/hilserl_sim/lerobot/src/lerobot/ycn_hilserl_sim/ycn_records_gamepad")  

    fps = 10  # 跟 env_config 里的保持一致

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=root,
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=0,
        features=features,
    )

    num_episodes = 30  # 想录多少，跟你 json 里一致

    print(f"[INFO] Start recording {num_episodes} episodes with keyboard teleop.")
    print("       Space 开启干预后，按键控制机械臂；成功/失败用环境默认的键盘操作结束一局。")

    for ep in range(num_episodes):
        print(f"\n========== Episode {ep} ==========")
        obs, info = env.reset()
        done = False
        t = 0

        # 上一次动作用于“没有干预”时填充
        last_action = np.zeros(env.action_space.shape, dtype=np.float32)

        while not done:
            start_t = time.perf_counter()

            dummy_action = np.zeros(env.action_space.shape, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(dummy_action)

            is_intervention = info.get("is_intervention", False)

            if is_intervention and "teleop_action" in info:
                raw_action = np.array(info["teleop_action"], dtype=np.float32)
                last_action = raw_action
            else:
                raw_action = last_action

            # ===== 实时打印 action =====
            print(
                f"[Step {t:04d}] "
                f"action = {raw_action} | "
                f"shape = {raw_action.shape} | "
                f"intervention = {is_intervention}"
            )

            action_to_log = raw_action

            frame = {
                "observation.images.front": obs["pixels"]["front"],
                "observation.images.wrist": obs["pixels"]["wrist"],
                "observation.state": obs["agent_pos"].astype(np.float32),
                "action": action_to_log,
                "next.reward": np.array([reward], dtype=np.float32),
                "next.done": np.array([terminated or truncated], dtype=bool),
                "task": "PandaPickCubeKeyboard-v0",
            }

            dataset.add_frame(frame)
            t += 1

            if terminated or truncated:
                print(f"[INFO] Episode {ep} ended at step {t}, reward={reward}")
                dataset.save_episode()
                done = True

            dt = time.perf_counter() - start_t
            sleep_t = max(0.0, 1.0 / fps - dt)
            time.sleep(sleep_t)

    env.close()

    # 如果你想 push_to_hub，就在这里调用
    print("[INFO] All episodes recorded. Pushing to hub...")
    dataset.push_to_hub()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()