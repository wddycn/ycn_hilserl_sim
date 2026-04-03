import torch
import gymnasium as gym
import numpy as np
import gym_hil

from lerobot.policies.sac.modeling_sac import SACPolicy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convert_obs(obs):
    """
    把 gym_hil 的 observation 转成训练时的 key 格式
    """

    return {
        "observation.images.front": torch.tensor(
            obs["pixels"]["front"], dtype=torch.float32
        ).permute(2, 0, 1).unsqueeze(0).to(DEVICE),  # HWC -> BCHW

        "observation.images.wrist": torch.tensor(
            obs["pixels"]["wrist"], dtype=torch.float32
        ).permute(2, 0, 1).unsqueeze(0).to(DEVICE),

        "observation.state": torch.tensor(
            obs["agent_pos"], dtype=torch.float32
        ).unsqueeze(0).to(DEVICE),
    }


def main():
    env = gym.make(
        "gym_hil/PandaPickCubeGamepad-v0",
        render_mode="human",
        image_obs=True,
        use_gripper=True,
        gripper_penalty=-0.02,
    )

    policy = SACPolicy.from_pretrained(
        "/home/ycn/outputs/train/2026-02-15/12-21-33_franka_sim/checkpoints/012000/pretrained_model"
    ).to(DEVICE)
    policy.eval()

    num_episodes = 100
    success_count = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = truncated = False

        while not (done or truncated):
            # ✅ 使用 policy.select_action，而非直接调 actor
            model_input = convert_obs(obs)
            action = policy.select_action(model_input)  # 返回 [4,] 或 [1,4]
            action = action.detach().cpu().numpy()
            
            # 如果 select_action 返回 [1,4]，则 squeeze
            if action.ndim == 2:
                action = action[0]

            obs, reward, done, truncated, info = env.step(action)

        if info.get("succeed", False):
            print(f"Episode {ep+1}: SUCCESS")
            success_count += 1
        else:
            print(f"Episode {ep+1}: FAILURE")

    print("=================================")
    print(f"Success rate: {success_count}/{num_episodes}")
    env.close()

if __name__ == "__main__":
    main()
