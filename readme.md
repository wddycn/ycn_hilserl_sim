# 配置环境
## lerobot
创建虚拟环境
```bash
conda create -n hilserl_sim python=3.10
```
install ffmpeg
```bash
conda install ffmpeg -c conda-forge
```
First, clone the repository and navigate into the directory:
```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
```
Then, install the library in editable mode. This is useful if you plan to contribute to the code.
```bash
pip install -e .
```
## hilserl
```bash
cd ..
git clone https://github.com/HuggingFace/gym-hil.git && cd gym-hil
pip install -e .
```
# 准备工作
## 开代理
终端输入：
```bash
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
```
## 终端配置wandb
安装
```bash
pip install wandb
```
[登录](https://wandb.ai/ycn-)，先创建一个项目，得到API
```bash
****************************************(自己创建)
```

终端配置wandb
```bash
wandb login

# 提示输入wandb的API，输入即可
****************************************(自己创建)
```
## 终端配置hugging face
安装
```bash
pip install huggingface_hub
```
[登录](https://huggingface.co/)注册账号，点击setting，点击access token，点击create new token，拿到API KEY（只会在创建的时候显示一次）
```
****************************************(自己创建)
```
如果忘了API KEY，可以在点击之前创建的token右边的三个点，随后invalidate and refresh

终端配置hugging face
```bash
hf auth login

# 提示输入hub的token，输入即可
****************************************(自己创建)
```
# 新建ycn_hilserl_sim
## 录制数据集record.py
运行
```bash
python -m lerobot.ycn_hilserl_sim.record
```
- 可以在gym-hil的`gym-hil/gym_hil/__init__.py`里面找各个环境当中的`max_episode_steps`参数进行修改，意思是一条episode最多需要多少step
- 如何计算每训练一次的步长呢，首先要清楚`一条训练数据 = 一个 episode`。而在`recoed.py`里面有`"fps": 10`参数，意思是一秒执行10step，如果修改`max_episode_steps=100`的话就意味着每次训练最多执行100step，那就是说“一次 episode 最长时间 = 100 / 10 = 10 秒”。
- 在`gym-hil/gym_hil/envs/panda_pick_gym_env.py`当中的环境初始化里面可以设置是否随机重置小方块位置
- 录制数据集时需要将小方块夹起来并提升超过0.1米
## 修改训练配置文件（train.json）
- 运行`state.py`脚本，**（运行前确保repo_id和root地址正确）**，会生成一个`dataset_stats.json`文件
```bash
python -m lerobot.ycn_hilserl_sim.state
```
- 根据官网教程写一个`train_gym_hil_env.json`文件，然后将`dataset_stats.json`文件当中的几个参数与`train_gym_hil_env.json`对齐（尤其是"observation.images.wrist"，"observation.state"，"action"，"observation.images.front"）
- 其余可修改：确保环境名一致，repo_id和root地址正确，step——总步长；batcg_size——批次大小（配置高可以适当提高）；job_name——wandb训练曲线名称
## 训练
- 先准备两个终端，一个跑learner，一个跑action
- 在两个终端里都开代理
```bash
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
```
- 在准备跑learner的终端里登录wandb，命令是
```bash
wandb login
# api是
****************************************(自己创建)
```
- 在准备跑learner的终端里登录huggingface，命令是
```bash
hf auth login
# token是
****************************************(自己创建)
```
- （先启动）在准备跑learner的终端里运行
```bash
python -m lerobot.rl.learner --config_path path/to/train_gym_hil_env.json
```
- （后启动）在准备跑action的终端里
```bash
python -m lerobot.rl.actor --config_path path/to/train_gym_hil_env.json
```
---
---
---
---
# 以上实验失败原因
主要原因是因为action的维度不匹配，主要的排查点是在`lerobot.policies.sac.modeling_sac`当中

在 SACPolicy.critic_forward() 方法中，有这段代码：

真正的问题可能出在这里：select_action 的拼接逻辑
看 select_action 方法：
```python
def select_action(self, batch: dict[str, Tensor]) -> Tensor:
    actions, _, _ = self.actor(batch, observations_features)

    if self.config.num_discrete_actions is not None:
        discrete_action_value = self.discrete_critic(batch, observations_features)
        discrete_action = torch.argmax(discrete_action_value, dim=-1, keepdim=True)
        actions = torch.cat([actions, discrete_action], dim=-1)  # ← 拼接！

    return actions
```
⚠️ 关键点：

    如果 num_discrete_actions is not None，就会把 连续 action（4D） + 离散 action（1D） → 5D
    但我的 train_gym_hil_env.json 中 num_discrete_actions 是多少？

✅ 找到了！问题根源就在这里！

我设置了：
```json
"num_discrete_actions": 3
```
这意味着：

    连续 action：4 维（dx, dy, dz, gripper）
    离散 action：1 维（比如 0/1/2 三种模式）
    总 action：4 + 1 = 5 维

但他的 环境只接受 4 维 action！

==>**因此只需要将3改成null即可**
⚠️ 但经过实验，这样把第四维变成连续的action的话，在训练的时候夹爪不会执行抓取动作，因为在录制数据集的时候第4维就是离散的

# 真正解决方法（一）

按照5维匹配不了4维的报错找到 buffer.py 当中的 concatenate_batch_transitions（）函数，让5维裁剪成4维（12345->1235）

但后面又报错195匹配不了196的错，于是我继续在这个函数当中修改，让他裁剪完成通过检测之后再继续恢复到5维，只不过现在让4=5（离散）

最后又报错说32key匹配不了64key，于是还是在这个函数当中做修改，将32key其他的补0补成64key

最终得到了函数：
```python
def concatenate_batch_transitions(
    left_batch_transitions: BatchTransition,
    right_batch_transition: BatchTransition,
) -> BatchTransition:

    # ========= 1️⃣ 处理 state =========
    left_batch_transitions["state"] = {
        key: torch.cat(
            [
                left_batch_transitions["state"][key],
                right_batch_transition["state"][key],
            ],
            dim=0,
        )
        for key in left_batch_transitions["state"]
    }

    # ========= 2️⃣ 处理 ACTION =========
    left_action = left_batch_transitions[ACTION]
    right_action = right_batch_transition[ACTION]

    # 如果是 5 维，则裁掉第 4 维 (index=3)
    if left_action.shape[1] == 5:
        left_action = torch.cat([left_action[:, :3], left_action[:, 4:]], dim=1)

    if right_action.shape[1] == 5:
        right_action = torch.cat([right_action[:, :3], right_action[:, 4:]], dim=1)

    # merge
    merged_action = torch.cat([left_action, right_action], dim=0)

    # 恢复成 5 维（index=3 补 0）
    # merged_action 现在是 4 维：[a0, a1, a2, discrete]

    discrete_column = merged_action[:, 3:4]  # 取 discrete

    merged_action = torch.cat(
        [merged_action[:, :3], discrete_column, merged_action[:, 3:]],
        dim=1,
    )

    left_batch_transitions[ACTION] = merged_action

    # ========= 3️⃣ reward =========
    left_batch_transitions["reward"] = torch.cat(
        [
            left_batch_transitions["reward"],
            right_batch_transition["reward"],
        ],
        dim=0,
    )

    # ========= 4️⃣ next_state =========
    left_batch_transitions["next_state"] = {
        key: torch.cat(
            [
                left_batch_transitions["next_state"][key],
                right_batch_transition["next_state"][key],
            ],
            dim=0,
        )
        for key in left_batch_transitions["next_state"]
    }

    # ========= 5️⃣ done =========
    left_batch_transitions["done"] = torch.cat(
        [
            left_batch_transitions["done"],
            right_batch_transition["done"],
        ],
        dim=0,
    )

    # ========= 6️⃣ truncated =========
    left_batch_transitions["truncated"] = torch.cat(
        [
            left_batch_transitions["truncated"],
            right_batch_transition["truncated"],
        ],
        dim=0,
    )

    # ========= 7️⃣ complementary_info（关键修复部分） =========
    left_info = left_batch_transitions.get("complementary_info")
    right_info = right_batch_transition.get("complementary_info")

    if left_info is None and right_info is None:
        return left_batch_transitions

    if left_info is None:
        left_info = {}
        left_batch_transitions["complementary_info"] = left_info

    if right_info is None:
        right_info = {}

    all_keys = set(left_info.keys()).union(set(right_info.keys()))

    for key in all_keys:

        left_value = left_info.get(key)
        right_value = right_info.get(key)

        # 如果某一侧没有该 key，自动补零
        if left_value is None and right_value is not None:
            left_value = torch.zeros_like(right_value)

        if right_value is None and left_value is not None:
            right_value = torch.zeros_like(left_value)

        # 两边都存在才拼接
        if left_value is not None and right_value is not None:
            left_info[key] = torch.cat([left_value, right_value], dim=0)

    return left_batch_transitions

```
# 真正解决方法（二）
## actor当中做裁剪
将315行
```python
        executed_action = new_transition[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"]
```
替换成：
```python
        #CZL
        # executed_action = new_transition[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"]

        # if isinstance(executed_action, torch.Tensor):
        #     # 有时是 (1, 4) / (1, 5)，先压掉 batch 维
        #     if executed_action.ndim == 2 and executed_action.shape[0] == 1:
        #         executed_action = executed_action.squeeze(0)

        #     # 如果是 5 维或更多，就保留前三维 + 最后一维
        #     if executed_action.ndim == 1 and executed_action.shape[0] > 4:
        #         dx_dy_dz = executed_action[:3]
        #         gripper = executed_action[-1].unsqueeze(0)
        #         executed_action = torch.cat([dx_dy_dz, gripper], dim=0)

        complementary = new_transition[TransitionKey.COMPLEMENTARY_DATA]

        # 1）优先用 teleop_action（人类干预）
        # 2）如果没有 teleop_action（例如 gym_hil + keyboard），就用策略动作
        if "teleop_action" in complementary:
            executed_action = complementary["teleop_action"]
        else:
            executed_action = new_transition[TransitionKey.ACTION]

        # --- 把 executed_action 强制整理成 4 维 [dx, dy, dz, gripper] ---
        # 目标：前三维是平移，最后一维是离散 gripper（0/1/2）
        if isinstance(executed_action, torch.Tensor):
            # 可能是 (1, 4) / (1, 5)，先压掉 batch 维
            if executed_action.ndim == 2 and executed_action.shape[0] == 1:
                executed_action = executed_action.squeeze(0)

            # 如果是 5 维（或更长），就取前三维 + 最后一维
            if executed_action.ndim == 1 and executed_action.shape[0] > 4:
                dx_dy_dz = executed_action[:3]
                gripper = executed_action[-1].unsqueeze(0)
                executed_action = torch.cat([dx_dy_dz, gripper], dim=0)

        elif isinstance(executed_action, (list, tuple)):
            # 有些路径里可能是 list/tuple，这里也统一成 tensor 再裁
            executed_action = torch.tensor(executed_action, dtype=torch.float32)
            if executed_action.ndim == 2 and executed_action.shape[0] == 1:
                executed_action = executed_action.squeeze(0)
            if executed_action.ndim == 1 and executed_action.shape[0] > 4:
                dx_dy_dz = executed_action[:3]
                gripper = executed_action[-1].unsqueeze(0)
                executed_action = torch.cat([dx_dy_dz, gripper], dim=0)

        else:
            logging.warning(f"[ACTOR] Unexpected executed_action type: {type(executed_action)}")
```
## 策略重构
1.替换critic_forward（）：
```python
    def critic_forward(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        use_target: bool = False,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through a critic network ensemble

        Args:
            observations: Dictionary of observations
            actions: Action tensor
            use_target: If True, use target critics, otherwise use ensemble critics

        Returns:
            Tensor of Q-values from all critics
        """
        dx_dy_dz = actions[..., :3]      # (..., 3)czl
        gripper = actions[..., -1:]      # (..., 1)
        actions = torch.cat([dx_dy_dz, gripper], dim=-1)  # (..., 4)

        critics = self.critic_target if use_target else self.critic_ensemble
        q_values = critics(observations, actions, observation_features)
        return q_values
```
2.注释284-288行
```python
        # 3- compute predicted qs
        # if self.config.num_discrete_actions is not None:
            # NOTE: We only want to keep the continuous action part
            # In the buffer we have the full action space (continuous + discrete)
            # We need to split them before concatenating them in the critic forward
            # actions: Tensor = actions[:, :DISCRETE_DIMENSION_INDEX]
```
3.将353行附近
```python
            # if discrete_penalties is not None:czl
            #     rewards_discrete = rewards + discrete_penalties
            # target_discrete_q = rewards_discrete + (1 - done) * self.config.discount * target_next_discrete_q
```
替换成：
```python
            if discrete_penalties is not None:
                # --- 安全检查：只有 batch 维度一致才叠加 penalty ---
                if discrete_penalties.shape[0] == rewards.shape[0]:
                    # 如果 penalty 少一维，比如 [B]，而 rewards 是 [B,1]，也可以兼容一下
                    if discrete_penalties.ndim == 1 and rewards_discrete.ndim == 2:
                        discrete_penalties = discrete_penalties.unsqueeze(-1)
                    rewards_discrete = rewards_discrete + discrete_penalties
                # 否则直接忽略 penalty，避免维度错误
            target_discrete_q = rewards_discrete + (1 - done) * self.config.discount * target_next_discrete_q
```

---
---
---
---
---
# 其他一些功能函数
## 新建franka_environment_quick_start.py
测试环境是否能打开，这个不会打开环境，只会输出一段视屏
## 新建joy_debug.py
可以查看手柄按键对应数据
## 新建check_gamepad.py
可以查看手柄类型名称（对应后面要用到的controller_config.json配置文件）
## 新建dof_test.py
查看数据集action的维度
## 新建upload_to_hub.py
将本地数据集上传到hub上
# 部署策略
## 新建eval_policy.py
看着训练的差不多了之后结束终端退出，模型会保存在learner终端一开始打印出来output的保存地址，一般一次训练会生成两套时间，选择时间靠前的文件夹打开，点击checkpoint，选择数字最大的文件夹打开，pretrained_model里面的文件就是训练好的策略，复制pretrained_model路径粘贴到eval_policy.py里面对应的策略路径位置

