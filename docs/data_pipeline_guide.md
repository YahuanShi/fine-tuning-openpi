# OpenPi 数据格式与模型输入指南

本文档梳理 OpenPi 项目中模型输入的完整定义、各机器人数据格式、LeRobot 存储格式，以及从原始数据到模型输入的完整流水线。

---

## 目录

1. [模型输入结构（Observation）](#1-模型输入结构observation)
2. [各机器人 State/Action 空间详解](#2-各机器人-stateaction-空间详解)
3. [LeRobot 数据格式](#3-lerobot-数据格式)
4. [数据转换脚本](#4-数据转换脚本)
5. [训练数据流水线](#5-训练数据流水线)
6. [归一化统计量](#6-归一化统计量)

---

## 1. 模型输入结构（Observation）

模型输入由 `Observation` 类定义（`src/openpi/models/model.py`），包含以下字段：

### 1.1 图像输入 `image`

模型**固定需要恰好 3 张图像**（`model.py` `IMAGE_KEYS`）：

| 键名 | 含义 | 分辨率 |
|------|------|--------|
| `base_0_rgb` | 第三视角（固定摄像头/俯视） | 224×224×3 |
| `left_wrist_0_rgb` | 左腕摄像头 | 224×224×3 |
| `right_wrist_0_rgb` | 右腕摄像头 | 224×224×3 |

- **数据类型**：float32，范围 `[-1, 1]`（uint8 自动转换：`img / 255 * 2 - 1`）
- **训练时图像增强**（`train=True` 时）：
  - 非腕部图像：RandomCrop(95%) + Rotate(±5°) + ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
  - 腕部图像：仅 ColorJitter

### 1.2 图像掩码 `image_mask`

与图像同键，`bool` 类型：
- `True`：该摄像头有效
- `False`：该摄像头缺失（用零张量填充）

### 1.3 机器人本体状态 `state`

- 形状：`[action_dim]`（模型默认 `action_dim=32`，不足的机器人用零填充）
- 类型：float32，归一化后的值

### 1.4 语言指令 `tokenized_prompt`

- Tokenizer：PaliGemma（SentencePiece）
- 形状：`[max_token_len]`，int32
- 最大长度：π₀ = 48 tokens，π₀.5 = 200 tokens
- 配套：`tokenized_prompt_mask`（bool，标记有效 token）

**π₀.5 特殊处理**：状态被离散化（256 bins）并拼入语言 prompt，格式为：
```
Task: {prompt}, State: {离散化状态};\nAction:
```

### 1.5 π₀-FAST 专用字段

| 字段 | 类型 | 用途 |
|------|------|------|
| `token_ar_mask` | int32 | 自回归掩码 |
| `token_loss_mask` | bool | 损失计算掩码 |

### 1.6 模型输出（Actions）

- 形状：`[action_horizon, action_dim]`
- 默认 `action_horizon=50`（π₀-FAST 为 10~16）
- 默认 `action_dim=32`

---

## 2. 各机器人 State/Action 空间详解

> **总体原则**（来自 `docs/norm_stats.md`）：
> - 关节角使用**弧度（radians）**
> - 夹爪位置归一化为 **[0.0, 1.0]**（0.0 = 完全打开，1.0 = 完全关闭）

### 2.1 ALOHA（Trossen Interbotix 双臂）

**数据来源**：HDF5 文件中的 `/observations/qpos`（关节位置）

| 维度 | 内容 | 单位 |
|------|------|------|
| dim 0–5 | 左臂 6 个关节角 | **弧度**，绝对值 |
| dim 6 | 左夹爪开合 | 归一化线性位置 [0, 1]；`adapt_to_pi=True` 时转换为归一化角度空间 |
| dim 7–12 | 右臂 6 个关节角 | **弧度**，绝对值 |
| dim 13 | 右夹爪开合 | 同 dim 6 |

**Action 格式（14D）**：

| 配置 | dim 0–5, 7–12 | dim 6, 13 |
|------|---------------|-----------|
| `use_delta_joint_actions=True`（默认） | **增量关节角（弧度 delta）** | 绝对归一化夹爪值 |
| `use_delta_joint_actions=False` | 绝对关节角（弧度） | 绝对归一化夹爪值 |

**`adapt_to_pi` 坐标转换**（`aloha_policy.py`）：
- 部分关节符号翻转（`_joint_flip_mask`）
- 夹爪：Aloha 线性空间 → pi0 角度空间（`arcsin` 几何转换）

---

### 2.2 DROID（Franka Panda，7-DoF）

**数据来源**：`robot_state/joint_positions`（关节位置）

| 维度 | 内容 | 单位 |
|------|------|------|
| dim 0–6 | Franka 7 个关节角 | **弧度**，绝对值 |
| dim 7 | 夹爪位置 | 归一化线性位置 [0, 1] |

**Action 格式（8D）**：

| 训练配置 | dim 0–6 | dim 7 |
|----------|---------|-------|
| 预训练（pi0.5-droid base） | **关节速度（rad/s）** | 夹爪绝对位置 |
| RLDS 全集微调（`pi0_fast_full_droid_finetune`） | **关节位置增量（弧度 delta）** | 夹爪绝对位置 |
| LeRobot 小数据微调（`pi05_droid_finetune`） | **关节速度（rad/s）** | 夹爪绝对位置 |

> **注意**：预训练用 joint velocity，控制频率 15Hz；joint position 更适合仿真评估。

---

### 2.3 LIBERO（Franka Panda，EEF 控制）

> **与 ALOHA/DROID 的根本区别**：LIBERO 的 state 和 action 都是**末端执行器（EEF/笛卡尔空间）**，而非关节空间。

**State（8D）** — 来源于 `main.py` 推理时构造：

```python
np.concatenate([
    obs["robot0_eef_pos"],                   # 3D EEF 位置 (x, y, z)，单位：米
    _quat2axisangle(obs["robot0_eef_quat"]), # 3D EEF 旋转（轴角表示，单位：弧度×轴方向）
    obs["robot0_gripper_qpos"],              # 2D 夹爪关节位置
])
```

| 维度 | 内容 | 单位 |
|------|------|------|
| dim 0–2 | EEF 位置 (x, y, z) | 米 |
| dim 3–5 | EEF 旋转（轴角） | 弧度 × 旋转轴 |
| dim 6–7 | 夹爪手指位置 | 关节空间（两个手指） |

**Action（7D）** — OSC（Operational Space Control）控制器，delta 形式：

| 维度 | 内容 | 单位 |
|------|------|------|
| dim 0–2 | EEF 位置增量 Δx/Δy/Δz | 米（相对当前位置） |
| dim 3–5 | EEF 旋转增量（轴角） | 弧度 delta |
| dim 6 | 夹爪控制 | binary（-1 开 / +1 关） |

> `config.py` 注释："In Libero, the raw actions in the dataset are already delta actions"

---

### 2.4 汇总对比

| 机器人 | State 空间 | Action 空间 | 控制频率 |
|--------|-----------|-------------|---------|
| ALOHA | 关节角（弧度）+ 归一化夹爪 | 增量关节角（默认）/ 绝对关节角 | 50 Hz |
| DROID | 关节角（弧度）+ 归一化夹爪 | 关节速度（预训练）/ 增量关节角（微调） | 15 Hz |
| LIBERO | **EEF 位置 + 轴角旋转** | **EEF delta（位置+旋转增量）** | 10 Hz |

---

## 3. LeRobot 数据格式

LeRobot 是 openpi 统一使用的数据存储格式（v2.1），基于 Parquet + MP4 + JSON。

### 3.1 目录结构

```
<dataset_root>/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet   # 每个 episode 一个文件，存储标量数据
│       ├── episode_000001.parquet
│       └── ...
├── meta/
│   ├── info.json              # 数据集元信息（fps、features 定义、总帧数等）
│   ├── episodes.jsonl         # 每个 episode 的索引和长度
│   ├── tasks.jsonl            # 任务语言描述列表（task_index → 自然语言）
│   └── episodes_stats.jsonl  # 每 episode 的统计信息（用于归一化）
└── videos/
    └── chunk-000/
        └── observation.images.cam_high/
            ├── episode_000000.mp4  # 图像以视频形式存储（更节省空间）
            └── ...
```

### 3.2 Parquet 字段

每条记录对应一个时间步，包含两类字段：

**系统固定字段（所有 LeRobot 数据集都有）**：

| 字段 | 类型 | 含义 |
|------|------|------|
| `timestamp` | float32 | 帧在 episode 中的时间（秒） |
| `frame_index` | int64 | 帧在 episode 内的序号 |
| `episode_index` | int64 | 所属 episode 编号 |
| `index` | int64 | 全局帧序号 |
| `task_index` | int64 | 对应任务索引（查询 tasks.jsonl 得到语言描述） |

**用户自定义字段（各机器人不同）**：

| 机器人 | 字段名 | 形状 | 内容 |
|--------|--------|------|------|
| ALOHA | `observation.state` | (14,) | 关节角 qpos |
| ALOHA | `action` | (14,) | 目标关节角 |
| ALOHA | `observation.images.cam_high` | 视频 | 第三视角 |
| ALOHA | `observation.images.cam_left_wrist` | 视频 | 左腕 |
| ALOHA | `observation.images.cam_right_wrist` | 视频 | 右腕 |
| DROID | `joint_position` | (7,) | Franka 关节角 |
| DROID | `gripper_position` | (1,) | 夹爪位置 |
| DROID | `actions` | (8,) | 关节速度 + 夹爪 |
| DROID | `exterior_image_1_left` | 视频 | 外部摄像头 |
| DROID | `wrist_image_left` | 视频 | 腕部摄像头 |
| LIBERO | `state` | (8,) | EEF 位置+旋转+夹爪 |
| LIBERO | `actions` | (7,) | EEF delta 动作 |
| LIBERO | `image` | 图像 | 第三视角 |
| LIBERO | `wrist_image` | 图像 | 腕部视角 |

### 3.3 delta_timestamps 机制

训练时，DataLoader 通过 `delta_timestamps` 自动从 parquet 中查找未来 `action_horizon` 步的 action，拼成 action chunk：

```python
delta_timestamps = {
    "actions": [t / fps for t in range(action_horizon)]
    # 例如 ALOHA 50Hz, horizon=50: [0, 0.02, 0.04, ..., 0.98]
}
```

输出 `actions` 的形状为 `[action_horizon, action_dim]`，这正是模型训练所需的 action sequence。

---

## 4. 数据转换脚本

这些脚本是**一次性离线预处理工具**，将各机器人原始数据转换为 LeRobot 格式后即可用于训练。

| 脚本 | 原始格式 | 主要转换内容 |
|------|---------|-------------|
| `examples/aloha_real/convert_aloha_data_to_lerobot.py` | HDF5 (`.hdf5`) | `qpos` → state, `/action` → action, 图像按摄像头存为视频 |
| `examples/droid/convert_droid_data_to_lerobot.py` | HDF5 (`.h5`) + MP4 | `robot_state/joint_positions` + `gripper_position` → state, `joint_velocity` → actions |
| `examples/libero/convert_libero_data_to_lerobot.py` | RLDS (TensorFlow) | `observation/state` → state, `action` → actions, 语言标注 → tasks.jsonl |

**使用方法**：

```bash
# ALOHA
uv run examples/aloha_real/convert_aloha_data_to_lerobot.py \
    --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>

# DROID
uv run examples/droid/convert_droid_data_to_lerobot.py \
    --data_dir /path/to/your/data

# LIBERO（需要 tensorflow_datasets）
uv run examples/libero/convert_libero_data_to_lerobot.py \
    --data_dir /path/to/your/data
```

---

## 5. 训练数据流水线

LeRobot 格式数据**不能直接输入模型**，每次读取时在线应用以下 transforms（`data_loader.py`）：

```
LeRobot Dataset (parquet + mp4)
    │
    ↓  LeRobotDataset.__getitem__
    │  ├── 读 parquet → 当前帧标量字段（state, action, task_index...）
    │  ├── 解码 mp4 → 当前帧图像
    │  └── delta_timestamps → 拼装 action_horizon 步的 actions
    │      ALOHA: actions.shape = [50, 14]
    │      DROID: actions.shape = [10, 8]
    │      LIBERO: actions.shape = [10, 7]
    │
    ↓  PromptFromLeRobotTask（若 prompt_from_task=True）
    │  task_index → 查询 tasks.jsonl → 填入 "prompt" 字符串
    │
    ↓  RepackTransform
    │  字段重命名：将数据集字段映射到统一键名
    │  例如 ALOHA: "observation.state" → "state"
    │         "observation.images.cam_high" → "images/cam_high"
    │
    ↓  Robot-specific Inputs（AlohaInputs / DroidInputs / LiberoInputs）
    │  ├── 图像格式转换：CHW float → HWC uint8
    │  ├── 缺失摄像头：填充零张量 + image_mask=False
    │  ├── ALOHA: 关节符号翻转 + 夹爪空间转换（adapt_to_pi）
    │  └── DROID: 拼接 joint_position + gripper_position → state
    │
    ↓  DeltaActions（可选，ALOHA/DROID joint position 模式）
    │  actions[arm_dims] -= state[arm_dims]  → 转为增量动作
    │
    ↓  Normalize
    │  ├── Z-score（π₀）：(x - mean) / (std + 1e-6)
    │  └── Quantile（π₀.5, π₀-FAST）：(x - q01) / (q99 - q01 + 1e-6) * 2 - 1
    │
    ↓  TokenizePrompt / TokenizeFASTInputs
    │  语言指令 → int32 token 序列 [max_token_len]
    │
    ↓  ResizeImages
    │  所有图像 resize 到 224×224
    │
    ↓  PadStatesAndActions
    │  state / actions 零填充到 model.action_dim（默认 32）
    │
    ↓  Observation + Actions
       └── 输入模型
```

### 5.1 数据加载器类型

| 加载器 | 使用场景 | 实现 |
|--------|---------|------|
| `TorchDataLoader` | ALOHA / LIBERO / 小规模 DROID | 基于 `torch.utils.data.DataLoader`，支持多进程 |
| `RLDSDataLoader` | 大规模 DROID（完整数据集） | 基于 TensorFlow `dlimp`，内置 shuffle buffer |

---

## 6. 归一化统计量

归一化统计量存储于 checkpoint 的 `assets/<asset_id>/` 目录下（`norm_stats.json`）。

### 6.1 统计量内容

| 字段 | 含义 |
|------|------|
| `mean` | 各维度均值 |
| `std` | 各维度标准差 |
| `q01` | 1% 分位数（Quantile 归一化用） |
| `q99` | 99% 分位数（Quantile 归一化用） |

### 6.2 各机器人预训练统计量（可复用）

| 机器人 | Asset ID | 适用模型 |
|--------|----------|---------|
| ALOHA（Trossen） | `trossen` | `pi0_base`, `pi0_fast_base` |
| Mobile ALOHA | `trossen_mobile` | `pi0_base`, `pi0_fast_base` |
| Franka（DROID） | `droid` | `pi0_base`, `pi0_fast_base` |
| Franka（非 DROID） | `franka` | `pi0_base`, `pi0_fast_base` |
| UR5e | `ur5e` | `pi0_base`, `pi0_fast_base` |
| UR5e 双臂 | `ur5e_dual` | `pi0_base`, `pi0_fast_base` |
| ARX | `arx` | `pi0_base`, `pi0_fast_base` |

### 6.3 使用预训练统计量（微调时）

```python
TrainConfig(
    data=LeRobotAlohaDataConfig(
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="trossen",
        ),
    ),
)
```

### 6.4 计算新统计量

```bash
uv run scripts/compute_norm_stats.py --config-name=<your-config>
```

---

## 关键文件索引

| 文件 | 作用 |
|------|------|
| `src/openpi/models/model.py` | `Observation` 类定义，图像预处理逻辑 |
| `src/openpi/models/pi0_config.py` | 模型配置（action_dim=32, horizon=50, token=48/200） |
| `src/openpi/transforms.py` | 所有数据变换类（Normalize、DeltaActions、TokenizePrompt 等） |
| `src/openpi/training/config.py` | 各机器人训练配置，transforms 组合定义 |
| `src/openpi/training/data_loader.py` | DataLoader 实现，transforms 应用逻辑 |
| `src/openpi/policies/aloha_policy.py` | ALOHA 14D 输入/输出处理，坐标转换 |
| `src/openpi/policies/droid_policy.py` | DROID 8D 输入/输出处理 |
| `src/openpi/policies/libero_policy.py` | LIBERO 8D state / 7D action 处理 |
| `src/openpi/training/droid_rlds_dataset.py` | DROID RLDS 大规模数据加载器 |
| `scripts/compute_norm_stats.py` | 计算归一化统计量 |
| `docs/norm_stats.md` | 归一化统计量说明与 action space 定义 |
