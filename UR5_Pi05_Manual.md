# UR5 Pi0.5 Fine-Tuning 操作手册

> 项目路径：`/mnt/4TBSSD/users/yahuanshi/OpenPi/Fine-Tuning-Pi0.5/`
> 以下所有命令均在该目录下执行。

---

## 1. 项目概览

| 项目 | 说明 |
|------|------|
| 模型 | Pi0.5（JAX/XLA），LoRA fine-tuning |
| GPU | RTX 6000 48GB，训练约占 41GB VRAM |
| 推理频率 | 10Hz（推荐，与 action_horizon=10 匹配） |
| 任务类型 | pick-and-place、assembly、pnpa |

### 目录结构

```
Fine-Tuning-Pi0.5/
├── dataset/
│   ├── raw/                    # 原始 HDF5 采集数据
│   ├── processed/              # 处理后数据（trimmed/smoothed）
│   │   └── trimmed/
│   │       └── 20260415/       # 按日期存放
│   └── for_training/           # LeRobot 格式数据（转换输出）
│       ├── ur5_dataset_20260415/
│       └── ...
├── checkpoints/pi05_ur5/       # 训练 checkpoint
├── assets/                     # 全局 norm stats（compute_norm_stats 输出）
├── examples/ur5/               # UR5 相关脚本
├── data_processing/            # 数据处理流水线
└── src/openpi/training/config.py  # 训练配置
```

---

## 2. 配置索引

| Config 名称 | repo_id（LeRobot 数据集） | 任务类型 | 主要 checkpoint |
|-------------|--------------------------|----------|----------------|
| `pi05_ur5` | 由 `UR5_REPO_ID` 环境变量指定 | pick-and-place | `ur5_pick_place_20260415/19999` |
| `pi05_ur5_assembly` | `ur5_dataset_20260402_assembly` | assembly | `ur5_pick_place_assembly_v1/19999` |
| `pi05_ur5_pnpa` | `ur5_dataset_20260402_pnpa` | pick-and-place-and-arrange | `ur5_pnpa_v2/19999` |

### Checkpoint 完整记录

| Checkpoint | 训练数据集 | 最终步数 | 备注 |
|-----------|-----------|---------|------|
| `ur5_pick_place_v3/19999` | `ur5_dataset_20260323` | 19999 | 早期版本 |
| `ur5_pick_place_v4/19999` | `ur5_dataset_20260331` | 19999 | 早期版本 |
| `ur5_pick_place_assembly_v1/19999` | `ur5_dataset_20260402` | 19999 | 混合任务 |
| `ur5_pnpa_v2/19999` | `ur5_dataset_20260402` | 19999 | pnpa 专项 |
| `ur5_pick_place_20260415/19999` | `ur5_dataset_20260415` | 19999 | 最新 pick-and-place |

---

## 3. 数据处理流程

### Step 1：数据采集
- 采集后存放于 `dataset/raw/` 或 `dataset/processed/trimmed/<DATE>/`
- 推荐采集频率：**20Hz**（实际约 18Hz 也可接受）

### Step 2：数据处理（trimming/smoothing）
```bash
bash data_processing/pipeline/pipeline.sh --input dataset/raw/<DATE>
# 输出到 dataset/processed/trimmed/<DATE>/
```

### Step 3：HDF5 → LeRobot 转换
```bash
HF_LEROBOT_HOME=$(pwd)/dataset/for_training \
uv run examples/ur5/convert_ur5_data_to_lerobot.py \
    --raw-dir dataset/processed/trimmed/<DATE> \
    --repo-id ur5_dataset_<DATE> \
    --fps 20 \
    --overwrite
```

> **注意**：
> - `--raw-dir` 和输出目录必须不同路径，否则原始数据会被覆盖删除
> - `--overwrite` 为显式覆盖开关，不加则输出目录已存在时报错退出
> - 转换前自动校验 HDF5 key 结构；空 episode 在结束时统一汇报

### Step 4：计算 Norm Stats
```bash
UR5_REPO_ID=ur5_dataset_<DATE> \
HF_LEROBOT_HOME=$(pwd)/dataset/for_training \
uv run scripts/compute_norm_stats.py --config-name pi05_ur5
```
输出至 `assets/pi05_ur5/ur5_dataset_<DATE>/norm_stats.json`

---

## 4. 训练

### 一键流水线（convert + norm stats + train）
```bash
bash examples/ur5/train_pipeline.sh \
    --raw-dir dataset/processed/trimmed/<DATE> \
    --repo-id ur5_dataset_<DATE> \
    --exp-name ur5_pick_place_<VERSION> \
    --fps 20 \
    --config pi05_ur5
```

可选参数：
- `--skip-convert`：跳过转换（已转换过）
- `--skip-stats`：跳过 norm stats（已计算过）
- `--resume`：继续中断的训练（默认为 `--overwrite` 从头开始）

Resume 示例：
```bash
bash examples/ur5/train_pipeline.sh \
    --repo-id ur5_dataset_<DATE> \
    --exp-name ur5_pick_place_<VERSION> \
    --resume --skip-convert --skip-stats
```

### 仅启动训练
```bash
UR5_REPO_ID=ur5_dataset_<DATE> \
HF_LEROBOT_HOME=$(pwd)/dataset/for_training \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
uv run scripts/train.py pi05_ur5 \
    --exp-name ur5_pick_place_<VERSION> \
    --overwrite
```

### Resume 训练
```bash
UR5_REPO_ID=ur5_dataset_<DATE> \
HF_LEROBOT_HOME=$(pwd)/dataset/for_training \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
uv run scripts/train.py pi05_ur5 \
    --exp-name ur5_pick_place_<VERSION> \
    --resume
```

### 关键训练参数（config.py）

| 参数 | 值 |
|------|----|
| `batch_size` | 32 |
| `num_train_steps` | 20,000 |
| `save_interval` | 2000 |
| `keep_period` | 5000 |
| `peak_lr` | 5e-5 |
| `warmup_steps` | 1000 |
| `action_horizon` | 10（硬限制） |
| `ema_decay` | None |

> 每次保存 checkpoint 时，会自动在 `assets/metadata.json` 中记录训练所用的 `repo_id`，供推理时自动匹配。

---

## 5. 推理（serve_policy）

### 启动推理服务（推荐）

使用 `serve.sh`，自动从 checkpoint 的 `assets/metadata.json` 读取 repo_id 并匹配正确的 config，无需手动指定：

```bash
./examples/ur5/serve.sh checkpoints/pi05_ur5/ur5_pick_place_20260415/19999
./examples/ur5/serve.sh checkpoints/pi05_ur5/ur5_pick_place_assembly_v1/19999
./examples/ur5/serve.sh checkpoints/pi05_ur5/ur5_pnpa_v2/19999
```

脚本会打印使用的 config 和 repo_id，方便确认。旧 checkpoint（无 metadata.json）会根据路径名自动推断。

### 手动指定（不推荐，需自行确认 config 与 checkpoint 匹配）
```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_ur5 \
    --policy.dir checkpoints/pi05_ur5/ur5_pick_place_20260415/19999
```

---

## 6. 常见问题 & 解决方案

### 问题 1：convert 时输出目录已存在报错

**现象：**
```
FileExistsError: Output already exists: .../for_training/ur5_dataset_xxx
Use --overwrite to replace it.
```

**解决：** 加 `--overwrite` 参数。

---

### 问题 2：训练时 `KeyError: 'UR5_REPO_ID'`

**原因：** `pi05_ur5` config 的 `repo_id` 现在必须通过环境变量传入，未设置时报错。

**解决：** 在命令前加 `UR5_REPO_ID=ur5_dataset_<DATE>`，或使用 `train_pipeline.sh`（自动设置）。

---

### 问题 3：训练时 OOM / GPU 显存不足

**原因：** 上次训练进程未完全退出，仍占用显存。

**解决：**
```bash
nvidia-smi   # 找到占用进程 PID
kill -9 <PID>
```

---

### 问题 4：`&` 在 exp-name 中导致后台执行

**现象：** `--exp-name ur5_P&P&A` 导致 shell 将任务放入后台。

**解决：** exp-name 使用下划线，如 `ur5_pick_place_and_arrange_v1`。

---

### 问题 5：推理时机械臂每隔一段有明显停顿（~400ms）

**原因：** ActionChunkBroker 同步推理，每执行完 `action_horizon=10` 步才请求下一批动作，此时机械臂等待。

**说明：** 这是当前架构的固有限制，异步预取会导致机械臂跳动（已测试并回退）。推荐使用 10Hz 推理以减少相对停顿感。

---

### 问题 6：HF_LEROBOT_HOME 未设置导致数据找不到

所有 convert / compute_norm_stats / train 命令都必须设置：
```bash
export HF_LEROBOT_HOME=$(pwd)/dataset/for_training
# 或在命令前添加
HF_LEROBOT_HOME=$(pwd)/dataset/for_training uv run ...
```

---

## 7. 关键参数参考

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 采集频率 | 20Hz | 实际 18Hz 也可接受 |
| 训练频率 | 20Hz（`--fps 20`） | 与采集一致 |
| 推理频率 | 10Hz | action_horizon=10，10Hz 停顿最不明显 |
| action_horizon | 10 | Pi0.5 硬限制，不可调高 |
| batch_size | 32 | RTX 6000 48GB 下的最大值 |
| XLA_PYTHON_CLIENT_MEM_FRACTION | 0.95 | 训练时必须设置 |
| 夹爪约定 | 0=开，1=闭 | 与 raw 数据相反，已在转换脚本中处理 |
| servoJ timeout | 500ms × 3次重试 | 相机抓帧超时设置 |

---

## 8. 修改 Config 流程

`pi05_ur5` 的 `repo_id` 通过环境变量控制，无需修改源码：

```bash
export UR5_REPO_ID=ur5_dataset_<NEW_DATE>
```

`pi05_ur5_assembly` 和 `pi05_ur5_pnpa` 的 `repo_id` 写死在 `config.py` 中（变动少）：
- assembly：`ur5_dataset_20260402_assembly`
- pnpa：`ur5_dataset_20260402_pnpa`
