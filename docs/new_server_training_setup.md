# 新 Server 从零部署训练指南

针对 **`Task_Graph_Dataset`**（405 episodes / 10 tasks / ~96 GB）。
覆盖：凭据准备 → 数据传输 → 仓库 clone → 环境 bootstrap → 镜像 build → 数据转换 → 启动训练 → 监控。

> **适用范围**：训练 only。不包含真机推理、ROS、数据采集端。
> **目标系统**：Ubuntu 22.04 + NVIDIA GPU（≥ 24 GB VRAM 推荐，48 GB 最佳）。

---

## 0. TL;DR 时间表

| 阶段 | 预计耗时 | 备注 |
|------|---------|------|
| 1. 凭据准备 | 5 min | wandb + HF token，本地浏览器操作 |
| 2. 数据 rsync | 30 min – 2 h | 取决于内网带宽，96 GB |
| 3. 仓库 clone | < 1 min | 不带 submodules |
| 4. Host bootstrap | 5 min | 安装 docker + nvidia-toolkit |
| 5. 镜像 build | 10–15 min | FFmpeg 编译 + `uv sync` |
| 6. 数据转换 | 20–40 min | HDF5 → LeRobot，405 episodes |
| 7. norm stats | 3–5 min | |
| 8. 启动训练 | < 1 min | JAX JIT 预热 ~60 s |
| **总计（含数据传输）** | **~1.5–3 h** | |

---

## 1. 凭据准备（本地预先做好）

### 1.1 WandB API key

1. 浏览器登录 https://wandb.ai
2. 右上角头像 → **User Settings**
3. 滚动到 **API keys** → **Reveal** 复制密钥（40 字符）
4. 临时存到本地某个安全文件，待会儿要填进 `.env.train`

### 1.2 HuggingFace Token

1. 浏览器登录 https://huggingface.co
2. 右上角头像 → **Settings** → **Access Tokens** → **New token**
3. **Token type**：`Read`（够用，下载权重）
4. **Repositories permissions**：勾选 `Public gated repos`（Pi0.5 base 是 gated）
5. 复制 token（`hf_xxxxxxxx...`）

> **重要**：先在浏览器访问 Pi0.5 base 模型页面，同意 license。访问后才能用 token 下载：
> https://huggingface.co/google/paligemma2-3b-pt-224 （或项目实际依赖的 base，看 [src/openpi/training/config.py](../src/openpi/training/config.py)）

### 1.3 SSH key（如果目标 server 用 SSH 拉 git repo）

```bash
# 在目标 server 上：
ssh-keygen -t ed25519 -C "<your-email>"
cat ~/.ssh/id_ed25519.pub
# 复制到 https://github.com/settings/keys
```

---

## 2. 数据传输（从当前机器 → 目标 server）

### 2.1 在目标 server 上准备落盘位置

```bash
# 选一个有 ≥ 200 GB 空闲的盘（dataset + LeRobot 转换产物 + checkpoints + image cache）
sudo mkdir -p /mnt/data/openpi
sudo chown $USER:$USER /mnt/data/openpi
mkdir -p /mnt/data/openpi/dataset/Task_Graph_Dataset
mkdir -p /mnt/data/openpi/checkpoints
```

### 2.2 在当前机器 rsync 数据集到目标 server

```bash
# 在当前机器执行（数据源 /mnt/4TBSSD/.../Task_Graph_Dataset 共 96 GB）
rsync -avhP --partial \
    /mnt/4TBSSD/users/yahuanshi/Projects/dataset/Task_Graph_Dataset/ \
    <user>@<target-server>:/mnt/data/openpi/dataset/Task_Graph_Dataset/
```

关键参数：
- `-a`：保留权限/时间戳
- `-h -P`：人类可读 + 进度 + `--partial`（断网可续传）
- 末尾两个路径都带 `/`：复制目录内容，不嵌套

> **校验**：传输完成后在目标 server 跑 `ls /mnt/data/openpi/dataset/Task_Graph_Dataset/*.hdf5 | wc -l` 应当返回 **405**。

---

## 3. Clone 仓库（目标 server）

```bash
cd ~
git clone git@github.com:YahuanShi/fine-tuning-openpi.git
# 若没配 SSH key：git clone https://github.com/YahuanShi/fine-tuning-openpi.git
cd fine-tuning-openpi
```

**不要**加 `--recurse-submodules`——训练用不到 `aloha` / `libero` 子模块。

---

## 4. Host bootstrap（一次性，幂等）

```bash
bash scripts/docker/bootstrap_host.sh
```

脚本会：
1. 验证 `nvidia-smi` 可用（驱动必须**手动**装好，脚本只检测）
2. 如缺 Docker → 装 docker-ce + compose plugin，把当前用户加到 `docker` 组
3. 如缺 `nvidia-ctk` → 装 NVIDIA Container Toolkit，配置 Docker runtime
4. 跑一次 `docker run --gpus all nvidia/cuda:12.2.2-base nvidia-smi` smoke test

**装完 docker 后必须**：

```bash
# 让 docker 组成员身份生效（否则下一步 docker 命令会 permission denied）
exec sudo su -l $USER     # 或者干脆 logout 再 ssh 回来
```

### 4.1 没有 sudo / 非 Ubuntu？

- **没有 sudo**：联系管理员预先装好 docker + nvidia-container-toolkit；之后所有步骤不需要 sudo
- **RHEL / Rocky / Debian**：手动替换 `scripts/docker/install_docker_ubuntu22.sh` 里的 apt 命令为 yum/dnf

### 4.2 验证

```bash
docker --version                                            # ≥ 24.x
nvidia-ctk --version                                        # 应输出版本号
docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi
# 最后一条应当显示 GPU 信息
```

---

## 5. 配置 `.env.train`

```bash
cp scripts/docker/.env.train.example scripts/docker/.env.train
$EDITOR scripts/docker/.env.train
```

按下表填写：

```bash
# ─── 必填 ───────────────────────────────────────────────────────────────
UR5_REPO_ID=ur5_task_graph_20260513         # 转换后的 LeRobot dataset 名字
WANDB_API_KEY=<粘贴 1.1 的 40 字符 key>

# ─── 强烈建议 ───────────────────────────────────────────────────────────
HF_TOKEN=<粘贴 1.2 的 hf_xxx token>

# ─── 路径（用 2.1 的目录）─────────────────────────────────────────────
DATASET_PATH=/mnt/data/openpi/dataset
CHECKPOINTS_PATH=/mnt/data/openpi/checkpoints
UV_CACHE_PATH=/mnt/data/openpi/.cache/uv
HF_CACHE_PATH=/mnt/data/openpi/.cache/huggingface

# ─── JAX 显存比例 ──────────────────────────────────────────────────────
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95         # 48 GB 卡用 0.95；24 GB 卡用 0.90
```

**REPO_ID 命名建议**：`ur5_task_graph_<YYYYMMDD>`——既能区分于其他实验，又有日期戳便于追溯（[README.md](../README.md) 的现有 checkpoint 命名风格）。

### 5.1 路径映射澄清

由于本数据集叫 `Task_Graph_Dataset`，转换脚本期望 `raw-dir` 直接指向包含 `episode_*.hdf5` 的文件夹。容器内：

| 宿主路径 | 容器内路径 |
|---------|-----------|
| `/mnt/data/openpi/dataset/Task_Graph_Dataset/` | `/app/dataset/Task_Graph_Dataset/` |
| `/mnt/data/openpi/checkpoints/` | `/app/checkpoints/` |

转换时 `--raw-dir dataset/Task_Graph_Dataset` 即可（[compose.train.yml](../scripts/docker/compose.train.yml) 已挂载好）。

---

## 6. Build 训练镜像（一次性）

```bash
docker compose -f scripts/docker/compose.train.yml \
               --env-file scripts/docker/.env.train \
               build
```

首次 ~10–15 min。`docker images` 应出现 `openpi_trainer:latest`，大小约 8–10 GB。

构建产物缓存在 Docker daemon，**不**占用 `/mnt/data/openpi`。如果 `/var/lib/docker` 所在盘空间紧张，需要先迁移 Docker 数据目录（见 §10）。

---

## 7. 数据转换 + norm stats + 训练（一次性执行）

`train_pipeline.sh` 顺序执行三步：转换 → norm stats → 训练，在 Docker 容器内运行。

> **关键**：SSH 断线会无声地杀死进程，留下不完整的数据集。必须用 `tmux` 或 `nohup`。

**Option A — tmux**（推荐，可实时看日志）：

```bash
# 如未安装 tmux：sudo apt install -y tmux
tmux new -s train
docker compose -f scripts/docker/compose.train.yml \
               --env-file scripts/docker/.env.train \
               run --rm trainer \
               ./examples/ur5/train_pipeline.sh \
                   --raw-dir dataset/Task_Graph_Dataset \
                   --repo-id Task_Graph_V1 \
                   --exp-name ur5_task_graph_v1
# Ctrl+B D 安全离开；tmux attach -t train 回来
```

**Option B — nohup**（关闭终端也安全）：

```bash
nohup docker compose -f scripts/docker/compose.train.yml \
               --env-file scripts/docker/.env.train \
               run --rm -T trainer \
               ./examples/ur5/train_pipeline.sh \
                   --raw-dir dataset/Task_Graph_Dataset \
                   --repo-id Task_Graph_V1 \
                   --exp-name ur5_task_graph_v1 \
    > /tmp/pipeline.log 2>&1 &
echo "PID: $!"
```

监控进度：

```bash
tail -f /tmp/pipeline.log
grep -c "Saved.*steps" /tmp/pipeline.log   # 已转换 episode 数（共 405）
```

时间估算：转换 ~20–40 min · norm stats ~5 min · 训练 ~数小时。

**跳过标志**（步骤已完成时使用）：

| 标志 | 效果 |
|------|------|
| `--skip-convert` | 跳过 HDF5 → LeRobot 转换 |
| `--skip-stats` | 跳过 norm stats 计算 |
| `--resume` | 从最新 checkpoint 续训 |

路径说明：`--raw-dir dataset/Task_Graph_Dataset` 是容器内路径（`/app/dataset` 由 `.env.train` 的 `DATASET_PATH` bind-mount）。

### 7.1 校验结果

```bash
ls /mnt/data/openpi/dataset/for_training/Task_Graph_V1/
# 应当看到 meta/、data/ 目录
ls ~/fine-tuning-openpi/assets/pi05_ur5/Task_Graph_V1/norm_stats.json
# 应当存在（norm stats 写入宿主机项目目录）
```

训练启动后前 60 秒是 JAX JIT 编译，看到 `step 0` 后进入正常循环。

### 7.2 训练关键日志位置

| 位置 | 内容 |
|------|------|
| tmux stdout | step / loss / throughput |
| `/tmp/pipeline.log` | nohup 模式下的完整日志 |
| WandB dashboard | https://wandb.ai/\<your-entity\>/openpi |
| `$CHECKPOINTS_PATH/pi05_ur5/ur5_task_graph_v1/<step>/` | 每 N 步的 checkpoint |

---

## 9. 监控

### 9.1 宿主侧 GPU 监控

```bash
watch -n 1 nvidia-smi
# 训练时 VRAM 应稳定在 ~41 GB（48 GB 卡 + batch 32）
```

### 9.2 训练日志（容器内 stdout）

`train_oneclick.sh` 用 `docker compose run`（非 detach），日志直接打在你 ssh 终端。**建议**：

```bash
# 用 tmux / screen 跑训练，断开 ssh 不影响
tmux new -s train
./scripts/train_oneclick.sh ur5_task_graph_20260513 ur5_task_graph_v1 --overwrite
# Ctrl+B D 断开；重连用 tmux attach -t train
```

### 9.3 检查点验证

```bash
ls -la /mnt/data/openpi/checkpoints/pi05_ur5/ur5_task_graph_v1/
# 期望每 N 步出现一个数字目录，最大数字 = 当前训练进度
```

---

## 10. 常见问题排查

### 10.1 `docker: Cannot connect to the Docker daemon`
刚装完 docker，没有 logout/login，docker 组身份未生效。
```bash
exec sudo su -l $USER
```

### 10.2 `nvidia-container-cli: initialization error`
```bash
sudo systemctl restart docker
# 仍失败：检查 /etc/docker/daemon.json 有没有 "default-runtime": "nvidia"
```

### 10.3 转换报 `No episode_*.hdf5 files found`
检查 `--raw-dir` 路径在**容器内**是否存在：
```bash
docker compose -f scripts/docker/compose.train.yml --env-file scripts/docker/.env.train \
    run --rm trainer ls /app/dataset/Task_Graph_Dataset | head
```
若空，回到 §5.1 检查 `DATASET_PATH` 是否正确指向了**包含** `Task_Graph_Dataset/` 这个文件夹的父目录。

### 10.4 训练启动时 OOM
```bash
nvidia-smi                              # 找占卡进程
kill -9 <PID>
# 仍 OOM：把 XLA_PYTHON_CLIENT_MEM_FRACTION 降到 0.9，或 batch_size 降到 16
```

### 10.5 wandb 报 `401 Unauthorized`
- `.env.train` 里 `WANDB_API_KEY` 拼写错误 / 多余空格 → 重检
- 或在宿主上执行 `wandb login`，然后 `cp -r ~/.netrc /mnt/data/openpi/`，并在 compose 里挂载

### 10.6 HuggingFace 下载 `gated repo` 401
浏览器登录 HF，到 base 模型页面手动点 **Agree and access repository**，再重跑。

### 10.7 `/var/lib/docker` 占满
迁移 Docker 数据目录到大盘：
```bash
sudo systemctl stop docker
sudo mv /var/lib/docker /mnt/data/docker
sudo ln -s /mnt/data/docker /var/lib/docker
sudo systemctl start docker
```

### 10.8 训练 throughput 异常低
- 数据盘是机械盘？把 `dataset/for_training/` 和 `checkpoints/` 都搬到 NVMe
- `XLA_PYTHON_CLIENT_MEM_FRACTION` 太小 → JIT 频繁 spill
- 别的进程占带宽 → 检查 `iotop` / `nvidia-smi`

---

## 11. 一页流程速查（cheatsheet）

```bash
# === 一次性 ===
git clone git@github.com:YahuanShi/fine-tuning-openpi.git
cd fine-tuning-openpi
bash scripts/docker/bootstrap_host.sh
exec sudo su -l $USER                 # 让 docker 组生效

cp scripts/docker/.env.train.example scripts/docker/.env.train
$EDITOR scripts/docker/.env.train     # 填 KEY / TOKEN / PATH

docker compose -f scripts/docker/compose.train.yml --env-file scripts/docker/.env.train build

# === 每个数据集只跑一次 ===
docker compose -f scripts/docker/compose.train.yml --env-file scripts/docker/.env.train run --rm trainer \
    bash -c "uv run examples/ur5/convert_ur5_data_to_lerobot.py --raw-dir dataset/Task_Graph_Dataset --repo-id \$UR5_REPO_ID --overwrite && \
             uv run scripts/compute_norm_stats.py --config-name pi05_ur5"

# === 训练 ===
tmux new -s train
./scripts/train_oneclick.sh ur5_task_graph_20260513 ur5_task_graph_v1 --overwrite
# Ctrl+B D 离开；tmux attach -t train 回来
```

---

## 12. 相关文档

- [TRAIN_DEPLOY.md](TRAIN_DEPLOY.md)：通用 Docker 部署说明（简版）
- [data_pipeline_guide.md](data_pipeline_guide.md)：数据处理管线（采集 → 清洗 → 切片）
- [norm_stats.md](norm_stats.md)：norm_stats 含义与重载
- [DEVELOPMENT.md](../DEVELOPMENT.md)：项目约定与已知陷阱
