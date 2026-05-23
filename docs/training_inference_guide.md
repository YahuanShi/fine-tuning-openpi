# OpenPi 训练输出与推理输出指南

本文档说明 OpenPi 模型在训练和推理两个阶段各自产生什么输出，以及 Flow Matching 的计算过程、Checkpoint 内容和推理反变换链。

---

## 目录

1. [训练输出](#1-训练输出)
   - [每步监控指标](#11-每步监控指标-train_step)
   - [Flow Matching 损失的计算过程](#12-flow-matching-损失的计算过程-compute_loss)
   - [Checkpoint 保存内容](#13-checkpoint-保存内容)
2. [推理输出](#2-推理输出)
   - [sample_actions：ODE 积分去噪](#21-sample_actionsode-积分去噪)
   - [Policy.infer：对外接口的完整返回](#22-policyinfer对外接口的完整返回)
   - [推理反变换链](#23-推理反变换链)
3. [训练 vs 推理输出对比](#3-训练-vs-推理输出对比)

---

## 1. 训练输出

### 1.1 每步监控指标 (`train_step`)

`train_step`（`scripts/train.py`）每步返回更新后的模型状态和一个 `info` 字典：

```python
return new_state, info
```

`info` 包含以下标量，每 `log_interval` 步记录到终端和 WandB：

| 字段 | 含义 | 计算方式 |
|------|------|---------|
| `loss` | Flow Matching 均方误差损失 | `mean(‖v_t − u_t‖²)` |
| `grad_norm` | 梯度全局 L2 范数 | `optax.global_norm(grads)` |
| `param_norm` | 模型权重全局 L2 范数（仅 kernel 层） | `optax.global_norm(kernel_params)` |

---

### 1.2 Flow Matching 损失的计算过程 (`compute_loss`)

`Pi0.compute_loss`（`src/openpi/models/pi0.py`）是训练的核心，实现 **Flow Matching** 目标：

```
┌─────────────────────────────────────────────────────────┐
│  输入                                                    │
│    actions        : 真实动作          [B, horizon, dim] │
│    noise ~ N(0,I) : 随机高斯噪声      [B, horizon, dim] │
│    t ~ Beta(1.5,1)× 0.999 + 0.001 : 流匹配时间步  [B]  │
├─────────────────────────────────────────────────────────┤
│  构造带噪动作（噪声与真实动作的线性插值）                  │
│    x_t = t · noise + (1 - t) · actions                  │
├─────────────────────────────────────────────────────────┤
│  构造真实速度场方向（目标）                               │
│    u_t = noise - actions                                 │
├─────────────────────────────────────────────────────────┤
│  一次完整前向传播（prefix + suffix 拼接）                 │
│    prefix：图像 token + 语言 token（双向注意力）          │
│    suffix：状态 token + 带噪动作 token + 时间步编码      │
│    → Transformer 输出后 action_horizon 个 token          │
│    → v_t = action_out_proj(suffix_out)  [B, horizon, dim]│
├─────────────────────────────────────────────────────────┤
│  损失                                                    │
│    loss = mean( ‖v_t − u_t‖² )    shape: [B, horizon]  │
│    最终对所有维度取均值 → 标量                           │
└─────────────────────────────────────────────────────────┘
```

**时间步采样分布**：使用 `Beta(1.5, 1)` 分布（偏向 `t=1` 附近，即较难的高噪声区域），在 `[0.001, 1.0]` 范围内采样，令模型更多地训练"去噪困难"的阶段。

**注意力结构**：
- prefix（图像 + 语言）：双向全注意力，token 间可互相看
- suffix（状态 + 动作）：因果注意力，动作 token 不能被 prefix 看到
- prefix 不能看 suffix（单向）

---

### 1.3 Checkpoint 保存内容

每隔 `save_interval` 步调用 `save_state`（`src/openpi/training/checkpoints.py`），保存目录结构：

```
checkpoints/<config_name>/<exp_name>/<step>/
├── params/
│   └── ...           ← 推理用权重（优先使用 EMA 参数）
├── train_state/
│   └── ...           ← 完整训练状态（params + optimizer state + step 计数）
└── assets/
    └── <asset_id>/
        └── norm_stats.json   ← 归一化统计量（mean / std / q01 / q99）
```

**EMA 参数**（`ema_decay=0.99`）：

```python
ema_params = ema_decay * ema_params + (1 - ema_decay) * new_params
```

Checkpoint 中 `params/` 目录优先保存 EMA 参数（而非瞬时参数），作为推理时加载的权重。EMA 参数对训练噪声更鲁棒，推理效果更稳定。

**保留策略**：默认只保留最新 1 个 checkpoint（`max_to_keep=1`），满足 `step % keep_period == 0` 的步骤永久保留。

---

## 2. 推理输出

### 2.1 `sample_actions`：ODE 积分去噪

`Pi0.sample_actions`（`src/openpi/models/pi0.py`）通过常微分方程（ODE）积分，从高斯噪声迭代去噪得到动作：

```
初始化：x_1 ~ N(0, I)    shape: [B, action_horizon, action_dim]

────────── ODE 积分（默认 num_steps=10 步）──────────
t = 1.0
while t >= dt/2:
    1. 用 KV cache 复用 prefix（图像+语言）的计算结果
    2. 计算 suffix（状态 + x_t + 时间步 t）
    3. 模型预测速度场：v_t = action_out_proj(suffix_out)
    4. 欧拉步：x_t ← x_t + dt · v_t    (dt = -1/num_steps)
    5. t ← t + dt
─────────────────────────────────────────────────────

输出：x_0    shape: [B, action_horizon, action_dim]
       归一化空间的动作序列，float32
```

**KV Cache 优化**：prefix（图像和语言 token）仅计算一次并缓存，每个去噪步骤只重新计算 suffix，大幅减少推理计算量。

**π₀-FAST 的不同之处**：π₀-FAST 不用 ODE 积分，而是自回归地逐 token 生成离散化的动作序列（基于 FAST tokenizer），单次前向传播即得动作。

---

### 2.2 `Policy.infer`：对外接口的完整返回

`Policy.infer`（`src/openpi/policies/policy.py`）是暴露给机器人环境的接口，返回 Python dict：

```python
{
    "state":   np.ndarray,   # shape: [action_dim]        当前机器人状态（原样透传）
    "actions": np.ndarray,   # shape: [action_horizon, action_dim]  反变换后的动作序列
    "policy_timing": {
        "infer_ms": float    # 模型推理耗时（毫秒）
    }
}
```

**`state` 字段**：从输入 `obs` 中直接透传，未经模型修改，供调用方参考当前状态。

**`actions` 字段**：经过完整反变换链（见下节），已还原为机器人可直接执行的物理量。

---

### 2.3 推理反变换链

`sample_actions` 的原始输出在归一化空间、填充到 32 维，不能直接执行。`output_transforms` 依次应用以下变换：

```
model.sample_actions 输出
  shape: [action_horizon, 32]    ← 归一化空间，含零填充维度
       ↓
  AbsoluteActions（若训练时使用了 DeltaActions）
       delta_action[arm_dims] + current_state[arm_dims]
       → 绝对关节角 / 绝对 EEF 位姿
       ↓
  Unnormalize
       Z-score 反变换：x * (std + 1e-6) + mean
       Quantile 反变换：(x + 1) / 2 * (q99 - q01 + 1e-6) + q01
       ↓
  Robot-specific Outputs
       ├── AlohaOutputs
       │     取前 14 维
       │     夹爪：pi0 角度空间 → Aloha 线性归一化空间
       │     关节：符号翻转（_joint_flip_mask）
       │
       ├── DroidOutputs
       │     取前 8 维（无坐标转换）
       │
       └── LiberoOutputs
             取前 7 维（无坐标转换）
       ↓
最终 actions（可直接发给机器人执行）

  ALOHA：  [action_horizon, 14]   关节角增量（弧度）× 12 + 归一化夹爪 × 2
  DROID：  [action_horizon, 8]    关节速度（rad/s）× 7 + 夹爪位置 × 1
  LIBERO： [action_horizon, 7]    EEF delta（Δx,Δy,Δz,Δax,Δay,Δaz）+ 夹爪 binary
```

---

## 3. 训练 vs 推理输出对比

| 维度 | 训练 | 推理 |
|------|------|------|
| **直接返回值** | 标量 `loss`、`grad_norm`、`param_norm` | `actions [horizon, dim]`、`state`、`timing` |
| **模型内部输出** | `v_t`：预测速度场向量 | `x_0`：ODE 积分后的动作 |
| **动作空间** | 归一化空间（无需反变换，仅计算损失） | 反归一化 → 机器人物理单位 |
| **图像处理** | 每步重新计算 prefix | prefix 用 KV cache 复用，只算一次 |
| **随机性** | `noise` 和 `t` 每步随机采样 | `noise` 从 N(0,I) 初始化，ODE 确定性积分 |
| **持久化产物** | Checkpoint（EMA params + optimizer state + norm_stats） | 无持久化，实时流式输出 |
| **Flow Matching 角色** | 学习速度场：`v_t → u_t = noise − action` | 沿速度场积分：`x_1 → x_0` |

---

## 相关文件索引

| 文件 | 作用 |
|------|------|
| `src/openpi/models/pi0.py` | `compute_loss`（Flow Matching 训练目标）和 `sample_actions`（ODE 推理） |
| `src/openpi/models/pi0_fast.py` | π₀-FAST 的自回归推理实现 |
| `src/openpi/models/model.py` | `BaseModel` 接口、`Observation` 类、`preprocess_observation` |
| `src/openpi/policies/policy.py` | `Policy.infer`：输入变换 → 模型推理 → 输出反变换 |
| `scripts/train.py` | `train_step`：一步训练的完整流程，含梯度更新和 EMA |
| `src/openpi/training/checkpoints.py` | Checkpoint 保存/加载，EMA 参数分离逻辑 |
| `src/openpi/training/utils.py` | `TrainState` 数据结构定义 |
| `src/openpi/policies/aloha_policy.py` | `AlohaOutputs`：推理后的动作反变换（夹爪空间、关节符号） |
| `src/openpi/policies/droid_policy.py` | `DroidOutputs`：截取前 8 维 |
| `src/openpi/policies/libero_policy.py` | `LiberoOutputs`：截取前 7 维 |
