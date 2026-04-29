# Git 主仓库 + 子模块管理指南

**主仓库：** `fine-tuning-openpi`

**子模块列表：**
- `teleoperation` → `git@github.com:YahuanShi/teleoperation.git`
- `data_processing` → `git@github.com:YahuanShi/data_processing.git`

---

## 1. 克隆主仓库（含子模块）

```bash
git clone git@github.com:YahuanShi/fine-tuning-openpi.git
cd fine-tuning-openpi

# 初始化并更新子模块
git submodule update --init --recursive
```

> `--init` 初始化子模块，`--recursive` 递归处理嵌套子模块（如第三方库）。

---

## 2. 查看子模块状态

```bash
git submodule status
```

输出示例：
```
4e8bfbc teleoperation (heads/main)
a542143 data_processing (heads/main)
d1dc83a third_party/aloha (d1dc83a)
f78abd6 third_party/libero (f78abd6)
```

> 每行显示当前 commit。`-` 表示未初始化，`+` 表示本地与主仓库记录不同步。

---

## 3. 更新子模块

**方法 A：更新到主仓库记录的 commit**
```bash
git submodule update --recursive
```

**方法 B：拉取子模块远程最新 commit**
```bash
git submodule update --remote --merge
```

更新完子模块后，在主仓库提交新的 commit 指针：
```bash
git add teleoperation data_processing
git commit -m "update submodules"
git push origin main
```

---

## 4. 修改子模块代码

```bash
# 进入子模块，修改并推送
cd teleoperation
git add .
git commit -m "fix UR5 servo control"
git push origin main

# 回到主仓库，更新 commit 指针
cd ..
git add teleoperation
git commit -m "update teleoperation submodule pointer"
git push origin main
```

> **注意：** 不要在主仓库目录直接修改子模块内部文件，否则容易出现 detached HEAD 或丢失 commit。

---

## 5. 添加新子模块

```bash
git submodule add <git-url> new_module
git commit -m "add new_module as submodule"
git push origin main
```

---

## 6. 删除子模块

```bash
git submodule deinit -f teleoperation
rm -rf .git/modules/teleoperation
git rm -f teleoperation
git commit -m "remove teleoperation submodule"
git push origin main
```

---

## 7. 备份方案

**rsync 备份（排除大数据目录）：**
```bash
rsync -av --exclude=dataset --exclude=checkpoints --exclude=wandb \
  fine-tuning-openpi/ fine-tuning-openpi_backup/
```

**Git bundle 备份：**
```bash
git bundle create backup.bundle --all
```

---

## 8. 关键原则

| 规则 | 说明 |
|------|------|
| 主仓库只跟踪子模块 commit | 不直接管理子模块文件内容 |
| 子模块修改必须在子模块内 commit & push | 避免 detached HEAD |
| 更新子模块指针后主仓库再 push | 保持引用同步 |
| clone 后执行 `update --init --recursive` | 保证工作区完整 |
| 增删子模块按流程操作 | 避免空目录或残留文件 |
