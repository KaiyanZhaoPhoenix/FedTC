# CIFAR-10 + CifarNet 的 40 轮实验

本页保留 40 轮 CifarNet 实验的兼容命令。当前服务器任务使用 ResNet-20 和 100 轮，见 [FedTCStep 实验指南](cifar10_resnet_100.md)。

实验入口是 `python -m diagnostics.fit_native_step_control --dataset Cifar10`。它使用仓库的三通道卷积模型 `CifarNet`，每 5 轮从 `0.5、1.0、1.5` 中选择服务器更新步长。完整实验需要先在 CIFAR-10 上拟合策略，然后冻结并评测。本地已用合成图像检查接口，真实 CIFAR-10 的准确率和 RL 收益待服务器实测。

## 1. 安装和设置路径

从仓库根目录执行。可沿用已有环境，项目依赖的验证版本为 Python 3.9、PyTorch 2.6、torchvision 0.21。此实验入口使用 CPU。`--jobs` 指同时运行的轨迹进程数，第一次可用 1。

```bash
cd code
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt

export DATA_ROOT="$PWD/dataset"
export RAW_ROOT="$PWD/dataset/Cifar10-raw/rawdata"
export RUN="$PWD/runs/cifar10_step_01"
```

可把前两个路径换成服务器数据盘。`DATA_ROOT` 指向 **包含 `Cifar10/` 的父目录**。Linux 上目录名区分大小写。`RUN` 是本次实验的新目录，由 `dev` 创建。

## 2. 接入数据，按手头的格式选一种

| 服务器已有的数据 | 在 `code/` 下执行 |
|---|---|
| 已分好客户端的 FedAA 格式 NPZ | `python dataset/prepare_cifar10.py --data-root "$DATA_ROOT" --verify-only` |
| 官方 Python 批次，已解压到 `$RAW_ROOT/cifar-10-batches-py/` | `python dataset/prepare_cifar10.py --data-root "$DATA_ROOT" --raw-root "$RAW_ROOT"` |
| 尚未下载，允许服务器联网 | `python dataset/prepare_cifar10.py --data-root "$DATA_ROOT" --raw-root "$RAW_ROOT" --download` |

生成分区时目标 `$DATA_ROOT/Cifar10/` 必须尚未存在。如果原始批次恰好存放在已有的 `Cifar10/rawdata/` 下，让 `RAW_ROOT` 指向这个 `rawdata` 目录，并另选一个 `DATA_ROOT` 保存生成的分区。

**已有客户端分区**须包含 `$DATA_ROOT/Cifar10/train/0.npz` 至 `95.npz`，以及 `test/` 下相同编号的文件。每个 NPZ 的 `data` 键保存字典，`x` 是预处理好的浮点数组，形状为 `(N, 3, 32, 32)`，`y` 是形状为 `(N,)` 的整数标签，取值 0–9。只读取可信来源的 NPZ。客户端训练文件至少 2 个样本，测试文件至少 1 个。训练文件有 21 个或更多样本时，留出 25% 验证后至少能组成一个大小为 16 的本地训练批次。`--verify-only` 检查格式；使用已有分区时，训练集与测试集的来源需由提供者确认。

**从官方批次生成**会创建 100 个客户端，按类使用 Dirichlet 分布（α=0.5），默认划分种子 3103。官方训练样本只写入 `train/`，官方测试样本只写入 `test/`。两者使用相同的类分配比例和独立的随机流。图像转换为 `float32`、通道在前，像素按 `x / 127.5 - 1` 归一化。分区配置写入 `Cifar10/config.json`。重复使用同一份分区时直接执行校验命令。上游 `generate_cifar10.py` 会先合并官方训练集和测试集，复现本页实验应使用上面的准备命令。

## 3. 先跑一次接入自检

```bash
python -m diagnostics.fit_native_step_control --phase check \
  --dataset Cifar10 --data-root "$DATA_ROOT"
```

兼容入口的单轮检查使用第一个 24 人池，从中选 12 个上传者完成本地训练和聚合，再在私有验证样本上给三个候选步长打分。要检查四个池共 96 人的文件，使用第 2 节的 `--verify-only`。单轮检查的最后一行应含 `"model": "CifarNet"`、`"uploads": 12` 和三个有限的 `private_gains`。这三个数是首轮私有成本的改善量。

## 4. 训练、留出对照和保存模型

按顺序执行。第二、三条使用第一条写出的同一个 `RUN`。每个种子跑 40 个通信轮，共 8 次决策。拟合时还需从同一起点展开候选分支，耗时高于单条 40 轮轨迹。

```bash
python -m diagnostics.fit_native_step_control --phase dev \
  --dataset Cifar10 --data-root "$DATA_ROOT" --output "$RUN" --jobs 1

python -m diagnostics.fit_native_step_control --phase held \
  --dataset Cifar10 --data-root "$DATA_ROOT" --output "$RUN" --jobs 1

python -m diagnostics.fit_native_step_control --phase deploy \
  --dataset Cifar10 --data-root "$DATA_ROOT" --output "$RUN" \
  --method rl --seed 16200 --checkpoint "$RUN/cifarnet_seed16200.pt"
```

- **dev** 用客户端 0–23 和 24–47、共 24 个种子采集同起点配对分支。另用客户端 48–71、8 个种子的私有验证选定对照策略，将冻结规则写入 `policy.json`。
- **held** 在客户端 72–95、12 个预先指定的种子上对比所有冻结方法。每对方法共享初始化、客户端分区、按轮公布的上传日程和本地 SGD 随机流。测试数据仅用于终点评分。编号 96–99 的客户端不参与本协议。
- **deploy** 用指定种子从头运行一条冻结策略轨迹，输出 8 个动作和终点指标。`.pt` 保存这条轨迹的 `CifarNet.state_dict()`。只需取得一个模型时，完成 `dev` 即可直接执行 `deploy`。更换数据盘路径时显式传 `--data-root`；使用同一套分区文件，程序会核对留出客户端的文件哈希。

## 5. 从输出判断 RL 是否有用

本实验无攻击，每轮有 12 个客户端上传。实际更新是 `新全局参数 = 原全局参数 + 所选步长 × FedTC 聚合增量`。所有对照都用 λ=0、τ=1.15，末五轮都比较 τ=1.15 和 1.55，按私有验证成本选终点。控制器只读取决策时可得的日程、轮次及候选更新的私有验证反馈。学习目标中的成本为验证错误率加 0.05 倍的截断交叉熵。`immediate` 用未来五轮的实测改善拟合；`rl` 在同一训练样本上再计入以步长 1 接续五轮的改善，折扣为 0.9，最后一个窗口直接用五轮改善。

| 输出文件 | 可以回答的问题 |
|---|---|
| `config.json`、`protocol.json`、`selection.json` | 新运行的种子列表、步长和客户端池，以及验证集选出的固定策略和时间表。 |
| `paired_training.json` | 每个配对起点的状态特征和三个候选的五轮、两窗口训练标签是多少？ |
| `held.json` | 每个种子实际执行了哪些动作？动作编号 0、1、2 依次对应步长 0.5、1.0、1.5；原始准确率为 0–1。 |
| `summary.json` | `[对照]["rl"]["accuracy"]` 的 `mean` 是 RL 最终测试准确率（%），`diff` 和 `ci95` 是 RL 减对照的配对差及其按种子重采样的 95% 区间，单位为百分点。`worst10` 是准确率最低的 3 个客户端的平均值。 |

看 `fixed` 可判断能否超过最佳固定步长，看 `schedule` 可判断状态反馈是否比按轮切换多提供收益，看 `immediate` 可判断第二个窗口的信息是否有贡献。若配对差的区间跨过零，这批种子对该项优势的证据仍不足。12 个种子共享同一留出客户端池。

```bash
python - "$RUN/summary.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    summary = json.load(f)
for ref in ("fixed", "schedule", "immediate", "direct", "incumbent"):
    row = summary[ref]["rl"]["accuracy"]
    lo, hi = row["ci95"]
    print(f"RL vs {ref:9s}  accuracy={row['mean']:.2f}%  diff={row['diff']:+.2f} pp  95%=[{lo:+.2f}, {hi:+.2f}]")
PY
```
