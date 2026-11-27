# FedTC 实验操作指南：怎么跑、怎么改、看哪个结果

这份指南对应 FedAA 主入口 `main.py` 中的 **`FedTCStep`**。默认任务用 CIFAR-10 和 ResNet-20 做 100 轮联邦训练，观察服务器每 5 轮根据私有验证反馈选择步长，能否提高终点精度。**目前已验证单轮启动，真实 CIFAR-10 的 100 轮结果待服务器实测。**

## 1. 从 GitHub 获取实验分支，先做单轮冒烟

首次使用，在服务器上准备存放项目的目录中执行：

```bash
git clone --branch codex/mnist-paired-feedback-results --single-branch https://github.com/KaiyanZhaoPhoenix/FedTC.git
cd FedTC/code
```

服务器已有该实验分支的本地副本时，从仓库根目录切到这个分支并更新，然后进入 `code/`：

```bash
git switch codex/mnist-paired-feedback-results
git pull --ff-only origin codex/mnist-paired-feedback-results
cd code
```

下文命令均从 `code/` 执行。参考环境为 Python 3.9、PyTorch 2.6.0、torchvision 0.21.0。已激活项目环境时可直接执行冒烟命令；首次安装可在该目录创建虚拟环境：

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
```

只检查能否启动时，执行：

```bash
DEVICE=cpu bash scripts/run_cifar10_resnet_100.sh check
```

省略 `check` 时脚本也只做单轮检查。设备从 JSON 的 `device` 读取（本配置为 CUDA），也可以用环境变量 `DEVICE` 临时覆盖。运行时可用 `git rev-parse HEAD` 记录本次实验对应的代码版本。

**成功标志**是输出中有 `"status": "ok"` 和 `"checked_rounds": 1`。默认参数下，`costs.uploads` 为 12，`costs.local_optimizer_steps` 为 48，`private_gains` 有三个有限数值。程序随后退出。`input_source` 显示本次用的数据：存在 `DATA_ROOT/Cifar10/` 时读真实分区；缺少该目录时使用临时合成图像。合成输入可检查模型、SGD 和反馈接口，实际 α 和客户端文件需要在真实分区上核验。

## 2. 默认实验到底跑什么

这里有三个不同的人数。**100** 是磁盘上的数据分区数，**24** 是一条训练轨迹的候选客户端数，**12** 是每轮实际训练并上传的人数。四个 24 人池分别用于两组拟合、校准和留出，编号 96–99 的四份分区在此配置中用不到。

| 阶段 | 默认使用的客户端和种子 | 做什么、得到什么 |
|---|---|---|
| `dev` 的拟合部分 | 0–23、24–47 两池轮流使用，拟合种子 16000–16023 | 默认每条轨迹选两个窗口，从相同起点分别运行步长 0.5、1.0、1.5，记录本窗口收益和以步长 1 接续一窗的收益。其余窗口以步长 1 推进。 |
| `dev` 的校准部分 | 48–71，校准种子 16100–16107 | 依据私有验证准确率选择固定步长和时间表等对照，生成冻结规则 `dev/policy.json`。 |
| `held` 留出评测 | 72–95，留出种子 16212、16213、16214 | 每个种子重新初始化分类模型。各方法共享初始模型、分区、上传日程和本地训练随机性，报告第 100 轮测试指标的三种子均值和样本方差。 |
| `deploy` 单独训练，可选 | 使用留出客户端池，种子由 `--seed` 指定 | 读取同一个冻结文件，运行一种策略，保存最终分类模型。 |

默认分组日程按客户端训练集的主导类别把 24 人分成两组。每个 5 轮窗口选定一个组，由于组内恰有 12 人、每轮也取 12 人，该窗内每轮由同一批客户端上传，顺序会打乱。若研究问题要求每轮从全池独立抽人，使用第 3 节的 `uniform`。

实现中的 `rl` 用配对收益离线拟合打分器，目标是本窗口的验证成本改善量加上 0.9 倍的接续改善量；最后一窗只计本窗。评测时打分器已经冻结，每窗仍会根据新的反馈选择动作。验证样本来自各客户端训练集的留出部分，模拟器集中做批量前向计算，策略读取按群体汇总的验证收益。

默认每个入选客户端每轮做 4 次 SGD 更新，batch 为 16，学习率为 0.1。100 个通信轮分成 20 个窗口；每窗首轮用同一批上传试探三个步长，再按选中步长推进该窗。开发时会运行候选分支，末窗还会比较两种 τ。**100 轮指一条分类模型轨迹的长度**，整套开发和多种子对照的计算量更大。当前 `FedTCStep` 使用诚实反馈和正常上传。FedAA、FedAWA 的基线成绩需另行运行。

## 3. 改实验：复制配置，编辑字段，再做冒烟

默认配置为 `configs/cifar10_resnet_100r.json`。保留它作参照，为自己的实验复制一份。

```bash
cp configs/cifar10_resnet_100r.json configs/cifar10_my_run.json
```

用编辑器打开 `configs/cifar10_my_run.json`，按下表改已有字段，保存后运行：

```bash
CONFIG=configs/cifar10_my_run.json DEVICE=cpu bash scripts/run_cifar10_resnet_100.sh check
```

| 想改什么 | JSON 字段及默认值 | 怎么改 |
|---|---|---|
| 数据集和模型 | `dataset="Cifar10"`，`model_name="resnet20"` | CIFAR-10 可直接换 `resnet18` 或 `cifarnet`。MNIST + MLP 见下面的独立命令。 |
| 磁盘分区总数 | `partition_clients=100` | 改为需要的总分区数，再生成一份数据。总数须能容纳用到的所有客户端池。 |
| 一条轨迹有多少客户端 | `num_clients=24`，`pool_offsets=[0,24,48,72]` | `pool_offsets` 顺序为拟合 A、拟合 B、校准、留出。每池从起始编号连续取 `num_clients` 人，最大起始编号加池大小不能超过总分区数。 |
| 每轮有多少人参与 | `num_sel_clients=12`，`cohort_schedule="grouped"` | `grouped` 将池分成两组，每轮人数最多为池大小的一半，向下取整。`uniform` 每轮从全池无放回随机抽样，允许 1 到全池人数。 |
| 哪些客户端做本地训练 | `training_cohort="sampled"` | `sampled` 只训练本轮入选者。`all` 会训练池中所有客户端，上传人数仍由 `num_sel_clients` 控制。 |
| 本地训练预算 | `local_steps=4`，`batch_size=16`，`local_learning_rate=0.1` | 每人改做 8 次更新就填 `local_steps=8`。要按本地 epoch 控制，删掉 `local_steps` 并添加 `local_epochs=1`。固定步数模式要求各客户端留出验证数据后仍能组成一个完整 batch。 |
| 全局轮数和控制频率 | `global_rounds=100`，`macro_length=5` | 改为 200 轮、每 10 轮选一次步长就填 `200` 和 `10`。末窗不足整窗时按剩余轮数执行。 |
| 数据异构性和划分种子 | `dirichlet_alpha=0.5`，`partition_seed=3103` | α 越小，通常各客户端的类别越集中。改任一项后生成新分区。FedAA 的 `--alpha` 表示服务器保留旧模型的比例；数据异构性用 `dirichlet_alpha`。 |
| 私有验证集 | `validation_fraction=0.25`，`split_seed=3103` | 在每个客户端自己的训练样本中划出验证集。`split_seed` 控制这一步，`partition_seed` 控制样本在客户端之间的划分。 |
| 实验种子和设备 | `fit_seeds`、`tune_seeds`、`held_seeds`；`device="cuda"`，`jobs=1` | 三组种子互不重叠。单卡使用 `jobs=1`。通过启动脚本选设备用 `DEVICE=cpu` 或 `DEVICE=cuda`；直接调用 `main.py` 可传 `--device cpu`。 |

**例 A：100 人中每轮随机抽 20 人。** 在复制的 JSON 里把以下五个字段设为下列值，其余字段保留。已有相同的 100 人数据分区可以复用。

```json
{
  "partition_clients": 100,
  "num_clients": 100,
  "num_sel_clients": 20,
  "pool_offsets": [0, 0, 0, 0],
  "cohort_schedule": "uniform"
}
```

这样拟合、校准、留出共用这 100 人，留出评测衡量对**新种子**的表现。如果希望四阶段使用互不重叠的客户端池、每池都有 100 人，则要准备至少 400 份分区并使用 `[0,100,200,300]`。

**例 B：只改轮数、本地步数或模型。** 要做 200 轮、每人 8 步、ResNet-18，就在副本中分别将 `global_rounds`、`local_steps`、`model_name` 设为 `200`、`8`、`"resnet18"`。若目的是判断哪一项影响精度，每份配置只改其中一项。真实分区就绪后，也可临时覆盖参数做一次首轮检查：

```bash
python main.py --config configs/cifar10_resnet_100r.json --data-root "$PWD/dataset" \
  --model resnet18 --global-rounds 200 --local-steps 8 --phase check --device cpu
```

**例 C：只改异构性 α。** 在自己的 CIFAR 配置里只把 `dirichlet_alpha` 从 `0.5` 改为 `0.3`。以下命令复用原始图像，在新目录生成客户端分区，再用真实分区做首轮检查。若原始批次已在本机，可省略 `--download`。

```bash
CFG=configs/cifar10_my_run.json
DATA="$HOME/datasets/fedtc/cifar10_a03_s3103"
RAW="$PWD/dataset/Cifar10-raw/rawdata"
python -m dataset.prepare_cifar10 --config "$CFG" --data-root "$DATA" --raw-root "$RAW" --download
CONFIG="$CFG" DATA_ROOT="$DATA" DEVICE=cpu bash scripts/run_cifar10_resnet_100.sh check
```

**例 D：换成 MNIST + MLP。** 仓库提供 `configs/mnist_native_40r.json`，采用已有的 FedAA MNIST 分区和每客户端 1 个本地 epoch。下面两行准备该版本的数据并做首轮检查；后续 `dev`、`held` 也从 `main.py` 进入。CIFAR 专用启动脚本用于 CIFAR 任务。

```bash
python dataset/prepare_mnist.py
python main.py --config configs/mnist_native_40r.json --phase check --device cpu
```

**改完是否要重做开发？** 官方数据集、磁盘分区数、划分 α 或划分种子变化时，先准备匹配的数据。修改模型、参与人数、训练预算、窗口等协议参数，可以继续用匹配的数据，但正式评测需要用新配置重新 `dev`。仅追加留出种子时，可以复用同一冻结策略；新种子要避开拟合和校准种子。候选步长与末窗 τ 目前是算法常量，见 [步长策略](../code/algorithms/fedtc/step_control.py)。

## 4. 正式提交时用哪些命令

**一条命令跑完默认实验。** 仅当准备好正式训练时执行。GPU 脚本会在需要时准备 CIFAR-10 数据，然后依次运行 `check`、`dev`、`held`。每次会自动创建新的结果目录。

```bash
bash scripts/run_cifar10_resnet_100.sh full
```

**正式跑自己的配置。** 例如例 C 已经准备好 α=0.3 的分区后，下面一条命令会让数据校验、开发和留出评测都使用修改过的配置。只想增加评测种子时，可以跳过这条整套命令，直接用下方的 `held` 命令读取已有策略。

```bash
CONFIG=configs/cifar10_my_run.json DATA_ROOT="$HOME/datasets/fedtc/cifar10_a03_s3103" \
  DEVICE=cuda bash scripts/run_cifar10_resnet_100.sh full
```

**按阶段提交或复用已冻结策略。** 以下示例使用默认配置。每个作业都先设置相同的 `CFG`、`DATA` 和 `RUN`；如果采用上面的 α=0.3 实验，`CFG`、`DATA` 换成例 C 的设置，`RUN` 指向该实验的新目录。`DATA` 指向**包含 `Cifar10/` 的父目录**。

```bash
CFG=configs/cifar10_resnet_100r.json
DATA="$PWD/dataset"
RUN="$PWD/runs/cifar10_resnet20_100r_v1"
```

首次准备真实 CIFAR-10 数据执行第一条；已有准备好的分区直接执行第二条校验。计算节点离线时，可在登录节点完成准备并将 `DATA` 放在共享盘。

```bash
python -m dataset.prepare_cifar10 --config "$CFG" --data-root "$DATA" --raw-root "$PWD/dataset/Cifar10-raw/rawdata" --download
```

```bash
python -m dataset.prepare_cifar10 --config "$CFG" --data-root "$DATA" --verify-only
```

然后依次执行训练入口的三个阶段。`dev` 成功后再提交依赖该策略的 `held` 作业。各阶段要求新的输出目录；如果只有留出作业需要重跑，复用已有 `dev/policy.json`，把 `--output` 改为新的目录即可。

```bash
python main.py --config "$CFG" --data-root "$DATA" --phase check --output "$RUN/check"
python main.py --config "$CFG" --data-root "$DATA" --phase dev  --output "$RUN/dev"
python main.py --config "$CFG" --data-root "$DATA" --phase held --policy "$RUN/dev/policy.json" --output "$RUN/held"
```

**已有同协议的冻结策略，只提交三种子评测。** 默认的 `fit_seeds` 有 24 个，`tune_seeds` 有 8 个。只改 JSON 的 `held_seeds` 或在命令行覆盖它，即可复用 `policy.json`。以下示例取之前 16200–16211 之后的连续三个编号；FedAA、FedAWA 等独立基线也应使用同三个种子并对齐客户端池、数据划分、训练预算与评测口径。

```bash
python main.py --config "$CFG" --data-root "$DATA" --phase held \
  --policy "$RUN/dev/policy.json" --held-seeds 16212 16213 16214 --output "$RUN/held_3seeds"
```

**只评一个种子**有两种用途。下列第一条只在留出种子 16212 上对比六种策略，适合看动作、耗时和指标。第二条只训练 `rl` 并保存模型，适合只要一个可加载的分类器。按需要选一条。单个种子的差值只描述该次运行。

```bash
python main.py --config "$CFG" --data-root "$DATA" --phase held --held-seeds 16212 \
  --policy "$RUN/dev/policy.json" --output "$RUN/held_16212"
```

```bash
python main.py --config "$CFG" --data-root "$DATA" --phase deploy --method rl --seed 16212 \
  --policy "$RUN/dev/policy.json" --output "$RUN/model_16212"
```

## 5. 结果怎么读

**三种子论文表直接取留出作业目录的 `seed_statistics.csv`。** 默认整套作业写到 `$RUN/held/`，上面的独立作业写到 `$RUN/held_3seeds/`。每个方法一行。全局精度取 `accuracy_mean_pct` 和 `accuracy_sample_variance_pp2` 两列；最差 10% 客户端精度取对应的 `worst10_*` 两列。每个种子贡献一个终点值，三个种子等权。均值单位为 %，样本方差以百分数为输入、除以 `n−1`（这里是 2），单位为百分点²。只有一个种子时样本方差在 CSV 留空、在 JSON 中为 `null`。

| 想知道什么 | 看哪个文件 | 读什么 |
|---|---|---|
| 实际用了什么配置 | 各阶段输出目录的 `config.json`、`protocol.json` | 生效的参数、客户端池、模型、窗口设置。逐种子 `manifest.json` 还记录真实上传名单。 |
| 策略怎么得到的 | `$RUN/dev/paired_training.json`、`selection.json`、`policy.json` | 拟合用的特征和配对收益、校准结果、最终冻结规则。`held` 与 `deploy` 读同一个 `policy.json`。 |
| 各种子选了什么步长 | `$RUN/held/held.json` | `steps` 是实际步长，`taus` 是每窗的 τ，`metrics` 是该种子终点指标，精度取值为 0–1。 |
| 多种子平均有没有提升 | `$RUN/held/summary.json` | 先索引对照名，再索引 `rl` 和指标名。`n`、`mean`、`sample_variance` 分别给出种子数、RL 均值和样本方差。`diff` 是 RL 减对照的平均配对差（百分点），`ci95` 是该差值按种子重采样的区间。 |
| 训练多花了多少算力 | 各种子目录的 `costs.json` | 实际上传数、本地 SGD 更新数和私有验证次数。留出阶段共享相同动作前缀；单条默认部署轨迹还要为末窗第二个 τ 多执行 4 个训练轮。 |
| 单种子模型存在哪里 | `$RUN/model_16212/deploy/16212/model.pt` | 包含模型参数和 BatchNorm 缓冲区的 PyTorch `state_dict`。 |

判断自适应是否有收益，先看 `summary["fixed"]["rl"]["accuracy"]`，它比较学习策略与校准选出的固定步长；再看 `summary["immediate"]["rl"]["accuracy"]`，它比较计入后续窗口收益和只用本窗口收益的两种学习目标。要判断**读取当前反馈**是否提供了超出预设时间表的增益，看 `summary["schedule"]["rl"]["accuracy"]`。`schedule` 从两条时间表和三个固定步长中按校准精度选出，实际选中了哪条可查 `dev/selection.json` 的 `lock.schedule`。另两组对照是恒定单位步长 `incumbent` 和直接挑首轮私有收益最高候选的 `direct`。全局 `accuracy` 按测试样本数加权，`worst10` 是最差 ⌈0.1N⌉ 个客户端的等权平均准确率，默认 N=24 时取 3 人。

需要修改训练加速时，从 [客户端执行与私有验证](../code/servers/step_execution.py) 和 [共享服务器的客户端训练循环](../code/servers/serverShared.py) 入手。窗口选择及配对运行在 [FedTCStep 服务器](../code/servers/serverFedTCStep.py)。ResNet 各候选分支分别保存参数和 BatchNorm 缓冲区；浮点运行统计按权重合并，整数计数取有权重客户端的最大值，服务器步长只缩放参数增量。已有服务器截图缺少确切 ResNet 变体和训练设置，历史精度的复现仍需原始配置。
