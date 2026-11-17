# FedTC 接入 FedAA 的实现说明

本工程基于官方 [Gp1g/FedAA](https://github.com/Gp1g/FedAA)，固定上游版本为 `1fba884934cbec1b3506812d9d612e6a181b1d79`。它直接使用 FedAA 的客户端与服务器基类，不调用旧独立工程 `code_fedtc` 的训练器。旧工程保留作公式审计参考。

## 复用与新增的边界

| 部分 | 当前实现 |
|---|---|
| 数据生成、客户端划分 | 原 `dataset/generate_*.py`、`dataset/dataset_utils.py`，文件未修改 |
| 数据读取 | 原 NPZ 格式和客户端编号；增加可指定的数据目录，修复服务器 NPZ 重复读取 |
| 本地训练 | `clientFedTC` 继承原 `clientFedAA.train()`，该文件未修改 |
| 优化器 | 原 SGD、交叉熵、梯度裁剪 60、local epochs、可选指数学习率衰减 |
| 模型 | 原 `utils/models.py` 和 `utils/resnet.py`，文件未修改；MNIST 仍用 784→100→10 全连接模型 |
| FedAA 策略网络 | 原 `DDPG/DDPG.py`，包括隐藏层噪声、动作归一化、actor/critic 更新，文件未修改 |
| 评测 | 继承原 `Server.evaluate()`、`Client.test_metrics()`、`Client.train_metrics()`；补上原来遗漏的 AUC 结果保存 |
| FedTC | `servers/serverFedTC.py` 接入前向校准、宏步反馈区间、PopArt、双 critic、门控 |
| 公共训练循环 | `servers/serverShared.py`，所有对比方法用同一份训练、攻击、广播和评测代码 |

上游未修改组件的 SHA-256 记录在 `SOURCE_PROVENANCE.json`。原入口和说明分别保存在 `upstream_main.py`、`README.upstream.md`。

## 两种运行协议

`--protocol shared` 是默认的公平对比协议。每轮执行：广播全局模型 → 原客户端本地训练 → 聚合 → 广播 → 全局模型评测。第 1 轮就有真实训练更新；每个 H5 记录对应明确的全局轮次。FedAA 的距离筛选和 DDPG 保持原实现，其上一条转移在下一轮距离状态出现后写入 replay，最后一条转移显式终止。所有方法按设定轮数跑完，不按服务器准确率提前结束。

`--protocol upstream --algorithm FedAA` 保留原 `FedAA.train()` 的时序：先聚合上轮模型再训练，首轮聚合尚未训练的模型；全局和本地评测交替写入原结果列表；服务器准确率达到 1 时可提前结束。该模式无本地验证集留出，结果继续保存到原 `results/` 和 `models/`。新入口统一采用指定 seed 初始化，因此这是原训练流程兼容模式，并非承诺逐比特复现原 main.py 的硬编码随机数行为。不可将 upstream 模式的指标直接与 shared 模式混作同一实验。

## 对比方法

| 参数 `--algorithm` | 聚合 / 控制 |
|---|---|
| `FedAA` | 原距离筛选选择 `aggre_num` 个客户端，再用原 DDPG 生成权重；使用原干净服务器数据计算 reward |
| `FedTC` | 正文前向校准 + 独立评估队列 + 区间双 critic + 执行门控 |
| `FedTCFixed` | 固定 `(lambda,tau)=(0,1.35)` 消融，不更新控制器 |
| `FedAvg` | 按参与客户端实际训练集样本数加权 |
| `RFA` | **均匀客户端权重**的几何中位数，15 次改进 Weiszfeld 迭代；并非样本量加权 RFA |
| `TrimmedMean` | 每维去掉两端各 f 个值后平均 |
| `MultiKrum` | 每个候选的最近 `n-f-2` 个其他更新平方距离求和，选择最低的 `aggre_num` 个等权平均 |

`--robust-f` 默认 `ceil(ad_ratio*num_sel_clients)`，是公开的配置参数，不从本轮恶意客户端真实身份计算。若仅抽取部分上传客户端，这个默认数不是概率保证的恶意数量上界；可显式设置更保守的 f。MultiKrum 要求 `n>2f+2` 且 `aggre_num<=n-f-2`。RFA 和 TrimmedMean 不伪造标量客户端权重，日志中 `weights=null`。

## 数据、训练与评测的共同约束

1. 保留原生成器及现有 `dataset/mnist` 的 100 客户端 Dirichlet(0.1) 划分。上游生成流程合并原始训练/测试数据、预留服务器数据，再分客户端并按 75/25 分训练/测试；本工程没有改为另外一种官方 train/test 口径。增加本地验证集前后，原 NPZ 文件均不改变。
2. shared 协议默认从每个原客户端**训练集**固定留出 10%，作为 FedTC 私有验证集。所有 baseline 也使用相同的剩余训练子集，不能让 baseline 多训练这 10%。`--split-seed` 控制这个固定划分，独立于运行 seed。原客户端测试集不参与训练、效用或 FedTC 奖励计算。
3. 正文中的前向效用用训练子集的逐样本截断损失下降计算：`clip((before-after)/M,0,1)`。宏步奖励使用独立留出的验证子集，比较宏步开始与结束的全局快照。恶意上报在仿真层修改；真实身份不传入校准函数、区间估计器或控制器。
4. `--training-cohort all` 默认保留 FedAA 每轮训练全部客户端的行为，再从中均匀抽 `num_sel_clients` 个上传。`--training-cohort sampled` 只训练抽到的客户端，对应正文的部分参与方式。两者都可运行，但同一对比必须一致。与正文不同的默认设置不应在论文中隐去。
5. 本地 minibatch 和模型随机运算由轮次、客户端 ID、seed 决定；控制器随机数不会改变训练顺序。攻击也使用独立且一致的随机流。公共对比脚本核对实际初始模型、原始数据、留出划分、恶意身份和每轮训练/上传名单。
6. 保留原 `drop_last=True`。训练样本少于 batch size 的客户端没有 SGD 批次；程序打印并在 manifest 记录这些客户端，不悄悄改变其训练行为。当前完整 MNIST 划分含极小客户端，设置实验 batch size 时需要考虑这一点。
7. 按原 FedAA 口径，只评测仿真中诚实客户端；准确率和 micro-AUC 按客户端测试样本数加权，标准差按客户端准确率计算。训练损失也使用原训练 loader。恶意身份在这里仅用于离线评测。H5 只记录全局模型；`--evaluate-local` 另存本地模型结果，二者不混合。
8. 原服务器广播和聚合仅处理模型 parameters，BatchNorm buffers 保持本地状态。本工程保留这个约定。MNIST/CifarNet 不依赖 BatchNorm；若改用含 BatchNorm 的模型，必须在论文中说明这种联邦 buffer 处理方式。
9. FedAA 保留原干净 `server_test.npz` 奖励。FedTC、FedAvg 等不读取这份服务器数据。这个信息条件差异是算法本身的区别，实验说明应如实列出。

## 攻击兼容性

`same_value`、`gaussian`、`sign_flip` 沿用上游对**整个客户端模型参数**的变换语义，而不是默认理解为对 delta 的攻击。特别是上游 `sign_flip` 实际乘以正的 `abs(N(0,10))`，并未反转符号；此行为保留用于原协议比较。新增 `delta_sign_flip` 才是 `delta <- -10*delta`。`none` 为无模型攻击。

FedTC 另支持 `--fake-utility` 以及 `--report-attack none|all_max|bimodal|echo`。echo 使用同一评估队列中诚实正反馈的中位数，属于仿真中的全知对手。所有方法共享模型攻击设置；无效用/评估上报接口的 baseline 不执行这些额外消息攻击。

本次提供的是 FedAA 原训练框架内的算法接入和常见 baseline。正文实验所列的全部攻击、非平稳阶段、后门任务和所有消融组合尚未在本工程中完整实现，不应将这次联跑写成论文实验已完成。

## FedTC 与公式的对应

`algorithms/fedtc/forward.py` 对应几何中位数、median 尺度、并列平均秩、tanh 信誉分数、解析截断半径与归一化权重。每个宏步 action 固定 H 轮，默认 H=5；下一步状态只使用已经完成的宏步统计。

`algorithms/fedtc/interval.py` 对应上下各保留 `m-b` 个反馈的均值和采样松弛项。用超几何尾概率选满足 `delta_bad` 的 b，且强制 `b<m/2`；采用有效 `eta=b/m-rho_max` 避免正文取整导致高估诚实样本数量。

`algorithms/fedtc/controller.py` 对应 PopArt、双 critic、actor、目标 critic 与门控。默认 `--td-mode paper` 保持正文 TD 形式。显式选 `--td-mode consistent` 才加入先前审计提出的 `gamma*(1-done)*mu/std` 补偿项；这是实现变体，不能混称正文原式。它也不解决区间太宽或门控可行性问题。

先前审计的区间松弛过大、独立上下界比较过严、保守动作导致探索不足以及神经 critic 缺少校准保证等问题仍然成立，参见 [回退审计](fallback_audit.md)。短自测减小 replay batch/warmup 是为触发并检查控制器更新；某次动作通过门控不代表其已获得可靠安全保证。迁移没有自动放宽门控，也未修改论文公式。

## 输出与可复现范围

- `manifest.json`：运行参数、上游版本、初始模型/数据/划分校验值、仿真恶意身份、零训练批次客户端。
- `private_split_indices.json`：本地仿真复现用索引，不作为服务器协议上传内容。
- `rounds.jsonl`：训练与上传客户端、聚合权重、FedTC 参数和前向统计。
- `macros.jsonl`：FedTC 评估队列、反馈、区间、门控原因、控制器训练损失。
- `evaluations.jsonl`、`metrics.h5`：统一全局模型评测；H5 保留原 `rs_*` 字段并增加 round。
- `model.pt`、`controller.pt` / FedAA 网络文件：最终模型与控制器状态；当前没有完整训练过程的中断续跑入口。
- `comparison.json`、`comparison.csv`：多个方法完成后核对协议并汇总结果。

现有验证在本机 CPU 完成，覆盖真实 MNIST 和小型原生 NPZ 接口测试；没有完成 CUDA、CIFAR-10、FashionMNIST 的正式实验。保留上游数据生成器不等同于验证所有外部数据下载源仍可用。
