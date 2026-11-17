# 环境、数据与实验复现

## 安装

从仓库根目录进入 `code/`。已验证 Python 3.9.6、CPU、PyTorch 2.6.0；其他平台需重新检查数值一致性。

```bash
cd code
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-tested.txt
.venv/bin/python dataset/prepare_mnist.py
.venv/bin/python -m unittest discover -s tests -v
```

`requirements.txt` 引用同一份已验证版本清单。Python 3.9 用户需要遵循该版本组合，不应无条件安装最新 NumPy/PyTorch。

## 精确恢复实验数据

`dataset/prepare_mnist.py` 默认从 FedAA 固定提交下载 `dataset/mnist` 的 202 个文件，逐个校验 `dataset/mnist_manifest.json` 中的 SHA-256，已存在且校验正确的文件会跳过。数据默认写入 `code/dataset/mnist/`。失败可直接重新运行；已有不匹配文件不会被静默覆盖。

```bash
.venv/bin/python dataset/prepare_mnist.py --verify-only
# 如果已经下载了同一 FedAA 版本，可从本地复制并核验：
.venv/bin/python dataset/prepare_mnist.py --from-checkout /path/to/FedAA
```

原始生成脚本仍保留，另行生成可运行 `cd dataset` 后 `../.venv/bin/python generate_mnist.py`。**重新生成是新划分，不能默认当成已发布实验的精确数据复现。** 精确复现优先使用固定快照。

上游合并 MNIST 官方训练/测试池后再划分客户端，并单独保留服务器数据。因此这里的测试准确率是客户端测试划分上的指标，不是官方 MNIST test split 指标。训练前的额外私有验证留出为 10%，`split_seed=2026`；所有方法共享。原生 `drop_last=True` 保留，100 客户端时四个小客户端没有完整 SGD batch。

## 小规模运行检查

```bash
.venv/bin/python run_comparison.py --output-root runs/smoke \
  --algorithms FedAA FedTC FedTCFixed FedTCSAC FedTCSACGate FedAvg RFA TrimmedMean MultiKrum \
  --seed 1 --device cpu --num_clients 12 --num_sel_clients 8 --aggre_num 4 \
  --global_rounds 6 --local_epochs 1 --batch_size 16 --eval_gap 2 \
  --eval-clients 10 --macro-length 2 --tc-batch-size 2 \
  --warmup-macros 1 --buffer_size 128 --attack same_value
```

该命令验证接口和比较条件，不检验学习收敛或算法优势。输出目录必须不存在或为空。

## 单种子攻击与轮数实验

```bash
.venv/bin/python run_scenarios.py --output-root runs/new_single_seed --jobs 2
.venv/bin/python report_scenarios.py runs/new_single_seed
```

共 57 次运行：12 客户端、8 场景、独立 20/100 轮预算，以及 100 客户端、3 场景、100 轮。方法为 FedAA、FedTCFixed、FedTC，固定组参数为 `(0,1.35)`，H=2、控制器 batch=2。小规模取原生客户端 0--11，不能与全量 100 客户端视为同一总体。

原归档运行于添加 SAC 之前。当前脚本可重新执行同一协议，但部分入口文件有新增可选参数，历史源码哈希不全部等于当前文件。历史原始哈希保留在归档 `plan.json`，未篡改成新版本。核心原控制器和前向计算未变。

## SAC 四组实验

```bash
.venv/bin/python run_sac_study.py --output-root runs/new_sac --jobs 4
.venv/bin/python report_sac_study.py runs/new_sac
```

先运行 9 个固定参数组合各 80 轮，按最后 20% 轮次的私有验证损失选参，不使用测试分数。随后 4 种方法 × 4 种反馈各运行 300 轮，顺序为真实反馈、再伪造反馈。全部保留同值模型投毒和虚假效用。H=2、控制器 batch=16、warmup=16 宏步、每宏步 4 次学习更新。主指标为第 250--300 轮六个评测点的均值。固定组选参额外花费 720 轮。

运行期间会保存配置、数据哈希、初始化、客户端名单、原始反馈、执行动作、模型及控制器断点。`report_sac_study.py` 针对完整新运行，要求这些文件及匹配的源码哈希。

## 核验已有结果和重画图

发布归档包含压缩原始日志，但不含模型/优化器断点。不要对发布目录运行要求完整断点的旧报告脚本。

```bash
python3 verify_published.py
.venv/bin/python report_published.py ../results/sac_fourway_seed1 --output runs/replot_sac
.venv/bin/python report_published.py ../results/single_seed_comparison --output runs/replot_scenarios
```

校验器核对归档文件哈希、准确率与原始评测日志、门控计数、数据/初始化/训练参与名单的一致性；不声称重新验证未随仓库发布的模型张量和 replay 断点。原运行时的这些核验结果仍作为历史记录保留。

`reproducibility/archive_manifest.json` 记录每个文件原始与发布哈希、压缩方式及遗漏断点的哈希。只有机器路径被改为 `@CODE@`、`@RUN@`、`@PYTHON@` 等标记，数值测量未改。记录中的命令保留原参数；标记不是可直接执行的 shell 变量，实际复现请使用上面的入口。

## 论文

```bash
cd ../paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

需要 TeX 发行版。PDF 是当前研究草稿，CIFAR、动态攻击和少数类别实验表仍待运行。
