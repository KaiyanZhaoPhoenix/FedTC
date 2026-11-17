# FedAA 接入验证记录

日期：2026-11-17。本机 CPU，Python 3.9.6、PyTorch 2.6.0、torchvision 0.21.0、NumPy 2.0.2、scikit-learn 1.6.1、h5py 3.14.0。以下为原工作区的历史检查记录，所列旧 `runs/` 路径未全部发布；本次整理后的验证见 [validation.md](validation.md)，运行入口均在 `code/`。

## 已完成的检查

| 检查 | 结果 |
|---|---|
| 单元与集成测试 | 32 项通过 |
| 六方法真实 MNIST 联跑 | FedAA / FedTC / FedAvg / RFA / TrimmedMean / MultiKrum 各完成 6 轮 |
| 原生 100 客户端 MNIST 联跑 | FedAA / FedTC 各完成 4 轮 |
| 原版 FedAA 兼容模式 | 12 客户端、2 轮，训练、两阶段评测、H5 保存完成 |
| 数据与协议核对 | 初始模型、原始数据文件、私有留出索引、公共训练评测配置、恶意身份、每轮参与名单一致 |
| 上游关键文件 | 数据生成器、本地 SGD 文件、DDPG、模型文件与固定上游提交逐字节相同 |
| Python 编译 / 修改检查 | 编译通过，`git diff --check` 通过 |

测试不仅检查程序退出：还验证本地 SGD 与相同批次的原生更新一致、训练不使用验证样本、FedTC 在完全没有服务器根数据文件时仍可训练、反馈不读取客户端测试 loader、控制器实际更新、H5 中全局轮次正确、最终模型广播一致。对比检查会拒绝被人为改动的学习率配置或参与名单。

数学回归覆盖几何中位数、信誉公式、截断半径、非塌缩、秩并列、区间上下界、超几何概率、PopArt 预测保持、门控记录、终止转移、actor/critic 更新和控制器状态恢复。这里的状态恢复测试不表示整个联邦训练已有续跑入口。

## 真实数据联跑细节

第一组使用原 MNIST 划分的客户端 0–11，每轮训练 12 个、上传 8 个，FedAA/MultiKrum 最终选 4 个；6 轮、local epochs=1、batch=16、seed=1、每 2 轮评测、10 个 FedTC 评估客户端。模型攻击 `same_value`，恶意比例 0.2；所有方法共同留出 10% 训练样本。FedTC 使用 H=2、replay batch=2、warmup=1，目的是短程触发学习路径。

结果与完整日志在 `runs/mnist_integration_smoke/`。FedTC 完成 3 个宏步、8 次控制器优化，其中 2 次因 warmup 回退，最后一次候选动作通过门控。小样本 critic 的通过判定不能当作可靠置信保证。

第二组使用全部原生 100 客户端，每轮训练/上传全部客户端，FedAA 选择 50 个；4 轮、local epochs=1、batch=16、seed=1、每 2 轮评测。FedTC 使用 H=2、20 个评估客户端、replay batch=1、warmup=1，同时启用 fake utility 与 all-max 评估反馈攻击。其余公共设置一致。

结果与完整日志在 `runs/mnist_100_clients_smoke/`。两方法均完成，FedTC 控制器实际更新。原 `drop_last=True` 下的小客户端零训练批次行为也被记录。该组用于完整数据和对抗反馈接口检查，不是正式性能评测。

原版兼容模式记录在 `runs/upstream_compatibility.log`，生成原目录格式的 H5 和模型。该模式交替记录全局/本地准确率，不与 shared 模式汇总。

## 复现命令

```sh
.venv/bin/python -m unittest discover -s tests -v

.venv/bin/python run_comparison.py --output-root runs/recheck_small \
  --device cpu --num_clients 12 --num_sel_clients 8 --aggre_num 4 \
  --global_rounds 6 --local_epochs 1 --batch_size 16 --eval_gap 2 \
  --eval-clients 10 --macro-length 2 --tc-batch-size 2 \
  --warmup-macros 1 --buffer_size 128 --attack same_value

.venv/bin/python run_comparison.py --algorithms FedAA FedTC \
  --output-root runs/recheck_full_data --device cpu --global_rounds 4 \
  --local_epochs 1 --batch_size 16 --eval_gap 2 --eval-clients 20 \
  --macro-length 2 --tc-batch-size 1 --warmup-macros 1 --buffer_size 128 \
  --attack same_value --report-attack all_max --fake-utility

.venv/bin/python main.py --protocol upstream --algorithm FedAA \
  --num_clients 12 --num_sel_clients 8 --aggre_num 4 --global_rounds 2 \
  --local_epochs 1 --batch_size 16 --buffer_size 128 --episodes 0
```

## 不能由这些检查得出的结论

不能据此宣称 FedTC 优于 baseline，或者区间和门控已完成可靠校准。没有运行论文所需的多 seed、完整训练轮数、全部数据集、全部攻击与消融。本次也没有修复正文的区间保守性和神经 critic 置信性问题；默认仍执行正文 TD 与门控。完整实现边界见 `INTEGRATION.md`。
