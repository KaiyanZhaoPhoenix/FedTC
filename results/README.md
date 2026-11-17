# 已完成实验与证据范围

| 目录 | 运行 | 指标 / 范围 |
|---|---|---|
| [single_seed_comparison](single_seed_comparison/README.md) | 57 次；FedAA、固定 FedTC、原 FedTC | seed=1；12/100 客户端；不同攻击与独立训练预算 |
| [sac_fourway_seed1](sac_fourway_seed1/README.md) | 9 次选参 + 16 次主实验 | seed=1；100 客户端；300 轮；四组控制器与四种反馈 |

这些都是探索和诊断结果，不是正文计划中的 CIFAR/长尾/动态攻击主基准。两批使用的控制器 batch、固定参数选择和主指标不同，不能拼接成同一实验。

各目录包含 `RESULTS.txt`、CSV/JSON 汇总、图表，以及 `reproducibility/`。压缩原始记录保留必要的配置、划分、名单、每轮动作、反馈、评测结果和来源哈希；模型和优化器断点留在原工作区。历史报告中的文件名采用当时的运行布局，发布位置以各目录 README 为准。

从 `code/` 执行 `python3 verify_published.py` 可核验现有归档；不需要下载数据或重新训练。新训练请写入 `code/runs/`，不要覆盖这里。

原来误启动的多种子运行、零散接口自测和大型断点未并入这两批结论。原始本地资料未删除。
