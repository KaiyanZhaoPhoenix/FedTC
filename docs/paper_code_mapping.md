# 正文公式与代码映射

正文入口为 `paper/main.tex`，以下公式标签位于 `paper/content/4_methodology.tex`。

| 正文部分 / 标签 | 代码 | 状态 |
|---|---|---|
| 几何中位数、`eq:delta_tilde` | `code/algorithms/fedtc/forward.py` | 已实现 |
| 排名效用、`eq:u_norm` | 同上 | 已实现 |
| `eq:calibration`、参数域与截断半径 | 同上 | 已实现；全部 FedTC 版本共享 |
| 截断本地验证改善 | `code/clients/clientFedTC.py` | 私有留出集计算，不能使用测试集 |
| 独立评估队列 | `code/servers/serverFedTC.py` | 从全部客户端抽取，与前向接受无关 |
| `eq:trimmed_bounds`、`eq:ci_interval` | `code/algorithms/fedtc/interval.py` | 以整数 `m-b` 实现诚实人数下界 |
| `eq:state_v2` | `forward.py` / `controller.py` | 8 维状态，包含上一反馈区间 |
| PopArt、`eq:dual_td`、`eq:actor_obj` | `code/algorithms/fedtc/controller.py` | `paper` 与显式 `consistent` 模式分开 |
| `eq:fallback_gate` | `IntervalController.select_action` | 原版按正文比较；Critic 并非已校准的价值置信界 |
| 固定参数消融 | `code/servers/serverFedTC.py` | `FedTCFixed`，参数由命令行指定 |
| SAC 控制器 | `code/algorithms/fedtc/sac.py` | 额外诊断扩展，不是当前正文算法 |
| 原 FedAA 基线 | `code/DDPG/DDPG.py` / `servers/serverFedAA.py` | 保留上游 DDPG 和干净服务器奖励 |
| 正文 CIFAR / ALIE / Min-Max / 长尾套件 | 暂无完整对应实现与结果 | 计划，不能由 MNIST 结果代替 |
| 配对相对收益门控 | `docs/next_version.md` | 设计提案，尚未实现 |

原稿的“诚实平均改善区间包含”不等于任意少数群体保护，也不等于神经价值估计或全轨迹安全。PopArt/TD 约定、门控保守策略含义、负报告保留表述等问题详见审计。整理仓库时未把这些待研究问题隐藏成已修复结论。
