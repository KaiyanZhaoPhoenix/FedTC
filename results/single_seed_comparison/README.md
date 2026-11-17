# 单种子攻击与轮数对照

原运行名：`single_seed_attack_horizons_20261117`。57 次运行，19 组比较，均为 seed=1。

- 12 客户端：8 场景 × 20/100 轮两个独立训练预算 × 3 方法。
- 100 客户端：3 场景 × 100 轮 × 3 方法。
- 方法：FedAA、固定 `(0,1.35)` 的 FedTC、原区间控制器 FedTC。
- 主指标：最终诚实客户端测试样本加权准确率。末 10 轮均值只作为时间平均描述。

小规模一些场景相对 FedAA 有明显差异，但不能归结为 RL 收益：完整 100 客户端的原 FedTC 全部回退，与固定保守动作结果相同。单种子没有显著性估计。完整数据说明和边界见 [RESULTS.txt](RESULTS.txt)。

| 文件 / 目录 | 内容 |
|---|---|
| `results.csv`、`summary.json` | 57 次运行的指标与汇总 |
| `paired_differences.csv` | 同场景方法间差异 |
| `learning_curves.csv`、`controller_history.csv` | 每个评测点及控制器轨迹 |
| `figures/` | 原始 PNG/PDF 图，无跨种子误差条 |
| `reproducibility/plan.json`、`status.json` | 原预设方案、完成状态；机器路径已替换为标记 |
| `reproducibility/raw/<experiment>/<algorithm>/` | 原始日志、manifest、指标；JSON/JSONL gzip 压缩 |
| `reproducibility/logs/` | 原 stdout 和报告日志 |
| `reproducibility/rl_duration/` | 在固定经验池上继续优化的历史诊断；没有新环境交互 |
| `reproducibility/archive_manifest.json` | 原始/发布文件哈希、遗漏断点清单 |

历史源码与当前增加 SAC 后的入口有三处文件差异；核心原版控制器未变，原哈希保留。读取公开结果用 `code/verify_published.py` / `report_published.py`。历史的完整 checkpoint/replay 核验记录不等于现在可以在没有断点的公开目录重新执行那些检查。
