# SAC 四组对照

原运行名：`sac_fourway_seed1_20261117`。9 次 80 轮固定参数搜索，随后 16 次 300 轮主实验，全部 seed=1，100 个 MNIST 原生客户端。

四种方法为调参固定 FedTC、原区间控制器、无门控 SAC、SAC 加门控。四种反馈为真实报告、All-Max、Echo、Bimodal；所有条件都保留同值模型投毒与虚假效用。

主要结果：原控制器及 SAC 加门控都在预热后 134/134 次回退。无门控 SAC 执行候选，但本次未超过保守执行路径。固定组按私有验证损失选为 `(0.5,1.35)`，与保守 `(0,1.35)` 不同。详细结论与限制见 [RESULTS.txt](RESULTS.txt)。

| 文件 / 目录 | 内容 |
|---|---|
| `results.csv` / `results.json` | 16 次主实验完整指标；主指标为末 20% 轮次六点评测均值 |
| `learning_curves.csv` / `controller_history.csv` | 原始曲线与执行参数 |
| `gate_diagnostics.json` / `shadow_checks.json` | 区间宽度、门控裕量、离线单步配对检查 |
| `figures/` | 原始准确率、执行次数、参数轨迹 PNG/PDF |
| `reproducibility/plan.json` / `fixed_selection.json` | 原预设参数、选参规则及结果 |
| `reproducibility/main_commands.json` / `tuning_commands.json` | 原运行参数，机器路径已替换为标记 |
| `reproducibility/raw/main/<report>/<algorithm>/` | 每次主实验原始记录 |
| `reproducibility/raw/tuning/grid_*/` | 9 次选参的原始记录 |
| `reproducibility/logs/` | 训练和测试日志 |
| `reproducibility/archive_manifest.json` | 文件校验和、压缩/路径变换、遗漏断点 |

历史报告里的 `source_snapshot.zip`、`model.pt`、`controller.pt` 未放入发布目录；源码见 `../../code/`，其与本次运行记录的源文件哈希一致。公开检查只覆盖已发布证据，不伪称重复了模型张量/replay 检查。该原运行的检查结论保留在 `reproducibility/VERIFIED.txt`。

重画曲线或核验无需训练，命令见 [复现指南](../../docs/reproduction.md)。
