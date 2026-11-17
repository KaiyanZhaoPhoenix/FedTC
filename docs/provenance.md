# 来源与整理记录

- 目标仓库：`KaiyanZhaoPhoenix/FedTC`，整理前 `main` 为 `1d74601b018a31e559347b93249b206265c36d9a`。
- 本次以新的提交替换旧文件布局，保留历史，不重写历史、不混用旧根目录 `src/` 实现。
- 主代码来自本地已运行的 `FedAA_FedTC`，框架上游为 `Gp1g/FedAA@1fba884934cbec1b3506812d9d612e6a181b1d79`。
- 正文来自现有 `_Part_FedTC` LaTeX 文件；保留 ICML 2026 模板、正文公式和待完成表格，添加仓库草稿状态说明。
- 原本地独立实现 `code_fedtc` 不再作为第二个运行入口；其诊断结论保存在 `fallback_audit.md`。原始独立实现仍保留在原工作区。
- 设计草稿保存在 [design_notes/](design_notes/README.md)，不作为现行实现规范。
- 45 个迁移时 Python 文件与原稿源文件的哈希见 [migration_manifest.json](migration_manifest.json)。数据、模型、环境与缓存未迁入版本控制。
- 上游未改组件和适配说明见 [source_provenance.json](source_provenance.json)。原 DDPG、客户端训练及 FedTC 核心计算按原字节保留。

## 实验归档

| 本地原实验名 | 发布位置 |
|---|---|
| `single_seed_attack_horizons_20261117` | `results/single_seed_comparison/` |
| `sac_fourway_seed1_20261117` | `results/sac_fourway_seed1/` |
| `rl_duration_diagnostic_20261117` | 第一组的 `reproducibility/rl_duration/` |

CSV 数值和图像原样保留。文本中的本机路径替换为具名标记；大型原始 JSON/JSONL 日志用 gzip 无损压缩。每批归档的 manifest 同时记录原始文件 SHA-256 与发布文件 SHA-256，并列出未发布的断点文件及其哈希。原始实验资料仍留在原工作区。

历史 `VERIFIED.txt` 记录原运行时包括模型断点在内的完整核验；公开目录的 `verify_published.py` 只重新核验现有归档证据。两者的范围不能混为一谈。

早期实验的入口源码哈希与新增 SAC 后的入口不全相同，原值保留。最新 SAC 实验所记录的运行源码应与 `code/` 中对应源文件匹配；新增数据准备和归档工具独立于这些运行源码。
