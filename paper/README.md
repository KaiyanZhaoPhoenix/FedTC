# 论文源文件与草稿状态

入口为 [main.tex](main.tex)，编译输出为 [main.pdf](main.pdf)。当前使用 ICML 2026 模板，保留原稿的匿名模板模式；这不表示实际投稿、接收或作者信息已经定稿。

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

需要 pdfLaTeX、BibTeX、latexmk。模板 `.sty` / `.bst` 随源文件保留，参考文献为 `reference.bib`。编译中间文件被 Git 忽略。

## 阅读顺序

`main.tex` 决定顺序：摘要 → 仓库草稿状态 → 引言 → 背景 → 方法 → 实验计划 → 相关工作 → 结论 → 影响声明与参考文献 → 附录。文件编号不完全等于阅读顺序。

## 当前状态

- 正文方法是原区间 Actor-Critic，SAC 属于代码中的额外对照，不静默替换正文算法。
- 实验部分的 CIFAR、动态攻击、少数类别和开销表格是待完成计划；`--` 不代表零或已验证结果。
- 最新已完成的是仓库 `../results/` 下的单种子 MNIST 诊断，不能填入这些不同数据集和协议的表格。
- 摘要后新增了显式草稿状态，说明自适应、少数群体保护和 Critic 安全解释仍待验证。其他科学推导保留待审计，详见 [限制](../docs/limitations.md) 与 [回退审计](../docs/fallback_audit.md)。
- 当前只引用 `content/figs/FedTC_framework.jpg`，其中旧版标签与当前公式不完全一致，图注明确标记待更新。其他历史图与占位图原样归档，见 [配图索引](content/figs/README.md)，不作为真实实验结果。
- 未凭空填写作者、单位或发表信息；原模板中的匿名作者占位项仍未定稿。

整理前所有源文件的哈希保存在 `../docs/migration_manifest.json`。本次稿件改动用于状态标注和必要的编译/布局修正，不增加新的性能或理论保证。
