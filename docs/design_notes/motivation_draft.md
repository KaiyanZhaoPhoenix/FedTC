# FedTC Motivation 重构草稿

主贡献定位：**RL-Guided Data-Free FL Trust Calibration**
三角顶点：**No server data × No attack prior × No blind trust**

---

## 一、定位合理性分析

### 1.1 Trust Calibration（问题重构）

相关文献逐篇列举：

- **FLTrust**, NDSS 2021（[arXiv:2012.13995](https://arxiv.org/abs/2012.13995)）：以服务器持有的 root dataset 训练参考更新，用余弦相似度定义 trust score。信任来源是服务器数据，与本文 zero-trust 设定相反。
- **Oort**, OSDI 2021（[arXiv:2010.06081](https://arxiv.org/abs/2010.06081)）：以客户端统计效用做客户端选择，信任等同于效用声明本身。
- **ACE**, USENIX Security 2024（[arXiv:2405.20975](https://arxiv.org/abs/2405.20975)）：首个针对 FL 贡献评估的投毒攻击，证明效用/贡献信号可被策略性伪造。
- **FedSelect**, KDD 2025（[GitHub](https://github.com/Binnnnnk/FedSelect_proxy)）：用 momentum meta-margin 内生评估样本重要性，属效用估计路线，不做鲁棒聚合。

本文定义：信任 = 不可验证效用声明与可验证更新行为的**单轮一致性度量**，经校准函数输出连续可信度。与上述三条脉络均无重叠。

**审稿人可能的质疑：**
1. "calibration" 与 confidence calibration 术语冲突——需在首次出现处显式区分；
2. "与 reputation weighting 的区别是什么"——需强调 credibility 是单轮双证据一致性，非跨轮累积分数，且有 Prop. 1/2 理论保证；
3. "trust calibration 是否只是换名的加权聚合"——需靠 Prop. 1（伪造声明封顶）与 Prop. 2（可分离区间）证明该形式化带来新性质。

### 1.2 Data-Free（约束设定）

必要性支撑文献：

- **Kairouz et al.**, FnT ML 2021（[arXiv:1912.04977](https://arxiv.org/abs/1912.04977)）：FL 综述，将服务器无验证数据列为现实部署约束。
- **FLShield**, ICDCS 2023（[arXiv:2308.05832](https://arxiv.org/abs/2308.05832)）：原文承认 "server's access to clean datasets — an assumption often impractical in real-life scenarios"。
- **FedSelect**, KDD 2025（[GitHub](https://github.com/Binnnnnk/FedSelect_proxy)）：以"消除对外部无偏数据的依赖"为核心动机。

现有方法无法满足该约束的文献：

- **FLTrust**, NDSS 2021（[arXiv:2012.13995](https://arxiv.org/abs/2012.13995)）：需 root dataset。
- **FedGreed**, arXiv 2025（[arXiv:2508.18060](https://arxiv.org/abs/2508.18060)）：2025 年最新聚合方法，摘要明确 "the central server is trusted and equipped with a reference dataset"。
- **PG-FFL**, arXiv 2022（[arXiv:2205.13415](https://arxiv.org/abs/2205.13415)）：RL 聚合权重分配，reward 由验证准确率构造。
- **FedAA**, AAAI 2025（[arXiv:2402.05541](https://arxiv.org/abs/2402.05541)）：RL 自适应聚合，奖励依赖服务器侧验证信号。
- **ProxyRL-FRS**, arXiv 2025（[arXiv:2508.10401](https://arxiv.org/abs/2508.10401)）：需额外部署代理模型估计客户端贡献。

**审稿人可能的质疑：**
1. "cross-silo 场景下服务器持有少量数据是合理的"——回应：本文面向 zero-trust/cross-device 部署，且强 non-IID 下服务器数据分布不代表客户端总体，验证信号有偏；
2. "FedSelect 已经去掉了外部数据依赖"——回应：它解决 client/sample selection，不做鲁棒聚合，也不处理伪造声明；
3. "data-free 奖励是否弱于验证数据奖励"——需补 source-free reward 与验证数据 reward 的上界对比实验。

### 1.3 RL-Guided（方法论）

相关文献逐篇列举：

- **PG-FFL**, arXiv 2022（[arXiv:2205.13415](https://arxiv.org/abs/2205.13415)）：策略梯度学习聚合权重，reward 含验证准确率。
- **FedAA**, AAAI 2025（[arXiv:2402.05541](https://arxiv.org/abs/2402.05541)）：DRL 动态调整聚合权重，"RL+聚合"最近的 CCF-A 先例。
- **ProxyRL-FRS**, arXiv 2025（[arXiv:2508.10401](https://arxiv.org/abs/2508.10401)）：代理模型 + RL 客户端选择（联邦推荐场景）。
- **FedCDA**, ICLR 2024（[OpenReview](https://openreview.net/forum?id=nbPGqeH3lt)）：跨轮 divergence 自适应聚合，自适应信号为固定规则、非学习式。

本文差异：RL 调节的是校准函数的**两个全局参数** $(\lambda_t,\tau_t)$，状态仅含聚合统计量（Eq. state），奖励为 source-free consistency reward（Eq. reward），无需任何数据。Prop. 2 表明可分离区间随攻击强度与异质性漂移，为在线调节提供理论动机。

**审稿人可能的质疑：**
1. "启发式自适应规则（如 δ 分位数调度）是否已足够"——**必须补非 RL 消融**，否则 RL 必要性不成立；
2. "5 维状态、2 维动作为何需要 DDPG"——需论证奖励非平稳（攻击者适应）使在线学习优于固定调度；
3. "RL 引入的训练不稳定与开销"——Efficiency 表已部分回应，建议补策略收敛稳定性分析；
4. "reward 的动量项被早期攻击污染怎么办"——需在论文中讨论或加敏感性实验。

### 1.4 总体判断

定位成立：Trust Calibration 给出问题定义，Data-Free 给出约束差异化，RL-Guided 给出解法。最强环节是 Data-Free（证据链完整），最薄弱环节是 RL 必要性（需消融兜底）。

---

## 二、Motivation 草稿（论文语言版）

### P1：问题设定——Zero-Trust 聚合

- 标准 FL 聚合（Eq. standard_agg）依赖三个隐含假设在现实部署中不成立：
  1. 服务器**无验证数据**评估客户端更新质量 [Kairouz 2019]；
  2. 客户端上报的效用 $u_k^{(t)}$ **不可验证**，可被任意伪造 [ACE, USENIX Sec'24]；
  3. non-IID 数据使良性更新天然偏离群体中心 [FedProx 2020]，更新几何与更新质量之间无稳定对应关系。
- 由此，聚合权重的确定构成一个 **zero-trust** 问题：任何单端信号（服务器侧、客户端侧）均不可独立采信。

【方法对应：Preliminaries 的设定与 notation；文献支撑：Kairouz 2019、ACE 2024、FedProx 2020】

### P2：不可能三角——现有方法不能满足的性质组合

理想聚合机制应同时满足三个性质：

| 性质 | 含义 | 缺失该性质的现有方法群 |
|---|---|---|
| **No server data** | 不依赖服务器端验证/参考数据 | FLTrust [NDSS'21]、FedGreed [arXiv'25]、PG-FFL ['22]、FedAA [AAAI'25] |
| **No attack prior** | 无需预知攻击比例/类型，随攻击强度与异质性在线调节 | Krum [NeurIPS'17]、Median/Trimmed-Mean [ICML'18]、DnC [NDSS'21]、Fed-NGA [NeurIPS'25] |
| **No blind trust** | 利用效用声明加速聚合，同时对伪造声明免疫 | 几何/统计规则放弃声明信息源；Oort [OSDI'21] 等效用加权方法静态消费声明且被 ACE 攻破 |

- 数据自由的方法（Krum → Fed-NGA）为固定规则，不消费效用声明，且在强 non-IID 下将良性长尾与恶意更新一并压制；
- 消费声明的方法（效用加权）为静态规则，声明可被伪造；
- 自适应的方法（FLTrust、PG-FFL、FedAA）的信任/奖励信号全部来自服务器数据，该假设被其同侪承认为不切实际 [FLShield, ICDCS'23]，去除该依赖已被列为独立研究目标 [FedSelect, KDD'25]。

**结论：现有方法至多满足其中两个性质（图 fig:trinity）。**

FedTC 对三个性质的同时满足方式：

| 性质 | FedTC 的满足机制 | 方法组件 |
|---|---|---|
| **No server data** | 信任信号与 RL 训练信号全部内生构造：行为偏差 $\delta_k^{(t)}$ 仅由客户端提交的 model delta 与鲁棒质心计算；RL 状态 $\mathbf{s}_t$ 仅含更新分布的聚合统计量；source-free consistency reward 由更新方向对齐与离散度构成 | Eq. delta_def、Eq. state、Eq. reward |
| **No attack prior** | 校准参数 $(\lambda_t,\tau_t)$ 不作为先验固定，而由 RL 元控制器每轮根据 $\mathbf{s}_t$ 在线输出（$\pi_\theta$，DDPG 优化），随攻击强度与异质性漂移自动调节宽严 | Eq. policy、Algorithm 1 |
| **No blind trust** | 消费效用声明但以行为证据封顶：激励项 $(1-\lambda_t)\tilde u_k^{(t)}$ 保留声明的加速作用，惩罚项 $\lambda_t\tanh(\delta_k^{(t)})$ 使伪造声明在行为偏差越界时可信度归零 | Eq. calibration、Prop. 1 |

【示意图：`content/figs/fig_trinity.tex`；文献支撑：上表各行】

### P3：问题重构——聚合作为信任校准

- 服务器每轮观测到两类性质不同的证据：不可验证的效用声明 $\tilde u_k^{(t)}$（Eq. u_norm）与可验证的行为偏差 $\delta_k^{(t)}=\|\Delta\mathbf{w}_k^{(t)}-\boldsymbol{\mu}_{\mathrm{rob}}^{(t)}\|_2$（Eq. delta_def）；
- 将聚合重构为**信任校准**：以双视角特征 $\mathbf{x}_k^{(t)}=(\tilde u_k^{(t)},\delta_k^{(t)})$（Eq. dual_view）为输入，由校准函数
  $$c_k^{(t)}=\big[(1-\lambda_t)\tilde u_k^{(t)}-\lambda_t\tanh(\delta_k^{(t)})-\tau_t\big]_+ \quad\text{(Eq. calibration)}$$
  输出连续可信度，经归一化得到聚合权重（Eq. fedtc_agg），替代"选或弃"的二值决策；
- 该形式化带来两个可证明的性质：
  1. **伪造声明封顶**（Prop. 1）：伪造 $\tilde u=1$ 的攻击者一旦 $\delta_k^{(t)}>\delta_{\mathrm{crit}}$ 则 $c_k^{(t)}=0$——声明操纵无法单独兑换影响力；
  2. **可分离性**（Prop. 2）：良性偏差有界且与恶意偏差有间隔时，存在门限区间同时保留全部良性客户端并剔除全部恶意客户端；连续调制保证偏离中心的长尾客户端在宽松校准下保留部分权重。

【方法对应：§4.2、§4.3 全部；文献支撑：行为证据思想最近者为 FLDetector [KDD'22]（跨轮一致性**检测剔除** vs 本文单轮**校准加权**），related work 需正面区分】

### P4：参数自适应——RL 元控制器与无数据奖励

- Prop. 2 的可分离区间随攻击强度与异质性漂移，固定 $(\lambda,\tau)$ 只能覆盖其一个切片，故校准参数需在线调节；
- 现有 RL 防御的奖励来自服务器代理数据 [PG-FFL'22; FedAA, AAAI'25]，违反 No server data；本文以 **source-free consistency reward**（Eq. reward）替代：全局更新与历史动量的余弦对齐度减去已接受更新的离散度，训练信号完全由更新几何内生构造；
- 元控制器以聚合统计量为状态 $\mathbf{s}_t$（Eq. state），输出 $(\lambda_t,\tau_t)=\pi_\theta(\mathbf{s}_t)$（Eq. policy），经 DDPG 在线优化；
- 框架作为即插即用元聚合器嵌入 FedAvg/FedProx，本地训练不变（§4.5, Algorithm 1）。

【方法对应：§4.4、§4.5 全部；文献支撑：PG-FFL、FedAA、FedSelect】

---

## 三、完整性分析：Motivation 声明 × 方法 × 实验证据

### 3.1 现有实验覆盖情况

| Motivation/方法声明 | 现有实验 | 覆盖度 |
|---|---|---|
| 抗 Byzantine（No attack prior 下仍鲁棒） | Robustness：3 种攻击 × ρ/强度扫描（Fig. 1） | ⚠️ 攻击覆盖够，但 baseline 错位（Ditto/lp-proj 是个性化方法） |
| No blind trust（抗伪造声明） | Fake Utility：Byzantine + 伪造报告（Fig. 2） | ⚠️ 缺"纯伪造声明"对照与自适应攻击 |
| 自适应能力 | Generalization 制度迁移（Fig. 3）+ Mechanism 轨迹（Fig. 4） | ⚠️ 缺动态攻击调度与 RL 消融 |
| 即插即用、低开销 | Efficiency（Table 1） | ✅ 基本够 |
| 几何规则压制长尾（P2 对现有方法的批判） | 无 | ❌ **完全缺失** |
| RL 必要性（P4 的核心） | 无 | ❌ **完全缺失** |

### 3.2 必须补充的实验（P0，缺失即审稿致命）

**E1. 修正 baseline 集合**
- 问题：当前对比 Ditto/lp-proj 为个性化 FL 方法，与 P2 批判的几何/信任引导路线不对应；
- 补充：Krum、Median/Trimmed-Mean（P2b 的直接代表）、FLTrust（数据依赖型代表，可作为"有数据上界"参照）、DnC 或 RFA（可选）；RL 方向如可行加 FedAA；
- 注意：FLTrust 需服务器数据，比较时说明其假设优势，FedTC 以 data-free 达到可比性能即为胜。

**E2. 良性长尾保留实验（支撑 P2 对几何规则的批判）**
- 设置：无攻击 + 高 non-IID（小 α Dirichlet），FedTC vs FedAvg/Krum/Trimmed-Mean；
- 指标：整体精度、收敛曲线、worst-client/worst-group 精度（量化长尾保留）；
- 对应声明：连续调制 + RL 宽松校准保留长尾贡献。

**E3. RL 必要性消融（支撑 P4）**
- 对照组：(i) 固定 $(\lambda,\tau)$ 网格最优值；(ii) 启发式调度（如 δ 分位数阈值规则）；(iii) 随机策略；
- 若启发式调度打平 RL，则 RL-Guided 定位不成立——**这是全论文最高风险实验**。

**E4. 自适应/知情攻击（支撑 Prop. 1 的实际效力）**
- 攻击者知晓 FedTC 机制，限制 $\delta$ 做优化型投毒（Min-Max/Min-Sum、AGR 类）；
- 对照：纯伪造声明 + 无害更新（预期：无伤害，验证"声明操纵无法单独兑换影响力"）。

### 3.3 强烈建议补充（P1）

**E5. Reward 消融**：$\beta=0$（无离散惩罚）、仅离散项无对齐项、source-free reward vs 验证数据 reward（oracle 上界）——直接回应"data-free 信号是否够用"；

**E6. 动态攻击调度**：训练中途开启/关闭/切换攻击类型，展示 $(\lambda_t,\tau_t)$ 轨迹的在线响应（与 Fig. 4 合并，强化 No attack prior 证据）；

**E7. 校准函数组件消融**：去效用项（纯行为，预期长尾受损）、去行为项（纯效用，预期 Fake Utility 下崩溃）、tanh 换线性（验证饱和设计的必要性）。

### 3.4 加分项（P2）

- **E8**：Prop. 1 的 $\delta_{\mathrm{crit}}$ 实证校验（机制图中叠加理论阈值线，验证理论与实现一致）；
- **E9**：规模与部分参与（客户端数 K、采样比例变化下的稳定性）；
- **E10**：RL 训练开销分解 + 多种子策略稳定性（回应 RL 过重质疑）；
- **E11**：$\beta$ 敏感性扫描。

### 3.5 结论

故事与方法已闭环，**所有缺口都在证据链**：P0 四项（baseline 修正、长尾保留、RL 消融、自适应攻击）决定论文生死；P1 三项决定说服力上限；P2 为锦上添花。当前实验体系只需保留 Robustness/Fake Utility/Generalization/Mechanism/Efficiency 的骨架，更换 baseline 后按 E1–E7 扩充即可支撑新 Motivation 的全部声明。
