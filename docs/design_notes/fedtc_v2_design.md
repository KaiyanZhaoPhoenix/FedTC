# FedTC v2.1 完整规划文档：故事、方法设计、代码架构与实验全流程

> **文档定位**：本文档对现有 `FedTC v2.1` 方案进行了全面审查与升级，将**论文叙事（Story & Abstract）**、**数学方法设计（Method v2.1）**、**代码工程实现蓝图（Code Architecture & PyTorch Spec）**与**全流程实验执行计划（Phase 0 $\to$ Stage A $\to$ Stage B $\to$ Full Benchmark）**整合为一份可直接指导代码编写与论文改写的总规划文档。

---

## 0. 对原版 `fedtc_v2_design.md` 的审查结论

经逐条核查，原版 `fedtc_v2_design.md` 中的**核心故事线与数学公式（v2.1）完全成立且就是我们当前的定稿方案**，具体优势如下：
1. **彻底修复了原稿 `_Part_FedTC` 的两处致命硬伤**：
   - 修复了前向 Proposition 1 中 $\tau_t \le 1 - 2\lambda_t$ 时无有限截断半径 $\delta_{\mathrm{crit}}$、且正文与附录公式符号（$\lambda_t$ vs. $\kappa$、$+\varepsilon_c$ 破坏权重归一化）不一致的数学漏洞；
   - 废除了原稿反向 RL 中鼓励“过度过滤、杀死诚实少数派”的自循环几何奖励 $r_t = \cos(\Delta\mathbf{w}_g^{(t)}, \mathbf{m}_{t-1}) - \beta\,\mathrm{disp}_t$。
2. **确立了不误伤诚实少数派的闭环反馈机制**：
   - 使用有界评估损失 $\ell_{\mathrm{eval}}(\mathbf{w};x,y) = \min(\ell_{\mathrm{CE}}(\mathbf{w};x,y), M_{\mathrm{loss}})$ 替代按客户端历史残差/中位数剔除评估者的做法（后者在“60% 诚实主流 + 20% 诚实少数派 + 20% 迎合型攻击者”下会系统性把诚实少数派的负向告警当成异常抹除）；
   - 使用宏步（Macro-step $H$ 轮）与双层置信区间 $[L_n, U_n]$ 覆盖全体诚实客户端的宏平均真实收益 $\bar{Z}_n^\star$，并通过执行层回退门控（Execution-Level Fallback Gate）保证安全底线。

**原文档缺失但本文档新增补全的三大板块**：
1. **完整可落地的算法状态空间 $\mathbf{s}_n$、动作映射与神经网络更新细节**；
2. **模块化 PyTorch 代码实现蓝图（含目录结构、核心类签名与关键张量实现）**；
3. **从 Phase 0 诊断实验（`K=60, Dir(0.1), CIFAR-100`）到主实验、消融实验与论文图表（Figures & Tables）的一一对应执行计划**。

---

## 1. 核心故事线与摘要（Abstract & Motivation）

### 1.1 一句话主线（One-Sentence Thesis）
在异构拜占庭联邦学习（Non-IID Byzantine FL）中，静态防御无法兼顾**抑制恶意投毒**与**保留长尾/罕见诚实客户端（Honest Minority）**；我们将鲁棒聚合重构为**基于局部验证集有界收益的闭环控制问题**，通过**解析可证明的前向软-硬信任校准层**与**抗恶意谎报的反向置信区间强化学习控制器**，在不依赖服务端干净验证集（Zero Clean Server Data）的前提下实现自适应鲁棒聚合。

### 1.2 中文摘要（定稿）
> 联邦学习（FL）在非独立同分布（Non-IID）数据下极易受到拜占庭投毒攻击。现有鲁棒聚合规则通常依赖静态过滤阈值或固定裁剪比例，陷入根本性的**鲁棒性–异构性两难困境（Robustness–Heterogeneity Dilemma）**：收紧过滤虽能抑制投毒攻击，却会误伤并长期饿死具有罕见数据分布的诚实少数派客户端；放宽过滤以包容数据异构性，又会放任贴近良性分布边界的自适应投毒更新渗透全局模型。解决该困境需要将聚合转化为动态序列决策问题，但服务端既无干净验证集，也无法直接信任客户端回传的局部验证反馈。
>
> 为此，本文提出 **FedTC**，一种无需服务端干净数据的动态信任校准框架，由**前向解析校准层（Forward Calibration Layer）**与**反向置信区间控制器（Backward Confidence-Interval Controller）**构成闭环。在前向层，FedTC 融合客户端自报质量分数与相对于几何中位数的无量纲距离，通过可微软衰减与精确解析截断半径抑制异常更新，严格保证单步聚合偏差有界且诚实中心客户端不发生全零崩溃。在反向层，针对客户端对局部验证损失改进量的策略性谎报，FedTC 在解耦的评估子集上采用有界评估损失，并结合宏步（Macro-step）评估周期与高概率恶意人数上界，构造覆盖全体诚实客户端真实宏平均进展的**非对称置信区间 $[L_n, U_n]$**——无需依赖高偏差的中位数残差剔除，从而完整保留诚实少数派的负面受损告警。基于该区间，FedTC 训练双价值网络（Dual Interval Critics）与独立保守参考策略，并通过执行层下界门控机制动态调节校准参数 $(\lambda_n, \tau_n)$。理论分析证明了前向校准层的单调性、Lipschitz 稳定性、精确截断半径与良恶分离条件，并建立了反向区间估计器对真实诚实收益的有限样本覆盖保证。实验表明，FedTC 在多种静态与动态切换投毒攻击下显著提升了全局准确率与少数派客户端保留率。

### 1.3 英文摘要（ICCV / ICML 规范版）
> Federated learning (FL) under non-IID data distributions is highly vulnerable to Byzantine poisoning attacks. Existing robust aggregation rules predominantly rely on static filtering thresholds or fixed trimming ratios, facing a fundamental *robustness–heterogeneity dilemma*: aggressive filtering suppresses poisoning but starves honest minority clients with rare data distributions, whereas permissive aggregation accommodates heterogeneity at the cost of admitting adaptive boundary-hugging attacks. Resolving this dilemma requires adaptive sequential control, yet the server possesses no clean validation data and cannot naively trust client-reported validation feedback.
>
> In this paper, we propose **FedTC**, a server-data-free dynamic trust calibration framework that closes the loop between a **Forward Calibration Layer** and a **Backward Confidence-Interval Controller**. In the forward layer, FedTC integrates self-reported client quality metrics with dimensionless deviations from the geometric median via smooth attenuation and an exact analytical cutoff radius, guaranteeing bounded single-step aggregation bias without weight-collapse. In the backward layer, to handle strategic misreporting of local validation loss improvements without silencing honest minority dissent, FedTC evaluates bounded loss improvements over decoupled macro-steps and constructs a high-probability **confidence interval $[L_n, U_n]$** via worst-case upper/lower tail trimming. Driving dual interval critics alongside an independently evaluated conservative reference policy and an execution-level fallback gate, FedTC dynamically adapts the calibration parameters $(\lambda_n, \tau_n)$ whenever optimistic exploration is certified by a positive lower-bound advantage. We establish theoretical guarantees for forward boundedness, separation, and backward finite-sample interval containment, and demonstrate empirically that FedTC consistently outperforms state-of-the-art defenses across static and non-stationary poisoning regimes.

---

## 2. Method 完整数学设计（FedTC v2.1 规范）

系统运行在总通信轮数 $T$ 上，每 $H$ 个联邦通信轮构成一个**控制宏步（Macro-step）** $n \in \{0, 1, \dots, N-1\}$（其中 $N = T/H$，$H \in \{1, 5, 10\}$）。在宏步 $n$ 内的每一轮 $t \in \mathcal{T}_n = \{nH, \dots, (n+1)H-1\}$，前向校准层保持动作 $\mathbf{a}_n = (\lambda_n, \tau_n)$ 不变；在宏步结束时，反向控制器根据 $H$ 轮累积的全局模型变化执行一次区间评估与策略更新。

### 2.1 前向信任校准层（Forward Calibration Layer，每轮 $t$ 执行）

设第 $t$ 轮被抽样的训练客户端集合为 $\mathcal{S}_t$（$|\mathcal{S}_t| = K_{\mathrm{train}}$），各客户端上传模型更新 $\Delta\mathbf{w}_k^{(t)} = \mathbf{w}_k^{(t)} - \mathbf{w}_t \in \mathbb{R}^d$ 及有界自报质量指标 $u_k^{(t)} \in [0, 1]$。

1. **鲁棒几何中心与无量纲偏离度**：
   通过 Weiszfeld 算法计算几何中位数作为鲁棒锚点：
   $$\boldsymbol{\mu}_{\mathrm{rob}}^{(t)} = \operatorname*{arg\,min}_{\boldsymbol{\mu}\in\mathbb{R}^d} \sum_{k\in\mathcal{S}_t} \|\Delta\mathbf{w}_k^{(t)} - \boldsymbol{\mu}\|_2.$$
   计算各客户端到锚点的欧氏距离 $\delta_k^{(t)} = \|\Delta\mathbf{w}_k^{(t)} - \boldsymbol{\mu}_{\mathrm{rob}}^{(t)}\|_2$，并以当轮距离中位数进行无量纲化：
   $$s_t = \operatorname{median}_{j\in\mathcal{S}_t} \delta_j^{(t)}, \qquad \tilde{\delta}_k^{(t)} = \frac{\delta_k^{(t)}}{s_t + \varepsilon_s} \ge 0 \quad (\varepsilon_s = 10^{-8}).$$

2. **组内排序归一化自报指标**：
   为消除不同客户端局部损失量纲差异并限制恶意夸大，对 $u_k^{(t)}$ 在 $\mathcal{S}_t$ 内做秩归一化（Rank Normalization）：
   $$\tilde{u}_k^{(t)} = \frac{\operatorname{rank}(u_k^{(t)} \text{ in } \mathcal{S}_t) - 1}{|\mathcal{S}_t| - 1} \in [0, 1].$$

3. **软衰减与硬截断校准函数（修复原稿 Prop. 1）**：
   给定当前宏步控制器的动作 $\mathbf{a}_n = (\lambda_t, \tau_t)$，其中动作空间严格约束为：
   $$\lambda_t \in [0, \lambda_{\max}] \;\; (\lambda_{\max} = 0.5), \qquad \tau_t \in [\tau_{\min}, \tau_{\max}] \;\; (\tau_{\min} = 1.05, \tau_{\max} = 1.80).$$
   定义温度常数 $\sigma_0 = 4.0$，客户端 $k$ 的未归一化信任得分为：
   $$c_k^{(t)} = \Big[(1-\lambda_t) + \lambda_t \tilde{u}_k^{(t)} - \tau_t \tanh\!\Big(\frac{\tilde{\delta}_k^{(t)}}{\sigma_0}\Big)\Big]_+, \quad [x]_+ = \max(0, x).$$
   归一化聚合权重与全局更新为：
   $$\alpha_k^{(t)} = \frac{c_k^{(t)}}{\sum_{j\in\mathcal{S}_t} c_j^{(t)}}, \qquad \Delta\mathbf{w}_g^{(t)} = \sum_{k\in\mathcal{S}_t} \alpha_k^{(t)} \Delta\mathbf{w}_k^{(t)}, \qquad \mathbf{w}_{t+1} = \mathbf{w}_t + \Delta\mathbf{w}_g^{(t)}.$$

4. **前向核心理论性质（4 条定理/命题，全部解析成立）**：
   - **性质 1（非崩溃保证 Non-Collapse）**：由中位数定义，至少 $\lceil |\mathcal{S}_t|/2 \rceil$ 个客户端满足 $\tilde{\delta}_k^{(t)} \le 1$。对这些中心客户端，即便 $\tilde{u}_k^{(t)} = 0, \lambda_t = 0.5, \tau_t = 1.8$，仍有：
     $$c_k^{(t)} \ge 0.5 - 1.8\tanh(1/4) \approx 0.5 - 1.8 \times 0.2449 = 0.0591 > 0,$$
     从而分母 $\sum_{j\in\mathcal{S}_t} c_j^{(t)} \ge 0.0591 \lceil |\mathcal{S}_t|/2 \rceil > 0$ 恒成立，**彻底消除了原稿分母加 $\varepsilon_c$ 导致权重和不为 1 的缺陷**。
   - **性质 2（单调性与 Lipschitz 稳定性）**：$c_k^{(t)}$ 关于 $\tilde{u}_k^{(t)}$ 单调不减（Lipschitz 常数 $\lambda_t$），关于 $\delta_k^{(t)}$ 单调不增（Lipschitz 常数 $\frac{\tau_t}{\sigma_0(s_t+\varepsilon_s)}$）。
   - **性质 3（精确解析截断半径 Exact Cutoff Radius）**：由于 $\tau_t \ge 1.05 > 1$ 且 $(1-\lambda_t) + \lambda_t \tilde{u}_k^{(t)} \in [1-\lambda_t, 1]$，任何满足
     $$\delta_k^{(t)} \ge R_t(\tau_t) \triangleq \sigma_0(s_t + \varepsilon_s)\operatorname{arctanh}\!\Big(\frac{1}{\tau_t}\Big)$$
     的更新必有 $c_k^{(t)} = 0$（无论其如何伪造 $u_k^{(t)} \in [0,1]$）。由凸组合性质，聚合偏移严格有界：
     $$\big\|\Delta\mathbf{w}_g^{(t)} - \boldsymbol{\mu}_{\mathrm{rob}}^{(t)}\big\|_2 < R_t(\tau_t).$$
   - **性质 4（良恶分离区间 Separation Condition）**：若诚实客户端最大无量纲距离 $\tilde{\delta}_{\max}^{\mathcal{H}} < \tilde{\delta}_{\min}^{\mathcal{B}}$（恶意客户端最小无量纲距离），且存在安全截断阈值 $\tau_t \in \big(\frac{1}{\tanh(\tilde{\delta}_{\min}^{\mathcal{B}}/\sigma_0)},\, \frac{1-\lambda_t}{\tanh(\tilde{\delta}_{\max}^{\mathcal{H}}/\sigma_0)}\big)$，则所有恶意客户端权重 $\alpha_b^{(t)} = 0$，且所有诚实客户端权重 $\alpha_h^{(t)} > 0$。

---

### 2.2 反向置信区间控制器（Backward Confidence-Interval RL Controller，每宏步 $n$ 执行）

#### (1) 真实控制目标：全体诚实客户端的宏平均有界收益
为防止单个极端 batch 或崩坏模型产生无界损失，定义**有界评估损失（Bounded Evaluation Loss）**：
$$\ell_{\mathrm{eval}}(\mathbf{w}; x, y) = \min\!\big(\ell_{\mathrm{CE}}(\mathbf{w}; x, y),\, M_{\mathrm{loss}}\big), \quad M_{\mathrm{loss}} = \ln C_{\mathrm{cls}} \text{ (或 } 2\ln C_{\mathrm{cls}}\text{)}.$$
（例如 CIFAR-100 上 $\ln 100 \approx 4.605$，设为 $M_{\mathrm{loss}} = 5.0$）。
在宏步 $n$（从模型 $\mathbf{w}_{nH}$ 到 $\mathbf{w}_{(n+1)H}$），客户端 $i$ 在其本地留出验证集 $\mathcal{V}_i$ 上的真实改进量为：
$$z_{i,n}^{\mathrm{true}} = \ell_{\mathrm{eval}}\big(\mathbf{w}_{nH}; \mathcal{V}_i\big) - \ell_{\mathrm{eval}}\big(\mathbf{w}_{(n+1)H}; \mathcal{V}_i\big) \in [-M_{\mathrm{loss}},\, +M_{\mathrm{loss}}].$$
系统的理想控制目标是最大化全体诚实客户端 $\mathcal{H}$ 的宏平均改进量（不可直接观测）：
$$\bar{Z}_n^\star = \frac{1}{|\mathcal{H}|}\sum_{i\in\mathcal{H}} z_{i,n}^{\mathrm{true}}.$$

#### (2) 解耦评估子集与双层置信区间 $[L_n, U_n]$
在宏步 $n$ 结束时，服务端**独立于前向入选结果**，从全体客户端中均匀无放回抽样评估子集 $\mathcal{E}_n$（$|\mathcal{E}_n| = m$）。每个 $i \in \mathcal{E}_n$ 报告并被服务端硬截断到合法区间：
$$z_{i,n} = \operatorname{clip}\!\big(\hat{z}_{i,n},\, -M_{\mathrm{loss}},\, +M_{\mathrm{loss}}\big).$$
- **诚实客户端**：如实报告 $z_{i,n} = z_{i,n}^{\mathrm{true}}$（含诚实主流与诚实少数派）；
- **恶意客户端**：可任意谎报 $z_{i,n} \in [-M_{\mathrm{loss}}, +M_{\mathrm{loss}}]$（如将受损谎报为 $+M_{\mathrm{loss}}$、或迎合主流中位数）。

将 $\{z_{i,n}\}_{i\in\mathcal{E}_n}$ 升序排列为 $z_{(1),n} \le z_{(2),n} \le \dots \le z_{(m),n}$。给定总体恶意比例上界 $\rho_{\max}$，由超几何分布尾界确定 $\mathcal{E}_n$ 中恶意客户端数量的高概率上界 $b = \lceil(\rho_{\max} + \eta_m)m\rceil$（满足 $b < m/2$）。定义**去上尾悲观均值**与**去下尾乐观均值**：
$$\hat{r}_n^{\mathrm{low}} = \frac{1}{m-b}\sum_{j=1}^{m-b} z_{(j),n}, \qquad \hat{r}_n^{\mathrm{high}} = \frac{1}{m-b}\sum_{j=b+1}^{m} z_{(j),n}.$$
补偿有限抽样误差 $\Delta_{\mathrm{samp}} = M_{\mathrm{loss}}\sqrt{\frac{2\ln(2/\delta_{\mathrm{samp}})}{m(1-\rho_{\max}-\eta_m)}}$ 后，得到完备置信区间：
$$L_n = \hat{r}_n^{\mathrm{low}} - \Delta_{\mathrm{samp}}, \qquad U_n = \hat{r}_n^{\mathrm{high}} + \Delta_{\mathrm{samp}}.$$

> **核心定理（Theorem 1: Honest-Minority-Preserving Interval Containment）**：
> 对任意策略性谎报攻击，只要 $\mathcal{E}_n$ 中实际恶意客户端数 $B_n \le b$，就有确定性不等式：
> $$\hat{r}_n^{\mathrm{low}} \le \frac{1}{|\mathcal{E}_n \cap \mathcal{H}|}\sum_{i\in\mathcal{E}_n \cap \mathcal{H}} z_{i,n}^{\mathrm{true}} \le \hat{r}_n^{\mathrm{high}}.$$
> 结合超几何抽样界，以至少 $1 - (\delta_{\mathrm{bad}} + \delta_{\mathrm{samp}})$ 的概率有 $\bar{Z}_n^\star \in [L_n, U_n]$。
> **关键意义**：由于 $\hat{r}_n^{\mathrm{low}}$ 完整保留了底部 $m-b$ 个最小值，**诚实少数派因被前向过度过滤而产生的负向反馈 $z_{i,n}^{\mathrm{true}} < 0$ 绝不会被当成异常值抹除，而是100%进入下界 $L_n$**，直接拉低当前过严动作 $(\lambda_n, \tau_n)$ 的价值评估！

#### (3) 状态空间 $\mathbf{s}_n$ 设计（8 维无量纲鲁棒特征）
宏步 $n$ 的控制器状态 $\mathbf{s}_n \in \mathbb{R}^8$ 由当前宏步内前向聚合的统计量与上一宏步的区间反馈构成（全部归一化到 $[-1, 1]$ 或 $[0, 1]$ 量级）：
1. $s_{n,1} = \log(1 + \bar{s}_n)$：宏步内平均中位数距离尺度 $\bar{s}_n = \frac{1}{H}\sum_{t\in\mathcal{T}_n} s_t$；
2. $s_{n,2} = \frac{\operatorname{IQR}(\{\tilde{\delta}_k\})}{\operatorname{median}(\{\tilde{\delta}_k\}) + \varepsilon_s}$：无量纲距离的四分位距离散度（反映更新分布的长尾/双峰程度）；
3. $s_{n,3} = \operatorname{frac}(\tilde{\delta}_k > 2.0)$：距离超过 2 倍中位数的外围客户端占比；
4. $s_{n,4} = \operatorname{corr}_{\mathrm{rank}}(\tilde{u}_k, -\tilde{\delta}_k) \in [-1, 1]$：自报质量指标 $\tilde{u}_k$ 与中心接近度 $-\tilde{\delta}_k$ 的 Spearman 秩相关系数（若攻击者伪造高 $u_k$ 但偏离中心，该相关性显著下降甚至转负，提示应降低 $\lambda_n$）；
5. $s_{n,5} = \bar{\rho}_{\mathrm{act}, n} \in (0, 1]$：上一宏步平均存活客户端比例 $\frac{1}{H|\mathcal{S}_t|}\sum_{t}\sum_k \mathbb{I}(c_k^{(t)} > 0)$；
6. $s_{n,6} = \tilde{L}_{n-1}$：上一宏步经 PopArt 标准化后的收益下界；
7. $s_{n,7} = \tilde{U}_{n-1} - \tilde{L}_{n-1} \ge 0$：上一宏步标准化置信区间宽度（反映反馈不确定性）；
8. $s_{n,8} = n / N \in [0, 1]$：训练进度归一化时间戳。

#### (4) PopArt 尺度自适应、双区间 Critic 与执行层安全回退门控
1. **PopArt 在线尺度归一化**：
   由于训练初期损失下降快、后期平缓，维护区间中点 $M_n = \frac{1}{2}(L_n + U_n)$ 的指数移动均值 $\mu_{R,n}$ 与标准差 $\sigma_{R,n}$：
   $$\tilde{L}_n = \frac{L_n - \mu_{R,n}}{\sigma_{R,n} + \varepsilon_r}, \qquad \tilde{U}_n = \frac{U_n - \mu_{R,n}}{\sigma_{R,n} + \varepsilon_r}.$$
   每当 $(\mu_{R,n}, \sigma_{R,n})$ 更新时，对双 Critic 输出线性层参数 $(\mathbf{W}_{\mathrm{out}}, b_{\mathrm{out}})$ 进行精确重缩放（PopArt），使隐藏层表征完全不受跨阶段量纲漂移影响。

2. **双区间价值网络（Dual Interval Critics）与保守参考策略 $\pi_{\mathrm{cons}}$**：
   - 定义固定保守参考策略 $\pi_{\mathrm{cons}}(\mathbf{s}) \equiv \mathbf{a}_{\mathrm{cons}} = (\lambda_{\mathrm{cons}}, \tau_{\mathrm{cons}}) = (0.0, 1.35)$（即关闭不可信自报指标 $\lambda=0$，采用中等稳健截断半径 $\tau=1.35 \implies R_t \approx 3.82 s_t$）。
   - 训练两个参数化价值网络 $Q_{\theta_{\mathrm{low}}}^{\mathrm{low}}(\mathbf{s}, \mathbf{a})$ 与 $Q_{\theta_{\mathrm{high}}}^{\mathrm{high}}(\mathbf{s}, \mathbf{a})$，其 TD 目标分别由标准化后的下界与上界驱动：
     $$y_n^{\mathrm{low}} = \tilde{L}_n + \gamma Q_{\bar{\theta}_{\mathrm{low}}}^{\mathrm{low}}\big(\mathbf{s}_{n+1}, \pi_\phi(\mathbf{s}_{n+1})\big), \qquad y_n^{\mathrm{high}} = \tilde{U}_n + \gamma Q_{\bar{\theta}_{\mathrm{high}}}^{\mathrm{high}}\big(\mathbf{s}_{n+1}, \pi_\phi(\mathbf{s}_{n+1})\big).$$

3. **风险感知 Actor 训练与执行层安全回退门控（Execution-Level Fallback Gate）**：
   - Actor $\pi_\phi(\mathbf{s}_n)$ 最大化风险加权目标，其中自适应悲观权重 $\omega_n = \operatorname{clip}\!\big(\omega_0 + \kappa \cdot (\tilde{U}_{n-1} - \tilde{L}_{n-1}),\, 0.5,\, 1.0\big)$ 随区间宽度单调递增：
     $$\mathcal{J}_{\mathrm{actor}}(\phi) = \mathbb{E}_{\mathbf{s}_n}\Big[\omega_n Q_{\theta_{\mathrm{low}}}^{\mathrm{low}}\big(\mathbf{s}_n, \pi_\phi(\mathbf{s}_n)\big) + (1-\omega_n) Q_{\theta_{\mathrm{high}}}^{\mathrm{high}}\big(\mathbf{s}_n, \pi_\phi(\mathbf{s}_n)\big)\Big].$$
   - **执行层门控（Execution Gate）**：在宏步 $n$ 实际下发动作前，比较候选动作 $\hat{\mathbf{a}}_n = \pi_\phi(\mathbf{s}_n)$ 的悲观价值下界与保守参考动作 $\mathbf{a}_{\mathrm{cons}}$ 的乐观价值上界（留出容忍裕度 $\xi_{\mathrm{tol}} = 0.05$）：
     $$\mathbf{a}_n = \begin{cases} \hat{\mathbf{a}}_n, & \text{if } Q_{\theta_{\mathrm{low}}}^{\mathrm{low}}(\mathbf{s}_n, \hat{\mathbf{a}}_n) \ge Q_{\theta_{\mathrm{high}}}^{\mathrm{high}}(\mathbf{s}_n, \mathbf{a}_{\mathrm{cons}}) - \xi_{\mathrm{tol}}, \\ \mathbf{a}_{\mathrm{cons}}, & \text{otherwise.} \end{cases}$$
   - **直观物理解释**：当攻击者进行疯狂谎报导致 $[L_n, U_n]$ 极宽，或者当 $\hat{\mathbf{a}}_n$ 过度激进导致诚实少数派受损（拉低 $Q^{\mathrm{low}}$）时，门控条件不满足，系统自动回退到 $\mathbf{a}_{\mathrm{cons}}$；只有当证据支持动态校准确实优于保守基线时，才放行 $\hat{\mathbf{a}}_n$。

---

## 3. 代码怎么写：工程架构与核心模块实现蓝图（Code Architecture）

为了保证代码结构清晰、可复现、易于跑完 Diagnostic $\to$ Stage A $\to$ Stage B $\to$ Full Benchmark 全部实验，建议在项目根目录下建立 `code_fedtc/` 目录，采用以下模块化架构。

### 3.1 推荐代码目录结构
```text
code_fedtc/
├── configs/
│   ├── diagnostic_cifar100.yaml     # Phase 0 诊断实验配置 (K=60, Dir(0.1), CIFAR-100)
│   ├── stage_a_ideal_rl.yaml        # Stage A 理想反馈与动态切换攻击配置
│   ├── stage_b_corrupted_ci.yaml    # Stage B 恶意谎报与宏步 H 消融配置
│   └── main_benchmark.yaml          # 主实验全基线对比配置
├── src/
│   ├── data/
│   │   ├── dirichlet_split.py       # Dir(alpha) 与 CIFAR-100-LT (IF=1, 10, 20) 划分器 + 本地 train/val 切分
│   │   └── minority_metrics.py      # 计算客户端偏离度 D_k、局部专属度 R_k 与 per-round alignment A_k^{(t)}
│   ├── models/
│   │   └── resnet_gn.py             # GroupNorm 版 ResNet-18 (避免 Non-IID 下 BatchNorm running stats 污染)
│   ├── attacks/
│   │   ├── model_poisoning.py       # Min-Max, Min-Sum, LIE (ALIE), Fang, IPM, Label-Flipping, Backdoor
│   │   ├── report_poisoning.py      # 恶意伪造 u_k 与策略性谎报 z_{i,n} (All-Max, Echo-Median, Bimodal-Jamming)
│   │   └── dynamic_scheduler.py     # 非平稳攻击调度器 (Warmup -> LIE -> Min-Max 切换)
│   ├── aggregators/
│   │   ├── baselines.py             # FedAvg, Multi-Krum, Trimmed-Mean, RFA, Bucketing+TM, NNM+Krum, FedLAW, FLTrust
│   │   └── fedtc_forward.py         # Weiszfeld 几何中位数 + 前向软-硬信任校准层 (Exact Cutoff Radius)
│   ├── rl_controller/
│   │   ├── ci_estimator.py          # 解耦评估抽样 E_n + 有界损失 l_eval + 双层置信区间 [L_n, U_n]
│   │   ├── popart.py                # PopArt 在线均值/方差归一化线性层
│   │   └── interval_sac_ddpg.py     # 8维状态提取 + 双区间 Critic + 保守参考策略 pi_cons + 执行层门控
│   └── trainer/
│       ├── fl_server.py             # 联邦服务端主循环 (支持 H 轮宏步与客户端权重/准确率的逐轮追踪)
│       └── fl_client.py             # 联邦客户端本地 SGD 训练 + 自报指标 u_k 计算 + 验证集有界损失评估
└── scripts/
    ├── run_phase0_diagnostic.py     # 一键跑 Phase 0 诊断实验并输出 3 张诊断图
    ├── run_stage_ab.py              # 一键跑 Stage A & B 控制器消融实验
    └── run_main_table.py            # 一键跑主实验表格
```

### 3.2 四大核心模块的 PyTorch 代码实现范式

#### 模块 1：前向信任校准层（`src/aggregators/fedtc_forward.py`）
```python
import torch
import numpy as np

def compute_weiszfeld_median(updates: torch.Tensor, max_iter: int = 15, eps: float = 1e-6) -> torch.Tensor:
    """
    updates: shape [K_train, d] (flattened client model updates)
    returns: mu_rob of shape [d]
    """
    mu = updates.mean(dim=0)
    for _ in range(max_iter):
        dists = torch.norm(updates - mu.unsqueeze(0), p=2, dim=1).clamp_min(eps)
        weights = 1.0 / dists
        weights = weights / weights.sum()
        new_mu = (weights.unsqueeze(1) * updates).sum(dim=0)
        if torch.norm(new_mu - mu, p=2) < eps:
            break
        mu = new_mu
    return mu

def fedtc_forward_calibrate(
    updates: torch.Tensor,       # [K_train, d]
    raw_u: torch.Tensor,         # [K_train], client self-reported quality in [0, 1]
    lam: float,                  # in [0.0, 0.5]
    tau: float,                  # in [1.05, 1.80]
    sigma_0: float = 4.0,
    eps_s: float = 1e-8,
):
    K = updates.size(0)
    mu_rob = compute_weiszfeld_median(updates)
    deltas = torch.norm(updates - mu_rob.unsqueeze(0), p=2, dim=1)  # [K_train]
    s_t = torch.median(deltas)
    delta_tilde = deltas / (s_t + eps_s)

    # Cohort-wise rank normalization of u_k into [0, 1]
    ranks = torch.argsort(torch.argsort(raw_u)).float()
    u_tilde = ranks / max(K - 1, 1)

    # Calibrated trust score c_k^{(t)}
    scores = (1.0 - lam) + lam * u_tilde - tau * torch.tanh(delta_tilde / sigma_0)
    c_t = torch.relu(scores)

    denom = c_t.sum()
    if denom > 0:
        alpha = c_t / denom
        agg_update = (alpha.unsqueeze(1) * updates).sum(dim=0)
    else:
        alpha = torch.zeros_like(c_t)
        agg_update = mu_rob

    # Exact theoretical cutoff radius R_t(tau)
    cutoff_radius = float(sigma_0 * (s_t.item() + eps_s) * np.arctanh(1.0 / tau))
    stats = {
        "s_t": s_t.item(),
        "delta_tilde": delta_tilde.detach().cpu(),
        "u_tilde": u_tilde.detach().cpu(),
        "alpha": alpha.detach().cpu(),
        "active_frac": (c_t > 0).float().mean().item(),
        "cutoff_radius": cutoff_radius,
    }
    return agg_update, stats
```

#### 模块 2：反向双层置信区间估计器（`src/rl_controller/ci_estimator.py`）
```python
import math
import torch

def compute_reward_confidence_interval(
    reports_z: torch.Tensor,     # [m], reported loss improvements from evaluator cohort E_n
    rho_max: float = 0.25,       # assumed max malicious fraction
    eta_m: float = 0.05,         # finite-sample hypergeometric slack
    M_loss: float = 5.0,         # bounded evaluation loss cap (e.g., >= ln(100))
    delta_samp: float = 0.05,    # sampling confidence parameter
    use_sampling_slack: bool = True,
):
    m = reports_z.numel()
    # 1. Server-side hard clip to [-M_loss, +M_loss]
    z_clipped = reports_z.clamp(-M_loss, M_loss)
    z_sorted, _ = torch.sort(z_clipped)

    # 2. High-probability upper bound on malicious evaluators in E_n
    b = min(int(math.ceil((rho_max + eta_m) * m)), (m // 2) - 1)
    keep = m - b

    # 3. Worst-case upper-tail / lower-tail trimming (preserves honest minority negative reports!)
    r_low = z_sorted[:keep].mean().item()
    r_high = z_sorted[b:].mean().item()

    # 4. Finite-sample concentration slack
    if use_sampling_slack:
        eff_honest = max(m * (1.0 - rho_max - eta_m), 1.0)
        delta_s = M_loss * math.sqrt(2.0 * math.log(2.0 / delta_samp) / eff_honest)
    else:
        delta_s = 0.0

    L_n = r_low - delta_s
    U_n = r_high + delta_s
    return L_n, U_n, r_low, r_high
```

#### 模块 3：PopArt 归一化与带安全回退门控的双区间控制器（`src/rl_controller/interval_sac_ddpg.py`）
```python
import torch
import torch.nn as nn

class PopArtQNetwork(nn.Module):
    def __init__(self, state_dim: int = 8, action_dim: int = 2, hidden_dim: int = 64, beta: float = 0.05):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.head = nn.Linear(hidden_dim, 1)
        self.register_buffer("mu", torch.tensor(0.0))
        self.register_buffer("sigma", torch.tensor(1.0))
        self.beta = beta

    def forward_normalized(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = self.trunk(torch.cat([state, action], dim=-1))
        return self.head(h).squeeze(-1)

    def forward_unnormalized(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.forward_normalized(state, action) * self.sigma + self.mu

    @torch.no_grad()
    def update_popart_stats(self, target_mean: float, target_std: float):
        old_mu, old_sigma = self.mu.clone(), self.sigma.clone()
        new_mu = (1.0 - self.beta) * old_mu + self.beta * target_mean
        new_sigma = torch.clamp((1.0 - self.beta) * old_sigma + self.beta * target_std, min=1e-4)
        # Preserve unnormalized outputs exactly: W_new = W_old * (old_sigma / new_sigma)
        scale = old_sigma / new_sigma
        self.head.weight.data.mul_(scale)
        self.head.bias.data.mul_(scale).add_((old_mu - new_mu) / new_sigma)
        self.mu.copy_(new_mu)
        self.sigma.copy_(new_sigma)

def select_action_with_fallback_gate(
    actor: nn.Module,
    q_low_net: PopArtQNetwork,
    q_high_net: PopArtQNetwork,
    state_n: torch.Tensor,       # [1, 8]
    a_cons: torch.Tensor,        # [1, 2], e.g. tensor([[0.0, 1.35]])
    xi_tol: float = 0.05,
):
    with torch.no_grad():
        a_cand = actor(state_n)  # outputs scaled to lam in [0, 0.5], tau in [1.05, 1.80]
        q_low_cand = q_low_net.forward_normalized(state_n, a_cand).item()
        q_high_cons = q_high_net.forward_normalized(state_n, a_cons).item()

        if q_low_cand >= q_high_cons - xi_tol:
            return a_cand.squeeze(0), False  # Fallback not triggered
        else:
            return a_cons.squeeze(0), True   # Fallback triggered -> safe conservative action
```

#### 模块 4：网络骨干与客户端自报指标注意事项
- **必须使用 `GroupNorm` 替代 `BatchNorm`**：在 `Dir(0.1)` 甚至长尾分布下，客户端本地 batch 的均值/方差极度偏移，且恶意客户端可通过污染 `BatchNorm` 的 `running_mean / running_var` 绕过梯度范数检测。因此 `ResNet-18` 骨干统一使用 `nn.GroupNorm(num_groups=8, num_channels=...)`。
- **诚实客户端自报指标 $u_k^{(t)}$ 的定义**：客户端 $k$ 在本地训练前后分别在本地留出集（或本地训练 batch 滑动平均）上计算有界相对损失下降率：
  $$u_k^{(t)} = \operatorname{sigmoid}\!\Big(\gamma_u \cdot \big(\ell_{\mathrm{eval}}(\mathbf{w}_t; \mathcal{B}_k) - \ell_{\mathrm{eval}}(\mathbf{w}_k^{(t)}; \mathcal{B}_k)\big)\Big) \in (0, 1),$$
  这几乎不增加额外计算开销，且能区分“已经充分收敛的高质量更新”与“高噪声振荡更新”。

---

## 4. 如何最终做实验：四阶段完整实验执行计划（Master Experimental Plan）

为确保论文逻辑环环相扣，实验严格按 **Phase 0（现象诊断） $\to$ Stage A（前向校准与动态控制必要性） $\to$ Stage B（抗谎报置信区间验证） $\to$ Phase 3（主实验全基线对比与开销分析）** 四个阶段推进。

### 4.1 Phase 0：Motivation 诊断实验（验证“鲁棒–异构两难困境”与 FedTC 解耦能力）
> 对应论文 **Section 1 (Introduction Fig. 1)** 与 **Section 2 (Motivation Fig. 2)**，详细操作规范已写入同目录下的 [diagnostic_exp_plan.md](file://@HOME@/1.MyPaper/ZT-FedTC2ICCV-3月/diagnostic_exp_plan.md)。

- **实验设置**：
  - 数据集：`CIFAR-100`，分别跑 **Setting I（标准均衡 `Dir(0.1)`）** 与 **Setting II（`CIFAR-100-LT (IF=20) + Dir(0.1)`，推荐作为主图，现象最强烈）**；
  - 客户端规模：总数 $K = 60$，每轮抽样 $K_{\mathrm{train}} = 20$（抽样率 $C = 1/3$）；
  - 恶意设置：$12$ 个恶意客户端（$20\%$），执行自适应边界投毒 **Min-Max**（或 **ALIE**）；
  - 诚实分组：将 $48$ 个诚实客户端按局部稀有度 $R_k = \sum_c p_{k,c}\big(\frac{1}{|\mathcal{H}|}\sum_{j\in\mathcal{H}} p_{j,c}\big)^{-1}$（或当轮孤立度 $1 - A_k^{(t)}$）切分为 **Top-25% 诚实少数派 $\mathcal{H}_{\mathrm{min}}$（12 个）**、**Bottom-25% 诚实主流派 $\mathcal{H}_{\mathrm{maj}}$（12 个）** 与中间组（24 个）。
- **对比算法**：`FedAvg`（无防御）、`Multi-Krum`、`Trimmed-Mean`、`FLShield`（或 `NNM+Krum`）、`Static Calibration (严阈值 / 宽阈值)`、以及 **`FedTC (Ours)`**。
- **核心产出图表（3 张子图构成论文 Figure 2）**：
  - **Fig. 2(a) 三类群体的累计聚合权重轨迹**：展示 `Multi-Krum / Trimmed-Mean` 中恶意权重虽接近 0，但**诚实少数派 $\mathcal{H}_{\mathrm{min}}$ 长期被赋予接近 0 的权重（Starvation）**；而 `FedTC` 同时将恶意权重压制到 0 附近并恢复 $\mathcal{H}_{\mathrm{min}}$ 的合理权重；
  - **Fig. 2(b) 客户端稀有度 $R_k$ vs. 累计入选率散点图 + 趋势线**：直观展示静态防御随 $R_k$ 升高而急剧跌落至 0 的负相关斜率；
  - **Fig. 2(c) 主流类准确率 vs. 少数派专属类准确率（Tail-20 Classes Accuracy）柱状图**：证明权重饿死直接导致尾部稀有类别准确率崩盘（差距可达 $15\%\sim 25\%$），而 `FedTC` 显著拉升尾部类别精度。

---

### 4.2 Stage A：理想反馈下验证“为什么需要动态 RL 控制 + 为什么需要 $u_k$”
> 对应论文 **Section 5.3 (Ablation Study Part 1)**。本阶段令评估子集 $\mathcal{E}_n$ 报告真实有界改进量 $z_{i,n}^{\mathrm{true}}$，纯粹检验前向层与动态序列决策的价值。

- **测试场景（非平稳动态攻击调度 Non-Stationary Attack Schedule）**：
  - 总轮数 $T = 300$（宏步 $H = 5$，共 $N = 60$ 个控制步）：
    - **Rounds 1–100**：无攻击（Clean Warmup，$\rho=0\%$），最优策略应当放宽 $\tau_n$ 并利用 $\lambda_n$ 加速异构收敛；
    - **Rounds 101–200**：突发 **ALIE (Little-is-Enough)** 隐蔽攻击（$\rho=20\%$）；
    - **Rounds 201–300**：切换为 **Min-Max** 自适应边界投毒攻击（$\rho=20\%$）。
- **对比控制器（5 组）**：
  1. `Fixed-Strict`：固定 $(\lambda=0.0, \tau=1.75)$；
  2. `Fixed-Permissive`：固定 $(\lambda=0.3, \tau=1.10)$；
  3. `Oracle Best Static`：网格搜索事后全剧最优的单一固定 $(\lambda^\star, \tau^\star)$；
  4. `Heuristic / Contextual Bandit ($\gamma=0$)`：只看单步即时收益或基于距离方差的启发式 PID 规则；
  5. `FedTC Dynamic RL ($\gamma=0.9$)`：完整长视距控制器。
- **同时进行的 $u_k$ 消融**：
  - 对比 `Distance-Only ($\lambda_n \equiv 0$)` vs. `Joint Calibration ($\lambda_n \in [0, 0.5]$)`，并在第 150 轮加入“恶意客户端伪造 $u_k = 1.0$”攻击，观测状态特征 $s_{n,4} = \operatorname{corr}_{\mathrm{rank}}(\tilde{u}_k, -\tilde{\delta}_k)$ 下降后控制器是否自动将 $\lambda_n$ 压回 $0$ 附近。

---

### 4.3 Stage B：恶意谎报反馈下验证“双层置信区间 $[L_n, U_n]$ 与宏步 $H$”
> 对应论文 **Section 5.3 (Ablation Study Part 2)**，直接回应审稿人关于“为什么不用中位数剔除评估者”与“置信区间是否过宽”的潜在质疑。

- **三种复合谎报攻击模式（前向模型投毒 + 反向评估谎报同时开启）**：
  1. **All-Max Inflation**：所有恶意评估者恒定报告 $+M_{\mathrm{loss}}$，试图欺骗控制器认为当前（哪怕已被攻破的）动作极好；
  2. **Minority-Silencing Echo（迎合主流中位数攻击）**：当过严的 $\tau_n$ 伤害了 20% 诚实少数派（其 $z_{i,n}^{\mathrm{true}} < 0$）而 60% 诚实主流略有正收益（$z_{i,n}^{\mathrm{true}} > 0$）时，20% 恶意评估者故意报告与诚实主流一致的正值，试图淹没少数派的负向告警；
  3. **Bimodal Jamming（双极干扰）**：一半恶意评估者报 $+M_{\mathrm{loss}}$，一半报 $-M_{\mathrm{loss}}$。
- **对比后端奖励估计器（4 组）**：
  1. `Raw Mean`：直接对 $\mathcal{E}_n$ 求均值；
  2. `Median / Huber Residual Filtering`：按与中位数残差剔除偏离评估者（展示其在 **Minority-Silencing Echo** 下误杀诚实少数派告警、导致 $\mathcal{H}_{\mathrm{min}}$ 准确率暴跌的实证反例！）；
  3. `Old FedTC Cosmetic Reward`：原稿的自循环几何奖励 $\cos(\Delta\mathbf{w}_g^{(t)}, \mathbf{m}_{t-1}) - \beta\,\mathrm{disp}_t$（展示其导致存活客户端数坍缩）；
  4. `FedTC CI Controller ($[L_n, U_n]$ + Fallback Gate)`（本文方法）。
- **宏步长度 $H \in \{1, 5, 10\}$ 敏感性实验**：
  - 统计并汇报四个核心指标：**① 真实收益覆盖率 $\mathbb{P}(\bar{Z}_n^\star \in [L_n, U_n])$**、**② 区间相对信噪比 $|M_n| / (U_n - L_n)$**、**③ 执行层门控回退触发率（Fallback Rate %）**、**④ 最终全局与少数派准确率**。验证 $H=5$ 在控制响应速度与信噪比之间达到最佳平衡。

---

### 4.4 Phase 3：主实验（Full Benchmark）与系统开销对比
> 对应论文 **Section 5.2 (Main Results Table 1 & Table 2)** 与 **Section 5.4 (System Overhead Table 3)**。

- **数据集与模型矩阵（3 个标准视觉基准）**：
  1. `CIFAR-10`（$\mathrm{Dir}(0.1)$ 与 $\mathrm{Dir}(0.05)$，GroupNorm `ResNet-18`，$K=60, K_{\mathrm{train}}=20$）；
  2. `CIFAR-100` 及 `CIFAR-100-LT (IF=20)`（$\mathrm{Dir}(0.1)$，GroupNorm `ResNet-18`，$K=60, K_{\mathrm{train}}=20$）；
  3. `Tiny-ImageNet`（或 `FEMNIST` 自然非独立同分布，验证大规模或自然异构场景）。
- **攻击类型（6 种）**：
  - 局域/隐蔽模型投毒：`LIE (ALIE)`, `Min-Max`, `Min-Sum`, `IPM (Inner Product Manipulation)`；
  - 数据级/目标级投毒：`Label Flipping`, `Model Replacement Backdoor`（额外汇报 ASR 攻击成功率）；
  - 动态复合攻击：`Dynamic Switching + Report Poisoning`。
- **对比基线（两大类别，分组清晰）**：
  - **主对比组（Zero Clean Server Data，公平同赛道）**：
    `FedAvg`, `RFA (Geometric Median)`, `Trimmed-Mean`, `Multi-Krum`, `Bucketing + Krum/TM` (Karimireddy et al., ICLR'22), `NNM (Nearest Neighbor Mixing)` (Allouah et al., ICML'23), `FedLAW` / `FLShield`；
  - **参考对比组（Requiring Clean Server Root Dataset，在表格中用 `†` 标注）**：
    `FLTrust` (Cao et al., NDSS'21), `FedAA` (He et al., AAAI'25)。证明 `FedTC` 在**无需任何服务端干净数据**的前提下，逼近甚至超越依赖干净根数据集的方法。
- **双核心评价指标**：
  - 每组实验同时汇报 **Global Test Accuracy (%)** 与 **Minority / Tail-20% Class Accuracy (%)**（这是紧扣全文 Story 的点睛之笔，审稿人一眼就能看出为何传统基线总体精度尚可但少数派精度惨不忍睹，而 FedTC 双指标全面领先）。

---

## 5. 论文图表与章节排布总表（6 Figures + 3 Tables Checklist）

| 编号 | 所在章节 | 图表内容概要 | 对应实验阶段 |
| :--- | :--- | :--- | :--- |
| **Fig. 1** | Sec. 1 Intro | **概念对比图 + 核心痛点速览**：上半部分展示静态防御“要么放进自适应投毒、要么饿死诚实少数派”；下半部分展示 FedTC 前向解析校准 + 反向置信区间闭环控制。 | 架构设计 |
| **Fig. 2** | Sec. 2 Motivation | **Phase 0 真实诊断三联图**：(a) 恶意/诚实主流/诚实少数派三组聚合权重随轮次演化；(b) 客户端稀有度 $R_k$ vs. 累计入选率；(c) 总体精度 vs. 尾部少数派类别精度。 | Phase 0 |
| **Fig. 3** | Sec. 4 Method | **FedTC 系统框架图**：左侧 Forward Calibration Layer（几何中位数无量纲偏离 $\tilde{\delta}_k$ + 秩归一化 $\tilde{u}_k$ + 精确截断半径 $R_t(\tau_t)$）；右侧 Backward CI Controller（解耦宏步有界损失 $\to [L_n, U_n] \to$ 双 Critic + 保守策略执行门控）。 | Method v2.1 |
| **Table 1** | Sec. 5.2 Main | **主实验大表 I（CIFAR-10 & CIFAR-100, Dir(0.1), 6 种攻击）**：汇报所有基线与 FedTC 的 `Global Acc / Minority Acc`。 | Phase 3 |
| **Table 2** | Sec. 5.2 Main | **主实验大表 II（长尾 CIFAR-100-LT & 动态切换复合攻击 & 不同恶意比例 $\rho \in \{10\%, 20\%, 30\%\}$）**。 | Phase 3 |
| **Fig. 4** | Sec. 5.3 Ablation | **Stage A 动态控制轨迹图**：在 `Clean -> ALIE -> Min-Max` 非平稳切换下，展示 $(\lambda_n, \tau_n)$、截断半径 $R_t(\tau_n)$、以及 `FedTC` 对比 `Oracle Static / Bandit` 的测试精度曲线。 | Stage A |
| **Fig. 5** | Sec. 5.3 Ablation | **Stage B 置信区间覆盖与抗谎报验证图**：(a) 真实诚实收益 $\bar{Z}_n^\star$ 被阴影带 $[L_n, U_n]$ 严密包裹的轨迹；(b) `Minority-Silencing Echo` 攻击下，中位数残差过滤 vs. 置信区间下界 $L_n$ 对少数派精度的保护对比。 | Stage B |
| **Table 3** | Sec. 5.4 Analysis | **宏步周期 $H \in \{1,5,10\}$ 消融与系统开销对比表**：对比每轮服务端耗时（ms）、额外验证通信量（仅每 $H$ 轮上传 1 个 float 标量，通信开销 $< 0.001\%$）、区间覆盖率与回退触发率。 | Stage B & Phase 3 |

---

## 6. 建议的下一步执行顺序（Actionable Roadmap）

1. **第一步（1–2 天，最高优先级）**：先搭建 `code_fedtc/` 基础框架，实现 `dirichlet_split.py`、`resnet_gn.py`、`fedtc_forward.py` 和 `run_phase0_diagnostic.py`，跑出 **Phase 0 诊断实验（Fig. 2）**，同时在真实数据流上跑通 `FedTC` 前向层。
2. **第二步（2–3 天）**：接入 `ci_estimator.py` 与 `interval_sac_ddpg.py`，在 `CIFAR-100 (K=60, Dir(0.1), H=5)` 上跑通 **Stage A 与 Stage B**，固定控制器超参数（$\sigma_0=4.0, \xi_{\mathrm{tol}}=0.05, H=5$）。
3. **第三步（并行开展）**：启动 **Phase 3 主实验批量脚本**，同时将本文档第 1、2 节的定稿数学公式与定理同步写入 LaTeX 源码 [`_Part_FedTC/content/0_abstract.tex`](file://@HOME@/1.MyPaper/ZT-FedTC2ICCV-3%E6%9C%88/_Part_FedTC/content/0_abstract.tex)、[`1_introduction.tex`](file://@HOME@/1.MyPaper/ZT-FedTC2ICCV-3%E6%9C%88/_Part_FedTC/content/1_introduction.tex)、[`4_methodology.tex`](file://@HOME@/1.MyPaper/ZT-FedTC2ICCV-3%E6%9C%88/_Part_FedTC/content/4_methodology.tex) 与 [`7_appendix.tex`](file://@HOME@/1.MyPaper/ZT-FedTC2ICCV-3%E6%9C%88/_Part_FedTC/content/7_appendix.tex)（顺带清理原稿中的重复 label 和未定义引用）。
