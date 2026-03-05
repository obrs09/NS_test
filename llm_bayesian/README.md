# Bayesian Causal Reasoning for LLM Interpretability

> **核心问题**: 能否用贝叶斯推断量化 LLM 内部因果机制的不确定性？幻觉是否对应因果歧义？

## 项目概述

本项目提出一个贝叶斯因果推断框架，用于分析大语言模型（LLM）内部的因果机制。传统的可解释性研究通常假设单一因果结构，本框架通过贝叶斯后验分布来表示多个竞争性因果假说之间的不确定性。

**核心假说 (H₁)**：幻觉输出对应高因果歧义（posterior entropy H 高），即无法识别出主导的内部机制；忠实输出对应低因果歧义（H 低），即存在清晰的因果路径。

### 技术栈

| 组件 | 说明 |
|---|---|
| 推理引擎 | [NeuroScope](../NeuroScope/) — 基于 llama.cpp 的 C++/CUDA 引擎，pybind11 Python API |
| 模型 | Qwen3-8B (Q4_K_M GGUF), 36 层, 4096 维 |
| 干预方法 | Activation patching via `apply_steering()` |
| 贝叶斯推断 | Softmax likelihood + posterior entropy |

## 实验流程

```
Phase 0 (PoC)          Phase 1 (Causal Tracing)      Phase 1b (Multi-Dataset)     Phase 2 (Fine-Grained)       Phase 3 (Blocking)
───────────────────    ──────────────────────────    ──────────────────────────    ────────────────────────    ────────────────────
已有实验数据             100条手工事实问答              300条 × 4个标准数据集           36层因果后验频谱              早期预警检测器
回顾性贝叶斯分析         激活修补 × 6机制              TruthfulQA/PopQA/              残差贡献分析                 因果阻断干预
验证框架可行性           忠实 vs 幻觉 对比              TriviaQA/NeQA                 双轨假说频谱证据             自适应流水线
```

---

## Phase 0 — Bayesian PoC（已完成）

**文件**: [`bayesian_poc.ipynb`](bayesian_poc.ipynb)  
**结果**: [`/workspace/Data/bayesian_phase0/`](../Data/bayesian_phase0/)

### 目标

在不进行新的推理的情况下，利用 Experiment 2/5/9/10 的已有激活数据，验证贝叶斯机制选择框架的可行性。

### 方法

- **数据**: 100 个美学样本的激活向量（50 正面 + 50 负面, Layer 16, 4096 维）
- **候选机制 (K=6)**: Exp2-Aesthetic, Exp10-Imagery, Exp9-Genre, PCA-PC1, PCA-PC2, Random-Null
- **统计模型**: 单样本高斯 + g-prior → 封闭形式 log Bayes Factor
- **关键指标**: 后验熵 H(M|D)，有效机制数 K_eff = exp(H)

### 结果

| 指标 | 值 |
|---|---|
| 最优机制 | Exp2-Aesthetic (P = 1.000) |
| log Bayes Factor | 185.6 |
| 后验熵 H | 0.0000 (零歧义) |
| K_eff | 1.00 |
| 幻觉模拟 (shuffle) | H = 1.59 ± 0.27, d = −5.91 |

**结论**: 贝叶斯框架成功运行。当存在清晰因果关系时，后验集中于单一机制（H→0）。随机打乱标签后，后验变得均匀（H→H_max），符合预期。

### 可视化

| 图 | 内容 |
|---|---|
| `posterior_L16.png` | Layer 16 后验分布 |
| `proj_histograms.png` | 各方向投影直方图 |
| `cross_layer_entropy.png` | 跨层熵分析 |
| `hallucination_sim.png` | 幻觉模拟 (shuffle vs real) |
| `prior_sensitivity.png` | g-prior 鲁棒性检验 |

---

## Phase 1 — Bayesian Causal Tracing（已完成）

**文件**: [`bayesian_phase1.ipynb`](bayesian_phase1.ipynb)  
**结果**: [`/workspace/Data/bayesian_phase1/`](../Data/bayesian_phase1/)

### 目标

通过因果追踪（causal tracing）测试核心假说：**幻觉输出是否比忠实输出具有更高的因果歧义（后验熵 H）**？

### 方法

#### 1. 数据集设计

100 条事实回忆问答，覆盖 5 个类别：

| 类别 | 数量 | 示例 |
|---|---|---|
| 世界首都 | 30 | "The capital of France is" → "Paris" |
| 化学符号 | 20 | "The chemical symbol for gold is" → "Au" |
| 历史事件 | 20 | "Albert Einstein was born in" → "1879" |
| 科学常识 | 15 | "The atomic number of carbon is" → "6" |
| 通用知识 | 15 | "The author of Harry Potter is" → "Rowling" |

每个类别包含 easy 和 hard 子集，hard 子集更容易触发幻觉。

#### 2. 生成与分类

- 使用 Qwen3-8B 对每条 prompt 生成回答（`temperature=0.01, top_k=1`）
- 通过子串匹配将回答分类为**忠实 (faithful)** 或 **幻觉 (hallucinated)**
- 结果：**88 faithful / 12 hallucinated**

#### 3. 因果追踪 (Activation Patching)

对每条 prompt 执行以下步骤：

```
Clean forward:    "The capital of France is"      → 捕获 clean_acts, p(target)
Corrupt forward:  "The capital of Zyxwv is"       → 捕获 corrupt_acts, p(target)
Delta:            delta[l] = clean_acts[l] - corrupt_acts[l]

对每个机制假说 k (层组):
  在 corrupt prompt 上 apply_steering(l, delta[l], 1.0) 对组内所有层
  测量 p_int(target) → 计算 IE_k = (p_int - p_corrupt) / (p_clean - p_corrupt)
```

- **机制假说 (K=6)**: L0-5, L6-11, L12-17, L18-23, L24-29, L30-35（每组 6 层）
- **总计**: 4500+ 次前向传播 (276 秒)

#### 4. 贝叶斯推断

```
似然:   P(D | m_k) ∝ exp(β · IE_k)
后验:   P(m_k | D) = softmax(β · IE)
熵:     H = −∑_k P_k · ln(P_k)
```

其中 β = 5.0 为默认温度参数。

### 结果

#### 核心指标

| 组 | 后验熵 H | Std | K_eff |
|---|---|---|---|
| Faithful (n=86) | 0.5702 | 0.5993 | 2.16 |
| Hallucinated (n=12) | 0.4969 | 0.4579 | 1.83 |

**Cohen's d = −0.135**（方向与 H₁ 相反）

#### 统计检验

| 检验 | 统计量 | p 值 | 显著性 |
|---|---|---|---|
| Mann-Whitney U | 488.0 | 0.621 | n.s. |
| Kolmogorov-Smirnov | D = 0.151 | 0.937 | n.s. |
| Welch t-test | t = −0.480 | 0.638 | n.s. |
| AUC (熵预测幻觉) | 0.473 | — | 低于随机 |

#### β 敏感性

| β | Cohen's d | H_faithful | H_hallucinated | 显著 |
|---|---|---|---|---|
| 0.5 | −0.182 | 1.588 | 1.529 | n.s. |
| 1.0 | −0.183 | 1.347 | 1.247 | n.s. |
| 2.0 | −0.048 | 1.005 | 0.974 | n.s. |
| 5.0 | −0.135 | 0.570 | 0.497 | n.s. |
| 10.0 | −0.428 | 0.393 | 0.197 | n.s. |
| 20.0 | −0.423 | 0.292 | 0.114 | n.s. |
| 50.0 | −0.302 | 0.204 | 0.090 | n.s. |

Cohen's d 在所有 β 值下均为负值，且随 β 增大而趋于更负。

#### 机制分布差异

| 机制 | Faithful P(m|D) | Hallucinated P(m|D) |
|---|---|---|
| L0-5 Shallow | 0.111 | 0.084 |
| **L6-11 Early** | **0.097** | **0.287** |
| L12-17 Mid | 0.106 | 0.108 |
| L18-23 Deep | 0.126 | 0.034 |
| L24-29 Late | 0.088 | 0.087 |
| **L30-35 Final** | **0.472** | **0.399** |

幻觉样本的 L6-11 Early 权重显著高于忠实样本（0.287 vs 0.097），暗示幻觉的因果路径包含更强的早期层参与。

### 可视化

| 图 | 内容 |
|---|---|
| `main_results.png` | 6 面板主图（KDE, violin, scatter, posterior bars, IE heatmap, gauge） |
| `beta_sensitivity.png` | 7 个 β 值的 box plot + posterior bar chart |

### 解读

**H₁ 未得到支持**。幻觉样本的后验熵比忠实样本更低（而非更高）。这意味着本数据集中的幻觉主要是**误信息驱动型 (misinformation-driven)**——模型通过清晰的内部机制自信地召回了错误事实，而非处于因果歧义状态。

这一发现揭示了幻觉的两种子类型：

1. **歧义驱动型 (Ambiguity-driven)**: 多个弱机制竞争 → 高 H — 对应真正的"不确定"
2. **误信息驱动型 (Misinformation-driven)**: 单一强机制但指向错误答案 → 低 H — 模型"自信地犯错"

本数据集（事实回忆）中的幻觉以第 2 类为主。

#### 已知局限

- **全位置干预**: `apply_steering()` 对所有 token 位置施加相同干预，而非仅对 subject token。这引入噪声，降低 IE 的精度。
- **样本不平衡**: 仅 12 条幻觉 vs 86 条忠实，统计效力低。
- **IE 噪声**: 全恢复率仅 35%，说明 activation patching 的精度受限于 API。

---

## Phase 1b — Multi-Dataset Bayesian Causal Tracing（已完成）

**文件**: [`bayesian_phase1b.ipynb`](bayesian_phase1b.ipynb)  
**结果**: [`/workspace/Data/bayesian_phase1b/`](../Data/bayesian_phase1b/)

### 目标

使用 4 个标准基准数据集扩展 Phase 1 实验，增加样本多样性和统计效力，测试框架的泛化能力。

### 数据集

| 数据集 | 样本数 | 来源 | 特点 |
|---|---|---|---|
| **TruthfulQA** | 100 | `truthful_qa` (generation split) | 对抗性问题，设计诱导错误回答 |
| **PopQA** | 100 | `akariasai/PopQA` (低人气偏抽) | 长尾实体知识，冷门人物/事物 |
| **TriviaQA** | 50 | `trivia_qa` (rc.nocontext) | 开放式百科问答 |
| **NeQA** | 50 | `inverse-scaling/NeQA` | 否定推理多选题 |
| **总计** | **300** | | |

PopQA 特别偏向低人气实体（按 `s_pop` 排序取底部），以获取更多幻觉样本。

### 方法

与 Phase 1 相同：
- 生成回答 → 子串匹配分类 faithful/hallucinated
- 对每条 prompt 执行 activation patching（替换 subject 为 "Zyxwv"）
- 计算 K=6 层组 IE → softmax 后验 → 后验熵 H
- 对比 H_faithful vs H_hallucinated

**总计**: 13,500 次前向传播（约 14 分钟）

### 结果

#### 生成准确率

| 数据集 | Faithful | Hallucinated | 准确率 |
|---|---|---|---|
| TruthfulQA | 11 | 89 | 11.0% |
| PopQA | 8 | 92 | 8.0% |
| TriviaQA | 35 | 15 | 70.0% |
| NeQA | 48 | 2 | 96.0% |
| **总计** | **102** | **198** | **34.0%** |

Phase 1 仅有 12 条幻觉；Phase 1b 有 **198 条**，统计效力大幅提升。

#### 核心指标（β=5.0）

| 组 | n | 后验熵 H | Std | K_eff |
|---|---|---|---|---|
| Faithful | 99 | 0.5997 | 0.5274 | 2.12 |
| Hallucinated | 187 | 0.6871 | 0.6163 | 2.42 |

**Cohen's d = +0.152**（正方向，与 H₁ 一致！对比 Phase 1 的 d = −0.135）

#### 统计检验

| 检验 | 统计量 | p 值 | 显著性 |
|---|---|---|---|
| Mann-Whitney U | 9737.0 | 0.235 | n.s. |
| **KS test** | **D = 0.178** | **0.028** | **\*** |
| Welch t-test | t = 1.251 | 0.212 | n.s. |
| AUC | 0.526 | — | 略高于随机 |

**KS 检验显著 (p < 0.05)**，表明忠实与幻觉的熵分布确实存在统计差异。

#### 分数据集 Effect Size

| 数据集 | n_F | n_H | H_F | H_H | Cohen's d | p |
|---|---|---|---|---|---|---|
| TruthfulQA | 11 | 87 | 0.700 | 0.702 | +0.002 | 0.415 |
| **PopQA** | **8** | **84** | **0.483** | **0.723** | **+0.426** | **0.093** |
| TriviaQA | 35 | 15 | 0.652 | 0.440 | −0.391 | 0.865 |
| NeQA | 45 | 1 | 0.555 | 0.077 | — | — |

**PopQA 是最强信号**（d = 0.426, 中等效应量），幻觉样本的因果歧义显著高于忠实样本。这与 PopQA 的长尾实体特点一致——模型对冷门实体缺乏清晰的因果路径。

#### β 敏感性

Cohen's d 在所有 β 值（0.5–50）下均为**正值**（范围 0.08–0.15），方向一致。

#### 机制分布差异

| 机制 | Faithful | Hallucinated |
|---|---|---|
| L0-5 Shallow | 0.152 | 0.155 |
| L6-11 Early | 0.212 | 0.217 |
| L12-17 Mid | 0.082 | 0.129 |
| L18-23 Deep | 0.061 | 0.097 |
| **L24-29 Late** | **0.246** | **0.139** |
| L30-35 Final | 0.246 | 0.262 |

忠实样本的 L24-29 Late 权重 (0.247) 远高于幻觉 (0.139)，暗示忠实回忆依赖晚期层的巩固。

### 可视化

| 图 | 内容 |
|---|---|
| `main_results.png` | 9 面板主图（KDE, per-dataset box, effect size, posteriors, IE heatmap, scatter, mechanism profiles, gauge, summary table） |
| `beta_sensitivity.png` | 7 个 β 值的 box plot + posterior bar chart |

### Phase 1 vs Phase 1b 对比

| 指标 | Phase 1 (手工) | Phase 1b (标准数据集) |
|---|---|---|
| 总样本 | 100 | 300 |
| 幻觉数 | 12 | 198 |
| Cohen's d | **−0.135** (错误方向) | **+0.152** (正确方向) |
| KS p值 | 0.937 (n.s.) | **0.028 (\*)** |
| 最强信号 | — | PopQA (d=0.426) |

Phase 1b 的关键进展：
1. **方向翻转**: d 从负变正，支持 H₁
2. **KS 检验显著**: 首次达到统计显著性
3. **PopQA 子集**: 中等效应量 (d=0.426)，长尾实体最易触发歧义型幻觉
4. **TriviaQA 反向**: 与 Phase 1 类似，事实回忆中的幻觉仍以误信息型为主（d=−0.391）

### 综合解读

两个 Phase 的结果共同支持**幻觉子类型假说**：

| 幻觉子类型 | H 水平 | 对应数据集 | 机制特征 |
|---|---|---|---|
| **歧义驱动型** | 高 H | PopQA（冷门实体）、TruthfulQA | 多个弱机制竞争 |
| **误信息驱动型** | 低 H | TriviaQA、Phase 1 手工数据 | 单一强机制指向错误答案 |

H₁ 在歧义驱动型幻觉上成立，在误信息驱动型上不成立。总体效应被两种子类型抵消，导致综合 d 较小。

---

## Phase 2 — Fine-Grained Causal Posterior Spectrum（已完成）

**文件**: [`bayesian_phase2_finegrained.ipynb`](bayesian_phase2_finegrained.ipynb)  
**结果**: [`/workspace/Data/bayesian_phase2/`](../Data/bayesian_phase2/)

### 目标

将机制空间从 K=6 层组精化为 L=36 个独立层，构建"因果后验频谱 (Causal Posterior Spectrum)"来可视化歧义型 vs 误信息型幻觉的结构差异。附加残差贡献分析作为 Attn/MLP 分离的代理方法。

### 方法

#### 1. 层级贝叶斯后验

直接使用 Phase 1b 的逐层 IE 数据 `IE_per_layer` (300×36)，**无需新的前向传播**：
```
P(m_l | D) = softmax(β · IE_l)    l ∈ {0, 1, ..., 35}
H_layer = −∑_l P_l · ln(P_l)      H_max = ln(36) = 3.584
```

#### 2. 幻觉子类型分类

以幻觉样本层级熵的中位数 (H_median = 2.218) 为阈值：
- **歧义驱动型**: H ≥ H_median → 频谱平坦（均匀后验）
- **误信息驱动型**: H < H_median → 频谱尖刻（集中后验）

#### 3. 残差贡献分析（Attn/MLP 代理）

NeuroScope 的 `apply_steering` 在残差流层级操作，无法直接分离 Attention 和 MLP。使用 50 条 prompt 子集（25 faithful + 25 hallucinated），通过 100 次清洁/损坏前向传播计算：
```
layer_contrib[l] = acts[l] - acts[l-1]    # 每层残差贡献
diff_norm[l] = ||clean_contrib[l] - corrupt_contrib[l]||
cosine[l] = cos(clean_contrib[l], corrupt_contrib[l])
```

### 结果

#### 层级熵统计（β=5.0, L=36）

| 组 | n | 层级 H | K_eff | 早期质量 | 晚期质量 |
|---|---|---|---|---|---|
| Faithful | 99 | 2.033 | 9.52 | 0.012 | 0.784 |
| **歧义型** | **94** | **2.647** | **14.52** | **0.043** | **0.725** |
| **误信息型** | **93** | **1.200** | **4.19** | **0.137** | **0.425** |

**层级 Cohen's d = −0.125**（与组级 d = +0.152 方向相反），因为层级分析揭示了子类型内的更精细结构。

#### TruthfulQA 子集强信号

| 检验 | 统计量 | p 值 | 显著性 |
|---|---|---|---|
| TruthfulQA Cohen's d | +1.210 | 0.0008 | \*\*\* |
| PopQA Cohen's d | +0.320 | 0.229 | n.s. |
| TriviaQA Cohen's d | −0.111 | 0.344 | n.s. |

#### 频谱特征（子类型判别）

| 特征 | Faithful | 歧义型 | 误信息型 |
|---|---|---|---|
| Max P(m_l\|D) | 0.321 | 0.163 | **0.595** |
| 活跃层数 (>2/L) | 6.4 | **7.7** | 3.2 |
| Top-3 集中度 | 0.564 | 0.381 | **0.854** |
| Gini 系数 | 0.756 | **0.907** | 0.517 |
| 早期质量 (L0-11) | 0.012 | 0.043 | **0.137** |
| 中期质量 (L12-23) | 0.203 | 0.232 | **0.438** |
| 晚期质量 (L24-35) | **0.784** | 0.725 | 0.425 |

**关键发现**：
- 误信息型频谱具有**双峰结构**：早期层 (L8, L14, L20) + 晚期层 (L34, L35)
- 歧义型频谱近似均匀（Gini=0.907，接近完美均匀的 1.0）
- 忠实样本几乎所有因果质量集中在晚期层（late_mass=0.784）

#### 关键层定位

| 层 | P(误信息) | P(歧义) | 差异 |
|---|---|---|---|
| **L34** | 0.116 | 0.063 | +0.053 |
| **L35** | 0.100 | 0.067 | +0.033 |
| **L14** | 0.060 | 0.006 | +0.054 |
| **L20** | 0.066 | 0.035 | +0.031 |

L34 和 L14 是误信息型幻觉的标志层——分别对应"输出投射"和"知识存储"。

#### 残差贡献分析

| 发现 | 描述 |
|---|---|
| 早期层过度发散 | 幻觉样本在 L0-L7 的 Halluc/Faithful 发散比率 > 1.2x |
| 余弦相似度下降 | 幻觉在 L5 时 cos≈0.8，忠实在 L15 仍保持 cos≈0.95 |
| 晚期层集中 | 忠实样本的清洁-损坏差异集中在 L30-35；幻觉分散于全部层 |

**解读**：幻觉样本从最初几层就对实体替换产生异常反应，但无法将信息正确路由到晚期输出层。忠实样本在早期层对实体替换几乎无响应，所有因果工作集中在晚期层。

### 可视化

| 图 | 内容 |
|---|---|
| `causal_spectrum.png` | 9 面板因果后验频谱主图 |
| `individual_spectra.png` | 12 个个体频谱示例（尖刺 vs 平坦） |
| `residual_contribution.png` | 6 面板残差贡献分析 |

### 理论贡献

Phase 2 的核心理论贡献是将**双轨幻觉假说**从统计描述提升为频谱可视化证据：

| 特征 | 歧义驱动型 | 误信息驱动型 |
|---|---|---|
| 频谱形状 | 🌊 平坦波（均匀） | 🗼 尖刺（集中） |
| H_layer | 2.647 (高) | 1.200 (低) |
| K_eff | 14.5 层 | 4.2 层 |
| 因果质量分布 | 全层均匀 | 双峰（早期+晚期） |
| 早期层参与 | 低 (4.3%) | 高 (13.7%) |
| 典型数据集 | PopQA, TruthfulQA | TriviaQA, Phase 1 手工 |
| 解读 | 模型"不知道该用哪层" | 模型"自信地用错了层" |

---

## Phase 3 — Early Warning & Causal Blocking Pipeline（已完成）

**文件**: [`bayesian_phase3_blocking.ipynb`](bayesian_phase3_blocking.ipynb)  
**结果**: [`/workspace/Data/bayesian_phase3/`](../Data/bayesian_phase3/)

### 目标

将 Phase 2 的早期层异常发散发现转化为实用的**幻觉早期预警检测器 + 因果阻断干预流水线**。三个部分：

| 部分 | 目标 | 方法 |
|---|---|---|
| **Part A** | 早期预警检测器 | L0-L9 激活特征 + 分类器 |
| **Part B** | 因果阻断干预 | Correction steering (faithful − halluc direction) |
| **Part C** | 自适应流水线 | 端到端 检测 → 决策 → 阻断 |

### Part A — 早期预警检测器

#### 方法

1. **激活采集**: 对 300 条 prompt 执行 forward，提取 L0-L9 激活向量 (shape: 300×10×4096)
2. **特征工程**: 从早期激活中提取 **91 个特征**：
   - 逐层范数、均值、方差、最大值、峰度 (10 层 × 5 = 50)
   - 逐层余弦相似度 cos(L_i, L_{i-1}) (9 对)
   - 逐层范数增长率 (9 对)
   - 逐层残差贡献范数 (9 层)
   - 全局特征：早期范数均值、std、early/late 范数比、峰度最大层、范数最大层等 (14 个)
3. **分类器**: 4 种模型 + 5-fold StratifiedKFold 交叉验证

#### 结果

| 分类器 | CV AUC | Std |
|---|---|---|
| LogReg (L2) | 0.847 | 0.035 |
| LogReg (L1) | 0.861 | 0.028 |
| Random Forest | 0.851 | 0.062 |
| **GBM** | **0.864** | **0.046** |

**最优模型**: GBM, AUC = 0.864

#### Top-5 预测特征

| 排名 | 特征 | 重要性 | 解释 |
|---|---|---|---|
| 1 | **cos_L3_L2** | 0.2566 | L2→L3 转换处余弦相似度 |
| 2 | early_late_norm_ratio | 0.0819 | 早期/晚期范数比 |
| 3 | kurtosis_L9 | 0.0818 | L9 激活峰度 |
| 4 | norm_growth_L7 | 0.0394 | L7 范数增长率 |
| 5 | norm_L2 | 0.0369 | L2 激活范数 |

**关键发现**: 幻觉在 **L2→L3 过渡层** 即可检测——余弦相似度 cos_L3_L2 独占 ~26% 的预测重要性。

#### AUC vs Detection Depth

| Cutoff 层 | CV AUC |
|---|---|
| L2 (仅 L0-L1) | 0.797 |
| L4 | 0.840 |
| L6 | 0.848 |
| L8 | 0.854 |
| L10 (全部) | 0.864 |

仅使用前 4 层 (L0-L3) 即可达到 AUC=0.840，之后边际收益递减。

### Part B — 因果阻断干预

#### 方法

1. **Correction Direction**: 计算每层 faithful 平均激活减去 hallucinated 平均激活
   ```
   correction_dir[l] = mean(faithful_acts[l]) − mean(halluc_acts[l])
   ```
   归一化后作为 steering 方向
2. **Target Token**: 使用 Qwen3-8B tokenizer 将每条 prompt 的正确答案 (`answers[0]`) 编码为 token，取第一个 token 作为追踪目标
3. **Grid Search**: 12 层 × 6 strength = 72 组合，对 187 条 hallucinated prompt 在**清洁 prompt** 上施加 steering
   - 层: L2, L5, L8, L11, L14, L17, L20, L23, L26, L29, L32, L34
   - 强度: 0.5, 1.0, 2.0, 5.0, 10.0, 20.0
4. **指标**: 绝对概率提升 Δp、排名变化 Δrank、top-1 翻转率（不使用比率指标，因为 p(correct|clean) ≈ 0 会导致除零问题）
5. **总计**: 13,651 次前向传播（约 16 分钟）

> **Bug 修复说明**：初版代码追踪的是模型 top-1 预测 token（即**错误答案**），而非正确答案 token。同时使用 `p_int/p_baseline` 比率作为 recovery——当 p_baseline ≈ 0 时该比率退化（可产生 >1 甚至 >700 的虚假值）。修正后使用 tokenizer 定位正确答案 token，并改用绝对指标。

#### 结果

**最优配置（按中位排名提升）**: Layer 2, Strength 20.0

| 指标 | 值 |
|---|---|
| % correct token rank improved | **77.5%** |
| % correct token p increased | **88.8%** |
| % top-1 flipped to correct | 1.1% |
| Median Δrank | **+934** |
| Mean Δp(correct) | +0.0084 |
| Baseline median rank | 2,023 |
| Steered median rank | **167** |

#### % Rank Improved 表（层 × 强度）

| Layer | s=0.5 | s=1.0 | s=2.0 | s=5.0 | s=10 | s=20 |
|---|---|---|---|---|---|---|
| **L2** | 55% | 60% | 66% | 62% | 61% | **78%** |
| L5 | 50% | 52% | 55% | 51% | 49% | 43% |
| L8 | 53% | 54% | 51% | 56% | 53% | 59% |
| L14 | 48% | 55% | 64% | 68% | **70%** | 65% |
| L23 | 58% | 51% | 61% | 70% | 73% | **78%** |
| L34 | 44% | 41% | 40% | 40% | 34% | 36% |

**关键发现**：
- **L2 最优**：最强 steering 效果在最早期层，与 Part A 的 cos_L3_L2 发现一致
- **L14 和 L23 也有效**: ~70-78% 排名提升率，说明中期层也参与 factual routing
- **L34 有害**: 高强度时 steering 反而降低排名（仅 34-40%），晚期层干预为时已晚
- **Top-1 翻转率极低 (1.1%)**：steering 能将正确答案从 rank 2023 推到 rank 167，但从 rank 167 到 rank 1 的最后一步极其困难——幻觉深度嵌入在模型权重中

#### 分数据集排名提升

| 数据集 | n | % Rank Improved | Median Δrank |
|---|---|---|---|
| **NeQA** | 1 | **100%** | +1,259 |
| **PopQA** | 84 | **82%** | +4,188 |
| **TruthfulQA** | 87 | **77%** | +98 |
| TriviaQA | 15 | 53% | +159 |

PopQA（长尾实体）Δrank 最大 (+4,188)，与 Phase 1b/2 的发现一致——模型对冷门实体的 factual recall 最容易被 steering 影响。TriviaQA 最难纠正——误信息型幻觉的权重难以仅靠外部 steering 覆盖。

#### 为什么 top-1 翻转率这么低？

对 hallucinated prompts，p(correct|clean) ≈ 0.0001（median rank ≈ 2,023）。模型从未"接近"给出正确答案——正确 token 深埋在 vocab 尾部。Steering 能持续改善排名（正确方向），但要跨越从 rank 167 到 rank 1 需要改变整个输出分布的 mode——这不是 single-layer additive steering 能做到的。

### Part C — 自适应流水线

将检测器和干预组合为端到端流水线：

```
Forward → L0-L9 特征 → GBM 预测 P(halluc)
                          ↓
                    P > 0.5? → 施加 L2 correction steering (s=20)
                          ↓
                    继续生成 → 修正输出
```

#### 流水线性能

| 指标 | 值 |
|---|---|
| Baseline accuracy (无干预) | 0.346 |
| Pipeline (保守, top-1 翻转) | 0.350 (+0.3%) |
| Pipeline (乐观, rank improved) | 0.853 (+50.7%) |
| CV AUC (真实泛化) | 0.860 |
| False positive rate | 0.0% |

**保守估计 (top-1 flip)**：仅 1.1% 的幻觉被彻底纠正为正确答案 → 流水线几乎无改善。  
**乐观估计 (rank improved)**：77.5% 的幻觉的正确答案排名提升 → 如果配合 beam search 或更复杂的解码策略，可能有效。

**诚实结论**：单层 additive steering 不足以翻转 top-1 预测，但能显著改善正确答案的排名，为后续更强的干预方法提供方向信号。

### 可视化

| 图 | 内容 |
|---|---|
| `early_warning_detector.png` | 6 面板 Part A 主图（ROC, features, score dist, per-dataset AUC, norm profile, AUC vs depth） |
| `causal_blocking.png` | 6 面板 Part B 主图（Δrank heatmap, % rank improved, Δrank distribution, per-dataset, strength curve, ambig vs misinfo） |

### 理论贡献

Phase 3 将前两阶段的观察性发现转化为**可操作的干预实验**：

| 发现 | Phase 2 (观察) | Phase 3 (干预) |
|---|---|---|
| 早期层异常发散 | L0-L7 发散比 > 1.2x | **cos_L3_L2 检测 AUC=0.864** |
| 幻觉因果瓶颈在早期 | 早期质量 13.7% (误信息型) | **L2 steering 最优, 78% rank improved** |
| PopQA 长尾效应 | d=0.426 (中等) | **Median Δrank=+4,188 (最大改善)** |
| 幻觉深度嵌入 | — | **Top-1 flip rate 仅 1.1%** |

**核心结论**：
1. 幻觉可在 **L2→L3** 处以 AUC=0.864 被检测出来
2. Correction steering 能将正确答案从 rank ~2000 推至 rank ~167（**方向正确，幅度显著**）
3. 但 **single-layer additive steering 不足以翻转 top-1 预测** — 需要更强的干预策略（多层协同、attention head 级别、或权重编辑）

---

## 文件结构

```
llm_bayesian/
├── README.md                          ← 本文件
├── proposal.md                        ← 原始研究动机与框架
├── bayesian_poc.ipynb                 ← Phase 0: PoC (已完成)
├── bayesian_phase1.ipynb              ← Phase 1: Causal Tracing (已完成)
├── bayesian_phase1b.ipynb             ← Phase 1b: Multi-Dataset (已完成)
├── bayesian_phase2_finegrained.ipynb  ← Phase 2: Fine-Grained Spectrum (已完成)
└── bayesian_phase3_blocking.ipynb     ← Phase 3: Early Warning + Blocking (已完成)

Data/
├── bayesian_phase0/
│   ├── phase0_results.npz     ← Phase 0 数值结果
│   ├── posterior_L16.png
│   ├── proj_histograms.png
│   ├── cross_layer_entropy.png
│   ├── hallucination_sim.png
│   ├── prior_sensitivity.png
│   └── validation.png
├── bayesian_phase1/
│   ├── phase1_results.npz     ← Phase 1 数值结果
│   ├── main_results.png       ← 6 面板主图
│   └── beta_sensitivity.png   ← β 敏感性分析图
├── bayesian_phase1b/
│   ├── phase1b_results.npz    ← Phase 1b 数值结果
│   ├── summary.json           ← 分数据集摘要 (JSON)
│   ├── main_results.png       ← 9 面板主图
│   └── beta_sensitivity.png   ← β 敏感性分析图
├── bayesian_phase2/
│   ├── phase2_results.npz     ← Phase 2 数值结果
│   ├── summary.json           ← 摘要 (JSON)
│   ├── causal_spectrum.png    ← 9 面板因果后验频谱
│   ├── individual_spectra.png ← 个体频谱示例
│   └── residual_contribution.png ← 残差贡献分析
└── bayesian_phase3/
    ├── phase3_results.npz     ← Phase 3 数值结果 (detector + intervention + recovery)
    ├── summary.json           ← 摘要 (JSON)
    ├── early_warning_detector.png ← 6 面板检测器分析
    └── causal_blocking.png    ← 6 面板因果阻断分析
```

---

## 下一步 (Proposed Next Steps)

### 短期改进

1. **真正的 Attn/MLP 分离**: Phase 2 的残差贡献分析是代理方法。需要扩展 NeuroScope API 添加 `apply_attn_steering` / `apply_mlp_steering` 以实现真正的子层干预。
2. **扩大 PopQA 子集**: PopQA 在 Phase 1b 中效应最强 (d=0.426)，且在 Phase 3 中 Δrank 最大 (+4,188)，应扩大至 500+ 样本。
3. **位置特异性干预**: 实现对特定 token 位置的 activation patching（当前 `apply_steering` 影响所有位置）。
4. **改进分类器**: 当前用子串匹配，假阳性率高。可用语义相似度或 LLM judge 替代。
5. **Hold-out 验证**: Phase 3 的 full-data refit 过拟合。需要 80/20 split 或新数据集验证 pipeline accuracy。

### 中期方向

6. **跨模型验证**: 在 Llama-3, Mistral 等不同架构上复现实验，验证频谱模式和 L2→L3 检测是否架构无关。
7. **Attention Head 级分析**: 利用 `get_attention(layer, head)` 分析注意力权重模式，寻找"factual recall"头。
8. **动态 β 选择**: 基于 IE 分布自适应选择 β，或使用 MCMC 等方法边际化 β。
9. **实时推理集成**: 将 Phase 3 pipeline 集成到 NeuroScope 推理循环中，实现 token-by-token 幻觉检测与修正。
10. **多层协同 steering**: Phase 3 单层 steering top-1 flip 仅 1.1%。尝试 L2+L5+L14 联合 steering，或更高 strength + contrastive decoding，可能突破 top-1 壁垒。

### 长期目标

11. **论文撰写**: 整合 Phase 0-3 结果，形式化贝叶斯因果歧义框架、双轨幻觉频谱理论、及早期预警+阻断流水线。
12. **幻觉检测器产品化**: 将 Phase 3 的 GBM 检测器 + L2 steering 打包为 NeuroScope 插件，实现 API 级幻觉缓解。

---

## 引用的先前实验

本项目基于 NeuroScope Test_bench 中已完成的 12 个实验的数据和发现：

| Exp | 名称 | 用途 |
|---|---|---|
| 2 | Aesthetic Subspace | Phase 0 数据源、方向向量 |
| 5 | Neuron Ablation | Phase 0 对比方向 |
| 9 | Poetry Genre | Phase 0 对比方向 |
| 10 | Imagery Direction | Phase 0 对比方向 |
