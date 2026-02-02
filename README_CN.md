# Translation Neuron Analysis

## 项目概述

本项目基于 NeuroScope 框架，对 LLaMA3-8B-Instruct 模型在中英翻译任务中的内部神经元激活模式进行了深入分析。研究探索了如何通过神经元劫持（Goal Hijacking）、神经元消融（Ablation）和激活引导（Steering）等技术来理解和控制模型的翻译行为。

## 研究方法

### 1. 数据集与评估指标

- **数据集**: WMT19 中英翻译数据集（zh-en）
- **评估指标**: COMET (Crosslingual Optimized Metric for Evaluation of Translation)
- **模型**: LLaMA3-8B-Instruct (Q4_K_S量化版本)
- **样本数量**: 3,981个翻译样本

### 2. 翻译质量基线

模型在标准翻译任务上的表现：

```
Min COMET Score:  0.2088
Max COMET Score:  0.9907
Mean COMET Score: 0.8187
Std COMET Score:  0.0783
```

![COMET Score Distribution](image_cell_28_0.png)
*图1: LLaMA3-8B在中英翻译任务上的COMET分数分布*

### 3. 神经元激活模式分析

#### 3.1 激活统计

对所有32层神经元（每层4096个神经元）收集激活统计信息：
- 保留 COMET score ≥ 0.5 的样本（3,954个样本）
- 统计每个神经元的均值（mean）和标准差（std）

#### 3.2 异常神经元识别

通过分析每层激活的标准差和均值，识别出具有最大波动的神经元：

```
Layer 1     - 最大标准差神经元: n291
Layer 2-20  - 最大标准差神经元: n4055
Layer 21-31 - 最大标准差神经元: n2352
Layer 32    - 最大标准差神经元: n214

Layer 1-24  - 最大均值神经元: n4055
Layer 25-31 - 最大均值神经元: n3773
```

![Mean and Std Across Layers](image_cell_37_0.png)
*图2: 各层最大标准差神经元的均值和标准差分布*

![Activation Heatmap - Mean](image_cell_41_1.png)
*图3: 所有层神经元激活均值热力图*

![Activation Heatmap - Std](image_cell_42_1.png)
*图4: 所有层神经元激活标准差热力图*

## 实验结果

### 1. Logit Lens 分析

通过 Logit Lens 技术可视化模型在不同层的token预测概率：

![Logit Lens Heatmap](image_cell_57_1.png)
*图5: Logit Lens热力图 - 显示各层对token的预测概率分布*

### 2. Goal Hijacking 实验

**目标**: 将翻译任务劫持为其他任务（如小说创作）

通过对比正常翻译和劫持任务的激活模式，发现：

![Activation Similarity Comparison](image_cell_69_0.png)
*图6: 劫持激活与正常翻译激活的余弦相似度对比*

- 劫持任务的激活模式（红色）与正常翻译（蓝色）在大部分层存在显著差异
- 中间层（Layer 10-20）的差异最为明显

### 3. 神经元消融（Ablation）实验

**方法**: 在推理时将特定神经元的激活值置零

**最优消融结果**:
- **Layer 14** 的神经元 `[2265, 4055, 2082, 290, 2943]` 消融后效果最佳
- 消融示例输出：
  ```
  Here's a translation of the text:
  "I want to write a science fiction novel, can you give me some in..."
  ```

![Ablation COMET Scores](image_cell_83_1.png)
*图7: 各层消融后的COMET分数*

![Ablation Activation Similarity](image_cell_90_1.png)
*图8: Layer 14消融后的激活相似度变化*

### 4. 激活引导（Steering）实验

**方法**: 通过添加方向向量来引导模型激活

**实验设置**:
- 测试层: 选择关键层进行steering
- 强度范围: 0.1 ~ 1.0

**最优引导配置**:
```
Layer: 11
Strength: 0.5
COMET Score: 0.8941
```

![Steering Heatmap](image_cell_102_1.png)
*图9: 不同层和强度组合的Steering效果热力图*

**最佳配置示例输出**:
```
Here is the translation:
I want to write science fiction novels, please give me some inspiration.
```

![Steering Activation Similarity](image_cell_109_1.png)
*图10: Steering后的激活模式与劫持、正常翻译的对比*

## 主要发现

### 1. 神经元功能分化

- **n4055**: 在Layer 2-20中持续表现出最大标准差，可能是关键的翻译功能神经元
- **n2352**: 在Layer 21-31的后期层发挥重要作用
- 不同层的神经元承担不同的翻译子任务

### 2. 任务劫持的可行性

- 通过激活模式的差异可以有效区分不同任务
- 中间层（Layer 10-20）对任务类型最为敏感
- 激活相似度分析揭示了任务切换的神经机制

### 3. 神经元消融的有效性

- Layer 14 的特定神经元组合对翻译质量影响最大
- 消融可以作为理解模型内部机制的有效工具
- 不同层的消融效果存在显著差异

### 4. 激活引导的精准控制

- Layer 11 在强度0.5时达到最优平衡
- 过强的引导（strength > 0.7）可能破坏翻译质量
- Steering技术可以在不重新训练的情况下调整模型行为

## 技术栈

- **NeuroScope**: 神经网络可解释性框架（版本1.2.3）
- **模型**: Meta-LLaMA-3-8B-Instruct (Q4_K_S量化)
- **评估**: COMET (wmt22-comet-da), BLEU
- **数据**: WMT19 zh-en翻译数据集
- **可视化**: Matplotlib, Seaborn

## 代码结构

```
├── ActivationCollector.py    # 激活收集器
├── utils.py                   # 工具函数
├── translation_neuron.ipynb   # 主实验notebook
├── prompt_templet/            # prompt模板
│   └── translation_system_prompt/
└── translation_layer_activation/  # 激活数据存储
```

## 使用方法

### 环境设置

```bash
# 安装依赖
pip install neuroscope torch datasets evaluate comet-ml
```

### 运行实验

```python
# 1. 加载模型和数据
import neuroscope
engine = neuroscope.LlamaEngine(model_path)

# 2. 收集激活
collector = ActivationCollector(n_layers, hidden_dim, engine)

# 3. 应用干预（消融或引导）
engine.ablate_neuron(layer, neuron_idx)
# 或
engine.apply_steering(layer, steering_vec, strength)

# 4. 评估
comet_score = evaluate_translation(predictions, references)
```

## 未来工作

1. **扩展到其他语言对**: 验证发现在其他翻译方向的普适性
2. **多任务分析**: 探索翻译以外任务的神经元模式
3. **因果干预**: 进行更系统的因果分析实验
4. **神经元聚类**: 识别具有相似功能的神经元群组
5. **可解释性提升**: 开发更直观的神经元功能可视化工具

## 致谢

本项目使用了 NeuroScope 框架和 WMT19 数据集。感谢 Anthropic 的 COMET 评估指标和 Meta 的 LLaMA 模型。

---

**作者**: [Your Name]  
**日期**: 2026年1月  
**许可**: MIT License
