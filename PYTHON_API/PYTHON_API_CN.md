# NeuroScope Python API 文档

> **版本**: 1.2.3  
> **更新日期**: 2026-01-29 (Batch 激活捕获修复、大批量支持)  
> **Python 支持**: 3.10+

---

## ⚠️ RTX 50 系列 (Blackwell) 用户注意

在 RTX 5090/5080 等 Blackwell 架构 GPU 上，llama.cpp 的 CUDA Graph 存在已知兼容性问题。
如果遇到 "CUDA error: unspecified launch failure" 错误（特别是在不同长度的 prompt 之间切换时），
请设置环境变量禁用 CUDA Graph：

```python
import os
os.environ['GGML_CUDA_DISABLE_GRAPHS'] = '1'

import neuroscope
# ... 正常使用
```

或在 PowerShell 中：
```powershell
$env:GGML_CUDA_DISABLE_GRAPHS = "1"
python your_script.py
```

---

## 快速开始

### 安装

```bash
# 从源码构建
cd NeuroScope/build
cmake --build . --target neuroscope_py --config Release

# 将 python 目录添加到 PYTHONPATH
# 或将 neuroscope.cp311-win_amd64.pyd 复制到你的项目
```

### 基础使用

```python
import neuroscope

# 加载模型
engine = neuroscope.Engine("model.gguf", n_ctx=4096, n_gpu_layers=-1)

# 运行前向传播
engine.forward("Hello, world!")

# 获取激活值 (返回 numpy 数组)
activations = engine.get_activations(0)  # 第 0 层
print(activations.shape)  # (4096,) for Llama-3-8B

# 获取 logits
logits = engine.get_logits()
top_token = logits.argmax()
```

---

## API 参考

### 模块级函数

```python
neuroscope.__version__: str
# 返回版本号，如 "1.0.0"

neuroscope.version() -> str
# 返回版本号

neuroscope.cuda_available() -> bool
# 检查 CUDA 是否可用 (编译时确定)
```

---

### 枚举类型

#### `neuroscope.State`

推理状态机状态：

| 值 | 描述 |
|----|------|
| `State.IDLE` | 空闲，等待 prompt |
| `State.PREFILL` | 处理输入 prompt |
| `State.DECODE` | 生成 token 中 |
| `State.PAUSED` | 已暂停，KV cache 保留 |
| `State.ERROR` | 错误状态 |
| `State.SHUTDOWN` | 正在关闭 |

#### `neuroscope.PromptMode`

Prompt 模式：

| 值 | 描述 |
|----|------|
| `PromptMode.COMPLETION` | 原始文本补全 |
| `PromptMode.CHAT` | 使用聊天模板 |

#### `neuroscope.ModelArchitecture` (v1.1.5)

模型架构类型，用于架构特定的功能和张量映射：

| 值 | 描述 |
|----|------|
| `ModelArchitecture.UNKNOWN` | 未知架构 |
| `ModelArchitecture.LLAMA` | LLaMA, Llama-2, Llama-3 |
| `ModelArchitecture.QWEN` | Qwen, Qwen-2 |
| `ModelArchitecture.QWEN_MOE` | Qwen-MoE |
| `ModelArchitecture.MISTRAL` | Mistral |
| `ModelArchitecture.MIXTRAL` | Mixtral (MoE) |
| `ModelArchitecture.DEEPSEEK` | DeepSeek |
| `ModelArchitecture.DEEPSEEK_MOE` | DeepSeek-MoE |
| `ModelArchitecture.GEMMA` | Gemma, Gemma-2 |
| `ModelArchitecture.PHI` | Phi-2, Phi-3 |
| `ModelArchitecture.YI` | Yi |
| `ModelArchitecture.INTERNLM` | InternLM |
| `ModelArchitecture.BAICHUAN` | Baichuan |
| `ModelArchitecture.CHATGLM` | ChatGLM |

---

### 数据类

#### `neuroscope.ModelInfo`

模型元数据（只读属性）：

| 属性 | 类型 | 描述 |
|------|------|------|
| `name` | `str` | 模型名称 |
| `architecture` | `ModelArchitecture` | 模型架构枚举 (v1.1.5) |
| `architecture_str` | `str` | 原始架构字符串 |
| `n_vocab` | `int` | 词汇表大小 |
| `n_ctx` | `int` | 上下文长度 |
| `n_embd` | `int` | 隐藏维度 |
| `n_layers` | `int` | Transformer 层数 |
| `n_heads` | `int` | 注意力头数 (Query) |
| `n_kv_heads` | `int` | KV 头数 (GQA 支持) |
| `n_ff` | `int` | FFN 中间维度 |
| `is_moe` | `bool` | 是否为 MoE 模型 (v1.1.5) |
| `n_experts` | `int` | 专家数量 (MoE) |
| `n_experts_used` | `int` | 每 token 使用的专家数 |
| `tie_word_embeddings` | `bool` | 是否共享输入/输出嵌入权重 |
| `is_loaded` | `bool` | 模型是否已加载 |

**示例**:
```python
info = engine.model_info
print(info)  # <ModelInfo name='...' arch=llama layers=32 hidden=4096 GQA=32Q/8KV>
print(f"Architecture: {info.architecture}")  # ModelArchitecture.LLAMA
print(f"Is MoE: {info.is_moe}")  # False
```

#### `neuroscope.InferenceStats`

推理统计（只读属性）：

| 属性 | 类型 | 描述 |
|------|------|------|
| `tokens_generated` | `int` | 已生成 token 数 |
| `prompt_tokens` | `int` | Prompt token 数 |
| `tokens_per_second` | `float` | 生成速度 |
| `prefill_time_ms` | `float` | Prefill 耗时 (ms) |
| `last_decode_time_ms` | `float` | 最后一次 decode 耗时 |

#### `neuroscope.TokenEvent`

Token 生成事件：

| 属性 | 类型 | 描述 |
|------|------|------|
| `token_id` | `int` | Token ID |
| `token_str` | `str` | Token 字符串 |
| `position` | `int` | 序列位置 |
| `logprob` | `float` | 对数概率 |

#### `neuroscope.EngineConfig`

引擎配置：

| 属性 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `model_path` | `str` | `""` | 模型路径 |
| `n_ctx` | `int` | `4096` | 上下文长度 |
| `n_batch` | `int` | `512` | 批处理大小 |
| `n_gpu_layers` | `int` | `-1` | GPU 层数 (-1=全部) |
| `seed` | `int` | `-1` | 随机种子 (-1=随机) |
| `temperature` | `float` | `0.8` | 采样温度 |
| `top_p` | `float` | `0.95` | Top-p 采样 |
| `top_k` | `int` | `40` | Top-k 采样 |
| `repeat_penalty` | `float` | `1.1` | 重复惩罚 |
| `use_mmap` | `bool` | `True` | 使用内存映射 |
| `use_mlock` | `bool` | `False` | 锁定内存 |
| `verbose` | `bool` | `False` | 详细输出 |
| `disable_flash_attention` | `bool` | `False` | 禁用 Flash Attention (v1.1.0) |

#### `neuroscope.ChatMessage`

聊天消息：

| 属性 | 类型 | 描述 |
|------|------|------|
| `role` | `str` | 角色: "system", "user", "assistant" |
| `content` | `str` | 消息内容 |

---

### Engine 类

主推理引擎类。

#### 构造函数

```python
neuroscope.Engine(
    model_path: str = "",
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,
    seed: int = -1,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 40,
    verbose: bool = False,
    disable_flash_attention: bool = False
) -> Engine
```

**参数**:
- `model_path`: GGUF 模型文件路径（可选，可后续调用 `load_model`）
- `n_ctx`: 上下文窗口大小
- `n_gpu_layers`: 卸载到 GPU 的层数 (-1 = 全部)
- `disable_flash_attention`: 禁用 Flash Attention 以捕获注意力权重 (v1.1.0)
- `seed`: 随机种子 (-1 = 随机)
- `temperature`: 采样温度 (**必须 ≥ 0**)
- `top_p`: Nucleus 采样阈值 (**必须在 [0, 1] 范围内**)
- `top_k`: Top-k 采样 (**必须 ≥ 0**)
- `verbose`: 启用详细输出

**异常**:
- `ValueError`: 如果 temperature < 0, top_p 不在 [0,1] 范围, 或 top_k < 0
- `RuntimeError`: 如果模型加载失败

**示例**:
```python
# 方式 1: 构造时加载
engine = neuroscope.Engine("model.gguf", n_ctx=2048)

# 方式 2: 后续加载
engine = neuroscope.Engine()
engine.load_model("model.gguf", n_ctx=2048)
```

---

#### 模型管理

##### `load_model(path, n_ctx=4096, n_gpu_layers=-1) -> bool`

加载 GGUF 模型文件。

```python
success = engine.load_model("path/to/model.gguf", n_ctx=4096, n_gpu_layers=-1)
```

##### `unload_model() -> None`

卸载当前模型并释放资源。

##### `is_loaded` (属性)

检查模型是否已加载。

```python
if engine.is_loaded:
    print("Model ready!")
```

##### `model_info` (属性)

获取模型元数据。

```python
info = engine.model_info
print(f"Model: {info.name}, Layers: {info.n_layers}, Hidden: {info.n_embd}")
```

---

#### 高级推理 API

##### `forward(prompt: str, add_special: bool = True) -> bool`

运行前向传播（prefill + 单步 decode）。

调用后可通过 `get_activations()` 获取激活值。

**参数**:
- `prompt`: 输入文本
- `add_special`: 是否自动添加 BOS/EOS tokens（默认 True）。
  如果你的 prompt 已经包含 special tokens（如 chat template），设为 False。

**注意**: 此方法会重置 KV cache。如需保留上下文，请使用 `forward_append()`。

**示例**:
```python
# 普通文本 - 自动添加 BOS
engine.forward("The capital of France is")
acts = engine.get_activations(0)  # 获取第 0 层激活

# Chat template - 禁用自动添加 BOS (避免 double BOS 警告)
chat_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"
engine.forward(chat_prompt, add_special=False)
```

##### `forward_append(text: str) -> bool`

将文本追加到现有上下文（保留 KV cache）。

与 `forward()` 不同，此方法不会重置 KV cache。新文本以正确的位置编码追加到现有上下文。

**用途**:
- 多轮对话
- 流式/增量输入
- KV cache 连贯性测试

**参数**:
- `text`: 要追加的文本

**返回**:
- `bool`: 成功返回 True

**异常**:
- `RuntimeError`: 未加载模型或无现有上下文

**示例**:
```python
# 分步输入与一次性输入结果一致
engine.forward("The capital of France")  # 重置 KV cache
engine.forward_append(" is Paris.")       # 追加，保留 KV cache
logits1 = engine.get_logits()

engine.forward("The capital of France is Paris.")  # 一次性输入
logits2 = engine.get_logits()

# logits1 和 logits2 应该几乎相同 (cosine sim > 0.999)
```

---

#### Batch 处理 API

> ⚡ **v1.1.0 新增** - 批量处理多个 prompt，高效进行大规模激活分析
>
> ⚡ **v1.2.1 新增** - `device` 参数支持 CPU 激活存储，节省 VRAM
>
> ⚡ **v1.2.3 新增** - 修复大批量 (>64) 和 ubatch 分片场景下的激活捕获

##### 内部实现原理

批处理采用**并行序列处理**模式，利用 llama.cpp 的多序列 KV cache 机制：

```
┌─────────────────────────────────────────────────────────────┐
│                    Batch Processing Flow                     │
├─────────────────────────────────────────────────────────────┤
│  1. Tokenize all prompts                                     │
│  2. Check batch size vs n_seq_max (default: 64)              │
│     ├─ batch_size ≤ 64  →  batch_forward()                   │
│     └─ batch_size > 64  →  batchForwardLarge() (auto-split)  │
│  3. Pack tokens into chunks (≤ n_batch tokens each)          │
│  4. llama_decode() may split chunks into ubatches            │
│  5. cb_eval callback captures activations per layer          │
│  6. Map captured activations back to original sequences      │
└─────────────────────────────────────────────────────────────┘
```

**关键限制与行为**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_seq_max` | 64 | KV cache 支持的最大并行序列数。超过此数量会自动分批 |
| `n_batch` | 512 | 单次 `llama_decode()` 的最大 token 数 |
| `n_ubatch` | 512 | llama.cpp 内部可能进一步分割的 micro-batch 大小 |

**激活捕获的特殊处理**:

llama.cpp 为了优化性能，对最后一层 (`l_out-{N-1}`) 做了稀疏化处理：
- **中间层**: 输出所有 tokens 的激活 (shape: `[n_tokens, n_embd]`)
- **最后一层**: 仅输出 `batch.logits[i]=1` 的 tokens（即每个序列的最后一个 token）

这意味着如果批次有 5 个序列，每个 4 tokens：
- Layer 0-30: 输出 20 个激活向量
- Layer 31: 仅输出 5 个激活向量（每序列一个）

NeuroScope 内部使用两套映射表 (`current_chunk_to_flat_map` 和 `current_chunk_logits_map`) 来正确处理这种不对称性。

##### `batch_forward(prompts: List[str], padding: str = "right", add_special: bool = True, device: str = "cuda") -> dict`

批量处理多个 prompt。每个 prompt 独立处理（KV cache 在序列间清空），结果存储在指定设备内存中。

**参数**:
- `prompts`: 字符串列表，包含要处理的多个 prompt
- `padding`: 填充方向，`"right"`（默认）或 `"left"`
- `add_special`: 是否自动添加 BOS/EOS tokens（默认 True）。
  如果 prompts 已包含 special tokens（如 chat template），设为 False。
- `device`: 激活存储设备（**v1.2.1 新增**）：
  - `"cuda"`（默认）: 存储在 GPU VRAM。适合后续需要激活分析/steering 的场景。
    VRAM 用量约为 `total_tokens * n_embd * n_layers * 4 bytes`。
  - `"cpu"`: 存储在系统内存（CPU offload）。适合长序列或大批量场景，显著节省 VRAM。
    读取时零拷贝返回 numpy 数组，速度很快。

**返回**:
- `dict`: 包含以下键值对：
  - `batch_size`: 批次大小
  - `seq_lengths`: 各序列的 token 长度列表
  - `max_seq_len`: 最长序列的长度
  - `total_tokens`: 总 token 数
  - `success`: 是否成功
  - `device`: 实际使用的存储设备（`"cuda"` 或 `"cpu"`）

**异常**:
- `RuntimeError`: 未加载模型
- `ValueError`: 空的 prompt 列表、无效的 padding 值或无效的 device 值

**大批量处理 (>64 prompts)**:

当 `batch_size > n_seq_max` (64) 时，自动调用 `batchForwardLarge()` 进行分批处理：

```python
# 处理 100 个 prompts - 自动分成 2 批 (64 + 36)
prompts = [f"Prompt {i}" for i in range(100)]
result = engine.batch_forward(prompts)

# 激活数据被合并到统一的缓冲区
# batch_get_activations(layer) 返回 shape (100, n_embd)
acts = engine.batch_get_activations(15)
print(acts.shape)  # (100, 4096)
```

**示例**:
```python
# 普通文本批处理 (GPU 存储)
prompts = ["Hello, world!", "The capital of France is", "Machine learning is"]
result = engine.batch_forward(prompts)
print(f"Processed {result['batch_size']} prompts, {result['total_tokens']} total tokens")

# CPU 存储模式 - 适合长序列，节省 VRAM
long_prompts = ["..." * 1000 for _ in range(100)]  # 大批量长序列
result = engine.batch_forward(long_prompts, device="cpu")
print(f"Stored on {result['device']}")  # "cpu"
acts = engine.batch_get_layer_activations(15)  # 快速零拷贝读取

# Chat template 批处理 (禁用自动 BOS)
chat_prompts = [
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWorld<|eot_id|>",
]
result = engine.batch_forward(chat_prompts, add_special=False)
```

##### `batch_get_logits() -> np.ndarray`

获取批处理的 logits（每个序列的最后一个 token）。

**返回**:
- `np.ndarray`: 形状 `(batch_size, n_vocab)` 的 float32 数组

**异常**:
- `RuntimeError`: 未执行 `batch_forward()` 或无数据可用

**示例**:
```python
engine.batch_forward(["Hello", "World", "Test"])
logits = engine.batch_get_logits()  # Shape: (3, 128256)

# 获取每个 prompt 的最可能下一个 token
next_tokens = np.argmax(logits, axis=1)
```

##### `batch_get_activations(layer: int) -> np.ndarray`

获取指定层的批处理激活（每个序列的最后一个 token）。

**参数**:
- `layer`: 层索引 (0 到 n_layers-1)

**返回**:
- `np.ndarray`: 形状 `(batch_size, n_embd)` 的 float32 数组

**异常**:
- `RuntimeError`: 未执行 `batch_forward()` 或无数据可用
- `ValueError`: 无效的层索引

**示例**:
```python
engine.batch_forward(prompts)
layer_15_acts = engine.batch_get_activations(15)  # Shape: (batch_size, 4096)

# 计算批内激活相似度
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(layer_15_acts)
```

##### `batch_get_all_activations() -> np.ndarray`

获取所有层的批处理激活（每个序列的最后一个 token）。

**返回**:
- `np.ndarray`: 形状 `(batch_size, n_layers, n_embd)` 的 float32 数组

**异常**:
- `RuntimeError`: 未执行 `batch_forward()` 或无数据可用

**示例**:
```python
engine.batch_forward(prompts)
all_acts = engine.batch_get_all_activations()  # Shape: (batch_size, 32, 4096)

# 分析激活随层变化的模式
for layer in range(all_acts.shape[1]):
    layer_mean = np.mean(all_acts[:, layer, :], axis=1)
    print(f"Layer {layer} mean activation: {layer_mean}")
```

**完整批处理示例**:
```python
# 批处理与单独处理的结果一致性验证
prompts = ["The sky is", "Water is", "Fire is"]

# 批处理
engine.batch_forward(prompts)
batch_logits = engine.batch_get_logits()
batch_acts = engine.batch_get_all_activations()

# 单独处理
for i, prompt in enumerate(prompts):
    engine.forward(prompt)
    single_logits = engine.get_logits()
    single_acts = engine.get_all_activations()
    
    # 验证一致性
    cos_sim = np.dot(batch_logits[i], single_logits) / (
        np.linalg.norm(batch_logits[i]) * np.linalg.norm(single_logits)
    )
    assert cos_sim > 0.9999, "Batch and single should match!"
```

---

#### 零拷贝 CPU 激活访问 API

> ⚡ **v1.2.2 新增** - 连续内存布局 + 真正的零拷贝 NumPy 访问

当使用 `device="cpu"` 进行批处理时，激活数据存储在连续内存缓冲区中。
以下方法提供对该缓冲区的直接零拷贝访问，无需任何数据复制。

##### `has_cpu_buffer() -> bool`

检查是否有可用的 CPU 激活缓冲区。

**返回**:
- `bool`: 如果执行了 `batch_forward(device="cpu")` 且数据可用，返回 True

**示例**:
```python
# 检查 CPU 缓冲区是否可用
if engine.has_cpu_buffer():
    activations = engine.get_cpu_activations()
else:
    print("No CPU buffer available - use batch_forward(device='cpu') first")
```

##### `get_cpu_buffer_shape() -> Tuple[int, int, int]`

获取 CPU 激活缓冲区的形状。

**返回**:
- `Tuple[int, int, int]`: `(n_layers, total_tokens, hidden_dim)`
  - `n_layers`: 模型层数
  - `total_tokens`: 批次中所有序列的 token 总数
  - `hidden_dim`: 隐藏维度 (n_embd)

**异常**:
- `RuntimeError`: 无可用的 CPU 缓冲区

**示例**:
```python
engine.batch_forward(prompts, device="cpu")
n_layers, total_tokens, hidden_dim = engine.get_cpu_buffer_shape()
print(f"Buffer shape: [{n_layers}, {total_tokens}, {hidden_dim}]")
# 例如: Buffer shape: [32, 150, 4096]
```

##### `get_cpu_activations() -> np.ndarray`

获取 CPU 激活缓冲区的零拷贝视图。

返回的 NumPy 数组直接引用 C++ 内存，无需复制。
这比逐层/逐序列调用 `batch_get_*` 方法更高效。

**返回**:
- `np.ndarray`: 形状 `(n_layers, total_tokens, hidden_dim)` 的 float32 数组
  - 数据布局: `[Layer 0 所有 tokens][Layer 1 所有 tokens]...`
  - **零拷贝**: 直接引用 C++ 内存
  - **只读安全**: 修改数组会影响原始数据

**异常**:
- `RuntimeError`: 无可用的 CPU 缓冲区（未调用 `batch_forward(device="cpu")`）

**示例**:
```python
import neuroscope
import numpy as np

engine = neuroscope.Engine("model.gguf")
prompts = ["Hello, world!", "The sky is blue", "Machine learning"]

# 使用 CPU 存储模式
result = engine.batch_forward(prompts, device="cpu")

# 零拷贝访问
activations = engine.get_cpu_activations()
print(f"Shape: {activations.shape}")  # (32, 15, 4096) for Llama-8B

# 标准 NumPy 切片操作
layer_15 = activations[15]           # 第 15 层所有 tokens, shape: (15, 4096)
all_layers_token_0 = activations[:, 0, :]   # 所有层的 token 0, shape: (32, 4096)
subset = activations[10:20, :5, :]   # 层 10-19, tokens 0-4, shape: (10, 5, 4096)

# 计算层间相似度
from sklearn.metrics.pairwise import cosine_similarity
layer_0 = activations[0].mean(axis=0)  # 第 0 层平均激活
layer_31 = activations[31].mean(axis=0)  # 最后层平均激活
print(f"Layer 0 vs 31 similarity: {cosine_similarity([layer_0], [layer_31])[0,0]:.4f}")
```

**性能比较**:
```python
import time

# 零拷贝方法 (推荐)
start = time.perf_counter()
acts = engine.get_cpu_activations()  # ~0.001 ms (仅创建视图)
layer_15 = acts[15]
zero_copy_time = time.perf_counter() - start

# 传统方法 (需要复制)
start = time.perf_counter()
layer_15_copy = engine.batch_get_layer_activations(15)  # 需要复制数据
copy_time = time.perf_counter() - start

print(f"Zero-copy: {zero_copy_time*1000:.3f} ms")
print(f"Copy method: {copy_time*1000:.3f} ms")
# 零拷贝通常快 10-100x
```

**注意事项**:
- 返回的数组生命周期与 Engine 对象绑定。在 Engine 对象被销毁或调用新的 `batch_forward()` 后，数组将失效。
- 数组是可写的，但不建议修改，因为这会影响 C++ 侧的原始数据。
- 如需持久化数据，使用 `np.copy(activations)` 创建独立副本。

---

##### `generate(prompt, max_tokens=256, temperature=0.8, top_p=0.95, top_k=40, seed=-1) -> str`

生成文本。

**参数**:
- `prompt`: 输入文本
- `max_tokens`: 最大生成 token 数 (**必须 ≥ 0，或 -1 表示无限制**)
- `temperature`: 采样温度 (**必须 ≥ 0**，0 = 贪心采样)
- `top_p`: Nucleus 采样阈值 (**必须在 [0, 1] 范围内**)
- `top_k`: Top-k 采样 (**必须 ≥ 0**，0 = 禁用)
- `seed`: 随机种子 (-1 = 随机)

**异常**:
- `ValueError`: 如果参数值无效
- `RuntimeError`: 如果未加载模型

**示例**:
```python
# 基本使用
text = engine.generate("Once upon a time", max_tokens=100)

# 可复现生成
text = engine.generate("Hello", max_tokens=50, seed=42, temperature=0.7)
```

##### `configure_sampler(temperature=0.8, top_p=0.95, top_k=40, seed=-1) -> None`

配置采样参数。可以在任何时候调用以更新采样行为。

**参数**:
- `temperature`: 采样温度 (**必须 ≥ 0**)
- `top_p`: Nucleus 采样阈值 (**必须在 [0, 1] 范围内**)
- `top_k`: Top-k 采样 (**必须 ≥ 0**)
- `seed`: 随机种子 (-1 = 随机)

**异常**:
- `ValueError`: 如果参数值无效

**示例**:
```python
# 设置为贪心采样
engine.configure_sampler(temperature=0)

# 设置高随机性
engine.configure_sampler(temperature=1.2, top_p=0.9, top_k=50)
```

##### `tokenize(text: str, add_special: bool = True) -> list[int]`

将文本转换为 token ID 列表。

```python
tokens = engine.tokenize("Hello, world!")
print(tokens)  # [9906, 11, 1917, 0]
```

##### `detokenize(tokens: list[int]) -> str`

将 token ID 列表转换回文本。

```python
text = engine.detokenize([9906, 11, 1917, 0])
print(text)  # "Hello, world!"
```

---

#### Chat Template API (v1.1.5)

自动使用模型内置的聊天模板格式化对话。

##### `apply_chat_template(messages: list, add_generation_prompt: bool = True) -> str`

应用模型的聊天模板格式化消息列表。

**参数**:
- `messages`: 消息列表，每条消息可以是：
  - dict 格式: `{"role": "user", "content": "Hello"}`
  - tuple 格式: `("user", "Hello")`
  - 有效 role: `"system"`, `"user"`, `"assistant"`
- `add_generation_prompt`: 是否添加 assistant 回复前缀 (默认 True)

**返回**:
- 格式化后的 prompt 字符串，可直接传给 `forward()`

**示例**:
```python
messages = [
    {"role": "system", "content": "你是一个专业翻译"},
    {"role": "user", "content": "Translate: Hello world"}
]
prompt = engine.apply_chat_template(messages)
# Llama-3 输出:
# <|start_header_id|>system<|end_header_id|>
# 
# 你是一个专业翻译<|eot_id|><|start_header_id|>user<|end_header_id|>
# 
# Translate: Hello world<|eot_id|><|start_header_id|>assistant<|end_header_id|>

# 模板不含 BOS，使用默认 add_special=True
engine.forward(prompt)
```

##### `has_chat_template() -> bool`

检查模型是否有内置聊天模板。

```python
if engine.has_chat_template():
    prompt = engine.apply_chat_template(messages)
else:
    # 手动构建 prompt
    prompt = f"User: {user_msg}\nAssistant:"
```

##### `get_chat_template() -> str`

获取原始聊天模板字符串 (Jinja2 格式)。

```python
template = engine.get_chat_template()
print(template[:100])  # "{% set loop_messages = messages %}{% for message in..."
```

---

#### 步进式推理 API

用于逐 token 控制生成过程。

##### `set_prompt(prompt: str) -> None`

设置 prompt，准备步进式生成。

##### `set_prompt_chat(user_message: str, keep_history: bool = False, system_prompt: str = "") -> None`

使用聊天模板设置 prompt。

##### `prefill() -> bool`

执行 prefill 阶段（处理 prompt）。

##### `step() -> TokenEvent | None`

生成恰好一个 token。返回 `TokenEvent` 或 `None`（如果结束）。

```python
engine.set_prompt("Once upon a time")
engine.prefill()

for _ in range(10):
    event = engine.step()
    if event is None:
        break
    print(event.token_str, end='', flush=True)
```

##### `decode(max_tokens: int = -1) -> int`

持续生成直到 EOS 或达到 max_tokens。返回生成的 token 数。

##### `reset() -> None`

重置推理状态，清空 KV cache。

##### `state` (属性)

获取当前推理状态。

```python
if engine.state == neuroscope.State.IDLE:
    print("Ready for new prompt")
```

##### `stats` (属性)

获取推理统计信息。

```python
print(f"Speed: {engine.stats.tokens_per_second:.1f} tok/s")
```

---

#### 激活值访问

所有激活值以 **numpy.ndarray** 返回，dtype 为 `float32`。

##### `get_activations(layer: int) -> np.ndarray`

获取指定层的激活值。

**参数**:
- `layer`: 层索引 (0-based，必须 < n_layers)

**返回**:
- `np.ndarray`: shape `(hidden_dim,)`，dtype `float32`

**示例**:
```python
engine.forward("Hello")
acts = engine.get_activations(0)  # 第 0 层
print(acts.shape)  # (4096,)
print(acts.dtype)  # float32
```

##### `get_all_activations() -> list[np.ndarray]`

获取所有层的激活值。

```python
all_acts = engine.get_all_activations()
print(len(all_acts))  # n_layers

# 堆叠为 2D 数组
import numpy as np
acts_matrix = np.stack(all_acts)
print(acts_matrix.shape)  # (n_layers, hidden_dim)
```

##### `get_final_hidden() -> np.ndarray`

获取最后一层的隐藏状态。

##### `get_logits() -> np.ndarray`

获取最后一个 token 的 logits。

```python
logits = engine.get_logits()
print(logits.shape)  # (n_vocab,)

# 获取 top-k tokens
top_k = 5
top_indices = np.argsort(logits)[-top_k:][::-1]
for idx in top_indices:
    print(f"{idx}: {engine.detokenize([idx])}")
```

##### `get_attention(layer: int, head: int = -1) -> np.ndarray`

获取注意力权重。**需要 `disable_flash_attention=True`**。

**参数**:
- `layer`: 层索引 (0 到 n_layers-1)
- `head`: 注意力头索引 (0 到 n_heads-1)，或 -1 获取所有头

**返回**:
- `head >= 0`: 2D 数组 `(seq_len, kv_len)` - 单头注意力矩阵
- `head == -1`: 3D 数组 `(n_heads, seq_len, kv_len)` - 所有头的注意力矩阵

**异常**:
- `RuntimeError`: 未加载模型、未运行 forward、或 Flash Attention 未禁用
- `IndexError`: 无效的 head 索引

**示例**:
```python
# 创建引擎时禁用 Flash Attention
engine = neuroscope.Engine("model.gguf", disable_flash_attention=True)
engine.forward("The capital of France is Paris.")

# 获取注意力元数据
info = engine.get_attention_info(0)
print(info)  # {'valid': True, 'n_heads': 32, 'seq_len': 8, 'kv_len': 256}

# 获取单头注意力 (2D)
attn = engine.get_attention(layer=0, head=0)
print(attn.shape)  # (8, 256)

# 获取所有头注意力 (3D)
attn_all = engine.get_attention(layer=0)
print(attn_all.shape)  # (32, 8, 256)

# 注意力权重是 softmax 输出，行和为 1
print(attn.sum(axis=1))  # [1.0, 1.0, ...]

# 可视化
import matplotlib.pyplot as plt
plt.imshow(attn[:, :attn.shape[0]], cmap='viridis')  # 显示 causal 部分
plt.colorbar()
plt.title("Attention Pattern (Layer 0, Head 0)")
```

##### `get_attention_info(layer: int) -> dict`

获取注意力元数据。

**参数**:
- `layer`: 层索引

**返回**:
- `dict`: 包含 `valid`, `n_heads`, `seq_len`, `kv_len`

---

#### 激活干预 / Steering

##### `apply_mask(layer: int, mask: np.ndarray) -> None`

对指定层应用乘法掩码：$H = H \odot M$

**参数**:
- `layer`: 目标层索引 (**必须在 [0, n_layers) 范围内**)
- `mask`: shape `(hidden_dim,)` 的 numpy 数组

**异常**:
- `IndexError`: 如果 layer 索引超出范围
- `ValueError`: 如果 mask 形状不匹配 hidden_dim
- `RuntimeError`: 如果未加载模型

**示例**:
```python
import numpy as np

# 将第 10 层的前 100 个神经元置零
mask = np.ones(4096, dtype=np.float32)
mask[:100] = 0
engine.apply_mask(10, mask)

# 运行推理
engine.forward("Hello")
```

##### `apply_steering(layer: int, direction: np.ndarray, strength: float = 1.0) -> None`

应用加性 steering 向量：$H = H + s \cdot D$

**参数**:
- `layer`: 目标层索引 (**必须在 [0, n_layers) 范围内**)
- `direction`: shape `(hidden_dim,)` 的方向向量
- `strength`: 缩放系数

**异常**:
- `IndexError`: 如果 layer 索引超出范围
- `ValueError`: 如果 direction 形状不匹配 hidden_dim
- `RuntimeError`: 如果未加载模型

**示例**:
```python
import numpy as np

# 应用一个 "积极" 方向到第 15 层
positive_direction = np.load("positive_vector.npy")
engine.apply_steering(15, positive_direction, strength=2.0)
```

##### `clear_interventions() -> None`

清除所有掩码和 steering 向量。

##### `has_interventions` (属性)

检查是否有活跃的干预。

---

#### 生成文本访问

##### `generated_text` (属性)

获取目前生成的所有文本。

```python
engine.generate("Hello", max_tokens=10)
print(engine.generated_text)
```

##### `get_generated_tokens() -> list[int]`

获取生成的 token ID 列表。

---

#### 回调函数

##### `set_token_callback(callback: Callable[[TokenEvent], None]) -> None`

设置 token 生成回调。

```python
def on_token(event):
    print(event.token_str, end='', flush=True)

engine.set_token_callback(on_token)
engine.generate("Hello", max_tokens=50)
```

---

#### 聊天历史

##### `add_assistant_response(response: str) -> None`

将助手响应添加到聊天历史。

##### `clear_chat_history() -> None`

清除对话历史。

##### `chat_history` (属性)

获取聊天历史。

```python
for msg in engine.chat_history:
    print(f"{msg.role}: {msg.content}")
```

---

#### 上下文管理器

Engine 支持上下文管理器协议：

```python
with neuroscope.Engine("model.gguf") as engine:
    text = engine.generate("Hello")
# 自动调用 unload_model()
```

---

## 完整示例

### 1. 基础推理

```python
import neuroscope

engine = neuroscope.Engine("llama-3-8b.gguf", n_ctx=2048)

# 简单生成
text = engine.generate("The meaning of life is", max_tokens=50)
print(text)
```

### 2. 激活分析

```python
import neuroscope
import numpy as np

engine = neuroscope.Engine("llama-3-8b.gguf")

# 运行前向传播
engine.forward("The capital of France is")

# 获取所有层激活
all_acts = engine.get_all_activations()
acts_matrix = np.stack(all_acts)
print(f"Activations shape: {acts_matrix.shape}")  # (32, 4096)

# 分析激活模式
layer_norms = np.linalg.norm(acts_matrix, axis=1)
print(f"Layer norms: {layer_norms}")
```

### 3. Activation Steering

```python
import neuroscope
import numpy as np

engine = neuroscope.Engine("llama-3-8b.gguf")

# 加载预计算的 steering 向量
steering_vec = np.load("happy_direction.npy")

# 应用到第 15 层
engine.apply_steering(15, steering_vec, strength=3.0)

# 生成
text = engine.generate("I feel", max_tokens=30)
print(text)

# 清除干预
engine.clear_interventions()
```

### 4. 步进式生成

```python
import neuroscope

engine = neuroscope.Engine("llama-3-8b.gguf")

engine.set_prompt("Once upon a time")
engine.prefill()

# 逐 token 生成并记录激活
activations_history = []
for i in range(20):
    event = engine.step()
    if event is None:
        break
    
    # 记录每个 token 的激活
    acts = engine.get_activations(15)  # 第 15 层
    activations_history.append(acts.copy())
    
    print(event.token_str, end='', flush=True)

print(f"\nRecorded {len(activations_history)} activation snapshots")
```

### 5. Chat Template 使用 (Llama-3)

```python
import neuroscope

engine = neuroscope.Engine("llama-3-8b-instruct.gguf", n_ctx=4096)

# 定义 system 和 user prompt
system_prompt = "你是一个专业翻译，你精通把中文翻译成英文，你会把user输入每一句话翻译成英文"
user_prompt = "我想写一个科幻小说，请给我一些灵感"

# 构建 Llama-3 chat template
# 格式: <|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
chat_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

# 重要: 使用 add_special=False，因为 prompt 已包含 <|begin_of_text|>
engine.forward(chat_prompt, add_special=False)

# 获取模型预测
top_tokens = engine.get_top_tokens(k=5)
print("模型预测的下一个 token:", top_tokens)

# 生成回复
response = engine.generate(chat_prompt, max_tokens=200, add_special=False)
print(response)
```

**Llama-3 Special Tokens 参考**:
| Token | ID | 用途 |
|-------|-----|------|
| `<\|begin_of_text\|>` | 128000 | BOS，对话开始 |
| `<\|end_of_text\|>` | 128001 | EOS，对话结束 |
| `<\|start_header_id\|>` | 128006 | 角色标记开始 |
| `<\|end_header_id\|>` | 128007 | 角色标记结束 |
| `<\|eot_id\|>` | 128009 | 单轮结束 (End of Turn) |

---

## 常见问题

### CUDA DLL 加载失败

在 Windows 上，如果遇到 CUDA DLL 加载问题：

```python
import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")
import neuroscope
```

或者设置环境变量 `CUDA_PATH`。

### 内存不足

对于大模型，可以减少 GPU 层数：

```python
engine = neuroscope.Engine("model.gguf", n_gpu_layers=20)  # 只卸载 20 层
```

### Batch 处理激活返回全零

如果 `batch_get_activations()` 返回全零，请检查：

1. **版本**: 确保使用 v1.2.3 或更高版本（修复了 ubatch 分片问题）
2. **RTX 50 系列**: 设置 `GGML_CUDA_DISABLE_GRAPHS=1` 环境变量
3. **批量大小**: 超过 64 个 prompts 时会自动分批处理，这是正常行为

```python
# 验证激活是否正常
result = engine.batch_forward(prompts)
acts = engine.batch_get_activations(15)
norms = np.linalg.norm(acts, axis=1)
print(f"Activation norms: {norms}")  # 应该都是非零值
```

### Batch 与 Single 模式结果不一致

批处理和单独处理的 cosine similarity 应该 > 0.999。如果差异较大：

1. 确保使用相同的 `add_special` 参数
2. 批处理使用并行 KV cache，可能有微小数值差异（这是正常的）
3. 对于需要精确一致的场景，使用单独 `forward()` 循环

### 上下文长度

如果 prompt 太长，会被截断。可以增加 `n_ctx`：

```python
engine = neuroscope.Engine("model.gguf", n_ctx=8192)
```

---

## 架构说明

```
┌─────────────────────────────────────────────┐
│              Python Layer                    │
│   import neuroscope                          │
│   engine = neuroscope.Engine(...)            │
└─────────────────┬───────────────────────────┘
                  │ pybind11
                  ▼
┌─────────────────────────────────────────────┐
│              CoreEngine (C++)               │
│   - Model loading                            │
│   - Tokenization                             │
│   - Inference (forward/generate)             │
│   - Activation capture                       │
│   - Steering/Masking                         │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│              llama.cpp                       │
│   - GGUF model loading                       │
│   - KV cache management                      │
│   - CUDA/CPU inference                       │
└─────────────────────────────────────────────┘
```

---

## Logit Lens 原理与使用

### 原理说明

Logit Lens 的核心思想是：将任意中间层的隐藏状态 $h^{(l)}$ 直接通过 unembedding 矩阵 $W_U$ 投影到词表空间：

$$\text{logits}^{(l)} = h^{(l)} \cdot W_U^T$$

其中：
- $h^{(l)} \in \mathbb{R}^{d_{model}}$ 是第 $l$ 层的隐藏状态（最后一个 token）
- $W_U \in \mathbb{R}^{V \times d_{model}}$ 是 unembedding（lm_head）权重矩阵
- $V$ 是词表大小，$d_{model}$ 是隐藏维度

通过这种方式，我们可以"窥视"模型在每一层的预测倾向，观察信息如何在层间流动。

### 使用示例

```python
engine.forward("The meaning of life is")

# 查看每层的预测
for layer in range(0, 32, 4):
    lens_logits = engine.logit_lens(layer)
    top_token = lens_logits.argmax()
    print(f"Layer {layer:2d}: {engine.detokenize([top_token])}")

# 获取 unembedding 权重用于自定义分析
W_unembed = engine.get_unembed_weights()  # (vocab_size, hidden_dim)
```

### 性能说明

> ✅ **v1.1.2 起支持所有模型类型**：F16/F32 及 Q4/Q8 量化模型均可使用 GPU 加速。

| 模型类型 | `logit_lens(layer)` | `logit_lens_all()` | 说明 |
|---------|---------------------|-------------------|------|
| F16/F32 | ~0.002s | ~0.045s | 原生 FP16 权重 |
| Q4_K_S | ~0.058s | ~0.045s | 预解压到 FP16 |
| Q8_0 | ~0.003s | ~0.047s | 预解压到 FP16 |

- **所有模型**: 统一使用 `cublasHgemm` 半精度矩阵乘法
- **量化模型**: 在 `loadModel()` 时自动预解压 unembedding 权重到 FP16 GPU 缓存
- **内存开销**: `n_vocab × n_embd × 2 bytes` (~1GB for Llama-3-8B)

### API 参考

| 方法 | 返回类型 | 描述 |
|------|---------|------|
| `logit_lens(layer)` | `np.ndarray (V,)` | 单层 Logit Lens |
| `logit_lens_all()` | `np.ndarray (L, V)` | 所有层 Logit Lens |
| `batch_logit_lens(layer)` | `np.ndarray (B, V)` | 批量 Logit Lens |
| `batch_logit_lens_all()` | `np.ndarray (B, L, V)` | 批量所有层 Logit Lens |
| `get_unembed_weights()` | `np.ndarray (V, H)` | 获取 unembedding 权重 |

---

## 版本历史

### v1.2.3 (2026-01-29)

**Batch 激活捕获修复**:
- 修复大批量 (>64 prompts) 场景下 `batch_get_activations()` 返回全零的问题
- 修复 llama.cpp ubatch 分片导致的激活映射错误
- 新增 `current_chunk_logits_map` 处理最后一层的稀疏化输出
- 新增 `ubatch_logits_offset` 原子变量追踪最后一层的 ubatch 偏移

**技术细节**:
- llama.cpp 的最后一层 (`l_out-{N-1}`) 仅输出 `batch.logits[i]=1` 的 tokens
- 中间层使用 `current_chunk_to_flat_map` 映射所有 tokens
- 最后层使用独立的 `current_chunk_logits_map` 映射稀疏 tokens
- `batchForwardLarge()` 现在正确处理跨子批次的激活合并

### v1.1.2 (2026-01-16)

**Q4/Q8 量化模型支持**:
- Logit Lens 现支持所有 GGUF 量化格式 (Q4_K, Q5_K, Q6_K, Q8_0 等)
- 在 `loadModel()` 时自动预解压 unembedding 权重到 FP16 GPU 缓存
- GPU 内存开销: ~1GB (Llama-3-8B)
- 性能: Q4_K_S ~0.05s, Q8_0 ~0.05s (32层 logit_lens_all)

### v1.1.1 (2026-01-16)

**cuBLAS GPU 加速**:
- F16/F32 模型 Logit Lens 全程 GPU 计算
- `logit_lens()`: 后续调用 ~0.002s (250x faster)
- `logit_lens_all()`: 后续调用 ~0.045s (356x faster)

### v1.1.0 (2026-01-16)

**Batch Processing API**:
- 新增 `batch_forward(prompts, padding)` - 批量前向传播
- 新增 `batch_get_logits()` - 获取批处理 logits
- 新增 `batch_get_activations(layer)` - 获取批处理激活
- 新增 `batch_get_all_activations()` - 获取所有层批处理激活

**Logit Lens API**:
- 新增 `get_unembed_weights()` - 获取 unembedding 权重
- 新增 `logit_lens(layer)` - 单层 Logit Lens
- 新增 `logit_lens_all()` - 所有层 Logit Lens
- 新增 `batch_logit_lens(layer)` - 批量 Logit Lens
- 新增 `batch_logit_lens_all()` - 批量所有层 Logit Lens

**Batch Steering API**:
- 新增 `batch_apply_steering(layer, direction, strength)` - 批量 steering
- 新增 `batch_apply_mask(layer, mask)` - 批量 mask

**Attention Map API**:
- 修复 `get_attention(layer, head)` 返回正确形状
  - 单头: `(seq_len, kv_len)`
  - 所有头: `(n_heads, seq_len, kv_len)`
- 新增 `get_attention_info(layer)` - 获取注意力元数据
- 新增 `disable_flash_attention` 参数 - 禁用 Flash Attention 以捕获注意力

**测试**: 114 tests passed, 12 skipped (Logit Lens 需要 F16 模型)

### v1.0.3 (2026-01-16)

- 新增 `forward_append()` 方法，支持 KV cache 续写模式
- 新增 KV Cache 连贯性测试套件 (11 tests)
- 支持多轮对话和流式输入

### v1.0.2 (2026-01-16)

- 修复 `max_tokens=-1` 边界检查：现在负数值正确抛出 `ValueError`
- 新增数值正确性测试套件（15 tests，与 HuggingFace 对比验证）
- 新增性能基准测试脚本 (`benchmarks/`)

### v1.0.1 (2026-01-16)

- 添加参数验证，无效参数现在会抛出明确异常
- `apply_mask()` / `apply_steering()`: 无效层索引抛出 `IndexError`
- `generate()`: 无效 max_tokens/temperature/top_p/top_k 抛出 `ValueError`
- `Engine()` 构造函数: 无效采样参数抛出 `ValueError`
- 新增 `configure_sampler()` 方法文档

### v1.0.0 (2026-01-16)

- 初始发布
- 支持 GGUF 模型加载
- 激活值提取 (numpy arrays)
- Activation steering 和 masking
- 步进式生成控制
- 聊天模板支持

---

## 异常参考

| 异常类型 | 触发条件 |
|---------|---------|
| `ValueError` | 参数值无效（如 temperature < 0, top_p > 1, max_tokens < -1） |
| `IndexError` | 层索引超出范围（如 `apply_mask(99999, mask)`） |
| `RuntimeError` | 未加载模型时调用推理方法，或模型加载失败 |
