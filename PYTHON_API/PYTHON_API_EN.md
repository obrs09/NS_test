# NeuroScope Python API Documentation

> **Version**: 1.2.3  
> **Last Updated**: 2026-01-29 (Batch activation capture fix, large batch support)  
> **Python Support**: 3.10+

---

## ⚠️ RTX 50 Series (Blackwell) User Notice

On RTX 5090/5080 and other Blackwell architecture GPUs, llama.cpp's CUDA Graph has known compatibility issues.
If you encounter "CUDA error: unspecified launch failure" errors (especially when switching between prompts of different lengths),
please disable CUDA Graph by setting the environment variable:

```python
import os
os.environ['GGML_CUDA_DISABLE_GRAPHS'] = '1'

import neuroscope
# ... normal usage
```

Or in PowerShell:
```powershell
$env:GGML_CUDA_DISABLE_GRAPHS = "1"
python your_script.py
```

---

## Quick Start

### Installation

```bash
# Build from source
cd NeuroScope/build
cmake --build . --target neuroscope_py --config Release

# Add python directory to PYTHONPATH
# Or copy neuroscope.cp311-win_amd64.pyd to your project
```

### Basic Usage

```python
import neuroscope

# Load model
engine = neuroscope.Engine("model.gguf", n_ctx=4096, n_gpu_layers=-1)

# Run forward pass
engine.forward("Hello, world!")

# Get activations (returns numpy array)
activations = engine.get_activations(0)  # Layer 0
print(activations.shape)  # (4096,) for Llama-3-8B

# Get logits
logits = engine.get_logits()
top_token = logits.argmax()
```

---

## API Reference

### Module-Level Functions

```python
neuroscope.__version__: str
# Returns version string, e.g. "1.0.0"

neuroscope.version() -> str
# Returns version string

neuroscope.cuda_available() -> bool
# Check if CUDA is available (determined at compile time)
```

---

### Enumerations

#### `neuroscope.State`

Inference state machine states:

| Value | Description |
|-------|-------------|
| `State.IDLE` | Idle, waiting for prompt |
| `State.PREFILL` | Processing input prompt |
| `State.DECODE` | Generating tokens |
| `State.PAUSED` | Paused, KV cache retained |
| `State.ERROR` | Error state |
| `State.SHUTDOWN` | Shutting down |

#### `neuroscope.PromptMode`

Prompt modes:

| Value | Description |
|-------|-------------|
| `PromptMode.COMPLETION` | Raw text completion |
| `PromptMode.CHAT` | Use chat template |

#### `neuroscope.ModelArchitecture` (v1.1.5)

Model architecture types for architecture-specific features and tensor mapping:

| Value | Description |
|-------|-------------|
| `ModelArchitecture.UNKNOWN` | Unknown architecture |
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

### Data Classes

#### `neuroscope.ModelInfo`

Model metadata (read-only properties):

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Model name |
| `architecture` | `ModelArchitecture` | Model architecture enum (v1.1.5) |
| `architecture_str` | `str` | Raw architecture string |
| `n_vocab` | `int` | Vocabulary size |
| `n_ctx` | `int` | Context length |
| `n_embd` | `int` | Hidden dimension |
| `n_layers` | `int` | Number of transformer layers |
| `n_heads` | `int` | Number of attention heads (Query) |
| `n_kv_heads` | `int` | Number of KV heads (GQA support) |
| `n_ff` | `int` | FFN intermediate dimension |
| `is_moe` | `bool` | Whether it's an MoE model (v1.1.5) |
| `n_experts` | `int` | Number of experts (MoE) |
| `n_experts_used` | `int` | Number of experts used per token |
| `tie_word_embeddings` | `bool` | Whether input/output embeddings are shared |
| `is_loaded` | `bool` | Whether the model is loaded |

**Example**:
```python
info = engine.model_info
print(info)  # <ModelInfo name='...' arch=llama layers=32 hidden=4096 GQA=32Q/8KV>
print(f"Architecture: {info.architecture}")  # ModelArchitecture.LLAMA
print(f"Is MoE: {info.is_moe}")  # False
```

#### `neuroscope.InferenceStats`

Inference statistics (read-only properties):

| Property | Type | Description |
|----------|------|-------------|
| `tokens_generated` | `int` | Number of tokens generated |
| `prompt_tokens` | `int` | Number of prompt tokens |
| `tokens_per_second` | `float` | Generation speed |
| `prefill_time_ms` | `float` | Prefill time (ms) |
| `last_decode_time_ms` | `float` | Last decode time |

#### `neuroscope.TokenEvent`

Token generation event:

| Property | Type | Description |
|----------|------|-------------|
| `token_id` | `int` | Token ID |
| `token_str` | `str` | Token string |
| `position` | `int` | Sequence position |
| `logprob` | `float` | Log probability |

#### `neuroscope.EngineConfig`

Engine configuration:

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `model_path` | `str` | `""` | Model path |
| `n_ctx` | `int` | `4096` | Context length |
| `n_batch` | `int` | `512` | Batch size |
| `n_gpu_layers` | `int` | `-1` | Number of GPU layers (-1=all) |
| `seed` | `int` | `-1` | Random seed (-1=random) |
| `temperature` | `float` | `0.8` | Sampling temperature |
| `top_p` | `float` | `0.95` | Top-p sampling |
| `top_k` | `int` | `40` | Top-k sampling |
| `repeat_penalty` | `float` | `1.1` | Repeat penalty |
| `use_mmap` | `bool` | `True` | Use memory mapping |
| `use_mlock` | `bool` | `False` | Lock memory |
| `verbose` | `bool` | `False` | Verbose output |
| `disable_flash_attention` | `bool` | `False` | Disable Flash Attention (v1.1.0) |

#### `neuroscope.ChatMessage`

Chat message:

| Property | Type | Description |
|----------|------|-------------|
| `role` | `str` | Role: "system", "user", "assistant" |
| `content` | `str` | Message content |

---

### Engine Class

Main inference engine class.

#### Constructor

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

**Parameters**:
- `model_path`: Path to GGUF model file (optional, can call `load_model` later)
- `n_ctx`: Context window size
- `n_gpu_layers`: Number of layers to offload to GPU (-1 = all)
- `disable_flash_attention`: Disable Flash Attention to capture attention weights (v1.1.0)
- `seed`: Random seed (-1 = random)
- `temperature`: Sampling temperature (**must be ≥ 0**)
- `top_p`: Nucleus sampling threshold (**must be in [0, 1] range**)
- `top_k`: Top-k sampling (**must be ≥ 0**)
- `verbose`: Enable verbose output

**Raises**:
- `ValueError`: If temperature < 0, top_p not in [0,1] range, or top_k < 0
- `RuntimeError`: If model loading fails

**Example**:
```python
# Method 1: Load during construction
engine = neuroscope.Engine("model.gguf", n_ctx=2048)

# Method 2: Load later
engine = neuroscope.Engine()
engine.load_model("model.gguf", n_ctx=2048)
```

---

#### Model Management

##### `load_model(path, n_ctx=4096, n_gpu_layers=-1) -> bool`

Load GGUF model file.

```python
success = engine.load_model("path/to/model.gguf", n_ctx=4096, n_gpu_layers=-1)
```

##### `unload_model() -> None`

Unload current model and free resources.

##### `is_loaded` (property)

Check if model is loaded.

```python
if engine.is_loaded:
    print("Model ready!")
```

##### `model_info` (property)

Get model metadata.

```python
info = engine.model_info
print(f"Model: {info.name}, Layers: {info.n_layers}, Hidden: {info.n_embd}")
```

---

#### Advanced Inference API

##### `forward(prompt: str, add_special: bool = True) -> bool`

Run forward pass (prefill + single decode step).

After calling this, you can retrieve activations via `get_activations()`.

**Parameters**:
- `prompt`: Input text
- `add_special`: Whether to automatically add BOS/EOS tokens (default True).
  If your prompt already contains special tokens (e.g., chat template), set to False.

**Note**: This method resets the KV cache. To preserve context, use `forward_append()`.

**Example**:
```python
# Plain text - automatically add BOS
engine.forward("The capital of France is")
acts = engine.get_activations(0)  # Get layer 0 activations

# Chat template - disable automatic BOS addition (avoid double BOS warning)
chat_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"
engine.forward(chat_prompt, add_special=False)
```

##### `forward_append(text: str) -> bool`

Append text to existing context (preserves KV cache).

Unlike `forward()`, this method does not reset the KV cache. New text is appended to the existing context with correct positional encoding.

**Use cases**:
- Multi-turn conversations
- Streaming/incremental input
- KV cache coherence testing

**Parameters**:
- `text`: Text to append

**Returns**: `True` if successful

**Example**:
```python
# Initial context
engine.forward("The capital of France is")

# Append without resetting KV cache
engine.forward_append(" Paris. The capital of Germany is")

# Coherent activation across turns
acts = engine.get_activations(15)
```

##### `get_activations(layer: int) -> np.ndarray`

Get activation values from specified layer.

**Parameters**:
- `layer`: Layer index (0 to n_layers-1)

**Returns**: NumPy array of shape `(n_embd,)` containing the last token's hidden state

**Raises**:
- `IndexError`: If layer index is out of range
- `RuntimeError`: If called before `forward()`

**Example**:
```python
engine.forward("The sky is blue")
layer_15_acts = engine.get_activations(15)
print(layer_15_acts.shape)  # (4096,) for Llama-3-8B
```

##### `get_all_activations() -> np.ndarray`

Get activations from all layers.

**Returns**: NumPy array of shape `(n_layers, n_embd)`

**Example**:
```python
engine.forward("Hello world")
all_acts = engine.get_all_activations()
print(all_acts.shape)  # (32, 4096) for Llama-3-8B
```

##### `get_logits() -> np.ndarray`

Get final layer logits.

**Returns**: NumPy array of shape `(n_vocab,)`

**Example**:
```python
engine.forward("The capital of France is")
logits = engine.get_logits()
top_5_tokens = logits.argsort()[-5:][::-1]
```

##### `get_top_tokens(k: int = 10) -> List[Tuple[int, str, float]]`

Get top-k predicted tokens with probabilities.

**Parameters**:
- `k`: Number of top tokens to return

**Returns**: List of tuples `(token_id, token_str, probability)`

**Example**:
```python
engine.forward("The meaning of life is")
top_tokens = engine.get_top_tokens(k=5)
for token_id, token_str, prob in top_tokens:
    print(f"{token_str}: {prob:.2%}")
```

---

#### Batch Processing API (v1.1.0)

##### `batch_forward(prompts: List[str], padding: str = "left", add_special: bool = True) -> bool`

Process multiple prompts in parallel.

**Parameters**:
- `prompts`: List of input texts
- `padding`: Padding strategy - "left" or "right" (default "left")
- `add_special`: Whether to add BOS/EOS tokens (default True)

**Returns**: `True` if successful

**Note**: For batches > 64 prompts, automatically splits into sub-batches

**Example**:
```python
prompts = [
    "The capital of France is",
    "The capital of Germany is",
    "The capital of Italy is"
]
engine.batch_forward(prompts, padding="left")
```

##### `batch_get_logits() -> np.ndarray`

Get logits for all prompts in the batch.

**Returns**: NumPy array of shape `(batch_size, n_vocab)`

##### `batch_get_activations(layer: int) -> np.ndarray`

Get activations for all prompts at specified layer.

**Parameters**:
- `layer`: Layer index

**Returns**: NumPy array of shape `(batch_size, n_embd)`

**Example**:
```python
engine.batch_forward(prompts)
acts = engine.batch_get_activations(15)
print(acts.shape)  # (3, 4096)
```

##### `batch_get_all_activations() -> np.ndarray`

Get activations for all prompts across all layers.

**Returns**: NumPy array of shape `(batch_size, n_layers, n_embd)`

---

#### Logit Lens API (v1.1.0)

##### `logit_lens(layer: int) -> np.ndarray`

Project specified layer's hidden state to vocabulary space.

**Parameters**:
- `layer`: Layer index

**Returns**: NumPy array of shape `(n_vocab,)` - logits from that layer

**Example**:
```python
engine.forward("The meaning of life is")
for layer in range(0, 32, 4):
    lens_logits = engine.logit_lens(layer)
    top_token = lens_logits.argmax()
    print(f"Layer {layer:2d}: {engine.detokenize([top_token])}")
```

##### `logit_lens_all() -> np.ndarray`

Get logit lens for all layers at once.

**Returns**: NumPy array of shape `(n_layers, n_vocab)`

##### `batch_logit_lens(layer: int) -> np.ndarray`

Batch version of `logit_lens()`.

**Returns**: NumPy array of shape `(batch_size, n_vocab)`

##### `batch_logit_lens_all() -> np.ndarray`

Batch version of `logit_lens_all()`.

**Returns**: NumPy array of shape `(batch_size, n_layers, n_vocab)`

##### `get_unembed_weights() -> np.ndarray`

Get unembedding (lm_head) weight matrix.

**Returns**: NumPy array of shape `(n_vocab, n_embd)`

---

#### Activation Steering & Masking

##### `apply_steering(layer: int, direction: np.ndarray, strength: float = 1.0) -> None`

Apply activation steering vector to specified layer.

**Parameters**:
- `layer`: Target layer index
- `direction`: Steering vector of shape `(n_embd,)`
- `strength`: Steering strength multiplier

**Raises**:
- `IndexError`: If layer index is out of range

**Example**:
```python
steering_vec = np.load("happy_direction.npy")
engine.apply_steering(15, steering_vec, strength=3.0)
text = engine.generate("I feel", max_tokens=30)
```

##### `apply_mask(layer: int, mask: np.ndarray) -> None`

Apply multiplicative mask to layer activations.

**Parameters**:
- `layer`: Target layer index
- `mask`: Binary mask of shape `(n_embd,)` (0 to ablate, 1 to keep)

**Example**:
```python
mask = np.ones(4096)
mask[1000:2000] = 0  # Ablate features 1000-2000
engine.apply_mask(15, mask)
```

##### `batch_apply_steering(layer: int, direction: np.ndarray, strength: float = 1.0) -> None`

Apply steering to all samples in current batch.

##### `batch_apply_mask(layer: int, mask: np.ndarray) -> None`

Apply mask to all samples in current batch.

##### `clear_interventions() -> None`

Remove all steering vectors and masks.

---

#### Attention Map API (v1.1.0)

##### `get_attention(layer: int, head: int = -1) -> np.ndarray`

Get attention weights from specified layer and head.

**Note**: Flash Attention must be disabled via `disable_flash_attention=True` in constructor.

**Parameters**:
- `layer`: Layer index
- `head`: Head index (-1 for all heads)

**Returns**: 
- Single head: `(seq_len, kv_len)`
- All heads: `(n_heads, seq_len, kv_len)`

**Example**:
```python
engine = neuroscope.Engine("model.gguf", disable_flash_attention=True)
engine.forward("Hello world")

# Get single head
attn = engine.get_attention(layer=15, head=0)
print(attn.shape)  # (seq_len, kv_len)

# Get all heads
attn_all = engine.get_attention(layer=15, head=-1)
print(attn_all.shape)  # (32, seq_len, kv_len)
```

##### `get_attention_info(layer: int) -> Tuple[int, int, int]`

Get attention metadata for specified layer.

**Returns**: Tuple of `(n_heads, seq_len, kv_len)`

---

#### Text Generation

##### `generate(prompt: str, max_tokens: int = 100, add_special: bool = True, stop_on_eos: bool = True) -> str`

Generate text from prompt.

**Parameters**:
- `prompt`: Input text
- `max_tokens`: Maximum tokens to generate (**must be ≥ 1**)
- `add_special`: Add BOS/EOS tokens
- `stop_on_eos`: Stop generation on EOS token

**Raises**:
- `ValueError`: If max_tokens < 1
- `RuntimeError`: If model not loaded

**Example**:
```python
text = engine.generate("Once upon a time", max_tokens=50)
print(text)
```

##### `generate_stream(prompt: str, max_tokens: int = 100, add_special: bool = True, stop_on_eos: bool = True)`

Generate text with streaming output (iterator).

**Yields**: `TokenEvent` objects

**Example**:
```python
for event in engine.generate_stream("Tell me a story", max_tokens=50):
    print(event.token_str, end='', flush=True)
```

---

#### Step-by-Step Control

##### `set_prompt(prompt: str, mode: PromptMode = PromptMode.COMPLETION, add_special: bool = True) -> bool`

Set prompt for step-by-step generation.

##### `prefill() -> bool`

Process the prompt (KV cache population).

##### `step() -> Optional[TokenEvent]`

Generate next token.

**Returns**: `TokenEvent` or `None` if generation complete

**Example**:
```python
engine.set_prompt("Once upon a time")
engine.prefill()

for i in range(20):
    event = engine.step()
    if event is None:
        break
    print(event.token_str, end='', flush=True)
```

##### `pause() -> None`

Pause generation (preserves KV cache).

##### `resume() -> None`

Resume generation.

##### `reset() -> None`

Reset engine state and clear KV cache.

---

#### Tokenization

##### `tokenize(text: str, add_special: bool = True) -> List[int]`

Convert text to token IDs.

**Parameters**:
- `text`: Input text
- `add_special`: Add BOS/EOS tokens

**Example**:
```python
tokens = engine.tokenize("Hello world")
print(tokens)  # [15496, 1917]
```

##### `detokenize(tokens: List[int]) -> str`

Convert token IDs to text.

```python
text = engine.detokenize([15496, 1917])
print(text)  # "Hello world"
```

##### `token_to_str(token_id: int) -> str`

Convert single token ID to string.

---

#### Configuration

##### `configure_sampler(temperature: float = None, top_p: float = None, top_k: int = None, repeat_penalty: float = None) -> None`

Update sampling parameters.

**Parameters**: All optional, only provided values are updated

**Raises**:
- `ValueError`: If temperature < 0, top_p not in [0,1], or top_k < 0

**Example**:
```python
engine.configure_sampler(temperature=0.9, top_k=50)
```

##### `get_config() -> EngineConfig`

Get current engine configuration.

---

#### Properties

##### `state` (read-only)

Current engine state (`State` enum).

```python
if engine.state == neuroscope.State.DECODE:
    print("Generating...")
```

##### `stats` (read-only)

Get inference statistics (`InferenceStats`).

```python
stats = engine.stats
print(f"Speed: {stats.tokens_per_second:.2f} tok/s")
```

---

## Usage Examples

### 1. Activation Analysis

```python
import neuroscope
import numpy as np

engine = neuroscope.Engine("llama-3-8b.gguf", n_ctx=2048)

# Get activations for different prompts
prompts = [
    "I am happy because",
    "I am sad because",
    "I am angry because"
]

activation_matrix = []
for prompt in prompts:
    engine.forward(prompt)
    acts = engine.get_activations(15)  # Layer 15
    activation_matrix.append(acts)

activation_matrix = np.array(activation_matrix)
print(activation_matrix.shape)  # (3, 4096)

# Compute cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(activation_matrix)
print(sim)
```

### 2. Logit Lens Visualization

```python
import neuroscope
import matplotlib.pyplot as plt

engine = neuroscope.Engine("llama-3-8b.gguf")
engine.forward("The meaning of life is")

# Get predictions from each layer
layer_predictions = []
for layer in range(32):
    lens_logits = engine.logit_lens(layer)
    top_token = lens_logits.argmax()
    token_str = engine.detokenize([top_token])
    layer_predictions.append(token_str)

# Visualize
plt.figure(figsize=(12, 6))
plt.plot(layer_predictions, 'o-')
plt.xlabel("Layer")
plt.ylabel("Top Prediction")
plt.title("Logit Lens: Prediction Evolution Across Layers")
plt.show()
```

### 3. Activation Steering

```python
import neuroscope
import numpy as np

engine = neuroscope.Engine("llama-3-8b.gguf")

# Load pre-computed steering vector
steering_vec = np.load("happy_direction.npy")

# Apply to layer 15
engine.apply_steering(15, steering_vec, strength=3.0)

# Generate
text = engine.generate("I feel", max_tokens=30)
print(text)

# Clear intervention
engine.clear_interventions()
```

### 4. Step-by-Step Generation

```python
import neuroscope

engine = neuroscope.Engine("llama-3-8b.gguf")

engine.set_prompt("Once upon a time")
engine.prefill()

# Generate token by token and record activations
activations_history = []
for i in range(20):
    event = engine.step()
    if event is None:
        break
    
    # Record activations for each token
    acts = engine.get_activations(15)  # Layer 15
    activations_history.append(acts.copy())
    
    print(event.token_str, end='', flush=True)

print(f"\nRecorded {len(activations_history)} activation snapshots")
```

### 5. Chat Template Usage (Llama-3)

```python
import neuroscope

engine = neuroscope.Engine("llama-3-8b-instruct.gguf", n_ctx=4096)

# Define system and user prompts
system_prompt = "You are a helpful AI assistant."
user_prompt = "What is the meaning of life?"

# Build Llama-3 chat template
# Format: <|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
chat_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

# Important: Use add_special=False because prompt already contains <|begin_of_text|>
engine.forward(chat_prompt, add_special=False)

# Get model prediction
top_tokens = engine.get_top_tokens(k=5)
print("Model's predicted next tokens:", top_tokens)

# Generate response
response = engine.generate(chat_prompt, max_tokens=200, add_special=False)
print(response)
```

**Llama-3 Special Tokens Reference**:
| Token | ID | Purpose |
|-------|-----|---------|
| `<\|begin_of_text\|>` | 128000 | BOS, conversation start |
| `<\|end_of_text\|>` | 128001 | EOS, conversation end |
| `<\|start_header_id\|>` | 128006 | Role marker start |
| `<\|end_header_id\|>` | 128007 | Role marker end |
| `<\|eot_id\|>` | 128009 | End of Turn |

---

## FAQ

### CUDA DLL Loading Failed

On Windows, if you encounter CUDA DLL loading issues:

```python
import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")
import neuroscope
```

Or set the `CUDA_PATH` environment variable.

### Out of Memory

For large models, reduce GPU layers:

```python
engine = neuroscope.Engine("model.gguf", n_gpu_layers=20)  # Only offload 20 layers
```

### Batch Activation Returns All Zeros

If `batch_get_activations()` returns all zeros, check:

1. **Version**: Ensure using v1.2.3 or higher (fixes ubatch slicing issue)
2. **RTX 50 Series**: Set `GGML_CUDA_DISABLE_GRAPHS=1` environment variable
3. **Batch Size**: Batches > 64 prompts are automatically split - this is normal behavior

```python
# Verify activations are working
result = engine.batch_forward(prompts)
acts = engine.batch_get_activations(15)
norms = np.linalg.norm(acts, axis=1)
print(f"Activation norms: {norms}")  # Should all be non-zero
```

### Batch vs Single Mode Inconsistency

Batch and individual processing should have cosine similarity > 0.999. If differences are large:

1. Ensure using same `add_special` parameter
2. Batch processing uses parallel KV cache, may have small numerical differences (this is normal)
3. For scenarios requiring exact consistency, use individual `forward()` loop

### Context Length

If prompt is too long, it will be truncated. Increase `n_ctx`:

```python
engine = neuroscope.Engine("model.gguf", n_ctx=8192)
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│              Python Layer                    │
│   import neuroscope                          │
│   engine = neuroscope.Engine(...)            │
└─────────────────┬───────────────────────────┘
                  │ pybind11
                  ▼
┌─────────────────────────────────────────────┐
│              CoreEngine (C++)                │
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

## Logit Lens: Principles and Usage

### Principles

The core idea of Logit Lens is to project intermediate layer hidden states $h^{(l)}$ directly through the unembedding matrix $W_U$ to vocabulary space:

$$\text{logits}^{(l)} = h^{(l)} \cdot W_U^T$$

Where:
- $h^{(l)} \in \mathbb{R}^{d_{model}}$ is the hidden state at layer $l$ (last token)
- $W_U \in \mathbb{R}^{V \times d_{model}}$ is the unembedding (lm_head) weight matrix
- $V$ is vocabulary size, $d_{model}$ is hidden dimension

This allows us to "peek" at the model's prediction tendencies at each layer, observing how information flows between layers.

### Usage Example

```python
engine.forward("The meaning of life is")

# View predictions at each layer
for layer in range(0, 32, 4):
    lens_logits = engine.logit_lens(layer)
    top_token = lens_logits.argmax()
    print(f"Layer {layer:2d}: {engine.detokenize([top_token])}")

# Get unembedding weights for custom analysis
W_unembed = engine.get_unembed_weights()  # (vocab_size, hidden_dim)
```

### Performance Notes

> ✅ **All model types supported from v1.1.2**: F16/F32 and Q4/Q8 quantized models all use GPU acceleration.

| Model Type | `logit_lens(layer)` | `logit_lens_all()` | Notes |
|------------|---------------------|-------------------|-------|
| F16/F32 | ~0.002s | ~0.045s | Native FP16 weights |
| Q4_K_S | ~0.058s | ~0.045s | Pre-dequantized to FP16 |
| Q8_0 | ~0.003s | ~0.047s | Pre-dequantized to FP16 |

- **All models**: Unified using `cublasHgemm` half-precision matrix multiplication
- **Quantized models**: Automatically pre-dequantize unembedding weights to FP16 GPU cache during `loadModel()`
- **Memory overhead**: `n_vocab × n_embd × 2 bytes` (~1GB for Llama-3-8B)

### API Reference

| Method | Return Type | Description |
|--------|------------|-------------|
| `logit_lens(layer)` | `np.ndarray (V,)` | Single layer Logit Lens |
| `logit_lens_all()` | `np.ndarray (L, V)` | All layers Logit Lens |
| `batch_logit_lens(layer)` | `np.ndarray (B, V)` | Batch Logit Lens |
| `batch_logit_lens_all()` | `np.ndarray (B, L, V)` | Batch all layers Logit Lens |
| `get_unembed_weights()` | `np.ndarray (V, H)` | Get unembedding weights |

---

## Version History

### v1.2.3 (2026-01-29)

**Batch Activation Capture Fix**:
- Fixed `batch_get_activations()` returning all zeros for large batches (>64 prompts)
- Fixed activation mapping errors caused by llama.cpp ubatch slicing
- Added `current_chunk_logits_map` to handle sparse output from last layer
- Added `ubatch_logits_offset` atomic variable to track last layer ubatch offset

**Technical Details**:
- llama.cpp's last layer (`l_out-{N-1}`) only outputs tokens where `batch.logits[i]=1`
- Intermediate layers use `current_chunk_to_flat_map` to map all tokens
- Last layer uses separate `current_chunk_logits_map` for sparse tokens
- `batchForwardLarge()` now correctly handles activation merging across sub-batches

### v1.1.2 (2026-01-16)

**Q4/Q8 Quantized Model Support**:
- Logit Lens now supports all GGUF quantization formats (Q4_K, Q5_K, Q6_K, Q8_0, etc.)
- Automatically pre-dequantizes unembedding weights to FP16 GPU cache during `loadModel()`
- GPU memory overhead: ~1GB (Llama-3-8B)
- Performance: Q4_K_S ~0.05s, Q8_0 ~0.05s (32-layer logit_lens_all)

### v1.1.1 (2026-01-16)

**cuBLAS GPU Acceleration**:
- F16/F32 models Logit Lens fully GPU-accelerated
- `logit_lens()`: Subsequent calls ~0.002s (250x faster)
- `logit_lens_all()`: Subsequent calls ~0.045s (356x faster)

### v1.1.0 (2026-01-16)

**Batch Processing API**:
- Added `batch_forward(prompts, padding)` - Batch forward pass
- Added `batch_get_logits()` - Get batch logits
- Added `batch_get_activations(layer)` - Get batch activations
- Added `batch_get_all_activations()` - Get all layer batch activations

**Logit Lens API**:
- Added `get_unembed_weights()` - Get unembedding weights
- Added `logit_lens(layer)` - Single layer Logit Lens
- Added `logit_lens_all()` - All layers Logit Lens
- Added `batch_logit_lens(layer)` - Batch Logit Lens
- Added `batch_logit_lens_all()` - Batch all layers Logit Lens

**Batch Steering API**:
- Added `batch_apply_steering(layer, direction, strength)` - Batch steering
- Added `batch_apply_mask(layer, mask)` - Batch mask

**Attention Map API**:
- Fixed `get_attention(layer, head)` to return correct shape
  - Single head: `(seq_len, kv_len)`
  - All heads: `(n_heads, seq_len, kv_len)`
- Added `get_attention_info(layer)` - Get attention metadata
- Added `disable_flash_attention` parameter - Disable Flash Attention to capture attention

**Testing**: 114 tests passed, 12 skipped (Logit Lens requires F16 models)

### v1.0.3 (2026-01-16)

- Added `forward_append()` method for KV cache continuation mode
- Added KV Cache coherence test suite (11 tests)
- Support for multi-turn conversations and streaming input

### v1.0.2 (2026-01-16)

- Fixed `max_tokens=-1` boundary check: negative values now correctly raise `ValueError`
- Added numerical correctness test suite (15 tests, verified against HuggingFace)
- Added performance benchmark scripts (`benchmarks/`)

### v1.0.1 (2026-01-16)

- Added parameter validation, invalid parameters now raise explicit exceptions
- `apply_mask()` / `apply_steering()`: Invalid layer index raises `IndexError`
- `generate()`: Invalid max_tokens/temperature/top_p/top_k raises `ValueError`
- `Engine()` constructor: Invalid sampling parameters raise `ValueError`
- Added `configure_sampler()` method documentation

### v1.0.0 (2026-01-16)

- Initial release
- GGUF model loading support
- Activation extraction (numpy arrays)
- Activation steering and masking
- Step-by-step generation control
- Chat template support

---

## Exception Reference

| Exception Type | Trigger Condition |
|---------------|------------------|
| `ValueError` | Invalid parameter values (e.g., temperature < 0, top_p > 1, max_tokens < -1) |
| `IndexError` | Layer index out of range (e.g., `apply_mask(99999, mask)`) |
| `RuntimeError` | Calling inference methods without loaded model, or model loading failure |
