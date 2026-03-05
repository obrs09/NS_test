# Research Proposals: Aesthetic Concepts in LLM Internal Representations

> Using NeuroScope to probe how large language models encode, process, and generate aesthetic judgments.

---

## Background

Aesthetic judgment — the ability to evaluate beauty, elegance, harmony, and taste — is a deeply human capacity that LLMs appear to exhibit at the behavioral level (e.g., writing poetry, rating art, discussing design principles). But how are these abstract concepts actually represented inside the model? Are there dedicated "aesthetic neurons"? Do aesthetic representations form coherent subspaces? Can we steer a model to be more or less "aesthetic" in its outputs?

NeuroScope provides the perfect toolkit: **activation extraction** across all layers, **logit lens** for tracking concept formation, **neuron ablation** for causal testing, and **activation steering** for controlled manipulation.

---

## Proposal 1: Mapping the Aesthetic Subspace

### Research Question
Do aesthetic concepts (beauty, elegance, harmony, ugliness, crudeness) occupy a coherent linear subspace in the model's activation space, analogous to how translation direction occupies a subspace?

### Methodology

**Phase 1: Contrastive Activation Collection**

1. **Dataset Construction** — Create paired prompts that differ only in aesthetic valence:
   - Positive: *"Describe a beautiful sunset over the ocean"*
   - Negative: *"Describe an ugly industrial wasteland"*
   - Neutral: *"Describe a parking lot"*
   - Design 200+ pairs across domains: visual (paintings, landscapes), auditory (music, sounds), literary (prose, poetry), design (architecture, fashion), gustatory (food, wine)

2. **Activation Collection** — For each prompt:
   ```python
   engine.reset()
   engine.forward(templated_prompt, add_special=True)
   acts_positive[i] = engine.get_all_activations()    # [n_layers, n_embd]
   ```
   Collect `acts_positive`, `acts_negative`, `acts_neutral` across all 200+ pairs.

3. **Subspace Extraction** — Compute the "aesthetic direction" per layer:
   ```
   aesthetic_direction[layer] = mean(acts_positive[layer]) - mean(acts_negative[layer])
   ```
   Apply PCA on the difference vectors to find the top-k aesthetic dimensions.

**Phase 2: Validation**

4. **Held-out Classification** — Project held-out prompts onto the aesthetic subspace. Can we predict aesthetic valence from activations alone? (ROC-AUC metric)

5. **Cross-domain Transfer** — Train the aesthetic direction on visual prompts, test on musical prompts. Does the subspace generalize across sensory modalities?

6. **Logit Lens Tracking** — Use `engine.logit_lens_all()` to track when aesthetic tokens (beautiful, elegant, ugly, crude) emerge in the layer-by-layer token predictions. At which layer does the model "decide" on aesthetic tone?

### Datasets
- **Custom paired prompts** (200+ pairs) across 5 aesthetic domains
- **WikiArt descriptions** — art criticism with known aesthetic judgments
- **Yelp/Amazon reviews** with aesthetic adjectives (filtered subset)
- **Poetry vs. Technical writing** — intrinsic aesthetic contrast

### Metrics
- **Subspace coherence**: Explained variance ratio of top-k PCA components of difference vectors
- **Classification accuracy**: ROC-AUC on held-out aesthetic valence prediction
- **Cross-domain transfer**: Accuracy drop when training on domain A, testing on domain B
- **Logit lens emergence layer**: First layer where aesthetic tokens appear in top-10 predictions

### NeuroScope API Usage
- `engine.forward()` + `engine.get_all_activations()` — contrastive activation pairs
- `engine.batch_forward()` + `engine.batch_get_all_activations()` — efficient batch collection
- `engine.logit_lens_all()` — track aesthetic concept emergence across layers
- `ActivationCollector` — online mean/std statistics

---

## Proposal 2: Aesthetic Neurons — Causal Identification via Ablation

### Research Question
Are there individual neurons (or small neuron groups) that are **causally necessary** for aesthetic judgment? If we ablate them, does the model lose its ability to distinguish beautiful from ugly?

### Methodology

**Phase 1: Candidate Identification**

1. Collect activation statistics for aesthetic vs. non-aesthetic prompts (reuse Proposal 1 data).
2. Identify **high-variance neurons** — neurons whose activation differs most between aesthetic-positive and aesthetic-negative prompts:
   ```
   aesthetic_selectivity[layer][neuron] = |mean_positive - mean_negative| / std_pooled
   ```
3. Rank neurons by selectivity score. Take top-50 candidates per layer.

**Phase 2: Ablation Testing**

4. For each candidate neuron (layer, index):
   ```python
   mask = np.ones(n_embd)
   mask[neuron_idx] = 0.0
   engine.apply_mask(layer, mask)
   output = engine.generate(aesthetic_prompt, max_tokens=256)
   engine.clear_interventions()
   ```
5. Evaluate output change:
   - Does the model still use aesthetic vocabulary? (aesthetic word frequency in output)
   - Does output sentiment shift? (sentiment classifier score)
   - Does output become more generic/bland?

**Phase 3: Group Ablation**

6. Ablate the top-N aesthetic neurons simultaneously. Is there a "tipping point" where aesthetic capacity collapses?
7. Compare with random neuron ablation of same count — is the effect specific to aesthetic neurons?

### Datasets
- Aesthetic judgment prompts: *"Rate the beauty of: [description]. Explain your reasoning."*
- Creative writing prompts: *"Write a beautiful poem about [topic]"* vs. *"Write a factual report about [topic]"*
- 100 test prompts × 36 layers × top-50 neurons = systematic ablation grid

### Metrics
- **Aesthetic vocabulary density**: Count of aesthetic adjectives (beautiful, elegant, graceful, stunning, hideous, crude, etc.) per 100 tokens in generated output
- **Sentiment shift**: Pre/post ablation sentiment score (using a classifier or COMET-like model)
- **Output quality**: BLEU/ROUGE between ablated and normal output (lower = more impact)
- **Perplexity change**: Does ablation increase perplexity on aesthetic text more than on technical text?

### NeuroScope API Usage
- `engine.apply_mask(layer, mask)` — single and group neuron ablation
- `engine.generate()` — output generation before/after ablation
- `engine.get_activations(layer)` — verify ablation is effective
- `engine.clear_interventions()` + `engine.reset()` — clean state between tests

---

## Proposal 3: Steering Aesthetic Tone via Activation Engineering

### Research Question
Can we use activation steering to make a model's outputs more or less aesthetically oriented? Can we transfer "aesthetic style" (e.g., make technical writing poetic, or make poetry clinical)?

### Methodology

**Phase 1: Steering Vector Construction**

1. Compute the aesthetic steering vector (from Proposal 1):
   ```
   steer_vec = mean(aesthetic_positive_activations) - mean(aesthetic_negative_activations)
   ```
   Normalize to unit length.

2. **Layer Selection** — Based on Proposal 1's logit lens analysis, select the layer(s) where aesthetic concepts crystallize (likely mid-to-late layers, ~layer 15-25 for a 36-layer model).

**Phase 2: Steering Experiments**

3. **Aesthetic Amplification**: Apply positive steering to neutral prompts:
   ```python
   engine.apply_steering(target_layer, steer_vec, strength=+α)
   output_beautiful = engine.generate(neutral_prompt, max_tokens=512)
   ```
   Sweep α ∈ {0.5, 1.0, 2.0, 5.0, 10.0, 20.0}

4. **Aesthetic Suppression**: Apply negative steering:
   ```python
   engine.apply_steering(target_layer, steer_vec, strength=-α)
   output_bland = engine.generate(neutral_prompt, max_tokens=512)
   ```

5. **Style Transfer**: Apply steering to domain-mismatched prompts:
   - Technical prompt + aesthetic steering → "poetic technical writing"
   - Poetry prompt + anti-aesthetic steering → "clinical poetry"

**Phase 3: Multi-Layer Steering**

6. Apply steering at multiple layers simultaneously. Does combining early + late layer steering produce different effects than single-layer?
7. Sweep (layer, strength) grid to find the optimal "aesthetic dial."

### Datasets
- **Neutral prompts**: 100 factual/descriptive prompts (*"Describe the process of making bread"*)
- **Mixed-domain**: 50 technical, 50 creative, 50 conversational prompts
- **Human evaluation set**: 30 selected outputs for human aesthetic preference rating

### Metrics
- **Aesthetic word ratio**: aesthetic_adjective_count / total_token_count
- **Style classifier accuracy**: Train a simple classifier to predict "aesthetic" vs. "clinical" style; measure how steering shifts the classification
- **Coherence**: Does the output remain coherent? (perplexity, self-BLEU)
- **Human preference**: A/B comparison — does steered output sound more/less beautiful? (small-scale annotation)
- **COMET scores** (if using translation context): Does aesthetic steering degrade translation accuracy?

### NeuroScope API Usage
- `engine.apply_steering(layer, vector, strength)` — core steering mechanism
- `engine.generate()` — steered generation
- `engine.get_logits()` — compare top token predictions before/after steering
- `engine.logit_lens(layer)` — observe how steering changes intermediate predictions

---

## Proposal 4: Aesthetic Judgment as an Emergent Computation — Layer-by-Layer Analysis

### Research Question
How does the model build up an aesthetic judgment across its layers? Is there a discrete "aesthetic judgment" computation at a specific layer, or does it emerge gradually?

### Methodology

**Phase 1: Probing at Every Layer**

1. Design prompts that require **explicit aesthetic judgment**:
   - *"Is this sentence beautiful or ugly: '[sentence]'. Answer: "*
   - Use 200 sentences with known human aesthetic ratings (from poetry/prose corpora).

2. After `engine.forward(prompt)`, extract activations at every layer and train **linear probes**:
   ```python
   for layer in range(n_layers):
       acts = engine.get_activations(layer)  # [n_embd]
       # ... accumulate for probe training
   ```
   Train a linear classifier that predicts human aesthetic rating from layer-k activations.

3. Plot **probing accuracy vs. layer** — this reveals where aesthetic information becomes linearly separable.

**Phase 2: Logit Lens for Aesthetic Tokens**

4. For prompts like *"This poem is [MASK]"*, use logit lens at each layer:
   ```python
   logits_all = engine.logit_lens_all()  # [n_layers, n_vocab]
   ```
   Track probability of aesthetic tokens (beautiful, ugly, elegant, crude) across layers.

5. Compare trajectories for genuinely beautiful vs. ugly input sentences — at which layer do the probability curves diverge?

**Phase 3: Attention Pattern Analysis**

6. For aesthetic judgment prompts, examine which input tokens the model attends to:
   ```python
   attn = engine.get_attention(layer, head)
   ```
   Do specific heads specialize in attending to aesthetic features (sensory adjectives, structural patterns, emotional words)?

### Datasets
- **Prose beauty ratings**: Excerpts from literary competitions with judge scores
- **Poetry corpus**: Poems with human beauty ratings (can use crowdsourced data)
- **Controlled sentences**: Minimal pairs differing only in aesthetic quality
  - *"The crystal vase caught the morning light"* vs. *"The plastic cup sat on the table"*
- **Cross-lingual**: Same content in Chinese (诗词 vs. 公文) and English — does aesthetic judgment emerge at the same layers?

### Metrics
- **Probing accuracy curve**: Accuracy at each layer (expect sigmoid-like growth)
- **Emergence layer**: First layer where probing accuracy exceeds chance + 2σ
- **Logit lens divergence layer**: Layer where P(beautiful|beautiful_input) - P(beautiful|ugly_input) > threshold
- **Attention entropy**: How concentrated is attention on aesthetic features vs. spread uniformly?

### NeuroScope API Usage
- `engine.forward()` + `engine.get_activations(layer)` — layer-wise probing
- `engine.logit_lens_all()` — full layer-by-layer prediction tracking
- `engine.get_attention(layer, head)` — attention pattern analysis
- `engine.batch_forward()` + `engine.batch_get_activations(layer)` — efficient batch probing

---

## Proposal 5: The Geometry of Aesthetic Spaces — Beauty, Symmetry, and Harmony

### Research Question
Do abstract aesthetic principles like symmetry, harmony, balance, and proportion have distinct geometric signatures in activation space? Are they organized hierarchically (e.g., "beauty" = composition of "symmetry" + "harmony" + "color")?

### Methodology

**Phase 1: Concept Vector Extraction**

1. For each aesthetic sub-concept, create contrastive prompt pairs:
   - **Symmetry**: *"A perfectly symmetric butterfly"* vs. *"An asymmetric broken branch"*
   - **Harmony**: *"Colors blending in perfect harmony"* vs. *"Clashing neon colors"*
   - **Proportion**: *"The golden ratio in architecture"* vs. *"A disproportionate building"*
   - **Rhythm**: *"The steady rhythm of ocean waves"* vs. *"Arrhythmic random noise"*
   - **Simplicity**: *"Elegant minimalist design"* vs. *"Cluttered overwhelming decoration"*

2. Extract concept vectors:
   ```
   symmetry_vec[layer] = mean(acts_symmetric) - mean(acts_asymmetric)
   harmony_vec[layer] = mean(acts_harmonious) - mean(acts_clashing)
   ...
   ```

**Phase 2: Geometric Analysis**

3. **Cosine similarity matrix** between all aesthetic concept vectors — are they orthogonal, aligned, or clustered?
4. **Composition test**: Is `beauty_vec ≈ α·symmetry_vec + β·harmony_vec + γ·proportion_vec`? Fit linear regression and measure R².
5. **Hierarchical clustering**: Do aesthetic concepts form a dendrogram? (e.g., visual aesthetics cluster separately from auditory aesthetics)

**Phase 3: Manipulation**

6. **Selective steering**: Apply only `symmetry_vec` steering — does the output become more symmetric in structure?
7. **Cocktail steering**: Combine multiple concept vectors with different weights — can we "mix" aesthetic properties?
   ```python
   combined_vec = 0.5 * symmetry_vec + 0.3 * harmony_vec + 0.2 * simplicity_vec
   engine.apply_steering(layer, combined_vec, strength=5.0)
   ```

### Datasets
- **Custom contrastive pairs** (50+ per concept × 5 concepts = 250+ pairs)
- **Design principles dataset**: Real-world design critique with labeled aesthetic properties
- **Music descriptions**: Annotated with harmony, rhythm, etc.

### Metrics
- **Inter-concept cosine similarity**: Pairwise similarity ∈ [-1, 1] for all aesthetic concept vectors
- **Decomposition R²**: How much of the "beauty" vector is explained by sub-concept vectors?
- **Cluster silhouette score**: Quality of concept clustering
- **Steering specificity**: When steering with `symmetry_vec`, does output symmetry increase without changing harmony? (measured by separate classifiers)

### NeuroScope API Usage
- `engine.forward()` + `engine.get_all_activations()` — concept vector extraction
- `engine.apply_steering(layer, vector, strength)` — selective and cocktail steering
- `engine.generate()` — steered output generation
- `engine.logit_lens(layer)` — track concept-specific predictions

---

## Proposal 6: Cross-Cultural Aesthetic Representations

### Research Question
Does the model encode different aesthetic standards for different cultures? For example, does it represent Japanese wabi-sabi (侘寂, beauty of imperfection) differently from Western classical beauty (symmetry, proportion)? Are these represented in the same or different subspaces?

### Methodology

**Phase 1: Cultural Aesthetic Prompt Design**

1. Design prompts invoking different cultural aesthetic frameworks:
   - **Western Classical**: symmetry, golden ratio, order (*"A Parthenon-like temple with perfect columns"*)
   - **Japanese Wabi-sabi**: imperfection, transience, simplicity (*"A cracked tea bowl with gold repair"*)
   - **Chinese Shanshui (山水)**: natural balance, emptiness, qi (*"Misty mountains with a lone pine"*)
   - **Islamic Geometric**: infinite patterns, mathematical beauty (*"Intricate tilework of the Alhambra"*)
   - **Minimalism**: reduction, essential form (*"A single stone in a zen garden"*)

2. Collect activations for each cultural aesthetic category (50+ prompts each).

**Phase 2: Subspace Comparison**

3. Extract per-culture aesthetic directions.
4. Measure **cross-cultural cosine similarity** — is there a universal "beauty" direction, or are they orthogonal?
5. **PCA of all cultural aesthetic activations** — do cultures cluster, spread, or overlap?

**Phase 3: Language × Culture Interaction**

6. Present the **same aesthetic concept** in different languages:
   - Wabi-sabi described in English vs. Japanese (侘び寂び)
   - Shanshui described in English vs. Chinese (山水意境)
   - Test: Does the aesthetic subspace shift with language, or is it language-invariant?

7. Use `engine.batch_forward()` for efficient multi-language comparison.

### Datasets
- **Custom cultural aesthetic prompts**: 50 per culture × 5 cultures × 2 languages = 500 prompts
- **Art description corpus**: WikiArt with culture labels
- **Poetry**: Haiku (Japanese), Tang poetry (Chinese), Sonnets (Western), Ghazals (Persian)

### Metrics
- **Cross-cultural cosine similarity matrix**: 5×5 matrix of pairwise similarities
- **Universal aesthetic component**: Variance explained by the first PC across all cultures
- **Language invariance**: Cosine similarity of same-concept vectors across languages
- **Cluster purity**: When clustering activations, do they cluster by culture, by language, or by aesthetic quality?

### NeuroScope API Usage
- `engine.batch_forward()` + `engine.batch_get_all_activations()` — efficient multi-prompt collection
- `engine.apply_chat_template()` — proper multi-language templating
- `engine.logit_lens_all()` — track cultural-aesthetic token emergence
- `engine.apply_steering()` — cross-cultural aesthetic transfer (e.g., apply wabi-sabi vector to classical art prompts)

---

## Proposal 7: Aesthetic Preference vs. Aesthetic Description — Representation Differences

### Research Question
Does the model use different internal representations when **describing** something beautiful vs. when **evaluating** whether something is beautiful? Is aesthetic perception separate from aesthetic judgment?

### Methodology

**Phase 1: Task-Contrastive Collection**

1. Design prompt pairs for the same content but different tasks:
   - **Description**: *"Describe this garden: [garden description]"*
   - **Evaluation**: *"Rate the beauty of this garden on a scale of 1-10: [same garden description]"*
   - **Generation**: *"Write a beautiful description of a garden"*

2. Collect activations for all three tasks across 100 garden/scene descriptions.

**Phase 2: Representation Analysis**

3. Compute **task-specific aesthetic directions**:
   ```
   describe_aesthetic_vec = mean(describe_beautiful) - mean(describe_ugly)
   evaluate_aesthetic_vec = mean(evaluate_beautiful) - mean(evaluate_ugly)
   generate_aesthetic_vec = mean(generate_beautiful) - mean(generate_neutral)
   ```

4. Compare the three vectors:
   - Are they aligned? (cosine similarity)
   - At which layers do they diverge? (layer-wise cosine similarity curve)
   - Does the model share early aesthetic features but diverge in later task-specific layers?

**Phase 3: Cross-Task Transfer**

5. Steering test: Apply `evaluate_aesthetic_vec` during a description task — does it make descriptions more evaluative?
6. Apply `generate_aesthetic_vec` during evaluation — does it bias ratings upward?

### Datasets
- **Scenes/objects** rated for beauty by humans (100+ items)
- **Three prompt templates** per item (describe, evaluate, generate)
- **Cross-validation**: 80/20 train/test split for direction extraction

### Metrics
- **Task vector alignment**: Cosine similarity between description, evaluation, and generation aesthetic vectors per layer
- **Divergence layer**: First layer where task vectors become significantly different (cosine < 0.8)
- **Cross-task steering effect**: Does applying one task's vector change behavior in another task's context?
- **Representation similarity analysis (RSA)**: Correlation between the similarity structure of items across tasks

### NeuroScope API Usage
- `engine.forward()` + `engine.get_all_activations()` — per-task activation collection
- `engine.apply_steering()` — cross-task transfer experiments
- `engine.generate()` — observe behavioral effects
- `engine.logit_lens()` — track task-specific prediction differences

---

## Implementation Priority & Resource Estimates

| Proposal | Difficulty | GPU Hours* | Key Insight | Recommended Order |
|----------|-----------|-----------|-------------|-------------------|
| 1. Aesthetic Subspace | ★★☆ | ~4h | Foundation for all others | **1st** |
| 4. Layer-by-Layer | ★★☆ | ~6h | When aesthetics emerge | **2nd** |
| 2. Aesthetic Neurons | ★★★ | ~12h | Causal mechanism | **3rd** |
| 5. Geometry | ★★★ | ~8h | Structure of beauty | **4th** |
| 3. Steering | ★★☆ | ~6h | Practical application | **5th** |
| 6. Cross-Cultural | ★★★★ | ~16h | Most novel finding | **6th** |
| 7. Perception vs. Judgment | ★★★ | ~10h | Theoretical depth | **7th** |

*Estimated on GB10 with Qwen3-8B Q4_K_M. Includes data preparation, forward passes, and analysis.

## Shared Infrastructure

All proposals share common building blocks that can be implemented once:

```python
# Core utilities to build
class AestheticCollector(ActivationCollector):
    """Extended collector with aesthetic-specific statistics"""
    def compute_contrastive_direction(self, positive_acts, negative_acts) -> np.ndarray
    def compute_concept_vector(self, concept_acts, baseline_acts) -> np.ndarray
    def project_onto_subspace(self, activations, directions) -> np.ndarray

class AestheticEvaluator:
    """Evaluate aesthetic quality of generated text"""
    def aesthetic_word_density(self, text: str) -> float
    def sentiment_score(self, text: str) -> float
    def style_classify(self, text: str) -> str  # "aesthetic" | "neutral" | "clinical"

class PromptFactory:
    """Generate contrastive prompt pairs for aesthetic experiments"""
    def make_pair(self, concept: str, domain: str) -> tuple[str, str]
    def make_cultural_pair(self, culture: str, concept: str) -> tuple[str, str]
    def make_task_set(self, item: str) -> dict[str, str]  # describe, evaluate, generate
```

## Quick-Start Experiment (Proof of Concept)

Before committing to a full proposal, run this 30-minute validation:

```python
# 1. Collect 20 aesthetic + 20 non-aesthetic activations
positive_prompts = ["Describe the beauty of a sunset", "Explain what makes a painting elegant", ...]
negative_prompts = ["Describe a rusty factory", "Explain what makes a room cluttered", ...]

positive_acts = []
for p in positive_prompts:
    engine.reset()
    msg = [{"role": "user", "content": p}]
    template = engine.apply_chat_template(msg) + NOTHINK_SUFFIX
    engine.forward(template, add_special=True)
    positive_acts.append(engine.get_all_activations())  # list of [n_layers][n_embd]

# Stack: [20, n_layers, n_embd]
positive_acts = np.array(positive_acts)
negative_acts = np.array(negative_acts)  # same collection process

# 2. Compute aesthetic direction
aesthetic_dir = positive_acts.mean(axis=0) - negative_acts.mean(axis=0)  # [n_layers, n_embd]

# 3. Quick validation: project held-out samples
# If projection sign predicts valence with >70% accuracy → green light for full study
```
