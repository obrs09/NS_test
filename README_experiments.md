# NeuroScope Interpretability Experiments

> Probing the internal representations of large language models using [NeuroScope](../NeuroScope/README.md) — a high-performance CUDA inference engine with activation extraction, logit lens, neuron ablation, and activation steering.

---

## Project Overview

This repository contains a series of mechanistic interpretability experiments that use NeuroScope's Python API to look inside LLMs at inference time. The work spans two major research threads:

| Experiment | Model | Status | Notebook |
|---|---|---|---|
| **Translation Neuron Analysis** | LLaMA-3-8B-Instruct Q4_K_S | ✅ Complete | `translation_neuron.ipynb` |
| **Aesthetic Subspace Mapping** | Qwen3-8B Q4_K_M | ✅ Complete | `aesthetic_subspace.ipynb` |
| **Layer-by-Layer Aesthetic Emergence** | Qwen3-8B Q4_K_M | ✅ Complete | `aesthetic_emergence.ipynb` |
| **Cross-Lingual Aesthetic Universality** | Qwen3-8B Q4_K_M | ✅ Complete | `cross_lingual_aesthetics.ipynb` |
| **Aesthetic Neurons via Ablation** | Qwen3-8B Q4_K_M | ✅ Complete | `aesthetic_neurons.ipynb` |
| ↳ Cross-Lingual Neuron Selectivity | Qwen3-8B Q4_K_M | ✅ Complete | `aesthetic_neurons.ipynb` (§11) |
| **Real Poetry Validation** | Qwen3-8B Q4_K_M | ✅ Complete | `poetry_aesthetics.ipynb` (Exp 6) |
| **Tang vs Song Dynasty Aesthetics** | Qwen3-8B Q4_K_M | ✅ Complete | `poetry_aesthetics.ipynb` (Exp 7) |
| **Bilingual Aesthetic Distillation** | Qwen3-8B Q4_K_M | ✅ Complete | `poetry_aesthetics.ipynb` (Exp 8) |
| **Poetry Replication** (Exp 2+3+5 redo) | Qwen3-8B Q4_K_M | ✅ Complete | `poetry_replication.ipynb` (Exp 9) |
| **Matched-Format Replication** (Dir A) | Qwen3-8B Q4_K_M | ✅ Complete | `matched_replication.ipynb` (Exp 10) |
| **Token-by-Token Dynamics** (Dir G) | Qwen3-8B Q4_K_M | ✅ Complete | `token_dynamics.ipynb` (Exp 11) |
| **Dynasty + Multilingual Geography** (Dir E) | Qwen3-8B Q4_K_M | ✅ Complete | `dynasty_multilingual.ipynb` (Exp 12) |

---

## Experiment 1: Translation Neuron Analysis

**Notebook**: `translation_neuron.ipynb` &nbsp;|&nbsp; **Model**: LLaMA-3-8B-Instruct (Q4_K_S, 32 layers, 4096 dim)

### Question
How do individual neurons contribute to Chinese→English translation? Can we identify, ablate, and steer translation-critical neurons?

### Method
1. **Baseline**: 3,981 WMT19 zh-en samples → mean COMET = 0.8187
2. **Activation statistics**: Per-neuron mean/std across 32 layers for COMET≥0.5 samples
3. **Goal hijacking**: Compare activation patterns between translation and hijacked tasks
4. **Neuron ablation**: Zero-out top-variance neurons, measure COMET impact
5. **Activation steering**: Add direction vectors at selected layers

### Key Findings
- **n4055** dominates layers 2–20 (largest std); **n2352** dominates layers 21–31
- Layers 10–20 are most sensitive to task type (hijacking detection)
- Ablating neurons `[2265, 4055, 2082, 290, 2943]` at **Layer 14** has the largest impact
- Steering at **Layer 11, strength 0.5** achieves optimal COMET = 0.8941

→ Full write-up and figures: see `README.md` (original) and `images/`

---

## Experiment 2: Mapping the Aesthetic Subspace

**Notebook**: `aesthetic_subspace.ipynb` &nbsp;|&nbsp; **Model**: Qwen3-8B (Q4_K_M, 36 layers, 4096 dim)  
**Output**: `/workspace/Data/aesthetic_experiment/` (activations, analysis results, plots)

### Question
Do aesthetic concepts (beauty, elegance, harmony, ugliness, crudeness) occupy a coherent **linear subspace** in the model's activation space? Is this subspace shared across sensory domains?

### Method

**Contrastive prompt dataset** — 50 pairs across 5 domains (visual, auditory, literary, design, gustatory) + 20 neutral prompts. Each prompt is wrapped with Qwen3's chat template + `<think>\n\n</think>\n\n` (no-think suffix) and forwarded through the engine to extract all-layer activations.

```
aesthetic_direction[layer] = mean(positive_acts) − mean(negative_acts)
```

### Results at a Glance

| Metric | Value |
|---|---|
| Best classification layer | **Layer 16** |
| ROC-AUC (5-fold CV) | **1.000 ± 0.000** |
| Accuracy (5-fold CV) | **1.000 ± 0.000** |
| Cross-domain transfer AUC | **1.000** (all 5 domains) |
| Inter-domain cosine similarity | 0.485 – 0.671 (mean 0.612) |
| Domain–global alignment | 0.81 – 0.85 |
| PCA top-5 explained variance (best layer) | 34.0% |
| Aesthetic/non-aesthetic logit ratio > 2× | **Layer 14** |
| Direction L2 norm range | 0.29 (L0) → 202 (L35) |

### Key Findings

#### 1. Aesthetic Direction Grows Monotonically
The L2 norm of the aesthetic direction vector increases from ~0.29 at layer 0 to ~202 at layer 35. The model progressively amplifies the aesthetic signal through its layers.

![Direction Magnitude](../Data/aesthetic_experiment/aesthetic_direction_magnitude.png)

#### 2. Perfect Linear Separability from Layer 16
A simple logistic regression achieves AUC=1.000 on held-out data from layer 16 onward. Even layer 0 already achieves AUC=0.938, suggesting aesthetic information is present early but crystallizes mid-network.

![Classification](../Data/aesthetic_experiment/classification_by_layer.png)

#### 3. Universal Cross-Domain Transfer
Training on 4 domains and testing on the held-out 5th domain yields AUC=1.000 everywhere (layer 16+). The aesthetic direction is **truly domain-general** — visual beauty and gustatory beauty share the same representation.

![Transfer](../Data/aesthetic_experiment/cross_domain_transfer.png)

#### 4. Domain Similarity Structure
Inter-domain cosine similarity at layer 16 ranges from 0.485 (auditory–design) to 0.671 (design–gustatory). All domains align 0.81–0.85 with the global aesthetic direction, indicating a **shared core** with domain-specific variation.

![Domain Similarity](../Data/aesthetic_experiment/domain_similarity.png)

#### 5. Logit Lens: Aesthetic Tokens Peak at Layer 14–16
Logit lens analysis on the prompt "Describe the beauty of a sunset over the ocean" shows aesthetic token probability mass peaks at layers 14–16, then collapses — the model "decides" on aesthetic tone at mid-layers.

![Logit Lens](../Data/aesthetic_experiment/logit_lens_aggregate.png)

#### 6. Activation Steering Works
Applying the aesthetic direction vector at layer 16 with varying strengths:

| Strength | Output Style | Aesthetic Density |
|---|---|---|
| −5.0 | *"a piece of land where plants are grown"* | 0.000 |
| −1.0 | Factual, no aesthetic adjectives | 0.000 |
| baseline | *"a carefully cultivated space"* | 0.018 |
| +1.0 | *"a peaceful and vibrant space, beauty, tranquility"* | 0.025 |
| +5.0 | *"a sanctuary of beauty, serenity, and life"* | 0.025 |
| +10.0 | *"Garden of Eden... unparalleled beauty"* | 0.012 |

Negative steering completely eliminates aesthetic vocabulary; positive steering increases poetic/aesthetic language.

![Steering](../Data/aesthetic_experiment/steering_aesthetic_density.png)

#### 7. 2D Visualization
PCA and t-SNE projections at layer 16 show clear three-cluster separation (aesthetic / non-aesthetic / neutral). Neutral prompts form a tight, distinct cluster.

![2D Projection](../Data/aesthetic_experiment/2d_projection.png)

### Saved Artifacts

```
/workspace/Data/aesthetic_experiment/
├── raw_activations.npz                 # 65.7 MB — [50,36,4096] × {pos,neg,neu}
├── aesthetic_analysis_results.npz      # Directions, PCA results, classification scores
├── aesthetic_direction_magnitude.png
├── pca_analysis.png
├── classification_by_layer.png
├── cross_domain_transfer.png
├── domain_similarity.png
├── logit_lens_aesthetic.png
├── logit_lens_aggregate.png
├── steering_aesthetic_density.png
└── 2d_projection.png
```

---

## Experiment 3: Layer-by-Layer Aesthetic Emergence

**Notebook**: `aesthetic_emergence.ipynb` &nbsp;|&nbsp; **Model**: Qwen3-8B (Q4_K_M, 36 layers, 4096 dim, 32 heads)  
**Output**: `/workspace/Data/aesthetic_emergence/` (activations, analysis results, plots)

### Question

At which layer does the model "decide" on aesthetic tone? Is there a discrete phase transition or a gradual emergence? How do perception (reading a sentence) and judgment (evaluating beauty) differ computationally?

### Method

**80 sentences** — 40 aesthetically rich (nature imagery, architectural beauty, literary prose, sensory experiences) + 40 plain/ugly (mundane descriptions, functional prose, unpleasant imagery).

Two probing conditions:
- **Content probing**: Forward the raw sentence → extract last-token activations → train linear probe at each layer
- **Judgment probing**: Wrap sentence in "Is this beautiful or ugly?" template → extract last-token activations → train linear probe at each layer

Additional analyses:
- **Logit lens**: Track P("beautiful") and P("ugly") token probabilities through layers for 10+10 judgment prompts
- **Attention analysis**: Collect attention patterns at 10 key layers, compute Shannon entropy and JS divergence per head
- **PCA trajectories**: Visualize activation space evolution at layers [0, 4, 8, 16, 24, 35]

Engine configured with `disable_flash_attention=True` for attention extraction, `n_ctx=4096`, `n_seq_max=1`.

### Results at a Glance

| Metric | Content Probing | Judgment Probing |
|---|---|---|
| AUC ≥ 0.75 | Layer 0 | Layer 0 |
| AUC ≥ 0.90 | Layer 0 | Layer 3 |
| AUC ≥ 0.95 | Layer 1 | Layer 6 |
| AUC ≥ 0.99 | Layer 3 | Layer 12 |
| AUC = 1.000 | **Layer 3** | **Layer 18** |
| Peak info gain layers | [0, 3, 2] | [3, 2, 6] |

| Attention Metric | Value |
|---|---|
| Most aesthetic-selective head | **Layer 28, Head 0** (JS = 0.695) |
| Second most selective | **Layer 24, Head 8** (JS = 0.617) |
| Logit lens "beautiful" diverges | **Layer 18** |
| Logit lens "ugly" diverges | **Layer 21** |

### Key Findings

#### 1. Two-Phase Emergence: Perception vs. Judgment

The most striking finding is a **15-layer gap** between content recognition and evaluative judgment:

- **Content probes** (raw sentence → "is it beautiful?") achieve AUC=1.000 by **layer 3–4**. The model *knows* whether a sentence is aesthetically rich almost immediately after embedding.
- **Judgment probes** (wrapped in evaluation template) only reach AUC=1.000 at **layer 18**. The explicit judgment computation takes significantly longer.

This suggests two distinct computational phases:
1. **Layers 0–4**: Rapid content categorization — aesthetic features (vivid imagery, poetic language) are already linearly separable
2. **Layers 4–18**: Judgment computation — the model builds its evaluative response, integrating content recognition with the task instruction

![Emergence Curves](../Data/aesthetic_emergence/emergence_curves.png)

#### 2. Convergence with Experiment 2

Overlaying Exp 2's contrastive classifier (AUC for "describe beauty" vs "describe ugliness" prompts) with Exp 3's probes reveals an aligned picture:
- Exp 2 contrastive AUC reaches 1.0 around **layer 3–4** (matches content probing)
- Exp 3 judgment probing lags behind, reaching 1.0 at **layer 18** (matches Exp 2's logit lens peak at L14–16)
- The **layer 16–18 window** is consistently the "decision boundary" across both experiments

![Exp2 vs Exp3](../Data/aesthetic_emergence/exp2_vs_exp3_comparison.png)

#### 3. Logit Lens Confirms Layer 18 as Decision Point

Tracking token probabilities through the logit lens for judgment prompts:
- P("beautiful" | beautiful input) overtakes P("ugly" | beautiful input) at **layer 18** (ratio > 2×)
- P("ugly" | ugly input) overtakes P("beautiful" | ugly input) at **layer 21**
- The asymmetry suggests the model resolves positive aesthetic judgments slightly faster than negative ones

![Logit Lens](../Data/aesthetic_emergence/logit_lens_judgment.png)

#### 4. Aesthetic Attention Heads

JS divergence analysis identified **specialized attention heads** that attend differently for beautiful vs. ugly inputs:

| Head | JS Divergence | Behavior |
|---|---|---|
| L28 H0 | 0.695 | Most selective — sharply focused on single position for beautiful, distributed for ugly |
| L24 H8 | 0.617 | Strong selective focus pattern |
| L24 H29 | 0.491 | Focused for beautiful, spread for ugly |
| L28 H23 | 0.441 | Late-layer aesthetic discriminator |
| L32 H11 | 0.422 | Near-final-layer aesthetic head |

The top aesthetic heads cluster in **layers 24–32** — deeper than the probing emergence layer (18), suggesting these heads refine the aesthetic representation after the initial decision.

Entropy analysis confirms: attention for beautiful inputs is **more focused** (lower entropy) than for ugly inputs at these heads, indicating the model "locks on" to specific aesthetic content tokens.

![Attention Selectivity](../Data/aesthetic_emergence/attention_aesthetic_selectivity.png)

#### 5. Activation Space Geometry

PCA visualizations show the progressive separation:
- **Layer 0**: Mixed clusters (AUC=0.919), ~17.4% PC1 variance
- **Layer 4**: Clear separation (AUC=1.000), ~19.5% PC1 variance  
- **Layer 16+**: Maximum geometric distance, PC1 captures 30–33% of variance
- **Layer 35**: Extreme spread (norm ~300), complete separation

![Activation Trajectory](../Data/aesthetic_emergence/activation_trajectory.png)

### Saved Artifacts

```
/workspace/Data/aesthetic_emergence/
├── content_activations.npz              # (80, 36, 4096) — raw sentence activations
├── judgment_activations.npz             # (80, 36, 4096) — judgment template activations
├── emergence_analysis_results.npz       # Probes, gradients, entropy, JS divergence
├── emergence_curves.png                 # Content vs judgment AUC by layer
├── emergence_gradient.png               # Information gain per layer
├── exp2_vs_exp3_comparison.png          # Overlay with Experiment 2
├── logit_lens_judgment.png              # Token probability trajectories
├── attention_entropy.png                # Per-head entropy heatmap
├── attention_aesthetic_selectivity.png  # JS divergence heatmap
├── top_aesthetic_heads_attention.png    # Detailed attention bar plots
└── activation_trajectory.png            # PCA evolution across layers
```

---

## Experiment 4: Cross-Lingual Aesthetic Universality

**Notebook**: `cross_lingual_aesthetics.ipynb` &nbsp;|&nbsp; **Model**: Qwen3-8B (Q4_K_M, 36 layers, 4096 dim)  
**Output**: `/workspace/Data/cross_lingual_aesthetics/` (activations, analysis results, plots)  
**Builds on**: Experiment 2 (aesthetic direction & best classification layer)

### Question

Do aesthetic representations transcend language? If the model encodes "beauty" as a linear direction in English activation space, does that **same direction** also separate beautiful from ugly sentences in Chinese, Japanese, French, and Spanish?

### Method

**150 sentences** — 30 per language (15 aesthetically rich + 15 plain/ugly) across 5 languages:
- **English** (EN) — nature, architecture, art, music, sensory experiences
- **中文 / Chinese** (ZH) — classical poetry imagery, garden descriptions, calligraphy, silk road, jade
- **日本語 / Japanese** (JA) — cherry blossoms, zen gardens, ukiyo-e, seasonal aesthetics, wabi-sabi
- **Français / French** (FR) — jardins de Versailles, Impressionism, haute couture, lavender, cathedral architecture
- **Español / Spanish** (ES) — Alhambra, flamenco, García Márquez imagery, tapas culture, Gaudí

Each sentence forwarded through the model with Qwen3 chat template + no-think suffix. Last-token activations extracted at all 36 layers.

Analyses:
1. **Zero-shot transfer**: Project all sentences onto Exp 2's English aesthetic direction → compute AUC per language per layer
2. **Per-language aesthetic directions**: Compute `mean(beautiful) − mean(ugly)` for each language → pairwise cosine similarity
3. **PCA visualization**: Across 4 representative layers, color by aesthetic quality vs. language identity
4. **Cross-lingual probe transfer**: Train LogisticRegression on English, test on each other language
5. **Bidirectional transfer matrix**: Full 5×5 train/test at best layer
6. **Silhouette analysis**: Quantify whether aesthetic quality or language identity dominates activation space geometry

### Results at a Glance

| Metric | Value |
|---|---|
| Zero-shot transfer AUC (Layer 16) | **1.000** for all 5 languages |
| Cross-lingual probe AUC (EN → others) | **0.996 – 1.000** |
| Bidirectional transfer AUC (mean off-diag) | **0.998** (min 0.982) |
| Per-language direction similarity | 0.505 – 0.876 (mean 0.619) |
| Silhouette (beauty grouping, L16) | 0.055 |
| Silhouette (language grouping, L16) | 0.293 |
| Dominant geometry axis | **Language** (but aesthetics perfectly linearly separable) |

### Key Findings

#### 1. Perfect Zero-Shot Cross-Lingual Transfer

The aesthetic direction discovered in Experiment 2 (trained on English contrastive prompts) achieves **AUC = 1.000** for ALL 5 languages at Layer 16. Even more strikingly:
- **Chinese and Japanese** achieve AUC = 1.000 from **Layer 0** — the model immediately recognizes aesthetic content in CJK languages
- **Spanish** reaches 1.000 by Layer 7
- **French** reaches 1.000 by Layer 15 (the slowest)

This means a single English-derived direction vector is sufficient to classify beauty in any language.

![Zero-Shot Transfer](../Data/cross_lingual_aesthetics/zero_shot_transfer.png)

#### 2. Bidirectional Transfer Is Near-Perfect

The full 5×5 transfer matrix shows that a probe trained in **any** language transfers to **every** other language:
- Within-language AUC: 1.000 (all diagonals)
- Cross-language AUC: mean = 0.998, minimum = 0.982 (Chinese → French)
- No language pair fails below 0.98

This is the strongest possible evidence for a **shared, language-universal aesthetic subspace**.

![Transfer Matrix](../Data/cross_lingual_aesthetics/bidirectional_transfer.png)

#### 3. Per-Language Aesthetic Directions Are Parallel But Not Identical

Cosine similarity between per-language aesthetic directions at Layer 16:
- **French ↔ Spanish**: 0.876 (highest — linguistically related)
- **English ↔ Spanish**: 0.662
- **English ↔ French**: 0.647
- **Chinese ↔ Japanese**: 0.615 (share aesthetic vocabulary — 漢字/kanji)
- **English ↔ Chinese**: 0.566
- **English ↔ Japanese**: 0.505 (lowest)

The similarity structure mirrors linguistic family relationships, suggesting the aesthetic direction has a **shared core** with language-specific modulation.

![Direction Similarity](../Data/cross_lingual_aesthetics/direction_similarity.png)

#### 4. PCA: From Language Clusters to Aesthetic Clusters

The most visually striking result. PCA across layers shows a dramatic transformation:
- **Layer 0**: Sentences cluster tightly by **language** (5 distinct clusters)
- **Layer 8**: Languages begin mixing, but aesthetic separation is weak
- **Layer 16**: Clear **aesthetic separation** with beautiful/ugly on opposite sides, languages mixed
- **Layer 35**: Extreme aesthetic separation, language barely detectable

The focused dual PCA at Layer 16 shows PC1 (30.1% variance) separates by aesthetic quality, not language.

![PCA Visualization](../Data/cross_lingual_aesthetics/pca_best_layer.png)

#### 5. Language Still Dominates Overall Geometry

Despite perfect aesthetic classification, silhouette analysis reveals that **language identity** always has higher silhouette scores than aesthetic grouping across all layers. At Layer 16: language silhouette = 0.293 vs. beauty silhouette = 0.055.

This is not contradictory — the 4096-dimensional space is rich enough to encode both. Language accounts for more **total variance** (grammar, script, vocabulary), but the aesthetic direction occupies a **specific linear subspace** that is shared across languages. A classifier can extract it perfectly even though it's not the dominant axis of variation.

![Silhouette Analysis](../Data/cross_lingual_aesthetics/silhouette_analysis.png)

#### 6. Aesthetic Direction Magnitude Follows Universal Growth Pattern

All 5 languages show the same exponential growth curve for aesthetic direction L2 norm:
- Near-zero until Layer ~15
- Exponential growth from Layer 20 to Layer 35
- Chinese leads (‖d‖ ≈ 430 at L35), followed by Japanese (≈ 395), Spanish (≈ 340), French (≈ 300), English (≈ 280)
- Exp 2's English contrastive direction tracks a similar trajectory at lower magnitude

![Direction Magnitude](../Data/cross_lingual_aesthetics/direction_magnitude.png)

### Interpretation

Qwen3-8B develops a **language-universal aesthetic representation**. The model doesn't have separate "beauty detectors" per language — instead, it maps aesthetic content from all languages into a shared geometric region. This shared representation emerges early (Layer 0 for CJK, Layer 16 at latest for all languages) and persists through the entire network.

The hierarchy of similarity (French–Spanish > EN–FR > ZH–JA > EN–JA) suggests the aesthetic direction is built on **semantic features** (vivid imagery, sensory language) rather than surface-level lexical cues, since scripts and grammars differ vastly between language families.

### Saved Artifacts

```
/workspace/Data/cross_lingual_aesthetics/
├── cross_lingual_activations.npz        # 82.1 MB — activations per language
├── cross_lingual_results.npz            # Transfer AUCs, similarity matrices, silhouette scores
├── zero_shot_transfer.png               # AUC curves (5 languages) using Exp2 direction
├── direction_similarity.png             # Cosine similarity heatmap + layer-wise curves
├── pca_4layers.png                      # PCA evolution (L0, L8, L16, L35)
├── pca_best_layer.png                   # Dual PCA: aesthetic vs language coloring
├── probe_transfer.png                   # EN-trained probe → other languages
├── bidirectional_transfer.png           # 5×5 transfer matrix heatmap
├── direction_magnitude.png              # Direction L2 norm by language
└── silhouette_analysis.png              # Beauty vs language silhouette scores
```

---

## Experiment 5: Aesthetic Neurons — Causal Identification via Ablation

**Notebook**: `aesthetic_neurons.ipynb` &nbsp;|&nbsp; **Model**: Qwen3-8B (Q4_K_M, 36 layers, 4096 dim)  
**Output**: `/workspace/Data/aesthetic_neurons/` (selectivity scores, ablation results, plots)  
**Builds on**: Experiment 2 (aesthetic direction, raw activations, best layer = 16)

### Question

Are there individual neurons (or small groups) that are **causally necessary** for aesthetic judgment? If we ablate them, does the model lose its ability to generate beautiful language?

### Method

**Three-phase approach**: offline analysis → online single-neuron ablation → group ablation with controls.

1. **Neuron selectivity scoring** from Exp 2's saved activations (50 positive + 50 negative prompts):
   - **Cohen's d**: |mean_pos − mean_neg| / pooled_std per neuron per layer
   - **Direction weight**: |component in aesthetic_direction_norm|
   - **Contribution**: |dir_weight| × |mean_diff| — how much each neuron contributes to projection separability

2. **Offline AUC analysis**: mask neurons in saved activations, recompute projection onto aesthetic direction, measure AUC change (instantaneous, no engine needed)

3. **Online generation ablation**: apply `engine.apply_mask(layer, mask)` during inference, generate from 5 aesthetic prompts (temperature=0, max_tokens=128), measure **aesthetic vocabulary density** (count of 78 aesthetic words per 100 words)

4. **Group ablation sweep**: ablate top-N neurons simultaneously (N = 5, 10, 20, 50, 100, 200, 500, 1000, 2048)

5. **Random control**: same N values but random neurons (2 seeds)

6. **Layer sensitivity**: ablate top-20 at each of 10 layers [0, 3, 8, 12, 16, 20, 24, 28, 32, 35]

### Results at a Glance

| Metric | Value |
|---|---|
| Baseline aesthetic density | **0.95%** ± 1.01% |
| Max single-neuron AUC drop | **0.0004** (n3179) |
| Top offline group ablation (N=2048) | AUC still **0.9984** |
| Single-neuron ablation effect | **All increase** density (+6% to +172%) |
| Peak group ablation density | **3.05%** at N=200 (+221% vs baseline) |
| Group collapse tipping point | **N ≈ 1000** (density → 0.0%) |
| Random ablation at N=1000 | **1.93%** (still functional) |
| Most sensitive layer (decrease) | **L35** (−40.1%) |
| Largest brake-removal layer | **L16** (+193.1%) |

### Key Findings

#### 1. The Aesthetic Representation Is Massively Distributed

The most fundamental finding: **no individual neuron matters**. The maximum AUC drop from ablating any single neuron is 0.0004 (out of a baseline of 0.999). Even ablating the top 2048 out of 4096 neurons (50%!) only drops AUC from 0.9992 to 0.9984. The aesthetic direction is spread uniformly across all 4096 dimensions with extreme redundancy.

![Selectivity Overview](../Data/aesthetic_neurons/selectivity_overview.png)

#### 2. Aesthetic Neurons Are Brakes, Not Drivers (at Mid-Layers)

The most surprising finding: ablating the top aesthetic neurons at Layer 16 **increases** aesthetic vocabulary density by up to 221%. Every single-neuron ablation produced MORE aesthetic words, not fewer. The most impactful neuron, n2276, increased density from 0.95% to 2.58% (+172%).

These neurons appear to function as **aesthetic regulators** — they normally constrain the model's use of aesthetic vocabulary. When removed, the model's aesthetic output becomes more effusive.

![Single-Neuron Ablation](../Data/aesthetic_neurons/single_neuron_ablation.png)

#### 3. Catastrophic Collapse at N ≈ 1000

Group ablation reveals a dramatic **phase transition**:
- **N = 5–500**: density increases to 2–3% (targeted ablation releases more aesthetic language)
- **N = 1000**: density drops to **0.0%** — the model produces empty/gibberish output
- **N = 2048**: complete collapse

Critically, **random ablation at N=1000 still produces coherent text** (density ≈ 1.93%). This proves the collapse is specific to removing aesthetic neurons, not just losing capacity.

![Group Ablation](../Data/aesthetic_neurons/group_ablation.png)

#### 4. Two-Phase Layer Architecture: Brakes vs Drivers

Ablating top-20 neurons at different layers reveals a striking dichotomy:

| Layer Range | Effect | Interpretation |
|---|---|---|
| **L0–L3** | Minimal (+5–26%) | Early embedding, little aesthetic specialization |
| **L8–L24** | Strong increase (+42–193%) | Mid-layer neurons act as **aesthetic brakes** |
| **L32–L35** | **Decrease** (−12–40%) | Late-layer neurons are **aesthetic drivers** |

Peak brake-removal effect is at **Layer 16** (the same layer where linear classification is perfect). Peak driver effect is at **Layer 35** (final layer), where ablation reduces density by 40%.

This suggests a control architecture: mid-layers constrain aesthetic expression (keeping it calibrated), while final layers amplify the aesthetic signal for vocabulary selection.

![Layer Sensitivity](../Data/aesthetic_neurons/layer_sensitivity.png)

#### 5. Offline Prediction Disconnects from Behavioral Impact

The offline AUC analysis (linear projection) shows nearly zero impact from any ablation — the 4096-dimensional space has infinite redundancy for linear classification. But **generation** (which involves 20 non-linear downstream layers) is highly sensitive. This demonstrates that:
- Linear probes capture the **information** but miss the **computation**
- A neuron can be unimportant for classification yet critical for generation
- Causal ablation experiments reveal structure that correlational analysis cannot

#### 6. Cross-Lingual Neuron Selectivity: Shared Core + Language-Specific Channels

Using multilingual activations from Experiment 4 (EN, ZH, JA, FR, ES × 30 sentences), we computed per-language Cohen's d selectivity for all 36 layers × 4096 neurons. This reveals whether the **same neurons** are aesthetically relevant across languages.

**Neuron-level agreement is moderate, not identical:**

| Metric | Value |
|---|---|
| Global Spearman ρ (all layers × neurons) | **0.533** |
| Peak layer agreement | **L21** (ρ = 0.677) |
| Lowest layer agreement | **L1** (ρ = 0.284) |
| FR–ES correlation | **0.796** (highest — same language family) |
| EN–JA correlation | **0.455** (lowest — most distant) |

**Universal vs Language-Specific neurons at Layer 16:**

| Category | Count | Description |
|---|---|---|
| Universal (CV < 0.57) | **2048** / 4096 | Consistent selectivity across all 5 languages |
| Language-specific (CV > 0.57, mean d > 0.5) | **1514** | High selectivity in one language, low in others |
| Specific → dominated by ZH | **39** / 50 top | Chinese activates unique aesthetic neurons |
| Specific → dominated by JA | **10** / 50 top | Japanese also has dedicated channels |
| Specific → EN / FR / ES | **1** / 50 top | Western languages share the common pool |

**Aesthetic direction alignment across languages:**

| Language Pair | Cosine @ L16 | Notes |
|---|---|---|
| FR ↔ ES | **0.876** | Same Romance family |
| EN ↔ ES | 0.662 | — |
| EN ↔ FR | 0.647 | — |
| ZH ↔ JA | 0.615 | Shared 漢字 / kanji aesthetic vocabulary |
| EN ↔ ZH | 0.566 | — |
| EN ↔ JA | 0.505 | Most distant |
| Mean cross-language | **0.619** | Peak alignment at L21 (cos = 0.734) |

The direction alignment follows linguistic family structure: Romance languages (FR-ES) are most aligned, CJK languages share an intermediate alignment, and cross-family pairs are the most divergent.

![Cross-Lingual Heatmap](../Data/aesthetic_neurons/cross_lingual_neuron_heatmap.png)
![Neuron Correlation](../Data/aesthetic_neurons/cross_lingual_neuron_correlation.png)
![Universal vs Specific](../Data/aesthetic_neurons/universal_vs_specific_neurons.png)
![Direction Alignment](../Data/aesthetic_neurons/cross_lingual_direction_alignment.png)

#### §12 Cross-Lingual Aesthetic Steering

With the neuron decomposition from §11, we built **disjoint neuron sets** and used them to transplant one language's aesthetic style into another language's generation.

**Neuron set decomposition (Layer 16):**

| Set | Count | Steering Norm | Criterion |
|---|---|---|---|
| Universal | **356** | 5.4 | Top-10% mean |d| AND CV < median |
| EN-specific | 285 | 4.5 | Top-20% own |d|, others < 50th pctile, minus universal |
| ZH-specific | **295** | **6.2** | (largest norm — strongest aesthetic signal) |
| JA-specific | 247 | 3.8 | — |
| FR-specific | 219 | 2.4 | — |
| ES-specific | 223 | 2.4 | — |

- Language-specific sets have **near-zero Jaccard overlap** (0.000) except FR-ES (0.097, Romance family)
- Total: 356 universal + 1269 specific = 1625 neurons involved in aesthetics (40% of 4096)

**Cross-lingual steering experiment** (suppress source-specific neurons, activate target-specific):

| Pair | Key Qualitative Effect |
|---|---|
| EN→ZH | Subtle: more texture detail ("canopy", "soft ground") |
| EN→JA | Added holistic metaphor ("breathtaking tapestry"), more contemplative |
| ZH→EN | More elaborate imagery ("如云似霞", "华丽的锦缎"), philosophical coda |
| **ZH→JA** | **Most dramatic: 3 paragraphs → 1 compact paragraph (Japanese minimalism)** |
| JA→ZH | Added concrete nature detail, reduced poetic abstraction |
| FR→JA | More nature-focused, simpler sentence structure |

The aesthetic density metric (English keyword-based) cannot capture these stylistic shifts — the real effect is in **structural and conceptual transformation**: paragraph count, metaphor density, emotional vs concrete imagery, and the balance between elaboration and restraint.

![Neuron Sets](../Data/aesthetic_neurons/cross_lingual_neuron_sets.png)

### Interpretation

The aesthetic representation in Qwen3-8B is not localized in "aesthetic neurons." Instead:

1. **Aesthetic information** is distributed across all 4096 dimensions (redundant linear code)
2. **Aesthetic control** is mediated by specific neurons that function as regulators:
   - Mid-layer (L8–L24) neurons **suppress** aesthetic vocabulary (brakes)
   - Late-layer (L32–L35) neurons **promote** aesthetic vocabulary (drivers)
3. The brake/driver architecture suggests the model learned a **calibration mechanism** for aesthetic expression — ensuring outputs are appropriately aesthetic for the prompt, rather than maxing out aesthetic vocabulary
4. The catastrophic collapse at N≈1000 targeted neurons (but not random ones) confirms these neurons carry a qualitatively different role than generic neurons
5. **Cross-lingual neuron analysis** reveals a dual structure: ~50% of neurons are **universally** aesthetic across all 5 languages, while the other ~50% are **language-specific** — overwhelmingly dominated by CJK languages (ZH: 78%, JA: 20%). This explains why zero-shot transfer works perfectly (Exp 4): the universal core is sufficient for classification, but the fine-grained neuron-level activity pattern is language-dependent
6. The **direction alignment** (mean cosine = 0.619) mirrors linguistic family structure, suggesting the aesthetic subspace has a shared geometric core (~60%) with language-specific modulation (~40%)
7. **Cross-lingual steering** (§12) demonstrates that language-specific neuron sets are causally linked to aesthetic *style*, not just vocabulary: suppressing ZH-specific neurons and activating JA-specific neurons transforms 3-paragraph Chinese prose into a single minimalist paragraph — reflecting Japanese aesthetic preferences (侘寂/wabi-sabi). This confirms that the neuron sets don't merely encode "which language to use" but rather **language-specific aesthetic conceptual frameworks**

### Saved Artifacts

```
/workspace/Data/aesthetic_neurons/
├── ablation_results.npz                    # All selectivity scores, ablation results, layer data
├── selectivity_overview.png                # Where aesthetic neurons live + distribution
├── offline_auc_analysis.png                # Per-neuron AUC + targeted vs random
├── single_neuron_ablation.png              # Top-10 single-neuron impact
├── group_ablation.png                      # Tipping point: generation + offline comparison
├── layer_sensitivity.png                   # Layer-by-layer ablation impact
├── neuron_heatmap_full.png                 # 36×4096 full selectivity + contribution heatmap
├── cross_lingual_neuron_heatmap.png        # 5-language selectivity heatmaps (sorted)
├── cross_lingual_neuron_correlation.png    # Spearman ρ: global matrix + per-layer + pair detail
├── universal_vs_specific_neurons.png       # Universal/specific scatter + per-language profiles
├── cross_lingual_direction_alignment.png   # Direction cosine by layer and language pair
├── cross_lingual_neuron_sets.png           # §12: neuron set membership, sizes, Jaccard overlap
└── cross_lingual_steering_results.npz      # §12: universal/specific sets, steering vectors, pairs
```

---

## Experiment 6: Real Poetry Validates Aesthetic Neurons

**Notebook**: `poetry_aesthetics.ipynb` (§1–§6) &nbsp;|&nbsp; **Model**: Qwen3-8B (Q4_K_M, 36 layers, 4096 dim)  
**Output**: `/workspace/Data/poetry_aesthetics/` (plots, saved results)  
**Builds on**: Experiment 2 (aesthetic direction), Experiment 5 (top-100 aesthetic neurons)

### Question

Do the aesthetic neurons and direction vector discovered from **synthetic contrastive prompts** (Exp 2) generalize to **real poetry**? Is the signal strong enough to separate genuine poems from news articles?

### Data

| Corpus | Source | N | Description |
|---|---|---|---|
| ZH Poetry | `larryvrh/Chinese-Poems` (HuggingFace) | 50 | Classical Chinese poems (Tang/Song/Yuan/Ming/Qing) |
| EN Poetry | `biglam/gutenberg-poetry-corpus` (HuggingFace) | 50 | 19th–early 20th century English poetry, ≥4 lines |
| ES Poetry | `andreamorgar/spanish_poetry` (HuggingFace) | 50 | Spanish poetry, diverse authors |
| ZH News | `wmt19` zh-en (HuggingFace) | 50 | Chinese news sentences |
| EN News | `wmt19` zh-en (HuggingFace) | 50 | English news sentences |

### Method

Each text forwarded through model with chat template + no-think suffix. Last-token activations at Layer 16 extracted and projected onto Exp 2's aesthetic direction vector. Compared poetry vs news via t-tests, Cohen's d, and AUC.

### Results

| Metric | Value |
|---|---|
| Poetry vs News AUC | **0.594** |
| Overall Cohen's d | **0.317** (small-medium) |
| ZH Poetry vs ZH News | d = **0.699**, p = 0.0008 |
| EN Poetry vs EN News | d = 0.099, p = 0.625 (n.s.) |
| ZH Poetry mean projection | **4.875** ± 1.334 |
| EN Poetry mean projection | 3.240 ± 1.584 |
| ZH News mean projection | 3.843 ± 1.607 |
| EN News mean projection | 3.077 ± 1.701 |
| Top-100 aesthetic neurons higher for poetry | 41/100 |

### Key Findings

#### 1. Chinese Poetry Strongly Activates the Aesthetic Subspace

ZH-Poetry has the highest mean aesthetic projection (4.875), significantly above ZH-News (d=0.699, p<0.001). This validates that the synthetic aesthetic direction from Exp 2 captures real aesthetic content in classical Chinese poetry.

#### 2. English Effect Is Weak

EN-Poetry and EN-News are nearly indistinguishable (d=0.099, p=0.625). The Exp 2 aesthetic direction — likely shaped by Qwen3's Chinese-dominant training — does not generalize well to English-language literary aesthetics. This is consistent with Exp 5 §11's finding that 78% of language-specific aesthetic neurons are dominated by Chinese.

#### 3. Top-100 Aesthetic Neurons Show Weak Signal

Only 41/100 aesthetic neurons had higher mean absolute activation for poetry than news. The bottom-left panel shows a weak negative correlation (r=−0.169) between Exp 5 contribution score and real poetry activation difference, suggesting neurons with the highest *ablation impact* (Exp 5's brake neurons) are actually *less* active for real poetry — consistent with the brake interpretation.

#### 4. The Direction Is Partially Valid

AUC=0.594 is above chance but modest. The aesthetic direction from synthetic contrastive prompts captures a **component** of real literary aesthetics, but real poetry is far more complex than the binary aesthetic/non-aesthetic distinction used in Exp 2.

![Exp 6 Results](../Data/poetry_aesthetics/exp6_poetry_validation.png)

---

## Experiment 7: Tang vs Song Dynasty Aesthetics

**Notebook**: `poetry_aesthetics.ipynb` (§7–§10) &nbsp;|&nbsp; **Model**: Qwen3-8B (Q4_K_M, 36 layers, 4096 dim)  
**Output**: `/workspace/Data/poetry_aesthetics/` (plots, saved results)

### Question

Can the model distinguish **Tang dynasty** (唐代, 618–907: 奔放/bold, expressive) from **Song dynasty** (宋代, 960–1279: 婉约/restrained, delicate) poetic styles? Is this encoded in the same neurons as the aesthetic subspace, or a separate one?

### Data

| Dynasty | N | Source | Style |
|---|---|---|---|
| Tang (唐代) | 50 | `larryvrh/Chinese-Poems` | Bold, majestic, outer-directed imagery |
| Song (宋代) | 50 | `larryvrh/Chinese-Poems` | Restrained, intimate, inner-directed emotion |

### Method

All-layer activations (36 layers × 4096 dim) collected for 100 poems. Layer-by-layer AUC classification, dynasty direction vector computed, neuron overlap analysis with Exp 2 aesthetic neurons.

### Results

| Metric | Value |
|---|---|
| Best dynasty classification layer | **L11** (AUC = **0.922**) |
| Dynasty–Aesthetic cosine (L16) | **−0.022** (orthogonal) |
| Dynasty–Aesthetic cosine (L11) | **0.008** (orthogonal) |
| Top-200 dynasty-only neurons (L11) | **195** |
| Top-200 aesthetic-only neurons (L11) | **95** |
| Shared neurons | **5** (2.5% overlap) |

### Key Findings

#### 1. Dynasty Style Is Perfectly Classifiable

AUC=0.922 at Layer 11 — the model cleanly separates Tang from Song poetry. The classification curve rises rapidly in layers 0–11, peaks at L11, then slowly declines. This is notably **earlier** than the aesthetic best layer (L16), suggesting dynasty style is a lower-level feature resolved before abstract aesthetic evaluation.

#### 2. Dynasty and Aesthetic Directions Are Orthogonal

Cosine similarity between the dynasty direction (Tang−Song) and the aesthetic direction is effectively zero (−0.022 at L16, 0.008 at L11). These are **independent subspaces** — the model encodes "beautiful vs ugly" and "bold vs restrained" as separate geometric directions.

#### 3. Minimal Neuron Overlap

Only 5 neurons appear in both the top-200 dynasty and top-200 aesthetic sets — 2.5% overlap. The vast majority are exclusive to one type. This confirms the subspace independence at the individual neuron level.

#### 4. Dynasty Steering Produces Subtle Stylistic Shifts

Applying the dynasty direction at L11 (strength=5.0) on poetry-generation prompts:

| Condition | Sample Output (Moonlight) |
|---|---|
| Baseline | 月光洒河面，波光映夜天。轻舟随水动，静影伴梦眠。 |
| Tang-steered | 月光洒江面，波光映**银辉**。轻舟随水动，静影伴梦**归**。 |
| Song-steered | 月光洒河面，波光映夜天。轻舟随水动，静影伴**星**眠。 |

Tang steering introduces grander imagery (银辉 = silver radiance), while Song steering selects more intimate/contemplative words (星眠 = star-sleep). Effects are subtle but consistent with dynasty poetic conventions.

![Exp 7 Results](../Data/poetry_aesthetics/exp7_dynasty_aesthetics.png)

---

## Experiment 8: Bilingual Aesthetic Distillation

**Notebook**: `poetry_aesthetics.ipynb` (§11–§13) &nbsp;|&nbsp; **Model**: Qwen3-8B (Q4_K_M, 36 layers, 4096 dim)  
**Output**: `/workspace/Data/poetry_aesthetics/` (plots, saved results)

### Question

When the **same poem** is presented in Chinese and English, which neurons activate consistently (encoding **content**) vs. differently (encoding language-specific **style**)? How do aesthetic neurons partition between content and style?

### Data

| Source | N | Description |
|---|---|---|
| `emmit989/chinese-poetry-trans-fine-tune` (HuggingFace) | 100 pairs | Classical Chinese poems + English translations |

### Method

For each of 100 poem pairs, forward both the ZH original and EN translation through the model. Extract Layer 16 activations. Compute Pearson correlation per neuron across the 100 pairs. Classify neurons:
- **Content** (r > 0.5): Activate similarly regardless of language → encode semantic content
- **Style** (r < 0.1): Activate differently → encode language-specific representation
- **Ambiguous** (0.1 ≤ r ≤ 0.5): Mixed signal

### Results

| Metric | Value |
|---|---|
| Content neurons (r > 0.5) | **10** / 4096 |
| Style neurons (r < 0.1) | **1554** / 4096 |
| Ambiguous neurons | 2532 / 4096 |
| Mean correlation | **0.135** |
| Aesthetic top-100: content | **4** |
| Aesthetic top-100: style | **29** |
| Aesthetic top-100: ambiguous | **67** |
| Content contribution to aesthetic dir | **0.5%** |
| Style contribution to aesthetic dir | **36.1%** |

### Key Findings

#### 1. Overwhelmingly Language-Specific Encoding

Only 10 out of 4096 neurons (0.24%) qualify as content neurons. The mean correlation is 0.135, indicating that even for the exact same poem, the model's activations are dominated by **language identity** rather than semantic content. This is consistent with Exp 4's finding that language silhouette > beauty silhouette.

#### 2. The Aesthetic Direction Is Mostly Style, Not Content

Style neurons contribute 36.1% of the aesthetic direction's total magnitude, while content neurons contribute only 0.5%. The aesthetic subspace from Exp 2 is predominantly a **Chinese-language aesthetic representation**, not a language-neutral "beauty detector." This explains why Exp 6 showed strong ZH-Poetry signal but weak EN-Poetry signal.

#### 3. Most Aesthetic Neurons Are Ambiguous

67% of the top-100 aesthetic neurons fall in the ambiguous range (0.1 ≤ r ≤ 0.5). These neurons respond partially to semantic content and partially to language — they likely encode **aesthetic features that are conceptually similar across languages but activated through language-specific pathways**.

#### 4. Example Pair Visualization

The bottom-right panel of the visualization shows top-30 aesthetic neuron activations for a single ZH-EN poem pair. Key neurons (n3470, n809) show dramatically different activations between languages, while a few (n3179, n1740) have similar magnitudes — a visual confirmation of the style-dominated partition.

![Exp 8 Results](../Data/poetry_aesthetics/exp8_bilingual_distillation.png)

### Interpretation

Experiments 6–8 together reveal that the aesthetic representation in Qwen3-8B, while powerful for classification (Exps 2–5), is fundamentally **language-entangled**:

1. The aesthetic direction works best on Chinese text (Exp 6) because it primarily captures Chinese-language aesthetic features
2. Literary style dimensions (Tang vs Song) are **orthogonal** to the aesthetic direction (Exp 7), showing the model disentangles multiple literary attributes
3. True cross-lingual content neurons are extremely rare (Exp 8) — the model processes "beauty" through language-specific computational pathways that converge on a shared subspace (explaining Exp 4's perfect transfer) but with different neuron-level mechanisms

### Saved Artifacts

```
/workspace/Data/poetry_aesthetics/
├── exp6_poetry_validation.png       # 4-panel: violin, heatmap, scatter, bar
├── exp7_dynasty_aesthetics.png      # 6-panel: AUC, cosine, projection, overlap, selectivity
├── exp8_bilingual_distillation.png  # 4-panel: correlation hist, scatter, pie, pair example
└── poetry_aesthetics_results.npz    # All numerical results
```

---

## Experiment 9: Poetry Replication — Redoing Exp 2+3+5 with Real Data

**Notebook**: `poetry_replication.ipynb` &nbsp;|&nbsp; **Model**: Qwen3-8B (Q4_K_M, 36 layers, 4096 dim)  
**Output**: `/workspace/Data/poetry_replication/` (plots, saved results)  
**Replicates**: Experiment 2 (Subspace), Experiment 3 (Emergence), Experiment 5 (Neurons)

### Question

Experiments 2, 3, and 5 used **synthetic contrastive prompts** ("Describe the beauty of…" vs "Describe the ugliness of…"). Do the core findings — linear separability, two-phase emergence, brake/driver neurons — replicate when using **real poetry** (4,900 Chinese poems, 1,000 English poems, 1,000 Spanish poems) vs **real news articles** (500 WMT pairs) as the contrastive dataset?

### Data

| Category | N (sampled) | Source |
|---|---|---|
| ZH Poetry | 50 | `larryvrh/Chinese-Poems` |
| EN Poetry | 50 | `biglam/gutenberg-poetry-corpus` |
| ES Poetry | 50 | `andreamorgar/spanish_poetry` |
| ZH News | 50 | `wmt19` zh-en |
| EN News | 50 | `wmt19` zh-en |
| **Total** | **250** | 150 poetry + 100 news |

### Part A: Aesthetic Subspace Replication (cf. Exp 2)

| Metric | Exp 2 (Synthetic) | Poetry (Real) |
|---|---|---|
| Best classification layer | **L16** | **L0** |
| Peak AUC (5-fold CV) | 1.000 | 1.000 |
| Direction cosine (new vs Exp 2 @ L16) | — | **0.310** |
| Direction L2 norm @ L0 | 0.29 | significant |
| PCA PC1 variance @ best | 34.0% (L16) | 38.2% (L0) |
| Cross-language transfer (ZH→EN) | 1.000 | 1.000 |
| Cross-language transfer (ZH+EN→ES) | — | 0.999 |
| Steering @ best layer, +10.0 | Aesthetic amplification | **Collapse** (gibberish) |

#### Key Findings — Part A

1. **Poetry vs news is trivially separable from Layer 0.** AUC=1.000 at *every* layer including L0. Unlike the synthetic pairs in Exp 2 (which shared the same prompt structure and only differed in aesthetic valence), real poetry and real news differ fundamentally in script distribution, sentence structure, vocabulary, and genre. The model recognizes this distinction from the embedding layer alone.

2. **The poetry direction is a different direction.** Cosine similarity with Exp 2's direction is only 0.310 at Layer 16. The synthetic aesthetic direction captured *aesthetic valence within matched formats*; the poetry direction captures *genre/register differences*.

3. **Cross-language transfer remains perfect.** A direction trained on ZH poetry + ZH news generalizes perfectly to EN and ES, confirming Exp 4's finding that genre distinctions are language-universal.

4. **Steering at L0 collapses.** Unlike Exp 2's L16 steering (which gracefully amplified aesthetic language), steering at L0 corrupts all downstream computation, producing repetitive gibberish at strength +10.0. This demonstrates that L0 features are too low-level for meaningful activation steering.

5. **t-SNE reveals language clusters, not aesthetic clusters.** The 2D projection shows 5 tight clusters organized by language (ZH-poetry, EN-poetry, ES-poetry, ZH-news, EN-news), with poetry and news from the same language closer to each other than poetry from different languages. This confirms the dominant geometry axis is **language**, not aesthetics.

![Part A Direction](../Data/poetry_replication/partA_direction.png)  
![Part A Classification](../Data/poetry_replication/partA_classification.png)  
![Part A Steering](../Data/poetry_replication/partA_steering.png)  
![Part A 2D](../Data/poetry_replication/partA_2d.png)

### Part B: Aesthetic Emergence Replication (cf. Exp 3)

| Metric | Exp 3 (Synthetic) | Poetry (Real) |
|---|---|---|
| Content AUC ≥ 0.95 | **L3** | **L0** |
| Judgment AUC ≥ 0.95 | **L18** | **L0** |
| Perception–Judgment gap | **15 layers** | **0 layers** |
| Content peak AUC | 1.000 | 1.000 |
| Judgment peak AUC | 1.000 | 1.000 |

#### Key Findings — Part B

1. **No emergence gap with real data.** Both content probing (raw text) and judgment probing ("is this beautiful?") achieve AUC ≥ 0.95 at Layer 0. The 15-layer perception–judgment gap from Exp 3 disappears because poetry vs news is recognizable from surface features alone — the model doesn't need to "compute" anything to distinguish them.

2. **The Exp 3 gap was real but specific.** The 15-layer gap was not a measurement artifact — it genuinely reflected the computational cost of distinguishing *matched synthetic sentences* that differed only in aesthetic valence. With unmatched real-world data (poetry vs news), that cost drops to zero.

3. **Logit lens on judgment prompts.** Tracking P("beautiful") and P("ugly") through layers for judgment-wrapped poetry vs news shows both probabilities declining through layers (all <10⁻⁶ by L15). The model's final tokens are dominated by other vocabulary, and the "beautiful"/"ugly" tokens never reach high probability — consistent with both inputs being easily categorized without needing evaluative deliberation.

![Part B Emergence](../Data/poetry_replication/partB_emergence.png)  
![Part B Logit Lens](../Data/poetry_replication/partB_logit_lens.png)

### Part C: Aesthetic Neurons Replication (cf. Exp 5)

| Metric | Exp 5 (Synthetic) | Poetry (Real) |
|---|---|---|
| Baseline aesthetic density | 0.95% | **2.48%** |
| Top-100 neuron overlap | — | **19/100** |
| Selectivity correlation (r) | — | **0.113** |
| Contribution correlation (r) | — | **0.813** |
| Top single-neuron ablation | n2276: +172% | n822: **−51.4%** |
| Group collapse tipping point | N ≈ 1000 | N ≈ 200 |
| Brake layers (>+10%) | L8–L24 | L0, L16, L20 |
| Driver layers (<−10%) | L32–L35 | L3, L12, L24, L28, L32 |

#### Key Findings — Part C

1. **Different neuron populations.** Only 19% of the top-100 neurons overlap between the synthetic (Exp 5) and poetry directions. Per-neuron selectivity correlation is near zero (r=0.113), confirming these are genuinely different feature directions.

2. **Contribution correlation is high despite different neurons.** The per-neuron *contribution* (|weight| × |mean_diff|) correlation is 0.813 — the overall *importance structure* is similar even though the specific neurons differ. This suggests the same "important neuron" geometry applies to different feature directions.

3. **Single-neuron ablation reveals drivers, not brakes.** Unlike Exp 5 (where every single-neuron ablation *increased* aesthetic density), the poetry direction has neurons that *decrease* density when ablated (n822: −51.4%). This shows the poetry-direction neurons at L0 include genuine **drivers** — consistent with the lower-level, genre-encoding nature of this direction.

4. **Earlier collapse.** Group ablation reaches 0% density at N=200 (vs N=1000 in Exp 5). The L0 poetry representation collapses more easily because embedding-level features are less redundant than L16 mid-layer features.

5. **Layer sensitivity is more distributed.** The brake/driver pattern from Exp 5 (clean mid-layer brakes + late-layer drivers) is replaced by a patchier landscape: L3, L24, and L32 are drivers, L20 is a brake, but many layers have moderate effects. This reflects the poetry direction's presence across all layers from L0.

6. **Random control validates specificity.** Random ablation at N=200 has negligible effect (density ≈ 2.26%, close to baseline), while targeted ablation at N=200 produces complete collapse (0.0%). This confirms the poetry-direction neurons carry a qualitatively distinct role.

![Part C Neurons](../Data/poetry_replication/partC_neurons.png)

### Overall Interpretation

The poetry replication reveals a fundamental distinction between **two types of "aesthetic" directions**:

| Property | Synthetic Direction (Exp 2) | Poetry Direction (Exp 9) |
|---|---|---|
| What it captures | Aesthetic *valence* (beautiful vs ugly) | Text *genre/register* (poetry vs news) |
| Best layer | L16 (mid-network) | L0 (embedding) |
| Classification difficulty | Moderate (needs computation) | Trivial (surface features suffice) |
| Emergence gap | 15 layers (real computation) | 0 layers (no computation needed) |
| Steering effect | Graceful aesthetic amplification | Collapse at high strength |
| Neuron function | Brakes (suppress aesthetics) | Drivers (promote genre features) |
| Direction cosine | 1.0 (self) | 0.310 vs synthetic |

This shows that the original Exp 2–5 findings were **not about genre detection** — they captured something more subtle: the model's internal computation for evaluating aesthetic quality within matched contexts. The poetry replication, by using *unmatched* real data, instead captures genre-level distinctions that the model resolves trivially. **Both are valid findings, but they operate at different levels of the representational hierarchy.**

### Saved Artifacts

```
/workspace/Data/poetry_replication/
├── poetry_replication_results.npz         # All numerical results
├── partA_direction.png                    # Direction magnitude + cosine with Exp 2
├── partA_pca.png                          # PCA analysis
├── partA_classification.png               # Layer-by-layer AUC
├── partA_transfer.png                     # Cross-language transfer
├── partA_logit_lens.png                   # Logit lens at best layer
├── partA_steering.png                     # Steering experiment
├── partA_2d.png                           # t-SNE / PCA 2D projection
├── partB_emergence.png                    # Content vs judgment AUC curves
├── partB_logit_lens.png                   # Logit lens on judgment prompts
├── partC_selectivity.png                  # Neuron selectivity overview
├── partC_offline_auc.png                  # Offline masking analysis
└── partC_neurons.png                      # Ablation + layer sensitivity
```

---

## Experiment 10: Matched-Format Poetry Replication (Direction A)

**Notebook**: `matched_replication.ipynb` &nbsp;|&nbsp; **Model**: Qwen3-8B (Q4_K_M, 36 layers, 4096 dim)  
**Output**: `/workspace/Data/direction_A/` (plots, saved results)  
**Replicates**: Experiments 2, 3, 5 — but with **matched-format** data (poem vs poem)

### Question

Exp 9 showed that poetry vs news is trivially separable at L0 because the formats differ completely. Exp 2–5 used synthetic prompts with matched formats. What happens when we use **real data with matched formats** — high-imagery poems vs low-imagery poems from the same corpus?

### Data

50 high-imagery Chinese poems (mean imagery score ≈ 0.18) vs 50 zero-imagery Chinese poems (score = 0.00), selected from `larryvrh/Chinese-Poems`. Both groups are classical Tang/Song poems with matched length (~38 chars). Imagery score = fraction of characters from a curated 33-character imagery set (月花风雪春秋山水云霞星夜雨露竹松柳桃兰菊荷梅鸟蝶泉湖海虹烟雾光影梦香). Effect size between group scores: d = 8.58.

### Part A: Aesthetic Subspace (cf. Exp 2)

| Metric | Exp 2 (Synthetic) | Exp 9 (Poetry vs News) | **Exp 10 (High vs Low Imagery)** |
|---|---|---|---|
| Best classification layer | L16 | L0 | **L4** |
| Peak AUC (5-fold CV) | 1.000 | 1.000 | **0.968** |
| Direction cosine vs Exp 2 | — | 0.310 | **0.166** |
| Direction cosine vs Exp 9 | — | — | **−0.265** |
| PCA PC1 variance | 34.0% (L16) | 38.2% (L0) | **12.1% (L4)** |

#### Key Findings — Part A

1. **Best layer is L4 — between L0 (trivial genre) and L16 (abstract valence).** Imagery distinction requires more computation than genre detection but less than aesthetic valence judgment. L4 sits in the early processing regime, consistent with imagery being a mid-level semantic feature.

2. **AUC = 0.968 — the first imperfect separation.** Every prior experiment achieved AUC = 1.000. The imagery direction is the first genuinely *difficult* classification task — some poems with no imagery words still "feel" imagistic, and vice versa.

3. **The direction is nearly orthogonal to both Exp 2 and Exp 9.** Cosine similarity of 0.166 (vs Exp 2) and −0.265 (vs Exp 9) shows this is a **third distinct direction** in activation space. Imagery ≠ aesthetic valence ≠ genre.

4. **Low PCA variance (12.1%) indicates distributed encoding.** Unlike genre encoding (38.2% at L0 in Exp 9), imagery information is spread across many dimensions at L4, not dominated by a single axis.

![Part A Subspace](../Data/direction_A/partA_subspace.png)

### Part B: Layer-by-Layer Emergence (cf. Exp 3)

| Metric | Exp 3 (Synthetic) | Exp 9 (Poetry vs News) | **Exp 10 (Imagery)** |
|---|---|---|---|
| Content AUC ≥ 0.95 | L3 | L0 | **L0** |
| Judgment AUC ≥ 0.95 | L18 | L0 | **L3** |
| Perception–Judgment gap | **15 layers** | 0 layers | **3 layers** |

#### Key Findings — Part B

1. **A 3-layer emergence gap partially returns.** Content probing ("which category?") reaches AUC ≥ 0.95 at L0, while judgment probing ("is this more beautiful?") needs L3. The gap is smaller than Exp 3's 15 layers but nonzero — confirming that evaluative judgment requires additional computation beyond pattern recognition, even for matched-format data.

2. **The gap scales with task difficulty.** Genre (0 layers) < imagery judgment (3 layers) < aesthetic valence (15 layers). This suggests the emergence gap is a **continuous function of the abstractness** of the required computation.

![Part B Emergence](../Data/direction_A/partB_emergence.png)

### Part C: Neuron Analysis (cf. Exp 5)

| Metric | Exp 5 (Synthetic) | Exp 9 (Poetry vs News) | **Exp 10 (Imagery)** |
|---|---|---|---|
| Baseline aesthetic density | 0.95% | 2.48% | **1.71%** |
| Top-100 neuron overlap vs Exp 5 | — | 19/100 | **10/100** |
| Top brake layer (>+10%) | L8–L24 | L0, L16, L20 | **L16 (+55.3%), L20, L28, L35** |
| Top driver layer (<−10%) | L32–L35 | L3, L12, L24, L28, L32 | **L0 (−100%), L8, L12 (−46.9%)** |

#### Key Findings — Part C

1. **Only 10/100 neuron overlap with Exp 5.** The imagery neurons are a largely distinct population from both the synthetic aesthetic neurons (Exp 5) and the genre neurons (Exp 9). Three directions, three neuron sets.

2. **L16 returns as a brake layer (+55.3%).** Despite the best classification layer being L4, L16 plays a strong inhibitory role for imagery — ablating L16 neurons dramatically *increases* the imagery signal in generated text. This echoes Exp 5's L16 brake finding.

3. **L0 is a 100% driver.** Ablating top L0 imagery neurons completely eliminates the imagery signal (density → 0%). This is consistent with imagery being anchored in early embedding features.

4. **L12 is a strong driver (−46.9%).** This layer contributes substantially to maintaining imagery representations, bridging the early (L0–L4) encoding with the mid-layer (L16) braking architecture.

![Part C Neurons](../Data/direction_A/partC_neurons.png)

### Saved Artifacts

```
/workspace/Data/direction_A/
├── dirA_results.npz            # All numerical results
├── partA_subspace.png          # Classification + direction analysis
├── partB_emergence.png         # Content vs judgment AUC curves
└── partC_neurons.png           # Ablation + layer sensitivity
```

---

## Experiment 11: Token-by-Token Temporal Dynamics (Direction G)

**Notebook**: `token_dynamics.ipynb` &nbsp;|&nbsp; **Model**: Qwen3-8B (Q4_K_M, 36 layers, 4096 dim)  
**Output**: `/workspace/Data/direction_G/` (plots, saved results)

### Question

All prior experiments extracted activations at the **final token** position. But aesthetic perception likely builds incrementally as the model reads through text. How does the aesthetic projection evolve token-by-token? Does it spike at imagery words (月, 花, 雪) or build smoothly?

### Method

**Progressive prefix forwarding**: For each text of length *n*, forward text[:k] for k = 1, 2, …, n. At each prefix, extract activations at 10 layers [L0, L4, L8, L12, L16, L20, L24, L28, L32, L35] and project onto Exp 2's aesthetic direction (from L16). This yields a **trajectory** of aesthetic projection values through the text.

**Texts**: 5 high-imagery Chinese poems, 3 low-imagery Chinese poems, 3 news articles = 11 texts total.

### Results

#### Token-Level Imagery Effect

| Metric | Value |
|---|---|
| Imagery tokens (N) | 38 |
| Non-imagery tokens (N) | 186 |
| Mean projection: imagery tokens | 60.21 ± 231.31 |
| Mean projection: non-imagery tokens | 40.39 ± 184.51 |
| Cohen's d | 0.095 |
| t-test p-value | 0.5669 (not significant) |
| Mann-Whitney U p-value | 0.0000 (rank-order significant) |

#### Trajectory Statistics (L16)

| Metric | Value |
|---|---|
| Mean autocorrelation | 0.324 ± 0.247 |
| Mean position slope | −9.24 ± 1.79 |
| Text-type mean: high-imagery poems | 48.55 |
| Text-type mean: low-imagery poems | 26.70 |
| Text-type mean: news articles | 51.67 |

#### Multi-Layer Imagery Gap

| Layer | Cohen's d |
|---|---|
| L0 | 0.229 |
| L4 | **−0.231** (reversal!) |
| L8 | 0.107 |
| L12 | 0.128 |
| L16 | 0.095 |
| L20 | 0.097 |
| L24 | 0.118 |
| L28 | 0.173 |
| L32 | 0.135 |
| L35 | **0.388** (peak) |

### Key Findings

1. **No token-level spike at imagery words.** The aesthetic projection does NOT spike when the model encounters imagery characters (月, 花, etc.). Cohen's d = 0.095 is negligible, and the t-test is non-significant (p = 0.57). Aesthetic representation is **context-dependent**, not word-triggered.

2. **Rank-order effect is significant.** Despite no mean difference, Mann-Whitney U (p < 0.0001) shows imagery tokens tend to have higher *ranks* in the projection distribution. The effect is real but subtle — imagery tokens shift the distribution shape without changing the mean.

3. **Aesthetic projection declines with position.** The mean slope is −9.24, meaning the aesthetic projection *decreases* as the model reads more text. This may reflect the model shifting from poetic feature encoding to prediction/completion computations.

4. **Weak autocorrelation (r = 0.324).** The trajectory is not smooth — aesthetic projection fluctuates substantially from token to token. This rules out a simple "accumulation" model where aesthetic evidence builds monotonically.

5. **L4 shows a reversal (d = −0.231).** At Layer 4, non-imagery tokens have *higher* aesthetic projection than imagery tokens. This inversion suggests L4 performs some kind of normalization or contrast computation on imagery features.

6. **Peak imagery effect is at L35 (d = 0.388).** The deepest layer shows the clearest imagery differentiation, suggesting that imagery information is refined through the full depth of the network before surfacing as a token-level effect at the output.

7. **High-imagery poems (48.55) vs low-imagery poems (26.70) shows text-level separation.** While individual token effects are weak, the average trajectory across entire poems differs by ≈22 units — consistent with Exp 10's classification findings.

### Interpretation

The temporal dynamics reveal that aesthetic processing in transformers is fundamentally **holistic, not compositional**:

- The model does not build an aesthetic percept by summing per-token "beauty signals"
- Instead, aesthetic information emerges from **global context integration** — the full trajectory matters, not individual token positions
- The declining slope and weak autocorrelation suggest the model "front-loads" aesthetic feature extraction and then transitions to other computations
- The L4 reversal and L35 peak imply a **bidirectional refinement process** across layers: early layers contrast imagery, late layers consolidate it

![Individual Trajectories](../Data/direction_G/trajectories_individual.png)  
![Aggregate Analysis](../Data/direction_G/aggregate_analysis.png)  
![Multi-Layer Dynamics](../Data/direction_G/multi_layer_dynamics.png)  
![Layer Gap Curve](../Data/direction_G/layer_gap_curve.png)

### Saved Artifacts

```
/workspace/Data/direction_G/
├── dirG_results.npz              # All numerical results
├── trajectories_individual.png   # Per-text trajectories
├── aggregate_analysis.png        # Token-level statistics
├── multi_layer_dynamics.png      # Multi-layer comparison
└── layer_gap_curve.png           # Layer-by-layer Cohen's d
```

---

## Experiment 12: Full Dynasty Trajectory + Multilingual Poetic Geography (Direction E)

**Notebook**: `dynasty_multilingual.ipynb` &nbsp;|&nbsp; **Model**: Qwen3-8B (Q4_K_M, 36 layers, 4096 dim)  
**Output**: `/workspace/Data/direction_E/` (plots, saved results)  
**Builds on**: Exp 2 (aesthetic direction), Exp 4 (cross-lingual), Exp 7 (Tang vs Song)

### Question

Two questions in one experiment:
1. Does the historical evolution of Chinese poetry (唐→宋→元→明→清) trace a continuous trajectory through activation space?
2. Across 14 languages from 6 language families and 6 geographic regions, does **language family** or **geographic/cultural proximity** better predict how the model represents poetry?

### Data

| Source | Languages | N per group |
|---|---|---|
| larryvrh/Chinese-Poems | ZHO × 5 dynasties | 30 per dynasty = 150 |
| PoetryMTEB/MultilingualPoetryDatabase | JPN, KOR, ARA, TUR, FAS, HIN, RUS, DEU, ENG, FRA, SPA, ITA, POR | 30 each (KOR: 15) |
| **Total** | **14 languages, 5 dynasties** | **~525 poems** |

**Language contrasts:**

| Language | Family | Sub-family | Geography |
|---|---|---|---|
| ZHO | Sino-Tibetan | Sinitic | East Asia |
| JPN | Japonic | — | East Asia |
| KOR | Koreanic | — | East Asia |
| ARA | Afro-Asiatic | Semitic | Middle East |
| TUR | Turkic | Oghuz | Middle East |
| FAS | Indo-European | Iranian | Middle East |
| HIN | Indo-European | Indo-Aryan | South Asia |
| RUS | Indo-European | Slavic | E. Europe |
| DEU | Indo-European | Germanic | W. Europe |
| ENG | Indo-European | Germanic | W. Europe |
| FRA | Indo-European | Romance | W. Europe |
| SPA | Indo-European | Romance | W. Europe |
| ITA | Indo-European | Romance | S. Europe |
| POR | Indo-European | Romance | S. Europe |

### Part A: Chinese Dynasty Trajectory

| Dynasty | Aesthetic Projection (L16) | Closest Neighbor | Cosine Distance |
|---|---|---|---|
| Tang (唐) | 5.47 ± 0.95 | Ming | 0.0118 |
| Song (宋) | 5.66 ± 1.09 | Tang | 0.0145 |
| Yuan (元) | 4.74 ± 1.62 | Qing | 0.0136 |
| Ming (明) | 5.66 ± 1.15 | Tang | 0.0118 |
| Qing (清) | 4.77 ± 1.26 | Yuan | 0.0136 |

#### Key Findings — Part A

1. **Two dynasty clusters, not a continuous path.** The PCA trajectory reveals two groups: **Tang-Song-Ming** (close, cosine ≈ 0.012–0.015) and **Yuan-Qing** (close, cosine = 0.014). The gap between groups is 3–5× larger (cosine ≈ 0.045–0.065).

2. **Chronological order does not predict activation distance.** Spearman r = 0.197, p = 0.585 (not significant). Ming (1368–1644) is closest to Tang (618–907) despite being separated by 450+ years, while Song (960–1279) — temporally adjacent to Yuan (1271–1368) — is far from it.

3. **Aesthetic intensity differs: Tang/Song/Ming > Yuan/Qing.** The high-aesthetic cluster (Tang 5.47, Song 5.66, Ming 5.66) has ≈18% higher aesthetic projection than the low-aesthetic cluster (Yuan 4.74, Qing 4.77). This aligns with literary history: Tang, Song, and Ming are considered the golden ages of Chinese classical poetry, while Yuan and Qing poetry emphasized different literary values.

4. **PCA explains 24% + 6% = 30%** of dynasty variance at L16, indicating substantial but not overwhelming stylistic differentiation.

![Dynasty Trajectory](../Data/direction_E/dynasty_trajectory.png)

### Part B: Multilingual Poetry Map

| Metric | Family | Geography | Sub-family |
|---|---|---|---|
| Within-group mean cosine | 0.124 ± 0.067 | 0.158 ± 0.090 | 0.079 ± 0.016 |
| Between-group mean cosine | 0.215 ± 0.083 | 0.182 ± 0.088 | 0.187 ± 0.087 |
| Cohen's d | **1.020** | 0.279 | **1.216** |
| Mann-Whitney U p | **< 0.0001** | 0.137 (NS) | **< 0.0001** |
| Mantel r (permutation) | **0.499** (p=0.016) | 0.098 (p=0.149) | — |

#### Key Findings — Part B

1. **Language family strongly predicts poetic distance.** The Mantel test confirms a significant correlation (r=0.50, p=0.016) between language family structure and activation distance. Same-family languages are 42% closer (0.124 vs 0.215).

2. **Geography does NOT significantly predict distance.** The geography effect is non-significant (Mantel p=0.149). Within-geography pairs (0.158) are only marginally closer than between-geography pairs (0.182).

3. **Sub-family is the strongest predictor.** Romance languages (FRA-SPA: 0.057, FRA-ITA: 0.076, SPA-POR: 0.072) form an extremely tight cluster (mean 0.079). Germanic (ENG-DEU: 0.108) is also compact. Cohen's d = 1.216 for sub-family is higher than family (1.020) — the model encodes fine-grained linguistic kinship.

4. **Family dominates at every layer.** The family effect (d ≈ 0.6–1.3) exceeds the geography effect (d ≈ 0.1–0.6) at all 10 layers tested. The ratio narrows at L32 (1.08×) but never reverses.

5. **Peak family effect is at L35 (d = 1.254)**, suggesting the deepest layers encode the most family-discriminative features.

![Multilingual Map](../Data/direction_E/multilingual_map.png)  
![Distance Analysis](../Data/direction_E/distance_analysis.png)

### Part C: Cultural Influence Analysis

| Pair | Relationship | Cosine Distance |
|---|---|---|
| FRA-SPA | Same sub-family (Romance) | **0.057** |
| FRA-ITA | Same sub-family (Romance) | **0.076** |
| SPA-POR | Same sub-family (Romance) | **0.072** |
| ENG-DEU | Same sub-family (Germanic) | **0.108** |
| ENG-FRA | Same geography, diff family | 0.090 |
| RUS-DEU | Diff family, diff geography | 0.087 |
| HIN-FRA | Same family IE, diff geography | 0.127 |
| ARA-FAS | Cultural influence (Arabic→Persian) | 0.163 |
| JPN-KOR | Geographic neighbors (East Asia) | 0.209 |
| ZHO-KOR | Cultural influence (Chinese→Korean) | 0.241 |
| FAS-TUR | Cultural influence (Persian→Turkish) | 0.226 |
| ARA-TUR | Cultural influence (Arabic→Turkish) | 0.285 |
| ZHO-JPN | Cultural influence (Chinese→Japanese) | **0.344** |
| ZHO-ARA | Control (unrelated) | **0.418** |

#### Key Findings — Part C

1. **Cultural influence ≠ representational proximity.** The mean distance for cultural-influence pairs (0.258) is closer to unrelated controls (0.229) than to same-sub-family pairs (0.078). In the model's representation of poetry, **linguistic genetics dominate cultural transmission.**

2. **East Asian languages are far apart despite cultural influence.** ZHO-JPN (0.344) is the most distant cultural-influence pair — further than many unrelated pairs. Despite centuries of Chinese literary influence on Japanese poetry (漢文, 漢詩), the model treats them as fundamentally different writing systems.

3. **Persian-Arabic is the closest cultural pair (0.163)**, likely because Persian poetry heavily borrowed Arabic vocabulary, script, and poetic forms (ghazal, qasida). Yet this is still 2× further than same-sub-family pairs.

4. **Indo-European is remarkably coherent across geography.** HIN-FRA (0.127) is closer than ZHO-JPN (0.344) despite being separated by 7,000 km, confirming that shared linguistic ancestry creates representational proximity even across different geographic regions.

5. **ENG-FRA (0.090) ≈ RUS-DEU (0.087)** — both cross-family European pairs are very close, but this likely reflects the model's training data bias (extensive European text) rather than pure linguistic kinship.

![Multi-Layer Cultural](../Data/direction_E/multilayer_cultural.png)

### Overall Interpretation

Experiment 12 delivers a clear answer: **language family, not geography or cultural influence, is the primary organizing principle** for how Qwen3-8B represents poetry across languages.

| Factor | Effect (Cohen's d, L16) | Significance |
|---|---|---|
| Sub-family (Romance, Germanic, ...) | **1.216** | p < 0.0001 |
| Language family | **1.020** | p < 0.0001 (Mantel p=0.016) |
| Geography | 0.279 | NS (p=0.137) |
| Cultural influence | ≈ control | — |

This has implications for the aesthetic universality found in Exp 4: the cross-lingual aesthetic transfer likely works **because aesthetics is more abstract than language identity**, and the aesthetic direction cuts across all language clusters in a shared subspace. Meanwhile, the dynasty analysis reveals that Chinese poetic traditions cluster by **style school** (Tang=Song=Ming vs Yuan=Qing) rather than chronological order, reflecting the model's sensitivity to literary-historical periods.

### Saved Artifacts

```
/workspace/Data/direction_E/
├── exp12_results.npz             # All numerical results
├── dynasty_trajectory.png        # PCA + aesthetic projection by dynasty
├── multilingual_map.png          # t-SNE/PCA 2×2 grid (14 languages)
├── distance_analysis.png         # Distance heatmaps + within/between bars
└── multilayer_cultural.png       # Layer curves + cultural influence comparison
```

---

## Shared Utilities

| File | Description |
|---|---|
| `ActivationCollector.py` | Online mean/std collector with cumulative statistics per layer |
| `utils.py` | `cosine_similarity`, `softmax`, `get_top_k_words_from_logits`, prompt I/O helpers |
| `prompt_templet/` | System prompts (JSONL format) for translation and other tasks |

---

## Environment

- **Hardware**: NVIDIA GB10 (aarch64), unified memory
- **Software**: CUDA 13.1, Python 3.12, NeuroScope 1.2.3
- **Key env vars**:
  ```bash
  GGML_CUDA_DISABLE_GRAPHS=1
  HF_HOME=/workspace/Data/huggingface_cache
  TORCH_HOME=/workspace/Data/torch_cache
  ```
- **Models**: `/workspace/Model/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf`

---

## What's Next

With 12 experiments completed, the picture has deepened significantly: the model encodes at least **three distinct directional axes** — aesthetic valence (L16), genre/register (L0), and imagery density (L4) — and organizes poetic representations primarily by **language family** (not geography or cultural influence), while Chinese dynasty styles cluster by **literary school** rather than chronological order. The following directions remain open.

### ~~Direction A: Matched-Format Poetry Replication~~ → ✅ Experiment 10

### Direction B: Purified Language-Neutral Aesthetic Direction

Exp 8 revealed the aesthetic direction is ~36% Chinese-style-specific. Three approaches to extract a clean, language-universal component:
1. **Projection**: Subtract the language-identity subspace (computed from same-content ZH-EN pairs) from the aesthetic direction
2. **CCA/DCCA**: Find the maximal correlation subspace between ZH and EN aesthetic activations — this is the shared content component
3. **Contrastive learning**: Train on (poem_ZH, translation_EN) pairs where the target is shared activation structure, not classification

Validation: the purified direction should achieve equal AUC for ZH-Poetry and EN-Poetry (currently d=0.699 vs d=0.099).

### Direction C: Fine-Grained Aesthetic Concept Decomposition

> *Is beauty = symmetry + harmony + proportion?*

Decompose the monolithic aesthetic direction into interpretable sub-components:
- Construct contrastive prompts for specific aesthetic dimensions: **symmetry**, **harmony**, **color richness**, **rhythm/cadence**, **emotional resonance**, **concreteness of imagery**
- Extract per-concept direction vectors and test whether `aesthetic_dir ≈ Σ wᵢ · concept_dirᵢ`
- Use Exp 7's dynasty data to test: Tang poetry should load on *grandeur/boldness* while Song poetry loads on *subtlety/restraint*

### Direction D: Multi-Model Comparison

Test whether the aesthetic subspace is universal across transformer LLMs:
- **LLaMA-3-8B** (already available as Q4_K_S GGUF in Exp 1)
- **Mistral-7B**, **DeepSeek-7B** (download GGUF quantizations)
- For each model, replicate Exp 2 (direction extraction) and Exp 3 (emergence timing)
- Key question: Is the 15-layer gap a property of transformers in general, or specific to Qwen3's training distribution?

### ~~Direction E: Full Dynasty Trajectory (元→明→清)~~ → ✅ Experiment 12

### Direction F: Aesthetic SAE (Sparse Autoencoder) Analysis

Train a sparse autoencoder on Layer 16 activations to discover monosemantic aesthetic features:
- Current neurons are **polysemantic** — each participates in many features (Cohen's d = 0.793, not >>1)
- SAE features should yield **sharper** aesthetic selectivity than raw neurons
- Predict: SAE features will decompose the distributed aesthetic code into a small number (10–50) of interpretable features (e.g., "visual beauty", "auditory harmony", "literary elegance")

### ~~Direction G: Temporal Dynamics — Token-by-Token Emergence~~ → ✅ Experiment 11

### Direction H: Causal Circuit Tracing

Exps 5 and 9 identified neurons but not the **circuits** connecting them:
- Use **activation patching** (path patching) from Exp 3's attention heads to Exp 5's brake neurons
- Map the information flow: Which attention heads at L24–28 write to which brake neurons at L16?
- Goal: reconstruct the minimal circuit that implements the brake/driver aesthetic control architecture

---

**Author**: Shuaizhou Wang  
**Date**: March 2026  
**License**: MIT License
