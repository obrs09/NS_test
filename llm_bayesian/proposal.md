# Motivation: Causal Claims and Their Uncertainty

Recent work on causal reasoning and interventions in large language
models has demonstrated that carefully designed manipulations of
internal components can lead to meaningful behavioral changes. These
results strongly suggest that internal representations play a causal
role in model behavior.

At the same time, causal analyses often implicitly assume a single
underlying mechanism. In complex systems like LLMs, however, multiple
competing internal explanations may be consistent with the same observed
intervention effects.

This motivates the question:

How confident should we be in a particular causal explanation of LLM
behavior?

# Research Question

I am interested in whether Bayesian causal reasoning can provide a
principled framework for representing uncertainty over competing
mechanistic hypotheses in LLMs, rather than committing to a single
causal structure early in the analysis.

# Probabilistic Causal Framing

From this perspective, internal components such as neurons, attention
heads, or subspaces can be viewed as latent causal variables.
Interventions then serve as evidence that updates beliefs over which
mechanisms are most plausible.

Conceptually, this corresponds to reasoning about:

$$\mathbb{P}(\text{mechanism} \mid \text{intervention outcomes})$$

rather than asking whether a particular intervention "works" in
isolation.

# Hypothesis: Hallucination as Causal Ambiguity

A tentative hypothesis is that hallucinations and unfaithful generations
correspond to high uncertainty over causal support: the output is
produced without a clearly identifiable or stable internal mechanism.

Under this view:

Faithful generations correspond to relatively well-identified causal
explanations.

Hallucinations correspond to causal ambiguity, where multiple weak
explanations compete.

This framing naturally connects behavioral evaluation with internal
causal analysis.

# Methodological Direction

One possible instantiation of this idea is to combine causal
intervention experiments with Bayesian inference, using repeated
interventions to update beliefs over candidate mechanisms. Importantly,
this does not replace causal testing, but instead structures it as an
iterative inference process.

# Practical Feasibility

To support such experiments, I have developed a high-performance
inference and intervention engine based on llama.cpp, enabling efficient
large-scale causal testing across internal components.

# Intended Contribution

Conceptual: Explicitly representing uncertainty over causal mechanisms
in LLMs.

Methodological: Integrating Bayesian inference with causal intervention
workflows.

Empirical: Studying hallucination and unfaithfulness through causal
ambiguity rather than purely behavioral metrics.
