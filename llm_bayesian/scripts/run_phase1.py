#!/usr/bin/env python3
"""
Phase 1 — Bayesian Causal Tracing with Hand-Crafted Prompts
============================================================
100 factual-recall prompts, activation patching (causal tracing),
Bayesian posterior entropy to compare faithful vs hallucinated.

Configurable variables are in the CONFIG section below.
"""

import os, sys, time, re, argparse
import numpy as np
from scipy import stats
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans']
import warnings; warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# CONFIG — Edit these variables to change experiment parameters
# ═══════════════════════════════════════════════════════════════════
CONFIG = dict(
    # ── Model ──
    model_path   = "/workspace/Model/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf",
    n_ctx        = 2048,
    n_gpu_layers = 99,

    # ── Output ──
    output_dir = '/workspace/Data/bayesian_phase1',

    # ── Causal tracing ──
    corrupt_entity = "Zyxwv",     # Nonsense entity for corruption
    gen_max_tokens = 20,          # Max tokens for answer generation
    gen_temperature = 0.01,
    gen_top_k      = 1,
    gen_seed       = 42,

    # ── Mechanism grouping ──
    # K layer groups (dict of name → list of layer indices)
    mechanisms = {
        'L0-5 Shallow':  list(range(0, 6)),
        'L6-11 Early':   list(range(6, 12)),
        'L12-17 Mid':    list(range(12, 18)),
        'L18-23 Deep':   list(range(18, 24)),
        'L24-29 Late':   list(range(24, 30)),
        'L30-35 Final':  list(range(30, 36)),
    },

    # ── Bayesian inference ──
    beta_default = 5.0,
    beta_sensitivity_values = [0.5, 1, 2, 5, 10, 20, 50],

    # ── Significance filter ──
    causal_effect_threshold = 0.01,  # min |p_clean - p_corrupt|
)
# ═══════════════════════════════════════════════════════════════════


# ── Default dataset (can be replaced by loading from JSON) ──
DATASET = [
    # (template_with_{}, subject, [expected_answers], category)
    # ════════════ World Capitals (30) ════════════
    ("The capital of {} is", "France", ["Paris"], "cap_easy"),
    ("The capital of {} is", "Japan", ["Tokyo"], "cap_easy"),
    ("The capital of {} is", "China", ["Beijing", "Peking"], "cap_easy"),
    ("The capital of {} is", "Germany", ["Berlin"], "cap_easy"),
    ("The capital of {} is", "Italy", ["Rome"], "cap_easy"),
    ("The capital of {} is", "Spain", ["Madrid"], "cap_easy"),
    ("The capital of {} is", "Russia", ["Moscow"], "cap_easy"),
    ("The capital of {} is", "India", ["New Delhi", "Delhi"], "cap_easy"),
    ("The capital of {} is", "Brazil", ["Brasilia"], "cap_easy"),
    ("The capital of {} is", "Canada", ["Ottawa"], "cap_easy"),
    ("The capital of {} is", "Australia", ["Canberra"], "cap_easy"),
    ("The capital of {} is", "Mexico", ["Mexico City"], "cap_easy"),
    ("The capital of {} is", "South Korea", ["Seoul"], "cap_easy"),
    ("The capital of {} is", "Egypt", ["Cairo"], "cap_easy"),
    ("The capital of {} is", "Argentina", ["Buenos Aires"], "cap_easy"),
    ("The capital of {} is", "Thailand", ["Bangkok"], "cap_easy"),
    ("The capital of {} is", "Turkey", ["Ankara"], "cap_easy"),
    ("The capital of {} is", "Sweden", ["Stockholm"], "cap_easy"),
    ("The capital of {} is", "Poland", ["Warsaw"], "cap_easy"),
    ("The capital of {} is", "Indonesia", ["Jakarta"], "cap_easy"),
    ("The capital of {} is", "Myanmar", ["Naypyidaw", "Nay Pyi Taw"], "cap_hard"),
    ("The capital of {} is", "Kazakhstan", ["Astana", "Nur-Sultan"], "cap_hard"),
    ("The capital of {} is", "Tanzania", ["Dodoma"], "cap_hard"),
    ("The capital of {} is", "Ivory Coast", ["Yamoussoukro"], "cap_hard"),
    ("The capital of {} is", "Belize", ["Belmopan"], "cap_hard"),
    ("The capital of {} is", "Burkina Faso", ["Ouagadougou"], "cap_hard"),
    ("The capital of {} is", "Palau", ["Ngerulmud", "Melekeok"], "cap_hard"),
    ("The capital of {} is", "Nauru", ["Yaren"], "cap_hard"),
    ("The capital of {} is", "Micronesia", ["Palikir"], "cap_hard"),
    ("The capital of {} is", "Eswatini", ["Mbabane", "Lobamba"], "cap_hard"),
    # ════════════ Chemical Symbols (20) ════════════
    ("The chemical symbol for {} is", "gold", ["Au"], "chem_easy"),
    ("The chemical symbol for {} is", "silver", ["Ag"], "chem_easy"),
    ("The chemical symbol for {} is", "iron", ["Fe"], "chem_easy"),
    ("The chemical symbol for {} is", "copper", ["Cu"], "chem_easy"),
    ("The chemical symbol for {} is", "sodium", ["Na"], "chem_easy"),
    ("The chemical symbol for {} is", "potassium", ["K"], "chem_easy"),
    ("The chemical symbol for {} is", "mercury", ["Hg"], "chem_easy"),
    ("The chemical symbol for {} is", "lead", ["Pb"], "chem_easy"),
    ("The chemical symbol for {} is", "tin", ["Sn"], "chem_easy"),
    ("The chemical symbol for {} is", "tungsten", ["W"], "chem_easy"),
    ("The chemical symbol for {} is", "antimony", ["Sb"], "chem_hard"),
    ("The chemical symbol for {} is", "strontium", ["Sr"], "chem_hard"),
    ("The chemical symbol for {} is", "molybdenum", ["Mo"], "chem_hard"),
    ("The chemical symbol for {} is", "vanadium", ["V"], "chem_hard"),
    ("The chemical symbol for {} is", "praseodymium", ["Pr"], "chem_hard"),
    ("The chemical symbol for {} is", "ytterbium", ["Yb"], "chem_hard"),
    ("The chemical symbol for {} is", "hafnium", ["Hf"], "chem_hard"),
    ("The chemical symbol for {} is", "rhenium", ["Re"], "chem_hard"),
    ("The chemical symbol for {} is", "thallium", ["Tl"], "chem_hard"),
    ("The chemical symbol for {} is", "gadolinium", ["Gd"], "chem_hard"),
    # ════════════ History (20) ════════════
    ("{} was born in", "Albert Einstein", ["1879"], "hist_easy"),
    ("{} was born in", "Isaac Newton", ["1642", "1643"], "hist_easy"),
    ("{} was born in", "Leonardo da Vinci", ["1452"], "hist_easy"),
    ("{} was born in", "William Shakespeare", ["1564"], "hist_easy"),
    ("{} was born in", "Napoleon Bonaparte", ["1769"], "hist_easy"),
    ("{} began in", "World War I", ["1914"], "hist_easy"),
    ("{} ended in", "World War II", ["1945"], "hist_easy"),
    ("{} was discovered in", "penicillin", ["1928"], "hist_easy"),
    ("{} first landed on the moon in", "Neil Armstrong", ["1969"], "hist_easy"),
    ("{} took place in", "the French Revolution", ["1789"], "hist_easy"),
    ("{} was signed in", "the Magna Carta", ["1215"], "hist_hard"),
    ("{} was completed in", "the Suez Canal", ["1869"], "hist_hard"),
    ("{} was born in", "Confucius", ["551"], "hist_hard"),
    ("{} died in", "Genghis Khan", ["1227"], "hist_hard"),
    ("{} was first performed in", "Beethoven's Ninth Symphony", ["1824"], "hist_hard"),
    ("{} was invented in", "the printing press", ["1440", "1450"], "hist_hard"),
    ("{} was founded in", "the Ottoman Empire", ["1299"], "hist_hard"),
    ("{} was built in", "the Colosseum", ["70", "72", "80"], "hist_hard"),
    ("{} was published in", "the Communist Manifesto", ["1848"], "hist_hard"),
    ("{} gained independence in", "Brazil", ["1822"], "hist_hard"),
    # ════════════ Science (15) ════════════
    ("The atomic number of {} is", "hydrogen", ["1"], "sci_easy"),
    ("The atomic number of {} is", "carbon", ["6"], "sci_easy"),
    ("The atomic number of {} is", "oxygen", ["8"], "sci_easy"),
    ("The atomic number of {} is", "iron", ["26"], "sci_easy"),
    ("The atomic number of {} is", "gold", ["79"], "sci_easy"),
    ("The boiling point of {} in degrees Celsius is", "water", ["100"], "sci_easy"),
    ("The freezing point of {} in degrees Celsius is", "water", ["0", "zero"], "sci_easy"),
    ("The number of chromosomes in a {} cell is", "human", ["46"], "sci_easy"),
    ("The atomic number of {} is", "uranium", ["92"], "sci_hard"),
    ("The atomic number of {} is", "plutonium", ["94"], "sci_hard"),
    ("The melting point of {} in degrees Celsius is", "iron", ["1538", "1535", "1536"], "sci_hard"),
    ("The half-life of {} in years is approximately", "carbon-14", ["5730", "5700"], "sci_hard"),
    ("The atomic number of {} is", "berkelium", ["97"], "sci_hard"),
    ("The atomic number of {} is", "lawrencium", ["103"], "sci_hard"),
    ("The atomic number of {} is", "seaborgium", ["106"], "sci_hard"),
    # ════════════ General Knowledge (15) ════════════
    ("The author of {} is", "Romeo and Juliet", ["Shakespeare"], "gen_easy"),
    ("The author of {} is", "Harry Potter", ["Rowling", "J.K."], "gen_easy"),
    ("The inventor of {} is", "the telephone", ["Bell", "Meucci"], "gen_easy"),
    ("The currency of {} is the", "Japan", ["yen"], "gen_easy"),
    ("The currency of {} is the", "United Kingdom", ["pound"], "gen_easy"),
    ("The largest planet in {} is", "the solar system", ["Jupiter"], "gen_easy"),
    ("The longest river in {} is the", "Africa", ["Nile"], "gen_easy"),
    ("The national language of {} is", "Brazil", ["Portuguese"], "gen_easy"),
    ("The author of {} is", "The Brothers Karamazov", ["Dostoevsky", "Dostoyevsky", "Dostoevsk"], "gen_hard"),
    ("The author of {} is", "One Hundred Years of Solitude", ["Marquez", "Garcia"], "gen_hard"),
    ("The composer of {} is", "The Four Seasons", ["Vivaldi"], "gen_hard"),
    ("The currency of {} is the", "Bhutan", ["ngultrum"], "gen_hard"),
    ("The national animal of {} is the", "Scotland", ["unicorn"], "gen_hard"),
    ("The architect of {} is", "the Sagrada Familia", ["Gaudi"], "gen_hard"),
    ("The inventor of {} is", "dynamite", ["Nobel"], "gen_easy"),
]


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def strip_think(text):
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)

def check_match(generated, expected_list):
    gen = generated.lower().strip()
    for exp in expected_list:
        if len(exp) <= 2:
            if re.search(r'(?<![a-zA-Z0-9])' + re.escape(exp.lower()) + r'(?![a-zA-Z0-9])', gen):
                return True
        else:
            if exp.lower() in gen:
                return True
    return False

def compute_posteriors_and_entropy(IE_matrix, beta):
    N, K = IE_matrix.shape
    posteriors = np.zeros((N, K))
    entropies = np.zeros(N)
    for i in range(N):
        log_liks = beta * IE_matrix[i]
        log_post = log_liks - logsumexp(log_liks)
        post = np.exp(log_post)
        posteriors[i] = post
        entropies[i] = -np.sum(post * np.log(post + 1e-30))
    return posteriors, entropies


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', dest='model_path')
    p.add_argument('--n-ctx', type=int)
    p.add_argument('--beta', type=float, dest='beta_default')
    p.add_argument('--output-dir')
    p.add_argument('--corrupt-entity')
    args = p.parse_args()
    for k, v in vars(args).items():
        if v is not None:
            CONFIG[k] = v


def main():
    parse_args()
    C = CONFIG
    os.environ['GGML_CUDA_DISABLE_GRAPHS'] = '1'
    os.makedirs(C['output_dir'], exist_ok=True)

    import neuroscope
    engine = neuroscope.Engine(C['model_path'], n_ctx=C['n_ctx'], n_gpu_layers=C['n_gpu_layers'])
    n_layers = engine.model_info.n_layers
    n_embd = engine.model_info.n_embd
    print(f"Model: {engine.model_info.name}")
    print(f"Layers: {n_layers}, Dim: {n_embd}, Vocab: {engine.model_info.n_vocab}")

    # Sanity check
    engine.reset()
    test_out = engine.generate("The capital of France is", max_tokens=10, temperature=0.01, top_k=1, seed=42)
    test_out = strip_think(test_out)
    print(f"Sanity: {test_out.strip()[:40]}")

    MECHANISMS = C['mechanisms']
    K = len(MECHANISMS)
    MECH_NAMES = list(MECHANISMS.keys())
    CORRUPT = C['corrupt_entity']
    N = len(DATASET)

    print(f"\nDataset: {N} prompts")
    print(f"Mechanisms K={K}: {MECH_NAMES}")

    # ── Step 1: Generate answers ──
    print("\nStep 1: Generating answers...")
    gen_results = []
    t0 = time.time()
    for i, (template, subject, answers, cat) in enumerate(DATASET):
        prompt = template.format(subject)
        engine.reset()
        output = engine.generate(prompt, max_tokens=C['gen_max_tokens'],
                                 temperature=C['gen_temperature'], top_k=C['gen_top_k'], seed=C['gen_seed'])
        output = strip_think(output)
        generated = output[len(prompt):] if output.startswith(prompt) else output
        generated = generated.strip()
        is_correct = check_match(generated, answers)
        gen_results.append(dict(idx=i, prompt=prompt, subject=subject,
                                expected=answers, generated=generated[:60],
                                correct=is_correct, category=cat))
        if (i+1) % 25 == 0:
            print(f"  [{i+1}/{N}] {time.time()-t0:.1f}s")

    is_faithful = np.array([r['correct'] for r in gen_results])
    print(f"\nClassification: {is_faithful.sum()} faithful, {(~is_faithful).sum()} hallucinated")

    # ── Step 2: Causal tracing ──
    print("\nStep 2: Causal tracing (activation patching)...")
    IE = np.zeros((N, K))
    IE_raw = np.zeros((N, K))
    IE_full = np.zeros(N)
    p_clean_arr = np.zeros(N)
    p_corrupt_arr = np.zeros(N)
    target_tokens = np.zeros(N, dtype=np.int64)
    IE_per_layer = np.zeros((N, n_layers))

    t0 = time.time()
    for i, (template, subject, answers, cat) in enumerate(DATASET):
        clean_prompt = template.format(subject)
        corrupt_prompt = template.format(CORRUPT)

        engine.reset(); engine.forward(clean_prompt)
        clean_logits = engine.get_logits().copy()
        clean_acts = [a.copy() for a in engine.get_all_activations()]

        engine.reset(); engine.forward(corrupt_prompt)
        corrupt_logits = engine.get_logits().copy()
        corrupt_acts = [a.copy() for a in engine.get_all_activations()]

        target = int(np.argmax(clean_logits))
        p_clean = float(softmax(clean_logits)[target])
        p_corrupt = float(softmax(corrupt_logits)[target])
        denom = p_clean - p_corrupt
        target_tokens[i] = target
        p_clean_arr[i] = p_clean
        p_corrupt_arr[i] = p_corrupt

        deltas = [clean_acts[l] - corrupt_acts[l] for l in range(n_layers)]

        for l in range(n_layers):
            engine.reset(); engine.clear_interventions()
            engine.apply_steering(l, deltas[l].astype(np.float32), 1.0)
            engine.forward(corrupt_prompt)
            p_l = float(softmax(engine.get_logits())[target])
            IE_per_layer[i, l] = (p_l - p_corrupt) / (denom + 1e-10) if abs(denom) > 1e-6 else 0.0
            engine.clear_interventions()

        for k, (mech_name, layers) in enumerate(MECHANISMS.items()):
            engine.reset(); engine.clear_interventions()
            for l in layers:
                engine.apply_steering(l, deltas[l].astype(np.float32), 1.0)
            engine.forward(corrupt_prompt)
            p_int = float(softmax(engine.get_logits())[target])
            IE[i, k] = (p_int - p_corrupt) / (denom + 1e-10) if abs(denom) > 1e-6 else 0.0
            IE_raw[i, k] = p_int - p_corrupt
            engine.clear_interventions()

        engine.reset(); engine.clear_interventions()
        for l in range(n_layers):
            engine.apply_steering(l, deltas[l].astype(np.float32), 1.0)
        engine.forward(corrupt_prompt)
        p_full = float(softmax(engine.get_logits())[target])
        IE_full[i] = (p_full - p_corrupt) / (denom + 1e-10) if abs(denom) > 1e-6 else 0.0
        engine.clear_interventions()

        if (i+1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed
            eta = (N - i - 1) / rate
            print(f"  [{i+1:3d}/{N}] {elapsed:.0f}s, ~{eta:.0f}s left")

    print(f"\nCausal tracing: {time.time()-t0:.1f}s")

    # ── Step 3: Bayesian inference ──
    causal_effect = np.abs(p_clean_arr - p_corrupt_arr)
    significant = causal_effect > C['causal_effect_threshold']

    beta = C['beta_default']
    posteriors, entropies = compute_posteriors_and_entropy(IE, beta)

    mask_f = is_faithful & significant
    mask_h = (~is_faithful) & significant
    H_faithful = entropies[mask_f]
    H_halluc = entropies[mask_h]
    H_max = np.log(K)

    sp = np.sqrt((H_faithful.var(ddof=1) + H_halluc.var(ddof=1)) / 2) if (len(H_faithful)>1 and len(H_halluc)>1) else 1e-10
    cohen_d = (H_halluc.mean() - H_faithful.mean()) / sp if sp > 1e-10 else 0

    print(f"\nbeta={beta}, Significant: {mask_f.sum()} faithful, {mask_h.sum()} hallucinated")
    if len(H_faithful) > 0:
        print(f"  Faithful: H = {H_faithful.mean():.4f} +/- {H_faithful.std():.4f}")
    if len(H_halluc) > 0:
        print(f"  Halluc:   H = {H_halluc.mean():.4f} +/- {H_halluc.std():.4f}")
    print(f"  Cohen d = {cohen_d:.3f}")

    # ── Step 4: Plots ──
    colors_f, colors_h = '#4ECDC4', '#FF6B6B'
    mech_colors = ['#FF6B6B', '#FFE66D', '#4ECDC4', '#95E1D3', '#A8D8EA', '#DDA0DD']
    mean_post_f = posteriors[mask_f].mean(axis=0) if mask_f.sum() > 0 else np.zeros(K)
    mean_post_h = posteriors[mask_h].mean(axis=0) if mask_h.sum() > 0 else np.zeros(K)

    fig = plt.figure(figsize=(18, 14))
    # (a) KDE
    ax1 = fig.add_subplot(2, 3, 1)
    x_range = np.linspace(0, H_max * 1.1, 200)
    if len(H_faithful) > 2:
        kde_f = stats.gaussian_kde(H_faithful, bw_method=0.3)
        ax1.fill_between(x_range, kde_f(x_range), alpha=0.4, color=colors_f, label=f'Faithful (n={len(H_faithful)})')
        ax1.plot(x_range, kde_f(x_range), color=colors_f, lw=2)
    if len(H_halluc) > 2:
        kde_h = stats.gaussian_kde(H_halluc, bw_method=0.3)
        ax1.fill_between(x_range, kde_h(x_range), alpha=0.4, color=colors_h, label=f'Halluc (n={len(H_halluc)})')
        ax1.plot(x_range, kde_h(x_range), color=colors_h, lw=2)
    ax1.axvline(H_max, color='grey', ls='--', lw=1)
    ax1.set_xlabel('Posterior Entropy H'); ax1.set_ylabel('Density')
    ax1.set_title(f'(a) Entropy KDE (d={cohen_d:.2f})', fontweight='bold'); ax1.legend(fontsize=8)

    # (b-f) same as notebook...
    ax4 = fig.add_subplot(2, 3, 4)
    x = np.arange(K); w = 0.35
    ax4.bar(x - w/2, mean_post_f, w, color=colors_f, edgecolor='black', lw=0.5, label='Faithful')
    ax4.bar(x + w/2, mean_post_h, w, color=colors_h, edgecolor='black', lw=0.5, label='Hallucinated')
    ax4.set_xticks(x); ax4.set_xticklabels(MECH_NAMES, fontsize=7, rotation=20, ha='right')
    ax4.set_ylabel('Mean P(m|D)'); ax4.set_title('(d) Mean Posterior by Group', fontweight='bold'); ax4.legend(fontsize=8)

    ax5 = fig.add_subplot(2, 3, 5)
    sig_idx = np.where(significant)[0]
    if len(sig_idx) > 0:
        order = sig_idx[np.argsort(entropies[sig_idx])]
        n_show = min(50, len(order))
        im = ax5.imshow(IE_per_layer[order[:n_show]], aspect='auto', cmap='RdYlBu_r', vmin=-0.1, vmax=0.5)
        ax5.set_xlabel('Layer'); ax5.set_ylabel('Prompt (sorted by entropy)')
        ax5.set_title('(e) Per-Layer IE', fontweight='bold')
        plt.colorbar(im, ax=ax5, shrink=0.8, label='IE')
        for b in range(6, n_layers, 6):
            ax5.axvline(b - 0.5, color='white', lw=1, ls='--')

    plt.tight_layout()
    plt.savefig(f'{C["output_dir"]}/main_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ── Step 5: Statistical tests ──
    print("\n" + "=" * 60)
    print("STATISTICAL TESTS")
    print("=" * 60)
    if len(H_faithful) > 1 and len(H_halluc) > 1:
        U, p_mw = stats.mannwhitneyu(H_halluc, H_faithful, alternative='greater')
        ks_stat, p_ks = stats.ks_2samp(H_faithful, H_halluc)
        t_stat, p_t = stats.ttest_ind(H_halluc, H_faithful, equal_var=False)
        y_true = np.concatenate([np.zeros(mask_f.sum()), np.ones(mask_h.sum())])
        y_score = np.concatenate([entropies[mask_f], entropies[mask_h]])
        auc_h = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else 0.5
        print(f"  Mann-Whitney U={U:.1f}, p={p_mw:.6f}")
        print(f"  KS D={ks_stat:.4f}, p={p_ks:.6f}")
        print(f"  Welch t={t_stat:.3f}, p={p_t:.6f}")
        print(f"  AUC={auc_h:.3f}")

    # Beta sensitivity
    print(f"\nBeta sensitivity:")
    print(f"{'beta':>5s}  {'Cohen d':>8s}  {'p-value':>10s}")
    print("-" * 30)
    for beta_val in C['beta_sensitivity_values']:
        post_b, ent_b = compute_posteriors_and_entropy(IE, beta_val)
        Hf = ent_b[mask_f]; Hh = ent_b[mask_h]
        sp_b = np.sqrt((Hf.var(ddof=1) + Hh.var(ddof=1)) / 2) if (len(Hf)>1 and len(Hh)>1) else 1e-10
        d_b = (Hh.mean() - Hf.mean()) / sp_b if sp_b > 1e-10 else 0
        _, p_b = stats.mannwhitneyu(Hh, Hf, alternative='greater') if (len(Hf)>1 and len(Hh)>1) else (0, 1)
        sig = '***' if p_b<0.001 else '**' if p_b<0.01 else '*' if p_b<0.05 else 'n.s.'
        print(f"{beta_val:5.1f}  {d_b:8.3f}  {p_b:10.6f}  {sig}")

    # ── Save ──
    np.savez(f'{C["output_dir"]}/phase1_results.npz',
        is_faithful=is_faithful, significant=significant,
        p_clean=p_clean_arr, p_corrupt=p_corrupt_arr, target_tokens=target_tokens,
        IE=IE, IE_raw=IE_raw, IE_full=IE_full, IE_per_layer=IE_per_layer,
        posteriors=posteriors, entropies=entropies,
        beta_default=beta, n_prompts=N, n_mechanisms=K,
        mechanism_names=np.array(MECH_NAMES),
    )
    print(f"\n{'='*60}")
    print("PHASE 1 COMPLETE")
    print(f"{'='*60}")
    print(f"Faithful: {mask_f.sum()}, Hallucinated: {mask_h.sum()}")
    print(f"Cohen d = {cohen_d:.3f}")
    print(f"Saved to {C['output_dir']}/")


if __name__ == '__main__':
    main()
