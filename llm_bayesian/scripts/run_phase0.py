#!/usr/bin/env python3
"""
Phase 0 — Bayesian Mechanism Selection PoC (Aesthetic Data)
===========================================================
Uses existing aesthetic experiment activations to demonstrate that
Bayesian model selection (g-prior BF) correctly identifies the most
causally relevant direction and that "hallucinated" (shuffled-label)
tasks produce high posterior entropy.

Configurable variables are in the CONFIG section below.
"""

import os, argparse
import numpy as np
from scipy import stats
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans']
import warnings; warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# CONFIG — Edit these variables to change experiment parameters
# ═══════════════════════════════════════════════════════════════════
CONFIG = dict(
    # ── Input data paths ──
    raw_activations_path = '/workspace/Data/aesthetic_experiment/raw_activations.npz',
    exp2_path  = '/workspace/Data/aesthetic_experiment/aesthetic_analysis_results.npz',
    exp9_path  = '/workspace/Data/poetry_replication/poetry_replication_results.npz',
    exp10_path = '/workspace/Data/direction_A/dirA_results.npz',
    exp5_path  = '/workspace/Data/aesthetic_neurons/ablation_results.npz',

    # ── Output ──
    output_dir = '/workspace/Data/bayesian_phase0',

    # ── Analysis parameters ──
    analysis_layer    = 16,        # Main layer for single-layer analysis
    pca_n_components  = 5,         # PCA components to extract
    g_prior           = 'N',       # g-prior scale: 'N', 'sqrtN', 'N2', or a float
    n_shuffles        = 100,       # Number of label-shuffled simulations
    g_sensitivity_values = [1, 'sqrtN', 'N', 'N2'],  # g values for sensitivity plot
    random_seed       = 42,
)
# ═══════════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw-activations', dest='raw_activations_path')
    p.add_argument('--analysis-layer', type=int)
    p.add_argument('--g-prior', dest='g_prior')
    p.add_argument('--n-shuffles', type=int)
    p.add_argument('--output-dir')
    p.add_argument('--seed', type=int, dest='random_seed')
    args = p.parse_args()
    for k, v in vars(args).items():
        if v is not None:
            CONFIG[k] = v


def resolve_g(g_spec, N):
    """Resolve g-prior spec to a float."""
    if isinstance(g_spec, (int, float)):
        return float(g_spec)
    s = str(g_spec).lower()
    if s == 'n':
        return float(N)
    elif s == 'sqrtn':
        return float(np.sqrt(N))
    elif s == 'n2':
        return float(N ** 2)
    else:
        return float(g_spec)


def unit(v):
    return v / np.linalg.norm(v)


def main():
    parse_args()
    C = CONFIG
    os.makedirs(C['output_dir'], exist_ok=True)

    # ── Load data ──
    raw = np.load(C['raw_activations_path'])
    acts_pos = raw['positive']   # (N_pos, 36, 4096)
    acts_neg = raw['negative']   # (N_neg, 36, 4096)
    N_pos, N_neg = acts_pos.shape[0], acts_neg.shape[0]
    N = N_pos + N_neg

    all_acts = np.concatenate([acts_pos, acts_neg], axis=0)
    labels = np.array([1]*N_pos + [0]*N_neg)

    exp2  = np.load(C['exp2_path'])
    exp9  = np.load(C['exp9_path'])
    exp10 = np.load(C['exp10_path'])
    exp5  = np.load(C['exp5_path'])

    print(f"Loaded {N} samples (pos={N_pos}, neg={N_neg})")
    print(f"Activation shape per sample: {all_acts.shape[1:]}")

    # ── Setup ──
    L = C['analysis_layer']
    acts_L = all_acts[:, L, :]  # (N, n_embd)
    rng = np.random.default_rng(C['random_seed'])
    g = resolve_g(C['g_prior'], N)
    n_eff = N_pos * N_neg / (N_pos + N_neg)

    # ── Candidate directions ──
    v_exp2  = unit(exp2['aesthetic_direction_norm'][L].astype(np.float64))
    v_exp10 = unit(exp10['aesthetic_dir_norm'][L].astype(np.float64))
    v_exp9  = unit(exp9['aesthetic_dir_norm'][L].astype(np.float64))

    pca = PCA(n_components=C['pca_n_components'], random_state=C['random_seed'])
    pca.fit(acts_L)
    v_pc1 = unit(pca.components_[0].astype(np.float64))
    v_pc2 = unit(pca.components_[1].astype(np.float64))
    v_null = unit(rng.standard_normal(acts_L.shape[1]))

    directions = {
        'Exp2\nAesthetic': v_exp2,
        'Exp10\nImagery':  v_exp10,
        'Exp9\nGenre':     v_exp9,
        'PCA-PC1':         v_pc1,
        'PCA-PC2':         v_pc2,
        'Random\nNull':    v_null,
    }
    K = len(directions)
    short = [n.replace('\n', ' ') for n in directions.keys()]

    # ── Cosine similarity matrix ──
    names = list(directions.keys())
    vecs = np.stack(list(directions.values()))
    cos_mat = vecs @ vecs.T
    print(f"\nCosine similarity matrix (K={K} directions):")
    header = "            " + "  ".join(f"{n.replace(chr(10),' '):>8s}" for n in names)
    print(header)
    for i, n in enumerate(names):
        row = "  ".join(f"{cos_mat[i,j]:8.3f}" for j in range(K))
        print(f"{n.replace(chr(10),' '):>12s}  {row}")

    # ── Measure class separation per direction ──
    results = {}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    colors_bars = ['#FF6B6B', '#FFE66D', '#4ECDC4', '#95E1D3', '#A8D8EA', '#CCCCCC']

    for idx, (name, v) in enumerate(directions.items()):
        projs = acts_L @ v
        p_pos = projs[labels == 1]
        p_neg = projs[labels == 0]
        sp = np.sqrt((p_pos.var(ddof=1) + p_neg.var(ddof=1)) / 2)
        delta = p_pos.mean() - p_neg.mean()
        d = delta / sp if sp > 0 else 0.0
        auc = roc_auc_score(labels, projs)
        auc = max(auc, 1 - auc)
        se = sp * np.sqrt(1/N_pos + 1/N_neg)
        t_stat = delta / se if se > 0 else 0.0
        results[name] = dict(projs=projs, p_pos=p_pos, p_neg=p_neg,
                             delta=delta, sp=sp, d=d, auc=auc, t_stat=t_stat)
        ax = axes[idx]
        bins = np.linspace(projs.min(), projs.max(), 30)
        ax.hist(p_neg, bins=bins, alpha=0.6, label='Neg', color='#4ECDC4')
        ax.hist(p_pos, bins=bins, alpha=0.6, label='Pos', color='#FF6B6B')
        ax.set_title(name.replace('\n', ' '), fontsize=11, fontweight='bold')
        ax.text(0.97, 0.95, f'd={d:.2f}\nAUC={auc:.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        ax.legend(fontsize=8)
        ax.set_xlabel('Projection')

    fig.suptitle(f'Class-Conditional Projection Distributions (L{L})', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{C["output_dir"]}/proj_histograms.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nEffect-size summary:")
    print(f"{'Direction':>18s}  {'delta':>8s}  {'Cohen d':>8s}  {'AUC':>6s}  {'t':>8s}")
    print("-" * 56)
    for name, r in results.items():
        print(f"{name.replace(chr(10),' '):>18s}  {r['delta']:8.3f}  {r['d']:8.3f}  {r['auc']:.3f}  {r['t_stat']:8.2f}")

    # ── Bayesian Inference ──
    log_bfs = {}
    for name, r in results.items():
        t2 = r['t_stat'] ** 2
        log_bf = -0.5 * np.log(1 + g) + 0.5 * t2 * g / (1 + g)
        log_bfs[name] = log_bf

    log_bf_arr = np.array(list(log_bfs.values()))
    log_posterior = log_bf_arr - logsumexp(log_bf_arr)
    posterior = np.exp(log_posterior)
    entropy = -np.sum(posterior * np.log(posterior + 1e-30))
    K_eff = np.exp(entropy)

    print(f"\nPosterior: {dict(zip(short, [f'{p:.4f}' for p in posterior]))}")
    print(f"Entropy H(M|D) = {entropy:.4f}  (H_max = {np.log(K):.4f})")
    print(f"K_eff = exp(H) = {K_eff:.2f}  (out of K={K})")

    # ── Posterior bar chart, BF, gauge ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(K); w = 0.35
    ax = axes[0]
    ax.bar(x - w/2, [1/K]*K, w, color='lightgrey', label='Prior (uniform)')
    ax.bar(x + w/2, posterior, w, color=colors_bars, edgecolor='black', linewidth=0.5, label='Posterior')
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8, rotation=25, ha='right')
    ax.set_ylabel('P(M | D)'); ax.set_title('(a) Prior -> Posterior', fontweight='bold')
    ax.legend(fontsize=8); ax.set_ylim(0, max(posterior)*1.15)

    ax = axes[1]
    ax.bar(x, log_bf_arr, color=colors_bars, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8, rotation=25, ha='right')
    ax.set_ylabel('log BF (vs null)'); ax.set_title('(b) Log Bayes Factor', fontweight='bold')
    ax.axhline(0, color='k', lw=0.5, ls='--')

    ax = axes[2]; H_max = np.log(K); frac = entropy / H_max
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', lw=2)
    angle = np.pi * (1 - frac)
    ax.annotate('', xy=(0.85*np.cos(angle), 0.85*np.sin(angle)), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
    ax.text(0, -0.15, f'H = {entropy:.3f}\nK_eff = {K_eff:.2f} / {K}',
            ha='center', va='top', fontsize=12, fontweight='bold')
    ax.text(-1.05, -0.02, 'Low\nambiguity', ha='center', fontsize=8, color='green')
    ax.text(1.05, -0.02, 'High\nambiguity', ha='center', fontsize=8, color='red')
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.35, 1.15)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('(c) Causal Ambiguity Gauge', fontweight='bold')
    fig.suptitle(f'Bayesian Mechanism Selection — Aesthetic Task (L{L})', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{C["output_dir"]}/posterior_L{L}.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ── Cross-layer sweep ──
    n_total_layers = all_acts.shape[1]
    layer_entropies = np.zeros(n_total_layers)
    layer_K_eff = np.zeros(n_total_layers)
    layer_posteriors = np.zeros((n_total_layers, K))
    layer_winner = []

    for l in range(n_total_layers):
        acts_l = all_acts[:, l, :]
        dirs_l = {
            'Exp2':  unit(exp2['aesthetic_direction_norm'][l].astype(np.float64)),
            'Exp10': unit(exp10['aesthetic_dir_norm'][l].astype(np.float64)),
            'Exp9':  unit(exp9['aesthetic_dir_norm'][l].astype(np.float64)),
        }
        pca_l = PCA(n_components=2, random_state=C['random_seed']).fit(acts_l)
        dirs_l['PC1'] = unit(pca_l.components_[0].astype(np.float64))
        dirs_l['PC2'] = unit(pca_l.components_[1].astype(np.float64))
        dirs_l['Null'] = v_null

        log_bfs_l = []
        for name_l, v in dirs_l.items():
            projs = acts_l @ v
            p1, p0 = projs[labels==1], projs[labels==0]
            sp = np.sqrt((p1.var(ddof=1) + p0.var(ddof=1)) / 2)
            se = sp * np.sqrt(1/N_pos + 1/N_neg)
            t = (p1.mean() - p0.mean()) / se if se > 1e-12 else 0.0
            t2 = t ** 2
            log_bf = -0.5 * np.log(1 + g) + 0.5 * t2 * g / (1 + g)
            log_bfs_l.append(log_bf)

        log_bfs_l = np.array(log_bfs_l)
        log_post = log_bfs_l - logsumexp(log_bfs_l)
        post = np.exp(log_post)
        H = -np.sum(post * np.log(post + 1e-30))
        layer_entropies[l] = H
        layer_K_eff[l] = np.exp(H)
        layer_posteriors[l] = post
        layer_winner.append(list(dirs_l.keys())[np.argmax(post)])

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [1, 1.2]})
    ax = axes[0]
    ax.plot(range(n_total_layers), layer_entropies, 'o-', color='#FF6B6B', lw=2, ms=4)
    ax.axhline(np.log(K), color='grey', ls='--', lw=1, label=f'H_max = ln({K}) = {np.log(K):.2f}')
    ax.fill_between(range(n_total_layers), 0, layer_entropies, alpha=0.15, color='#FF6B6B')
    ax.set_ylabel('Posterior Entropy H(M|D)')
    ax.set_title('(a) Causal Ambiguity across Layers', fontweight='bold')
    ax.legend(fontsize=9)
    best_l = np.argmin(layer_entropies)
    ax.annotate(f'L{best_l}: lowest ambiguity\nK_eff={layer_K_eff[best_l]:.2f}',
                xy=(best_l, layer_entropies[best_l]),
                xytext=(best_l+3, layer_entropies[best_l]+0.15),
                arrowprops=dict(arrowstyle='->', color='green'), fontsize=9, color='green')

    ax = axes[1]
    dir_names = ['Exp2', 'Exp10', 'Exp9', 'PC1', 'PC2', 'Null']
    ax.stackplot(range(n_total_layers), layer_posteriors.T, labels=dir_names, colors=colors_bars, alpha=0.85)
    ax.set_xlabel('Layer'); ax.set_ylabel('P(M | D)')
    ax.set_title('(b) Posterior Distribution across Layers', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, ncol=3)
    ax.set_xlim(-0.5, n_total_layers-0.5); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f'{C["output_dir"]}/cross_layer_entropy.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nBest layer (lowest ambiguity): L{best_l}, H={layer_entropies[best_l]:.4f}")
    print(f"Winner at L{best_l}: {layer_winner[best_l]}")

    # ── Shuffled-label simulation ──
    n_shuffles = C['n_shuffles']
    shuffle_entropies = np.zeros(n_shuffles)
    rng_shuf = np.random.default_rng(123)

    for s in range(n_shuffles):
        labels_shuf = rng_shuf.permutation(labels)
        log_bfs_s = []
        for name, v in directions.items():
            projs = acts_L @ v
            p1 = projs[labels_shuf == 1]; p0 = projs[labels_shuf == 0]
            sp = np.sqrt((p1.var(ddof=1) + p0.var(ddof=1)) / 2)
            se = sp * np.sqrt(1/N_pos + 1/N_neg)
            t = (p1.mean() - p0.mean()) / se if se > 1e-12 else 0.0
            log_bf = -0.5 * np.log(1 + g) + 0.5 * t ** 2 * g / (1 + g)
            log_bfs_s.append(log_bf)
        log_bfs_s = np.array(log_bfs_s)
        log_post = log_bfs_s - logsumexp(log_bfs_s)
        post = np.exp(log_post)
        shuffle_entropies[s] = -np.sum(post * np.log(post + 1e-30))

    d_cohen_shuf = (entropy - shuffle_entropies.mean()) / shuffle_entropies.std()
    p_val_shuf = (shuffle_entropies <= entropy).mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.hist(shuffle_entropies, bins=25, alpha=0.7, color='#CCCCCC', edgecolor='grey',
            label='Shuffled labels (x{})'.format(n_shuffles))
    ax.axvline(entropy, color='#FF6B6B', lw=3, label=f'True labels: H={entropy:.3f}')
    ax.axvline(np.log(K), color='grey', ls='--', lw=1, label=f'H_max = {np.log(K):.3f}')
    ax.set_xlabel('Posterior Entropy H(M|D)'); ax.set_ylabel('Count')
    ax.set_title('(a) Faithful vs Ambiguous', fontweight='bold'); ax.legend(fontsize=9)

    # Representative shuffled posterior
    med_idx = np.argmin(np.abs(shuffle_entropies - np.median(shuffle_entropies)))
    rng_repr = np.random.default_rng(123)
    for _ in range(med_idx + 1):
        labels_repr = rng_repr.permutation(labels)
    log_bfs_repr = []
    for name, v in directions.items():
        projs = acts_L @ v
        p1 = projs[labels_repr == 1]; p0 = projs[labels_repr == 0]
        sp = np.sqrt((p1.var(ddof=1) + p0.var(ddof=1)) / 2)
        se = sp * np.sqrt(1/N_pos + 1/N_neg)
        t = (p1.mean() - p0.mean()) / se if se > 1e-12 else 0.0
        log_bf = -0.5 * np.log(1 + g) + 0.5 * t ** 2 * g / (1 + g)
        log_bfs_repr.append(log_bf)
    post_repr = np.exp(np.array(log_bfs_repr) - logsumexp(log_bfs_repr))

    ax = axes[1]
    ax.bar(x - w/2, posterior, w, color=colors_bars, edgecolor='black', lw=0.5, label='True labels')
    ax.bar(x + w/2, post_repr, w, color=colors_bars, edgecolor='black', lw=0.5, alpha=0.4, label='Shuffled labels')
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8, rotation=25, ha='right')
    ax.set_ylabel('P(M | D)'); ax.set_title('(b) Posterior: Faithful vs Ambiguous', fontweight='bold')
    ax.legend(fontsize=9)
    fig.suptitle(f'Hallucination ~ Causal Ambiguity (L{L})', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{C["output_dir"]}/hallucination_sim.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nTrue-label entropy:    H = {entropy:.4f}")
    print(f"Shuffled-label entropy: H = {shuffle_entropies.mean():.4f} +/- {shuffle_entropies.std():.4f}")
    print(f"Separation:  Cohen's d = {d_cohen_shuf:.2f}")

    # ── Validation: posterior vs AUC ──
    validation = {name: max(roc_auc_score(labels, acts_L @ v), 1 - roc_auc_score(labels, acts_L @ v))
                  for name, v in directions.items()}
    aucs_arr = np.array([validation[n] for n in directions.keys()])
    r_corr, p_corr = stats.pearsonr(aucs_arr, posterior)

    abl_sizes = exp5['group_sizes_offline']
    abl_aucs = exp5['group_targeted_offline']
    abl_random = exp5['group_random_offline_mean']
    abl10_sizes = exp10['group_sizes']
    abl10_results = exp10['group_results']
    abl10_random = exp10['random_results']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    for i, (name, auc, p) in enumerate(zip(short, aucs_arr, posterior)):
        ax.scatter(auc, p, s=120, c=colors_bars[i], edgecolors='black', zorder=3)
        ax.annotate(name, (auc, p), textcoords="offset points", xytext=(8, 5), fontsize=8)
    ax.set_xlabel('Empirical AUC'); ax.set_ylabel('Bayesian Posterior P(M|D)')
    ax.set_title(f'(a) Posterior vs AUC (r={r_corr:.3f})', fontweight='bold'); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(abl_sizes, abl_aucs, 'o-', color='#FF6B6B', label='Targeted')
    ax.plot(abl_sizes, abl_random, 's--', color='grey', label='Random')
    ax.set_xscale('log'); ax.set_xlabel('# Neurons Ablated'); ax.set_ylabel('Offline AUC')
    ax.set_title('(b) Exp 5 Ablation Curve', fontweight='bold'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(abl10_sizes, abl10_results, 'o-', color='#FFE66D', label='Targeted')
    ax.plot(abl10_sizes, abl10_random, 's--', color='grey', label='Random')
    ax.set_xscale('log'); ax.set_xlabel('# Neurons Ablated'); ax.set_ylabel('Density Change')
    ax.set_title('(c) Exp 10 Ablation Curve', fontweight='bold'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{C["output_dir"]}/validation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Validation: posterior vs AUC r = {r_corr:.3f}, p = {p_corr:.4f}")

    # ── g-prior sensitivity ──
    g_values = {}
    for gv in C['g_sensitivity_values']:
        label = f'g={gv}' if isinstance(gv, (int, float)) else f'g={gv}'
        g_values[label] = resolve_g(gv, N)

    fig, axes = plt.subplots(1, len(g_values), figsize=(4*len(g_values), 4), sharey=True)
    if len(g_values) == 1:
        axes = [axes]
    for ax_idx, (g_label, g_val) in enumerate(g_values.items()):
        log_bfs_g = []
        for name, r in results.items():
            t2 = r['t_stat'] ** 2
            log_bf = -0.5 * np.log(1 + g_val) + 0.5 * t2 * g_val / (1 + g_val)
            log_bfs_g.append(log_bf)
        log_bfs_g = np.array(log_bfs_g)
        post_g = np.exp(log_bfs_g - logsumexp(log_bfs_g))
        H_g = -np.sum(post_g * np.log(post_g + 1e-30))
        ax = axes[ax_idx]
        ax.bar(range(K), post_g, color=colors_bars, edgecolor='black', lw=0.5)
        ax.set_xticks(range(K)); ax.set_xticklabels(short, fontsize=7, rotation=30, ha='right')
        ax.set_title(f'{g_label}\nH={H_g:.3f}, K_eff={np.exp(H_g):.2f}', fontsize=10, fontweight='bold')
        if ax_idx == 0: ax.set_ylabel('P(M|D)')
    fig.suptitle('Prior Sensitivity: Effect of g on Posterior', fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(f'{C["output_dir"]}/prior_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ── Save results ──
    np.savez(f'{C["output_dir"]}/phase0_results.npz',
        direction_names=np.array(short),
        direction_vectors=np.stack(list(directions.values())),
        cohens_d=np.array([results[n]['d'] for n in directions.keys()]),
        aucs=np.array([results[n]['auc'] for n in directions.keys()]),
        t_stats=np.array([results[n]['t_stat'] for n in directions.keys()]),
        log_bayes_factors=log_bf_arr,
        posterior=posterior,
        entropy=entropy, K_eff=K_eff, sigma2=results['Random\nNull']['sp']**2, g=g,
        layer_entropies=layer_entropies, layer_K_eff=layer_K_eff,
        layer_posteriors=layer_posteriors, layer_winner=np.array(layer_winner),
        shuffle_entropies=shuffle_entropies,
    )

    print("\n" + "=" * 60)
    print("PHASE 0 — RESULTS SUMMARY")
    print("=" * 60)
    print(f"Task: Aesthetic classification ({N_pos} pos + {N_neg} neg, L{L})")
    print(f"Candidate mechanisms K = {K}")
    print(f"\n{'Direction':>18s}  {'P(M|D)':>8s}  {'Cohen d':>8s}  {'AUC':>6s}")
    print("-" * 48)
    for i, name in enumerate(short):
        print(f"{name:>18s}  {posterior[i]:8.4f}  {results[list(directions.keys())[i]]['d']:8.3f}  "
              f"{results[list(directions.keys())[i]]['auc']:.3f}")
    print(f"\nH = {entropy:.4f}, K_eff = {K_eff:.2f}")
    print(f"Best layer: L{best_l}, H = {layer_entropies.min():.4f}")
    print(f"Shuffle d = {d_cohen_shuf:.2f}")
    print(f"\nSaved to {C['output_dir']}/")


if __name__ == '__main__':
    main()
