#!/usr/bin/env python3
"""
Phase 2 — Fine-Grained Causal Posterior Spectrum
=================================================
Refines the mechanism space from K=6 layer groups to L=36 individual
layers, constructing a "Causal Posterior Spectrum." Classifies
hallucinations into ambiguity-driven (flat spectrum) vs
misinformation-driven (spike spectrum).

Uses Phase 1b per-layer IE data (zero new forward passes for core
analysis). New forward passes only for residual contribution analysis.
"""

import os, sys, time, re, json, argparse
import numpy as np
from scipy import stats
from scipy.special import logsumexp
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans']
import warnings; warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# CONFIG — Edit these variables to change experiment parameters
# ═══════════════════════════════════════════════════════════════════
CONFIG = dict(
    # ── Model (needed only for residual contribution analysis) ──
    model_path   = "/workspace/Model/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf",
    n_ctx        = 2048,
    n_gpu_layers = 99,

    # ── Input (Phase 1b results) ──
    phase1b_results_path = '/workspace/Data/bayesian_phase1b/phase1b_results.npz',

    # ── Output ──
    output_dir = '/workspace/Data/bayesian_phase2',

    # ── Bayesian ──
    beta = 5.0,

    # ── Residual contribution analysis ──
    n_residual_samples  = 50,   # total prompts for residual analysis (half faithful, half halluc)
    corrupt_entity      = "Zyxwv",

    # ── Dataset reconstruction (needs same counts as Phase 1b) ──
    n_truthfulqa = 100,
    n_popqa      = 100,
    n_triviaqa   = 50,
    n_neqa       = 50,
    popqa_low_popularity_pool = 2000,

    # ── HuggingFace cache ──
    hf_home = '/workspace/Data/huggingface_cache',

    # ── Random seed ──
    random_seed = 42,
    residual_sample_seed = 123,
)
# ═══════════════════════════════════════════════════════════════════


PROP_TEMPLATES = {
    'occupation': "The occupation of {} is",
    'genre': "The genre of {} is",
    'country': "The country of {} is",
    'capital': "The capital of {} is",
    'place of birth': "The birthplace of {} is",
    'author': "The author of {} is",
    'director': "The director of {} is",
    'producer': "The producer of {} is",
    'screenwriter': "The screenwriter of {} is",
    'composer': "The composer of {} is",
    'color': "The color of {} is",
    'religion': "The religion of {} is",
    'sport': "The sport of {} is",
    'continent': "The continent of {} is",
    'has_part': "{} contains",
    'mother': "The mother of {} is",
    'father': "The father of {} is",
    'child': "A child of {} is",
}


def layer_posteriors_and_entropy(IE_layer, beta):
    N, L = IE_layer.shape
    posteriors = np.zeros((N, L))
    entropies = np.zeros(N)
    for i in range(N):
        log_liks = beta * IE_layer[i]
        log_post = log_liks - logsumexp(log_liks)
        post = np.exp(log_post)
        posteriors[i] = post
        entropies[i] = -np.sum(post * np.log(post + 1e-30))
    return posteriors, entropies


def reconstruct_prompts(C):
    """Re-load and sample exactly the same prompts as Phase 1b (same seed)."""
    from datasets import load_dataset
    rng_r = np.random.RandomState(C['random_seed'])
    all_prompts = []

    tqa = load_dataset('truthful_qa', 'generation', split='validation')
    tqa_prompts = []
    for row in tqa:
        q = row['question']; correct = row['correct_answers']; best = row['best_answer']
        m = re.search(r'(?:What|Where|When|Who|How)\s+(?:is|are|was|were|did)\s+(?:the\s+)?(.+?)(?:\?|$)', q, re.I)
        if not m: continue
        subject = m.group(1).strip().rstrip('?').strip()
        if not subject or len(subject) < 3 or len(subject) > 60: continue
        expected = correct if correct else [best]
        tqa_prompts.append((f"Q: {q}\nA:", subject, expected, 'TruthfulQA'))
    rng_r.shuffle(tqa_prompts)
    all_prompts.extend(tqa_prompts[:C['n_truthfulqa']])

    pqa = load_dataset('akariasai/PopQA', split='test')
    pqa_prompts = []
    for row in pqa:
        prop = row['prop']; subj = row['subj']; obj = row['obj']
        template = PROP_TEMPLATES.get(prop, f"The {prop} of " + "{}" + " is")
        prompt = template.format(subj)
        try: answers = json.loads(row['possible_answers'])
        except: answers = [obj]
        if obj not in answers: answers.insert(0, obj)
        pqa_prompts.append((prompt, subj, answers, 'PopQA', row['s_pop']))
    pqa_prompts.sort(key=lambda x: x[4])
    pqa_low = pqa_prompts[:C['popqa_low_popularity_pool']]
    rng_r.shuffle(pqa_low)
    all_prompts.extend([(p, s, a, d) for p, s, a, d, _ in pqa_low[:C['n_popqa']]])

    trivia = load_dataset('trivia_qa', 'rc.nocontext', split='validation')
    trivia_prompts = []
    for row in trivia:
        q = row['question']; answer = row['answer']
        aliases = answer.get('aliases', []); value = answer.get('value', '')
        m = re.search(r"(?:What|Where|When|Who|Which|How)\s+(?:is|are|was|were|did)\s+(?:the\s+)?(.+?)(?:\?|$)", q, re.I)
        if not m: continue
        subject = m.group(1).strip().rstrip('?').strip()
        if not subject or len(subject) < 3 or len(subject) > 60: continue
        expected = aliases if aliases else [value]
        if value and value not in expected: expected.insert(0, value)
        trivia_prompts.append((f"Q: {q}\nA:", subject, expected, 'TriviaQA'))
    rng_r.shuffle(trivia_prompts)
    all_prompts.extend(trivia_prompts[:C['n_triviaqa']])

    neqa = load_dataset('inverse-scaling/NeQA', split='train')
    neqa_prompts = []
    for row in neqa:
        prompt_text = row['prompt']; classes = row['classes']; answer_idx = row['answer_index']
        m = re.search(r'Question:\s*(.+?)(?:\nA\.)', prompt_text, re.DOTALL)
        if not m: continue
        question = m.group(1).strip()
        m2 = re.search(r'(?:a |an |the )?(\w[\w\s]{2,30}?)(?:\s+(?:has|is|are|was|were|does|do|did|can|will|isn|aren|doesn|don|wasn|weren|cannot|won))', question, re.I)
        subject = m2.group(1).strip() if m2 else question[:30]
        if not subject or len(subject) < 2: subject = question[:30]
        neqa_prompts.append((prompt_text, subject, [classes[answer_idx].strip()], 'NeQA'))
    rng_r.shuffle(neqa_prompts)
    all_prompts.extend(neqa_prompts[:C['n_neqa']])

    print(f"Reconstructed {len(all_prompts)} prompts")
    return all_prompts


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', dest='model_path')
    p.add_argument('--phase1b', dest='phase1b_results_path')
    p.add_argument('--beta', type=float)
    p.add_argument('--n-residual', type=int, dest='n_residual_samples')
    p.add_argument('--output-dir')
    args = p.parse_args()
    for k, v in vars(args).items():
        if v is not None:
            CONFIG[k] = v


def main():
    parse_args()
    C = CONFIG
    os.environ['GGML_CUDA_DISABLE_GRAPHS'] = '1'
    os.environ['HF_HOME'] = C['hf_home']
    os.makedirs(C['output_dir'], exist_ok=True)
    beta = C['beta']

    # ── Load Phase 1b data ──
    print("Loading Phase 1b results...")
    data = np.load(C['phase1b_results_path'], allow_pickle=True)
    IE_per_layer = data['IE_per_layer']
    IE_group     = data['IE']
    is_faithful  = data['is_faithful']
    significant  = data['significant']
    ds_labels    = data['ds_labels']
    p_clean_arr  = data['p_clean']
    p_corrupt_arr = data['p_corrupt']
    entropies_grp = data['entropies']

    N, L = IE_per_layer.shape
    print(f"Loaded: {N} prompts, {L} layers")
    print(f"Faithful: {is_faithful.sum()}, Halluc: {(~is_faithful).sum()}")
    print(f"Significant: {significant.sum()}/{N}")

    # ── Load model for residual analysis ──
    import neuroscope
    engine = neuroscope.Engine(C['model_path'], n_ctx=C['n_ctx'], n_gpu_layers=C['n_gpu_layers'])
    n_layers = engine.model_info.n_layers
    n_embd = engine.model_info.n_embd
    print(f"Model: {n_layers} layers, {n_embd} dim")

    # ══════════════════════════════════════════════════════════
    # Core Analysis (no forward passes needed)
    # ══════════════════════════════════════════════════════════
    layer_post, layer_ent = layer_posteriors_and_entropy(IE_per_layer, beta)
    H_max_layer = np.log(L)

    mask_f = is_faithful & significant
    mask_h = (~is_faithful) & significant
    H_f_layer = layer_ent[mask_f]
    H_h_layer = layer_ent[mask_h]

    sp = np.sqrt((H_f_layer.var(ddof=1) + H_h_layer.var(ddof=1)) / 2) if (len(H_f_layer) > 1 and len(H_h_layer) > 1) else 1e-10
    cohen_d_layer = (H_h_layer.mean() - H_f_layer.mean()) / sp if sp > 1e-10 else 0

    print(f"\nLayer-Level Analysis (L={L}, beta={beta})")
    print(f"H_max = ln({L}) = {H_max_layer:.4f}")
    print(f"  Faithful: H={H_f_layer.mean():.4f} +/- {H_f_layer.std():.4f}")
    print(f"  Halluc:   H={H_h_layer.mean():.4f} +/- {H_h_layer.std():.4f}")
    print(f"  Cohen d = {cohen_d_layer:.3f}")

    U, p_mw = stats.mannwhitneyu(H_h_layer, H_f_layer, alternative='greater')
    ks, p_ks = stats.ks_2samp(H_f_layer, H_h_layer)
    t_stat, p_t = stats.ttest_ind(H_h_layer, H_f_layer, equal_var=False)
    print(f"  Mann-Whitney p={p_mw:.6f}")
    print(f"  KS p={p_ks:.6f}")
    print(f"  Welch t p={p_t:.6f}")

    # Per-dataset
    per_ds_layer = {}
    print(f"\n{'Dataset':>12s}  {'n_F':>4s}  {'n_H':>4s}  {'H_F':>7s}  {'H_H':>7s}  {'d':>6s}  {'p':>8s}")
    print("-" * 55)
    for ds in ['TruthfulQA', 'PopQA', 'TriviaQA', 'NeQA']:
        dm = ds_labels == ds
        mf = dm & mask_f; mh = dm & mask_h
        Hf = layer_ent[mf]; Hh = layer_ent[mh]
        if len(Hf) > 1 and len(Hh) > 1:
            s = np.sqrt((Hf.var(ddof=1) + Hh.var(ddof=1)) / 2)
            d = (Hh.mean() - Hf.mean()) / s if s > 1e-10 else 0
            _, p = stats.mannwhitneyu(Hh, Hf, alternative='greater')
        else:
            d, p = 0, 1.0
        per_ds_layer[ds] = {'n_f': int(mf.sum()), 'n_h': int(mh.sum()),
                             'H_f': float(Hf.mean()) if len(Hf) > 0 else 0,
                             'H_h': float(Hh.mean()) if len(Hh) > 0 else 0,
                             'd': float(d), 'p': float(p)}
        sig = '*' if p < 0.05 else ''
        print(f"{ds:>12s}  {mf.sum():4d}  {mh.sum():4d}  {Hf.mean():7.4f}  {Hh.mean() if len(Hh) > 0 else 0:7.4f}  {d:6.3f}  {p:7.4f} {sig}")

    # ── Hallucination subtypes ──
    H_halluc_all = layer_ent[mask_h]
    H_median = np.median(H_halluc_all)
    ambiguity_mask = mask_h & (layer_ent >= H_median)
    misinfo_mask = mask_h & (layer_ent < H_median)
    print(f"\nSubtypes (split at H_median={H_median:.4f}):")
    print(f"  Ambiguity-driven: {ambiguity_mask.sum()}")
    print(f"  Misinfo-driven:   {misinfo_mask.sum()}")

    spectrum_faithful = layer_post[mask_f].mean(axis=0)
    spectrum_ambig = layer_post[ambiguity_mask].mean(axis=0)
    spectrum_misinfo = layer_post[misinfo_mask].mean(axis=0)

    # Peak analysis
    def analyze_spectrum(spectrum, label):
        peaks, _ = find_peaks(spectrum, height=2.0 / L, prominence=0.5 / L)
        max_layer = np.argmax(spectrum)
        max_val = spectrum[max_layer]
        concentration = np.sum(np.sort(spectrum)[-3:])
        print(f"  {label:>20s}: peak L{max_layer} ({max_val:.3f}), top3={concentration:.3f}, n_peaks={len(peaks)}")
        return peaks, max_layer

    print("\nSpectral analysis:")
    analyze_spectrum(spectrum_faithful, 'Faithful')
    analyze_spectrum(spectrum_ambig, 'Ambiguity-type')
    analyze_spectrum(spectrum_misinfo, 'Misinfo-type')

    # Spectral features
    features = {}
    features['max_P'] = layer_post.max(axis=1)
    features['argmax_P'] = layer_post.argmax(axis=1)
    features['n_active'] = (layer_post > 2 / L).sum(axis=1)
    features['top3_conc'] = np.sort(layer_post, axis=1)[:, -3:].sum(axis=1)
    features['gini'] = 1 - np.sum(layer_post ** 2, axis=1)
    features['early_mass'] = layer_post[:, :12].sum(axis=1)
    features['mid_mass'] = layer_post[:, 12:24].sum(axis=1)
    features['late_mass'] = layer_post[:, 24:].sum(axis=1)

    print(f"\n{'Feature':>15s}  {'Faithful':>10s}  {'Ambiguity':>10s}  {'Misinfo':>10s}")
    print("-" * 52)
    for fname, fval in features.items():
        print(f"{fname:>15s}  {fval[mask_f].mean():10.4f}  {fval[ambiguity_mask].mean():10.4f}  {fval[misinfo_mask].mean():10.4f}")

    # Critical layers
    print("\nCritical layers for each group:")
    for label, mask in [('Faithful', mask_f), ('Ambiguity', ambiguity_mask), ('Misinfo', misinfo_mask)]:
        spec = layer_post[mask].mean(axis=0)
        top5 = np.argsort(spec)[-5:][::-1]
        print(f"  {label}: {', '.join(f'L{l} ({spec[l]:.4f})' for l in top5)}")

    # ══════════════════════════════════════════════════════════
    # MEGA FIGURE
    # ══════════════════════════════════════════════════════════
    colors_map = {'faithful': '#4ECDC4', 'ambiguity': '#FF6B6B', 'misinfo': '#FFE66D'}
    ds_colors = {'TruthfulQA': '#E74C3C', 'PopQA': '#3498DB', 'TriviaQA': '#2ECC71', 'NeQA': '#9B59B6'}
    layers = np.arange(L)

    fig = plt.figure(figsize=(24, 20))

    # (a) Causal Posterior Spectrum
    ax1 = fig.add_subplot(3, 3, 1)
    for lbl, spec, clr, mk in [('Faithful', spectrum_faithful, colors_map['faithful'], '-o'),
                                 ('Ambiguity-halluc', spectrum_ambig, colors_map['ambiguity'], '-s'),
                                 ('Misinfo-halluc', spectrum_misinfo, colors_map['misinfo'], '-^')]:
        ax1.fill_between(layers, spec, alpha=0.25, color=clr)
        n_lbl = mask_f.sum() if 'Faith' in lbl else ambiguity_mask.sum() if 'Ambig' in lbl else misinfo_mask.sum()
        ax1.plot(layers, spec, mk, color=clr, lw=2, ms=3, label=f'{lbl} (n={n_lbl})')
    ax1.axhline(1 / L, color='grey', ls=':', lw=1, label=f'Uniform (1/L={1/L:.3f})')
    ax1.set_xlabel('Layer'); ax1.set_ylabel('P(m_l | D)')
    ax1.set_title('(a) Causal Posterior Spectrum', fontweight='bold', fontsize=11)
    ax1.legend(fontsize=7, loc='upper left'); ax1.set_xlim(-0.5, L - 0.5)
    for b in [6, 12, 18, 24, 30]:
        ax1.axvline(b - 0.5, color='grey', ls='--', lw=0.5, alpha=0.3)

    # (b) Posterior Heatmap
    ax2 = fig.add_subplot(3, 3, 2)
    sig_idx = np.where(significant)[0]
    order = sig_idx[np.argsort(layer_ent[sig_idx])]
    im = ax2.imshow(layer_post[order], aspect='auto', cmap='hot', vmin=0, vmax=0.2, interpolation='nearest')
    ax2.set_xlabel('Layer'); ax2.set_ylabel('Prompt (sorted by H, low→high)')
    ax2.set_title('(b) Posterior Heatmap', fontweight='bold', fontsize=11)
    plt.colorbar(im, ax=ax2, shrink=0.8, label='P(m_l|D)')
    for b in [6, 12, 18, 24, 30]:
        ax2.axvline(b - 0.5, color='cyan', lw=0.5, ls='--', alpha=0.5)
    halluc_positions = [np.where(order == idx)[0][0] for idx in np.where(mask_h)[0] if idx in order]
    for hp in halluc_positions:
        ax2.plot(L + 0.3, hp, 'r|', ms=2)

    # (c) Faithful | Ambig | Misinfo side-by-side
    ax3 = fig.add_subplot(3, 3, 3)
    n_show = 30
    rng = np.random.RandomState(42)
    f_idx = np.where(mask_f)[0]; a_idx = np.where(ambiguity_mask)[0]; m_idx = np.where(misinfo_mask)[0]
    f_sample = f_idx[rng.choice(len(f_idx), min(n_show, len(f_idx)), replace=False)]
    a_sample = a_idx[rng.choice(len(a_idx), min(n_show, len(a_idx)), replace=False)]
    m_sample = m_idx[rng.choice(len(m_idx), min(n_show, len(m_idx)), replace=False)]
    combined = np.concatenate([f_sample, a_sample, m_sample])
    im3 = ax3.imshow(layer_post[combined], aspect='auto', cmap='hot', vmin=0, vmax=0.2, interpolation='nearest')
    ax3.axhline(len(f_sample) - 0.5, color='cyan', lw=2)
    ax3.axhline(len(f_sample) + len(a_sample) - 0.5, color='yellow', lw=2)
    ax3.set_xlabel('Layer'); ax3.set_ylabel('Prompt')
    ax3.set_title('(c) Faithful | Ambig | Misinfo', fontweight='bold', fontsize=11)
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    ax3.text(-3, len(f_sample) // 2, 'F', fontweight='bold', color='cyan', ha='center', va='center', fontsize=10)
    ax3.text(-3, len(f_sample) + len(a_sample) // 2, 'A', fontweight='bold', color='red', ha='center', va='center', fontsize=10)
    ax3.text(-3, len(f_sample) + len(a_sample) + len(m_sample) // 2, 'M', fontweight='bold', color='orange', ha='center', va='center', fontsize=10)

    # (d) Entropy KDE
    ax4 = fig.add_subplot(3, 3, 4)
    x_range = np.linspace(0, H_max_layer * 1.1, 200)
    for label, arr, color in [('Faithful', H_f_layer, colors_map['faithful']),
                               ('Ambiguity', layer_ent[ambiguity_mask], colors_map['ambiguity']),
                               ('Misinfo', layer_ent[misinfo_mask], colors_map['misinfo'])]:
        if len(arr) > 2:
            kde = stats.gaussian_kde(arr, bw_method=0.3)
            ax4.fill_between(x_range, kde(x_range), alpha=0.3, color=color, label=f'{label} (n={len(arr)})')
            ax4.plot(x_range, kde(x_range), color=color, lw=2)
    ax4.axvline(H_max_layer, color='grey', ls='--', lw=1, label=f'H_max={H_max_layer:.2f}')
    ax4.axvline(H_median, color='black', ls=':', lw=1.5, label=f'Split H={H_median:.2f}')
    ax4.set_xlabel('Layer-Level Posterior Entropy H'); ax4.set_ylabel('Density')
    ax4.set_title(f'(d) Entropy KDE (d={cohen_d_layer:.3f})', fontweight='bold', fontsize=11)
    ax4.legend(fontsize=7)

    # (e) Spike Count vs Max Posterior
    ax5 = fig.add_subplot(3, 3, 5)
    threshold = 2.0 / L
    spike_counts = (layer_post > threshold).sum(axis=1)
    max_P = layer_post.max(axis=1)
    for label, mask, color, marker in [('Faithful', mask_f, colors_map['faithful'], 'o'),
                                        ('Ambiguity', ambiguity_mask, colors_map['ambiguity'], 's'),
                                        ('Misinfo', misinfo_mask, colors_map['misinfo'], '^')]:
        ax5.scatter(spike_counts[mask], max_P[mask], c=color, alpha=0.5, s=30, marker=marker,
                    edgecolors='white', lw=0.3, label=f'{label} (n={mask.sum()})')
    ax5.set_xlabel(f'# Layers with P > {threshold:.3f}'); ax5.set_ylabel('Max P(m_l | D)')
    ax5.set_title('(e) Spike Count vs Max Posterior', fontweight='bold', fontsize=11); ax5.legend(fontsize=7)

    # (f) Spectrum by Dataset
    ax6 = fig.add_subplot(3, 3, 6)
    for ds_name in ['TruthfulQA', 'PopQA', 'TriviaQA', 'NeQA']:
        dm = (ds_labels == ds_name) & significant
        if dm.sum() > 0:
            spec = layer_post[dm].mean(axis=0)
            ax6.plot(layers, spec, '-o', ms=3, lw=2, color=ds_colors[ds_name], label=ds_name)
    ax6.axhline(1 / L, color='grey', ls=':', lw=1)
    ax6.set_xlabel('Layer'); ax6.set_ylabel('Mean P(m_l | D)')
    ax6.set_title('(f) Spectrum by Dataset', fontweight='bold', fontsize=11)
    ax6.legend(fontsize=8); ax6.set_xlim(-0.5, L - 0.5)

    # (g) Differential Spectrum
    ax7 = fig.add_subplot(3, 3, 7)
    diff_spectrum = spectrum_misinfo - spectrum_ambig
    ax7.bar(layers, diff_spectrum, color=np.where(diff_spectrum > 0, '#FFE66D', '#FF6B6B'), edgecolor='black', lw=0.3)
    ax7.axhline(0, color='black', lw=1)
    ax7.set_xlabel('Layer'); ax7.set_ylabel('P(misinfo) - P(ambiguity)')
    ax7.set_title('(g) Differential Spectrum', fontweight='bold', fontsize=11)
    top_pos = np.argsort(diff_spectrum)[-3:][::-1]
    top_neg = np.argsort(diff_spectrum)[:3]
    for l in top_pos:
        if diff_spectrum[l] > 0.005:
            ax7.annotate(f'L{l}', xy=(l, diff_spectrum[l]), fontsize=8, ha='center', va='bottom', color='darkgoldenrod')
    for l in top_neg:
        if diff_spectrum[l] < -0.005:
            ax7.annotate(f'L{l}', xy=(l, diff_spectrum[l]), fontsize=8, ha='center', va='top', color='red')

    # (h) Group vs Layer Entropy
    ax8 = fig.add_subplot(3, 3, 8)
    for label, mask, color in [('Faithful', mask_f, colors_map['faithful']), ('Halluc', mask_h, '#FF6B6B')]:
        ax8.scatter(entropies_grp[mask], layer_ent[mask], c=color, alpha=0.5, s=25, edgecolors='white', lw=0.3, label=label)
    r, _ = stats.pearsonr(entropies_grp[significant], layer_ent[significant])
    ax8.set_xlabel('Group-Level H (K=6)'); ax8.set_ylabel('Layer-Level H (L=36)')
    ax8.set_title(f'(h) Group vs Layer Entropy (r={r:.3f})', fontweight='bold', fontsize=11)
    ax8.legend(fontsize=8)
    lims_max = max(entropies_grp[significant].max(), layer_ent[significant].max()) * 1.1
    ax8.plot([0, lims_max], [0, lims_max * H_max_layer / np.log(6)], 'grey', ls='--', lw=1, alpha=0.5)

    # (i) 3-Way Gauge
    ax9 = fig.add_subplot(3, 3, 9)
    theta = np.linspace(np.pi, 0, 200)
    ax9.plot(np.cos(theta), np.sin(theta), 'k-', lw=2)
    for label, H_mean, color, radius in [('Faithful', H_f_layer.mean(), colors_map['faithful'], 0.7),
                                           ('Ambiguity', layer_ent[ambiguity_mask].mean(), colors_map['ambiguity'], 0.8),
                                           ('Misinfo', layer_ent[misinfo_mask].mean(), colors_map['misinfo'], 0.9)]:
        frac = H_mean / H_max_layer if H_max_layer > 0 else 0
        angle = np.pi * (1 - frac)
        ax9.annotate('', xy=(radius * np.cos(angle), radius * np.sin(angle)), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
    ax9.text(0, -0.15, f'Faithful: H={H_f_layer.mean():.3f}\nAmbiguity: H={layer_ent[ambiguity_mask].mean():.3f}\nMisinfo: H={layer_ent[misinfo_mask].mean():.3f}',
             ha='center', va='top', fontsize=9, fontweight='bold')
    ax9.text(-1.05, -0.02, 'Low H\n(spike)', ha='center', fontsize=8, color='green')
    ax9.text(1.05, -0.02, 'High H\n(flat)', ha='center', fontsize=8, color='red')
    ax9.set_xlim(-1.3, 1.3); ax9.set_ylim(-0.35, 1.15); ax9.set_aspect('equal'); ax9.axis('off')
    ax9.set_title('(i) 3-Way Ambiguity Gauge', fontweight='bold', fontsize=11)

    fig.suptitle(f'Phase 2: Fine-Grained Causal Posterior Spectrum (L={L}, beta={beta}, N={N})',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{C["output_dir"]}/causal_spectrum.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved causal_spectrum.png")

    # ── Individual spectra figure ──
    fig, axes = plt.subplots(3, 4, figsize=(20, 12), sharey=True)
    f_by_ent = np.argsort(layer_ent[mask_f]); f_indices_arr = np.where(mask_f)[0]
    h_by_ent = np.argsort(layer_ent[mask_h]); h_indices_arr = np.where(mask_h)[0]
    for col in range(4):
        ax = axes[0, col]
        if col < len(f_by_ent):
            idx = f_indices_arr[f_by_ent[col]]
            ax.bar(range(L), layer_post[idx], color=colors_map['faithful'], edgecolor='black', lw=0.3)
            ax.axhline(1 / L, color='grey', ls=':', lw=1)
            ax.set_title(f'Faithful #{idx} [{ds_labels[idx][:6]}]\nH={layer_ent[idx]:.3f}', fontsize=9, fontweight='bold')
        ax.set_xlim(-0.5, L - 0.5)
        if col == 0: ax.set_ylabel('P(m_l|D)')
    for col in range(4):
        ax = axes[1, col]
        if col < len(h_by_ent):
            idx = h_indices_arr[h_by_ent[col]]
            ax.bar(range(L), layer_post[idx], color=colors_map['misinfo'], edgecolor='black', lw=0.3)
            ax.axhline(1 / L, color='grey', ls=':', lw=1)
            ax.set_title(f'Misinfo #{idx} [{ds_labels[idx][:6]}]\nH={layer_ent[idx]:.3f}', fontsize=9, fontweight='bold')
        ax.set_xlim(-0.5, L - 0.5)
        if col == 0: ax.set_ylabel('P(m_l|D)')
    for col in range(4):
        ax = axes[2, col]
        rev_idx = len(h_by_ent) - 1 - col
        if rev_idx >= 0:
            idx = h_indices_arr[h_by_ent[rev_idx]]
            ax.bar(range(L), layer_post[idx], color=colors_map['ambiguity'], edgecolor='black', lw=0.3)
            ax.axhline(1 / L, color='grey', ls=':', lw=1)
            ax.set_title(f'Ambiguity #{idx} [{ds_labels[idx][:6]}]\nH={layer_ent[idx]:.3f}', fontsize=9, fontweight='bold')
        ax.set_xlim(-0.5, L - 0.5)
        if col == 0: ax.set_ylabel('P(m_l|D)')
        ax.set_xlabel('Layer')
    fig.suptitle('Individual Causal Posterior Spectra\n(Top: Faithful | Mid: Misinfo | Bot: Ambiguity)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{C["output_dir"]}/individual_spectra.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved individual_spectra.png")

    # ══════════════════════════════════════════════════════════
    # Residual Contribution Analysis (requires forward passes)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("RESIDUAL CONTRIBUTION ANALYSIS")
    print("=" * 60)

    rng_sub = np.random.RandomState(C['residual_sample_seed'])
    f_indices_all = np.where(mask_f)[0]
    h_indices_all = np.where(mask_h)[0]
    n_half = C['n_residual_samples'] // 2

    def stratified_sample(indices, ents, n):
        quartiles = np.percentile(ents[indices], [25, 50, 75])
        q_masks = [
            ents[indices] <= quartiles[0],
            (ents[indices] > quartiles[0]) & (ents[indices] <= quartiles[1]),
            (ents[indices] > quartiles[1]) & (ents[indices] <= quartiles[2]),
            ents[indices] > quartiles[2]
        ]
        selected = []
        per_q = n // 4
        for qm in q_masks:
            q_idx = indices[qm]
            rng_sub.shuffle(q_idx)
            selected.extend(q_idx[:per_q].tolist())
        remaining = [i for i in indices if i not in selected]
        rng_sub.shuffle(remaining)
        selected.extend(remaining[:n - len(selected)])
        return selected[:n]

    sample_f = stratified_sample(f_indices_all, layer_ent, n_half)
    sample_h = stratified_sample(h_indices_all, layer_ent, n_half)
    sample_indices = sorted(sample_f + sample_h)
    print(f"Residual analysis: {len(sample_indices)} prompts ({len(sample_f)} F, {len(sample_h)} H)")

    # Reconstruct prompts
    all_prompts = reconstruct_prompts(C)
    CORRUPT = C['corrupt_entity']

    layer_contrib_clean = np.zeros((len(sample_indices), L, n_embd), dtype=np.float32)
    layer_contrib_corrupt = np.zeros((len(sample_indices), L, n_embd), dtype=np.float32)
    contrib_diff_norm = np.zeros((len(sample_indices), L))
    contrib_cosine = np.zeros((len(sample_indices), L))

    t0 = time.time()
    for si, idx in enumerate(sample_indices):
        prompt, subject, answers, ds_name = all_prompts[idx]
        corrupt_prompt = prompt.replace(subject, CORRUPT, 1)
        if corrupt_prompt == prompt:
            corrupt_prompt = CORRUPT + " " + prompt

        engine.reset(); engine.forward(prompt)
        clean_acts = [a.copy() for a in engine.get_all_activations()]
        engine.reset(); engine.forward(corrupt_prompt)
        corrupt_acts = [a.copy() for a in engine.get_all_activations()]

        for l in range(L):
            c_contrib = clean_acts[l] if l == 0 else clean_acts[l] - clean_acts[l - 1]
            x_contrib = corrupt_acts[l] if l == 0 else corrupt_acts[l] - corrupt_acts[l - 1]
            layer_contrib_clean[si, l] = c_contrib
            layer_contrib_corrupt[si, l] = x_contrib
            diff = c_contrib - x_contrib
            contrib_diff_norm[si, l] = np.linalg.norm(diff)
            norm_c = np.linalg.norm(c_contrib); norm_x = np.linalg.norm(x_contrib)
            contrib_cosine[si, l] = np.dot(c_contrib, x_contrib) / (norm_c * norm_x) if (norm_c > 1e-10 and norm_x > 1e-10) else 1.0

        if (si + 1) % 10 == 0:
            print(f"  [{si+1}/{len(sample_indices)}] {time.time()-t0:.1f}s")
    print(f"Done in {time.time()-t0:.1f}s")

    sample_is_f = is_faithful[sample_indices]
    sample_ent = layer_ent[sample_indices]
    f_mask_s = sample_is_f; h_mask_s = ~sample_is_f

    # ── Residual figure ──
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    ax = axes[0, 0]
    ax.plot(range(L), contrib_diff_norm[f_mask_s].mean(axis=0), '-o', ms=4, color=colors_map['faithful'], lw=2, label='Faithful')
    ax.plot(range(L), contrib_diff_norm[h_mask_s].mean(axis=0), '-s', ms=4, color='#FF6B6B', lw=2, label='Hallucinated')
    ax.set_xlabel('Layer'); ax.set_ylabel('||clean - corrupt contrib||')
    ax.set_title('(a) Residual Contribution Difference', fontweight='bold'); ax.legend(fontsize=9)
    for b in [6, 12, 18, 24, 30]: ax.axvline(b - 0.5, color='grey', ls='--', lw=0.5, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(range(L), contrib_cosine[f_mask_s].mean(axis=0), '-o', ms=4, color=colors_map['faithful'], lw=2, label='Faithful')
    ax.plot(range(L), contrib_cosine[h_mask_s].mean(axis=0), '-s', ms=4, color='#FF6B6B', lw=2, label='Hallucinated')
    ax.set_xlabel('Layer'); ax.set_ylabel('Cosine Similarity')
    ax.set_title('(b) Within-Layer Computation Similarity', fontweight='bold'); ax.legend(fontsize=9); ax.set_ylim(-0.1, 1.1)

    ax = axes[0, 2]
    order_s = np.argsort(sample_ent)
    im = ax.imshow(contrib_diff_norm[order_s], aspect='auto', cmap='magma', interpolation='nearest')
    ax.set_xlabel('Layer'); ax.set_ylabel('Prompt (sorted by H)')
    ax.set_title('(c) Contribution Diff Heatmap', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='||diff||')
    for b in [6, 12, 18, 24, 30]: ax.axvline(b - 0.5, color='cyan', lw=0.5, ls='--', alpha=0.5)

    ax = axes[1, 0]
    mean_diff_f = contrib_diff_norm[f_mask_s].mean(axis=0)
    mean_diff_h = contrib_diff_norm[h_mask_s].mean(axis=0)
    ratio = mean_diff_h / (mean_diff_f + 1e-10)
    ax.bar(range(L), ratio - 1, color=np.where(ratio > 1, '#FF6B6B', '#4ECDC4'), edgecolor='black', lw=0.3)
    ax.axhline(0, color='black', lw=1); ax.set_xlabel('Layer'); ax.set_ylabel('Halluc/Faithful Ratio - 1')
    ax.set_title('(d) Excess Divergence in Hallucinated', fontweight='bold')
    top_div = np.argsort(ratio)[-3:][::-1]
    for l in top_div: ax.annotate(f'L{l}', xy=(l, ratio[l] - 1), fontsize=8, ha='center', va='bottom', fontweight='bold')

    ax = axes[1, 1]
    cos_diff = contrib_cosine[f_mask_s].mean(axis=0) - contrib_cosine[h_mask_s].mean(axis=0)
    ax.bar(range(L), cos_diff, color=np.where(cos_diff > 0, '#FF6B6B', '#4ECDC4'), edgecolor='black', lw=0.3)
    ax.axhline(0, color='black', lw=1); ax.set_xlabel('Layer'); ax.set_ylabel('cos(F) - cos(H)')
    ax.set_title('(e) Cosine Divergence (F-H)', fontweight='bold')

    ax = axes[1, 2]
    clean_norm_f = np.linalg.norm(layer_contrib_clean[f_mask_s], axis=2).mean(axis=0)
    clean_norm_h = np.linalg.norm(layer_contrib_clean[h_mask_s], axis=2).mean(axis=0)
    ax.plot(range(L), clean_norm_f, '-o', ms=4, color=colors_map['faithful'], lw=2, label='Faithful')
    ax.plot(range(L), clean_norm_h, '-s', ms=4, color='#FF6B6B', lw=2, label='Hallucinated')
    ax.set_xlabel('Layer'); ax.set_ylabel('||layer contribution|| (clean)')
    ax.set_title('(f) Clean Contribution Magnitude', fontweight='bold'); ax.legend(fontsize=9)

    fig.suptitle('Residual Contribution Analysis (Attn+MLP Proxy)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{C["output_dir"]}/residual_contribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved residual_contribution.png")

    # ── Save all results ──
    np.savez(f'{C["output_dir"]}/phase2_results.npz',
        layer_posteriors=layer_post, layer_entropies=layer_ent,
        IE_per_layer=IE_per_layer, is_faithful=is_faithful, significant=significant,
        ds_labels=ds_labels, ambiguity_mask=ambiguity_mask, misinfo_mask=misinfo_mask,
        mask_f=mask_f, mask_h=mask_h,
        spectrum_faithful=spectrum_faithful, spectrum_ambig=spectrum_ambig, spectrum_misinfo=spectrum_misinfo,
        contrib_diff_norm=contrib_diff_norm, contrib_cosine=contrib_cosine,
        sample_indices=np.array(sample_indices),
        beta=beta, cohen_d_layer=cohen_d_layer,
    )

    summary = {
        'phase': '2', 'n_prompts': int(N), 'n_layers': int(L), 'beta': beta,
        'layer_level': {'H_faithful': float(H_f_layer.mean()), 'H_hallucinated': float(H_h_layer.mean()),
                         'cohen_d': float(cohen_d_layer), 'KS_p': float(p_ks), 'MW_p': float(p_mw)},
        'subtypes': {'n_ambiguity': int(ambiguity_mask.sum()), 'n_misinfo': int(misinfo_mask.sum()),
                      'H_ambiguity': float(layer_ent[ambiguity_mask].mean()),
                      'H_misinfo': float(layer_ent[misinfo_mask].mean())},
        'per_dataset': {ds: {k: float(v) if isinstance(v, (float, np.floating)) else int(v)
                              for k, v in r.items()} for ds, r in per_ds_layer.items()},
    }
    with open(f'{C["output_dir"]}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("PHASE 2 COMPLETE")
    print(f"{'='*60}")
    print(f"L={L}, beta={beta}, Cohen d={cohen_d_layer:.3f}")
    print(f"Ambiguity: {ambiguity_mask.sum()}, Misinfo: {misinfo_mask.sum()}")
    print(f"Saved to {C['output_dir']}/")


if __name__ == '__main__':
    main()
