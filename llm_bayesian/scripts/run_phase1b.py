#!/usr/bin/env python3
"""
Phase 1b — Multi-Dataset Bayesian Causal Tracing
=================================================
Loads prompts from 4 HuggingFace benchmarks (TruthfulQA, PopQA,
TriviaQA, NeQA), runs activation patching causal tracing,
computes Bayesian posterior entropy per prompt.

Configurable: model, sample counts per dataset, beta, mechanisms.
Results are used by Phase 2 and Phase 3.
"""

import os, sys, time, re, json, argparse
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
    model_path     = "/workspace/Model/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf",
    n_ctx          = 2048,
    n_gpu_layers   = 99,

    # ── Output ──
    output_dir = '/workspace/Data/bayesian_phase1b',

    # ── Dataset sample counts (change these to use more data) ──
    n_truthfulqa = 100,     # how many TruthfulQA prompts to use
    n_popqa      = 100,     # how many PopQA prompts to use
    n_triviaqa   = 50,      # how many TriviaQA prompts to use
    n_neqa       = 50,      # how many NeQA prompts to use
    popqa_low_popularity_pool = 2000,  # pre-filter low-popularity from this pool

    # ── Causal tracing ──
    corrupt_entity  = "Zyxwv",
    gen_max_tokens  = 30,
    gen_temperature = 0.01,
    gen_top_k       = 1,
    gen_seed        = 42,

    # ── Mechanism grouping ──
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
    causal_effect_threshold = 0.01,

    # ── HuggingFace cache ──
    hf_home = '/workspace/Data/huggingface_cache',

    # ── Random seed ──
    random_seed = 42,
)
# ═══════════════════════════════════════════════════════════════════


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


def load_datasets(C):
    """Load and sample prompts from 4 benchmarks. Returns list of (prompt, subject, [expected], dataset_name)."""
    from datasets import load_dataset
    rng = np.random.RandomState(C['random_seed'])
    all_prompts = []

    # ═══ 1. TruthfulQA ═══
    print("Loading TruthfulQA...")
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
    rng.shuffle(tqa_prompts)
    all_prompts.extend(tqa_prompts[:C['n_truthfulqa']])
    print(f"  Selected: {min(C['n_truthfulqa'], len(tqa_prompts))} (from {len(tqa_prompts)} valid)")

    # ═══ 2. PopQA (low-popularity bias) ═══
    print("Loading PopQA...")
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
    rng.shuffle(pqa_low)
    all_prompts.extend([(p, s, a, d) for p, s, a, d, _ in pqa_low[:C['n_popqa']]])
    print(f"  Selected: {min(C['n_popqa'], len(pqa_low))} (low-popularity bias)")

    # ═══ 3. TriviaQA ═══
    print("Loading TriviaQA...")
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
    rng.shuffle(trivia_prompts)
    all_prompts.extend(trivia_prompts[:C['n_triviaqa']])
    print(f"  Selected: {min(C['n_triviaqa'], len(trivia_prompts))} (from {len(trivia_prompts)} valid)")

    # ═══ 4. NeQA ═══
    print("Loading NeQA...")
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
    rng.shuffle(neqa_prompts)
    all_prompts.extend(neqa_prompts[:C['n_neqa']])
    print(f"  Selected: {min(C['n_neqa'], len(neqa_prompts))} (from {len(neqa_prompts)} valid)")

    N = len(all_prompts)
    print(f"\nTOTAL: {N} prompts")
    ds_counts = {}
    for _, _, _, ds in all_prompts:
        ds_counts[ds] = ds_counts.get(ds, 0) + 1
    for ds, n in ds_counts.items():
        print(f"  {ds}: {n}")

    return all_prompts


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', dest='model_path')
    p.add_argument('--n-ctx', type=int)
    p.add_argument('--n-truthfulqa', type=int)
    p.add_argument('--n-popqa', type=int)
    p.add_argument('--n-triviaqa', type=int)
    p.add_argument('--n-neqa', type=int)
    p.add_argument('--beta', type=float, dest='beta_default')
    p.add_argument('--output-dir')
    p.add_argument('--seed', type=int, dest='random_seed')
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

    import neuroscope
    engine = neuroscope.Engine(C['model_path'], n_ctx=C['n_ctx'], n_gpu_layers=C['n_gpu_layers'])
    n_layers = engine.model_info.n_layers
    n_embd = engine.model_info.n_embd
    print(f"Model: {engine.model_info.name}")
    print(f"Layers: {n_layers}, Dim: {n_embd}")

    # Sanity check
    engine.reset()
    test_out = engine.generate("The capital of France is", max_tokens=10, temperature=0.01, top_k=1, seed=42)
    print(f"Sanity: {strip_think(test_out).strip()[:40]}")

    MECHANISMS = C['mechanisms']
    K = len(MECHANISMS)
    MECH_NAMES = list(MECHANISMS.keys())
    CORRUPT = C['corrupt_entity']

    all_prompts = load_datasets(C)
    N = len(all_prompts)

    # ── Step 1: Generate answers ──
    print("\nStep 1: Generating answers...")
    gen_results = []
    t0 = time.time()
    for i, (prompt, subject, answers, ds_name) in enumerate(all_prompts):
        engine.reset()
        output = engine.generate(prompt, max_tokens=C['gen_max_tokens'],
                                 temperature=C['gen_temperature'], top_k=C['gen_top_k'], seed=C['gen_seed'])
        output = strip_think(output)
        generated = output[len(prompt):] if output.startswith(prompt) else output
        generated = generated.strip()
        is_correct = check_match(generated, answers)
        gen_results.append(dict(idx=i, prompt=prompt, subject=subject,
                                expected=answers, generated=generated[:80],
                                correct=is_correct, dataset=ds_name))
        if (i+1) % 50 == 0:
            n_correct = sum(r['correct'] for r in gen_results)
            print(f"  [{i+1}/{N}] {n_correct}/{i+1} correct ({n_correct/(i+1)*100:.0f}%) | {time.time()-t0:.1f}s")

    is_faithful = np.array([r['correct'] for r in gen_results])
    ds_labels = np.array([r['dataset'] for r in gen_results])

    print(f"\nOverall: {is_faithful.sum()} faithful / {(~is_faithful).sum()} hallucinated")
    for ds_name in ['TruthfulQA', 'PopQA', 'TriviaQA', 'NeQA']:
        mask_ds = ds_labels == ds_name
        if mask_ds.sum() > 0:
            n_f = is_faithful[mask_ds].sum()
            print(f"  {ds_name}: {n_f}/{mask_ds.sum()} = {n_f/mask_ds.sum()*100:.1f}%")

    # ── Step 2: Causal tracing ──
    print("\nStep 2: Causal tracing...")
    IE = np.zeros((N, K))
    IE_raw = np.zeros((N, K))
    IE_full = np.zeros(N)
    p_clean_arr = np.zeros(N)
    p_corrupt_arr = np.zeros(N)
    target_tokens = np.zeros(N, dtype=np.int64)
    IE_per_layer = np.zeros((N, n_layers))

    t0 = time.time()
    for i, (prompt, subject, answers, ds_name) in enumerate(all_prompts):
        corrupt_prompt = prompt.replace(subject, CORRUPT, 1)
        if corrupt_prompt == prompt:
            corrupt_prompt = CORRUPT + " " + prompt

        engine.reset(); engine.forward(prompt)
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

        if (i+1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed; eta = (N - i - 1) / rate
            print(f"  [{i+1:3d}/{N}] {elapsed:.0f}s, ~{eta:.0f}s left")

    print(f"\nCausal tracing: {time.time()-t0:.1f}s")

    causal_effect = np.abs(p_clean_arr - p_corrupt_arr)
    significant = causal_effect > C['causal_effect_threshold']
    print(f"Significant: {significant.sum()}/{N}")

    # ── Step 3: Bayesian inference ──
    beta = C['beta_default']
    posteriors, entropies = compute_posteriors_and_entropy(IE, beta)

    mask_f = is_faithful & significant
    mask_h = (~is_faithful) & significant
    H_faithful = entropies[mask_f]
    H_halluc = entropies[mask_h]
    H_max = np.log(K)

    sp = np.sqrt((H_faithful.var(ddof=1) + H_halluc.var(ddof=1)) / 2) if (len(H_faithful)>1 and len(H_halluc)>1) else 1e-10
    cohen_d = (H_halluc.mean() - H_faithful.mean()) / sp if sp > 1e-10 else 0

    print(f"\nbeta={beta}, mask_f={mask_f.sum()}, mask_h={mask_h.sum()}")
    print(f"Cohen d (overall) = {cohen_d:.3f}")

    # Per-dataset breakdown
    per_ds_results = {}
    for ds_name in ['TruthfulQA', 'PopQA', 'TriviaQA', 'NeQA']:
        ds_mask = ds_labels == ds_name
        mf = ds_mask & is_faithful & significant
        mh = ds_mask & (~is_faithful) & significant
        Hf_ds = entropies[mf]; Hh_ds = entropies[mh]
        if len(Hf_ds) > 1 and len(Hh_ds) > 1:
            sp_ds = np.sqrt((Hf_ds.var(ddof=1) + Hh_ds.var(ddof=1)) / 2)
            d_ds = (Hh_ds.mean() - Hf_ds.mean()) / sp_ds if sp_ds > 1e-10 else 0
            _, p_ds = stats.mannwhitneyu(Hh_ds, Hf_ds, alternative='greater')
        else:
            d_ds, p_ds = 0, 1.0
        per_ds_results[ds_name] = dict(n_f=int(mf.sum()), n_h=int(mh.sum()),
                                        H_f=float(Hf_ds.mean()) if len(Hf_ds)>0 else 0,
                                        H_h=float(Hh_ds.mean()) if len(Hh_ds)>0 else 0,
                                        d=float(d_ds), p=float(p_ds))
        sig = '***' if p_ds<0.001 else '**' if p_ds<0.01 else '*' if p_ds<0.05 else 'n.s.'
        print(f"  {ds_name}: n_f={mf.sum()}, n_h={mh.sum()}, d={d_ds:.3f}, p={p_ds:.4f} {sig}")

    # ── Step 4: Main figure ──
    colors_f, colors_h = '#4ECDC4', '#FF6B6B'
    ds_colors = {'TruthfulQA': '#E74C3C', 'PopQA': '#3498DB', 'TriviaQA': '#2ECC71', 'NeQA': '#9B59B6'}
    mean_post_f = posteriors[mask_f].mean(axis=0) if mask_f.sum() > 0 else np.zeros(K)
    mean_post_h = posteriors[mask_h].mean(axis=0) if mask_h.sum() > 0 else np.zeros(K)

    fig = plt.figure(figsize=(22, 18))

    ax1 = fig.add_subplot(3, 3, 1)
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
    ax1.set_title(f'(a) Overall KDE (d={cohen_d:.2f})', fontweight='bold'); ax1.legend(fontsize=8)

    ax3 = fig.add_subplot(3, 3, 3)
    ds_d = [per_ds_results[d]['d'] for d in ['TruthfulQA', 'PopQA', 'TriviaQA', 'NeQA']]
    ds_p = [per_ds_results[d]['p'] for d in ['TruthfulQA', 'PopQA', 'TriviaQA', 'NeQA']]
    ds_names_plot = ['TruthfulQA', 'PopQA', 'TriviaQA', 'NeQA']
    ax3.bar(range(4), ds_d, color=[ds_colors[d] for d in ds_names_plot], edgecolor='black', lw=0.5)
    ax3.axhline(0, color='black', lw=1); ax3.axhline(0.2, color='green', ls='--', lw=1, alpha=0.5)
    ax3.set_xticks(range(4)); ax3.set_xticklabels(ds_names_plot, fontsize=8, rotation=15)
    ax3.set_ylabel("Cohen's d"); ax3.set_title("(c) Effect Size by Dataset", fontweight='bold')
    for i, (d, p) in enumerate(zip(ds_d, ds_p)):
        sig_s = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
        ax3.text(i, d + 0.02 * np.sign(d), f'{d:.2f}{sig_s}', ha='center', va='bottom' if d>=0 else 'top', fontsize=9)

    ax4 = fig.add_subplot(3, 3, 4)
    x = np.arange(K); w = 0.35
    ax4.bar(x - w/2, mean_post_f, w, color=colors_f, edgecolor='black', lw=0.5, label='Faithful')
    ax4.bar(x + w/2, mean_post_h, w, color=colors_h, edgecolor='black', lw=0.5, label='Hallucinated')
    ax4.set_xticks(x); ax4.set_xticklabels(MECH_NAMES, fontsize=7, rotation=20, ha='right')
    ax4.set_ylabel('Mean P(m|D)'); ax4.set_title('(d) Mean Posterior by Group', fontweight='bold'); ax4.legend(fontsize=8)

    ax5 = fig.add_subplot(3, 3, 5)
    sig_idx = np.where(significant)[0]
    if len(sig_idx) > 0:
        order = sig_idx[np.argsort(entropies[sig_idx])]
        n_show = min(80, len(order))
        im = ax5.imshow(IE_per_layer[order[:n_show]], aspect='auto', cmap='RdYlBu_r', vmin=-0.1, vmax=0.5)
        ax5.set_xlabel('Layer'); ax5.set_ylabel('Prompt (sorted by H)')
        ax5.set_title('(e) Per-Layer IE', fontweight='bold')
        plt.colorbar(im, ax=ax5, shrink=0.8, label='IE')

    fig.suptitle(f'Phase 1b: Multi-Dataset Bayesian Causal Tracing (N={N}, beta={beta})',
                 fontsize=14, fontweight='bold', y=1.01)
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
        auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else 0.5
        print(f"  Mann-Whitney U={U:.1f}, p={p_mw:.6f}")
        print(f"  KS D={ks_stat:.4f}, p={p_ks:.6f}")
        print(f"  Welch t={t_stat:.3f}, p={p_t:.6f}")
        print(f"  AUC={auc:.3f}")
        print(f"  Cohen d={cohen_d:.3f}")

    # Beta sensitivity
    print(f"\nBeta sensitivity:")
    print(f"{'beta':>5s}  {'Cohen d':>8s}  {'p':>10s}")
    for beta_val in C['beta_sensitivity_values']:
        post_b, ent_b = compute_posteriors_and_entropy(IE, beta_val)
        Hf = ent_b[mask_f]; Hh = ent_b[mask_h]
        sp_b = np.sqrt((Hf.var(ddof=1) + Hh.var(ddof=1)) / 2) if (len(Hf)>1 and len(Hh)>1) else 1e-10
        d_b = (Hh.mean() - Hf.mean()) / sp_b if sp_b > 1e-10 else 0
        _, p_b = stats.mannwhitneyu(Hh, Hf, alternative='greater') if (len(Hf)>1 and len(Hh)>1) else (0, 1)
        sig_s = '***' if p_b<0.001 else '**' if p_b<0.01 else '*' if p_b<0.05 else 'n.s.'
        print(f"{beta_val:5.1f}  {d_b:8.3f}  {p_b:10.6f}  {sig_s}")

    # ── Save ──
    np.savez(f'{C["output_dir"]}/phase1b_results.npz',
        is_faithful=is_faithful, significant=significant, ds_labels=ds_labels,
        p_clean=p_clean_arr, p_corrupt=p_corrupt_arr, target_tokens=target_tokens,
        IE=IE, IE_raw=IE_raw, IE_full=IE_full, IE_per_layer=IE_per_layer,
        posteriors=posteriors, entropies=entropies,
        beta_default=beta, n_prompts=N, n_mechanisms=K,
        mechanism_names=np.array(MECH_NAMES),
    )

    summary = {
        'phase': '1b', 'model': os.path.basename(C['model_path']),
        'n_prompts': int(N), 'beta': beta,
        'overall': {'n_faithful': int(mask_f.sum()), 'n_hallucinated': int(mask_h.sum()),
                     'H_faithful': float(H_faithful.mean()) if len(H_faithful)>0 else None,
                     'H_hallucinated': float(H_halluc.mean()) if len(H_halluc)>0 else None,
                     'cohen_d': float(cohen_d)},
        'per_dataset': per_ds_results,
    }
    with open(f'{C["output_dir"]}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("PHASE 1b COMPLETE")
    print(f"{'='*60}")
    print(f"N={N}, Faithful={mask_f.sum()}, Hallucinated={mask_h.sum()}")
    print(f"Cohen d={cohen_d:.3f}")
    print(f"Saved to {C['output_dir']}/")


if __name__ == '__main__':
    main()
