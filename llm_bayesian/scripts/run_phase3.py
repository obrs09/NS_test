#!/usr/bin/env python3
"""
Phase 3 — Early Warning & Causal Blocking Pipeline
====================================================
Part A: Train an early-layer activation classifier to predict
        hallucinations before generation completes.
Part B: Grid-search correction steering on clean prompts to
        restore correct-answer probability and rank.
Part C: Combine detection + intervention into a pipeline and
        evaluate end-to-end accuracy improvement.

Requires Phase 1b and Phase 2 results.
"""

import os, sys, time, re, json, argparse
import numpy as np
from scipy import stats
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, classification_report, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
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
    tokenizer_path = "/workspace/Model/Qwen3-8B",
    n_ctx          = 2048,
    n_gpu_layers   = 99,

    # ── Input (Phase 1b & 2 results) ──
    phase1b_results_path = '/workspace/Data/bayesian_phase1b/phase1b_results.npz',
    phase2_results_path  = '/workspace/Data/bayesian_phase2/phase2_results.npz',

    # ── Output ──
    output_dir = '/workspace/Data/bayesian_phase3',

    # ── Part A: Early Warning Detector ──
    early_cutoff = 10,        # use layers L0..L(cutoff-1) for features
    cv_folds     = 5,
    classifier_seed = 42,
    # Classifier hyperparameters (change to tune)
    logreg_C           = 1.0,
    logreg_max_iter    = 1000,
    rf_n_estimators    = 100,
    rf_max_depth       = 5,
    gbm_n_estimators   = 100,
    gbm_max_depth      = 3,
    gbm_learning_rate  = 0.1,

    # ── Part B: Causal Blocking Intervention ──
    corrupt_entity       = "Zyxwv",
    intervention_layers  = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 34],
    strengths            = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0],

    # ── Part C: Pipeline ──
    detection_threshold  = 0.5,
    threshold_sweep      = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],

    # ── Dataset reconstruction (same counts as Phase 1b) ──
    n_truthfulqa = 100,
    n_popqa      = 100,
    n_triviaqa   = 50,
    n_neqa       = 50,
    popqa_low_popularity_pool = 2000,

    # ── HuggingFace cache ──
    hf_home = '/workspace/Data/huggingface_cache',

    # ── Random seed ──
    random_seed = 42,
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


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


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


def extract_early_features(early_acts, all_acts_norm, EARLY_CUTOFF):
    """Extract features from L0..L(CUTOFF-1) activations for hallucination prediction."""
    N = early_acts.shape[0]
    features = {}

    for l in range(EARLY_CUTOFF):
        features[f'norm_L{l}'] = np.linalg.norm(early_acts[:, l, :], axis=1)

    for l in range(1, EARLY_CUTOFF):
        contrib = early_acts[:, l, :] - early_acts[:, l - 1, :]
        features[f'contrib_norm_L{l}'] = np.linalg.norm(contrib, axis=1)

    for l in range(1, EARLY_CUTOFF):
        cos = np.sum(early_acts[:, l, :] * early_acts[:, l - 1, :], axis=1) / (
            np.linalg.norm(early_acts[:, l, :], axis=1) * np.linalg.norm(early_acts[:, l - 1, :], axis=1) + 1e-10)
        features[f'cos_L{l}_L{l-1}'] = cos

    for l in range(EARLY_CUTOFF):
        features[f'var_L{l}'] = np.var(early_acts[:, l, :], axis=1)

    for l in range(EARLY_CUTOFF):
        features[f'max_L{l}'] = np.max(early_acts[:, l, :], axis=1)
        features[f'mean_L{l}'] = np.mean(early_acts[:, l, :], axis=1)

    norms = np.array([features[f'norm_L{l}'] for l in range(EARLY_CUTOFF)]).T
    for l in range(1, EARLY_CUTOFF):
        features[f'norm_growth_L{l}'] = norms[:, l] / (norms[:, l - 1] + 1e-10)

    features['early_norm_mean'] = norms.mean(axis=1)
    features['early_norm_std'] = norms.std(axis=1)
    features['early_norm_max'] = norms.max(axis=1)
    features['early_norm_range'] = norms.max(axis=1) - norms.min(axis=1)

    early_mean_norm = all_acts_norm[:, :EARLY_CUTOFF].mean(axis=1)
    late_mean_norm = all_acts_norm[:, -EARLY_CUTOFF:].mean(axis=1)
    features['early_late_norm_ratio'] = early_mean_norm / (late_mean_norm + 1e-10)

    for l in range(1, EARLY_CUTOFF):
        features[f'dist_from_L0_L{l}'] = np.linalg.norm(early_acts[:, l, :] - early_acts[:, 0, :], axis=1)

    for l in range(EARLY_CUTOFF):
        features[f'kurtosis_L{l}'] = stats.kurtosis(early_acts[:, l, :], axis=1)

    return features


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', dest='model_path')
    p.add_argument('--tokenizer', dest='tokenizer_path')
    p.add_argument('--phase1b', dest='phase1b_results_path')
    p.add_argument('--phase2', dest='phase2_results_path')
    p.add_argument('--early-cutoff', type=int, dest='early_cutoff')
    p.add_argument('--cv-folds', type=int, dest='cv_folds')
    p.add_argument('--threshold', type=float, dest='detection_threshold')
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
    EARLY_CUTOFF = C['early_cutoff']

    # ── Load Phase 1b/2 data ──
    data = np.load(C['phase1b_results_path'], allow_pickle=True)
    IE_per_layer = data['IE_per_layer']
    is_faithful  = data['is_faithful']
    significant  = data['significant']
    ds_labels    = data['ds_labels']
    p_clean_arr  = data['p_clean']
    p_corrupt_arr = data['p_corrupt']

    data2 = np.load(C['phase2_results_path'], allow_pickle=True)
    layer_post     = data2['layer_posteriors']
    layer_ent      = data2['layer_entropies']
    ambiguity_mask = data2['ambiguity_mask']
    misinfo_mask   = data2['misinfo_mask']
    mask_f         = data2['mask_f']
    mask_h         = data2['mask_h']

    N, L = IE_per_layer.shape
    print(f"Loaded: {N} prompts, {L} layers")
    print(f"  Faithful: {mask_f.sum()}, Halluc: {mask_h.sum()}")
    print(f"  Ambiguity: {ambiguity_mask.sum()}, Misinfo: {misinfo_mask.sum()}")

    # ── Load model ──
    import neuroscope
    engine = neuroscope.Engine(C['model_path'], n_ctx=C['n_ctx'], n_gpu_layers=C['n_gpu_layers'])
    n_layers = engine.model_info.n_layers
    n_embd = engine.model_info.n_embd
    print(f"Model: {n_layers} layers, {n_embd} dim")

    # ── Reconstruct prompts ──
    all_prompts = reconstruct_prompts(C)

    # ══════════════════════════════════════════════════════════
    # Part A: Collect early-layer activations
    # ══════════════════════════════════════════════════════════
    print(f"\nPart A — Collecting activations (L0-L{EARLY_CUTOFF-1})...")
    early_acts = np.zeros((N, EARLY_CUTOFF, n_embd), dtype=np.float32)
    all_acts_norm = np.zeros((N, L), dtype=np.float32)

    t0 = time.time()
    for i in range(N):
        prompt = all_prompts[i][0]
        engine.reset()
        engine.forward(prompt)
        acts = engine.get_all_activations()
        for l in range(L):
            all_acts_norm[i, l] = np.linalg.norm(acts[l])
        for l in range(EARLY_CUTOFF):
            early_acts[i, l] = acts[l]
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{N}] {time.time()-t0:.1f}s")
    print(f"Done in {time.time()-t0:.1f}s")

    # ── Feature engineering ──
    features = extract_early_features(early_acts, all_acts_norm, EARLY_CUTOFF)
    feature_names = list(features.keys())
    X = np.column_stack([features[f] for f in feature_names])
    y = (~is_faithful).astype(int)

    sig_mask = significant.astype(bool)
    X_sig = X[sig_mask]
    y_sig = y[sig_mask]
    print(f"Features: {X_sig.shape[1]}, Samples (significant): {X_sig.shape[0]}")
    print(f"  Halluc: {y_sig.sum()}, Faithful: {(1-y_sig).sum()}")

    # ── Classifier training ──
    classifiers = {
        'LogReg (L2)': LogisticRegression(C=C['logreg_C'], max_iter=C['logreg_max_iter'], class_weight='balanced'),
        'LogReg (L1)': LogisticRegression(C=C['logreg_C'], max_iter=C['logreg_max_iter'], penalty='l1', solver='saga', class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=C['rf_n_estimators'], max_depth=C['rf_max_depth'],
                                                 class_weight='balanced', random_state=C['classifier_seed']),
        'GBM': GradientBoostingClassifier(n_estimators=C['gbm_n_estimators'], max_depth=C['gbm_max_depth'],
                                           learning_rate=C['gbm_learning_rate'], random_state=C['classifier_seed']),
    }

    skf = StratifiedKFold(n_splits=C['cv_folds'], shuffle=True, random_state=C['classifier_seed'])

    print(f"\n{'='*70}")
    print(f"EARLY WARNING DETECTOR — {C['cv_folds']}-Fold Stratified CV")
    print(f"Features: L0-L{EARLY_CUTOFF-1} ({X_sig.shape[1]} dims)")
    print(f"{'='*70}")

    results = {}
    best_model = None
    best_auc = 0

    for name, clf in classifiers.items():
        pipe = make_pipeline(StandardScaler(), clf)
        auc_scores = cross_val_score(pipe, X_sig, y_sig, cv=skf, scoring='roc_auc')
        f1_scores = cross_val_score(pipe, X_sig, y_sig, cv=skf, scoring='f1')
        acc_scores = cross_val_score(pipe, X_sig, y_sig, cv=skf, scoring='accuracy')

        results[name] = {
            'auc': float(auc_scores.mean()), 'auc_std': float(auc_scores.std()),
            'f1': float(f1_scores.mean()), 'f1_std': float(f1_scores.std()),
            'acc': float(acc_scores.mean()), 'acc_std': float(acc_scores.std()),
        }
        print(f"\n{name}:")
        print(f"  AUC: {auc_scores.mean():.3f} +/- {auc_scores.std():.3f}")
        print(f"  F1:  {f1_scores.mean():.3f} +/- {f1_scores.std():.3f}")
        print(f"  Acc: {acc_scores.mean():.3f} +/- {acc_scores.std():.3f}")
        if auc_scores.mean() > best_auc:
            best_auc = auc_scores.mean()
            best_model = name

    print(f"\n>>> Best: {best_model} (AUC={best_auc:.3f})")

    # Full-data refit for feature importance and pipeline estimation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sig)
    final_clf = classifiers[best_model]
    final_clf.fit(X_scaled, y_sig)
    y_proba = final_clf.predict_proba(X_scaled)[:, 1]
    y_pred = final_clf.predict(X_scaled)
    print(f"Full-data refit AUC: {roc_auc_score(y_sig, y_proba):.3f}")
    print(classification_report(y_sig, y_pred, target_names=['Faithful', 'Hallucinated']))

    if hasattr(final_clf, 'feature_importances_'):
        importances = final_clf.feature_importances_
    elif hasattr(final_clf, 'coef_'):
        importances = np.abs(final_clf.coef_[0])
    else:
        importances = np.zeros(len(feature_names))

    top_k = 15
    top_idx = np.argsort(importances)[-top_k:][::-1]
    print(f"\nTop-{top_k} features:")
    for rank, idx in enumerate(top_idx):
        print(f"  {rank+1:2d}. {feature_names[idx]:30s} imp={importances[idx]:.4f}")

    # ── Part A Figure ──
    colors = {'faithful': '#4ECDC4', 'halluc': '#FF6B6B', 'misinfo': '#FFE66D'}
    ds_colors_map = {'TruthfulQA': '#E74C3C', 'PopQA': '#3498DB', 'TriviaQA': '#2ECC71', 'NeQA': '#9B59B6'}

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # (a) ROC curves
    ax = axes[0, 0]
    for name, clf in classifiers.items():
        pipe = make_pipeline(StandardScaler(), clf)
        y_cv_proba = cross_val_predict(pipe, X_sig, y_sig, cv=skf, method='predict_proba')[:, 1]
        fpr, tpr, _ = roc_curve(y_sig, y_cv_proba)
        ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC={results[name]["auc"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('(a) ROC Curves — Early Warning', fontweight='bold'); ax.legend(fontsize=8)

    # (b) Feature importance
    ax = axes[0, 1]
    top_fi = np.argsort(importances)[-top_k:]
    ax.barh(range(top_k), importances[top_fi], color='#3498DB', edgecolor='black', lw=0.3)
    ax.set_yticks(range(top_k)); ax.set_yticklabels([feature_names[i] for i in top_fi], fontsize=8)
    ax.set_xlabel('Importance'); ax.set_title(f'(b) Top-{top_k} Features ({best_model})', fontweight='bold')

    # (c) Score distribution
    ax = axes[0, 2]
    for label, msk, color in [('Faithful', mask_f[sig_mask], colors['faithful']),
                                ('Ambiguity', ambiguity_mask[sig_mask], '#FF6B6B'),
                                ('Misinfo', misinfo_mask[sig_mask], '#FFE66D')]:
        if msk.sum() > 2:
            scores = y_proba[msk]
            kde = stats.gaussian_kde(scores, bw_method=0.3)
            xs = np.linspace(0, 1, 200)
            ax.fill_between(xs, kde(xs), alpha=0.3, color=color, label=f'{label} (n={msk.sum()})')
            ax.plot(xs, kde(xs), color=color, lw=2)
    ax.axvline(0.5, color='black', ls='--', lw=1)
    ax.set_xlabel('P(Hallucinated)'); ax.set_ylabel('Density')
    ax.set_title('(c) Score Distribution', fontweight='bold'); ax.legend(fontsize=8)

    # (d) Per-dataset AUC
    ax = axes[1, 0]
    ds_aucs = {}
    for ds_name in ['TruthfulQA', 'PopQA', 'TriviaQA', 'NeQA']:
        dm = ds_labels[sig_mask] == ds_name
        if dm.sum() > 5 and len(np.unique(y_sig[dm])) > 1:
            ds_aucs[ds_name] = roc_auc_score(y_sig[dm], y_proba[dm])
    if ds_aucs:
        dn = sorted(ds_aucs, key=lambda x: ds_aucs[x], reverse=True)
        ax.bar(range(len(dn)), [ds_aucs[d] for d in dn],
               color=[ds_colors_map[d] for d in dn], edgecolor='black', lw=0.5)
        ax.set_xticks(range(len(dn))); ax.set_xticklabels(dn)
        for i, d in enumerate(dn):
            ax.text(i, ds_aucs[d] + 0.02, f'{ds_aucs[d]:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.axhline(0.5, color='grey', ls='--', lw=1); ax.set_ylim(0, 1.05)
    ax.set_ylabel('AUC'); ax.set_title('(d) Per-Dataset AUC', fontweight='bold')

    # (e) Activation norm profile
    ax = axes[1, 1]
    ax.plot(range(L), all_acts_norm[mask_f].mean(axis=0), '-o', ms=3, color=colors['faithful'], lw=2, label='Faithful')
    if ambiguity_mask.sum() > 0:
        ax.plot(range(L), all_acts_norm[ambiguity_mask].mean(axis=0), '-s', ms=3, color='#FF6B6B', lw=2, label='Ambiguity')
    if misinfo_mask.sum() > 0:
        ax.plot(range(L), all_acts_norm[misinfo_mask].mean(axis=0), '-^', ms=3, color='#FFE66D', lw=2, label='Misinfo')
    ax.axvspan(0, EARLY_CUTOFF - 0.5, alpha=0.1, color='red', label=f'Early window')
    ax.set_xlabel('Layer'); ax.set_ylabel('Mean Act Norm')
    ax.set_title('(e) Activation Norm Profile', fontweight='bold'); ax.legend(fontsize=8)

    # (f) AUC vs detection depth
    ax = axes[1, 2]
    cutoff_aucs = []
    cutoff_layers = list(range(2, EARLY_CUTOFF + 1))
    for cutoff in cutoff_layers:
        feats_cut = extract_early_features(early_acts[:, :cutoff, :], all_acts_norm, cutoff)
        X_cut = np.column_stack([feats_cut[f] for f in feats_cut.keys()])
        X_cut_sig = X_cut[sig_mask]
        try:
            pipe = make_pipeline(StandardScaler(),
                                 LogisticRegression(C=C['logreg_C'], max_iter=C['logreg_max_iter'], class_weight='balanced'))
            auc_cut = cross_val_score(pipe, X_cut_sig, y_sig, cv=skf, scoring='roc_auc').mean()
        except:
            auc_cut = 0.5
        cutoff_aucs.append(auc_cut)
    ax.plot(cutoff_layers, cutoff_aucs, '-o', color='#E74C3C', lw=2, ms=5)
    ax.axhline(0.5, color='grey', ls='--', lw=1, label='Random')
    ax.axhline(best_auc, color='blue', ls=':', lw=1, label=f'Best ({best_model})')
    ax.set_xlabel('Early Cutoff Layer'); ax.set_ylabel('CV AUC')
    ax.set_title('(f) AUC vs Detection Depth', fontweight='bold'); ax.legend(fontsize=8)

    fig.suptitle('Phase 3A: Early Warning Hallucination Detector', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{C["output_dir"]}/early_warning_detector.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved early_warning_detector.png")

    # ══════════════════════════════════════════════════════════
    # Part B: Causal Blocking Intervention
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Part B — CAUSAL BLOCKING INTERVENTION")
    print(f"{'='*60}")

    print("Collecting per-layer activation stats...")
    mean_acts_faithful = np.zeros((L, n_embd), dtype=np.float64)
    mean_acts_halluc = np.zeros((L, n_embd), dtype=np.float64)
    count_f, count_h = 0, 0

    t0 = time.time()
    for i in range(N):
        engine.reset()
        engine.forward(all_prompts[i][0])
        acts = engine.get_all_activations()
        for l in range(L):
            if is_faithful[i]:
                mean_acts_faithful[l] += acts[l]
            else:
                mean_acts_halluc[l] += acts[l]
        if is_faithful[i]: count_f += 1
        else: count_h += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{N}] {time.time()-t0:.1f}s")

    mean_acts_faithful /= max(count_f, 1)
    mean_acts_halluc /= max(count_h, 1)
    correction_dir = mean_acts_faithful - mean_acts_halluc
    correction_norms = np.linalg.norm(correction_dir, axis=1)
    correction_dir_normed = correction_dir / (correction_norms[:, None] + 1e-10)
    print(f"Stats collected: {count_f} F, {count_h} H ({time.time()-t0:.1f}s)")

    # ── Intervention grid search ──
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(C['tokenizer_path'])
    print("Loaded tokenizer")

    halluc_indices = np.where(~is_faithful & significant)[0]
    N_halluc = len(halluc_indices)
    intervention_layers = C['intervention_layers']
    strengths_arr = C['strengths']
    n_layers_int = len(intervention_layers)
    n_strengths = len(strengths_arr)

    target_token_ids = np.zeros(N_halluc, dtype=np.int64)
    p_correct_clean = np.zeros(N_halluc)
    rank_correct_clean = np.zeros(N_halluc, dtype=np.int64)
    top1_clean = np.zeros(N_halluc, dtype=np.int64)
    p_correct_steer = np.zeros((N_halluc, n_layers_int, n_strengths))
    rank_correct_steer = np.zeros((N_halluc, n_layers_int, n_strengths), dtype=np.int64)
    top1_steer = np.zeros((N_halluc, n_layers_int, n_strengths), dtype=np.int64)

    skipped = []
    print(f"Grid: {n_layers_int} layers x {n_strengths} strengths, {N_halluc} halluc prompts")
    print(f"Total fwd passes: {N_halluc * (1 + n_layers_int * n_strengths)}")

    t0 = time.time()
    for hi, idx in enumerate(halluc_indices):
        prompt, subject, answers, ds_name = all_prompts[idx]
        expected = answers[0] if answers else ""
        target_ids = tokenizer.encode(expected, add_special_tokens=False)
        if not target_ids:
            target_ids = tokenizer.encode(" " + expected, add_special_tokens=False)
        if not target_ids:
            skipped.append(hi); continue
        target_tid = target_ids[0]
        target_token_ids[hi] = target_tid

        # Clean baseline
        engine.reset(); engine.forward(prompt)
        logits_clean = engine.get_logits()
        probs_clean = softmax(logits_clean)
        p_correct_clean[hi] = probs_clean[target_tid]
        sorted_indices = np.argsort(probs_clean)[::-1]
        rank_correct_clean[hi] = np.where(sorted_indices == target_tid)[0][0]
        top1_clean[hi] = sorted_indices[0]

        # Clean + steering grid
        for li, layer in enumerate(intervention_layers):
            for si, strength in enumerate(strengths_arr):
                engine.reset(); engine.clear_interventions()
                engine.apply_steering(layer, correction_dir_normed[layer].astype(np.float32), strength)
                engine.forward(prompt)
                logits_int = engine.get_logits()
                probs_int = softmax(logits_int)
                p_correct_steer[hi, li, si] = probs_int[target_tid]
                sorted_int = np.argsort(probs_int)[::-1]
                rank_correct_steer[hi, li, si] = np.where(sorted_int == target_tid)[0][0]
                top1_steer[hi, li, si] = sorted_int[0]
                engine.clear_interventions()

        if (hi + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (hi + 1) * (N_halluc - hi - 1)
            print(f"  [{hi+1}/{N_halluc}] {elapsed:.1f}s (ETA: {eta:.0f}s)")

    valid = np.ones(N_halluc, dtype=bool)
    valid[skipped] = False
    N_valid = int(valid.sum())
    print(f"Done in {time.time()-t0:.1f}s ({N_valid} valid, {len(skipped)} skipped)")

    # ── Metrics ──
    delta_p = p_correct_steer - p_correct_clean[:, None, None]
    delta_rank = rank_correct_clean[:, None, None].astype(np.int64) - rank_correct_steer
    top1_became_correct = (top1_steer == target_token_ids[:, None, None])

    mean_delta_p = delta_p[valid].mean(axis=0)
    median_delta_rank = np.median(delta_rank[valid], axis=0)
    mean_delta_rank = delta_rank[valid].astype(float).mean(axis=0)
    frac_rank_improved = (delta_rank[valid] > 0).mean(axis=0)
    frac_p_improved = (delta_p[valid] > 0).mean(axis=0)
    frac_top1_flip = top1_became_correct[valid].mean(axis=0)

    # Display grid
    print(f"\nMean Δp per layer x strength:")
    print(f"{'Layer':>6s}", end='')
    for s in strengths_arr: print(f"  s={s:5.1f}", end='')
    print()
    for li, layer in enumerate(intervention_layers):
        print(f"  L{layer:2d}", end='  ')
        for si in range(n_strengths):
            v = mean_delta_p[li, si]
            print(f"  {v:+.1e}" if abs(v) < 0.001 else f"  {v:+6.4f}", end='')
        print()

    best_li, best_si = np.unravel_index(median_delta_rank.argmax(), median_delta_rank.shape)
    opt_layer = intervention_layers[best_li]
    opt_strength = strengths_arr[best_si]
    opt_drank = delta_rank[valid, best_li, best_si].astype(float)
    opt_dp = delta_p[valid, best_li, best_si]
    opt_top1_flip = top1_became_correct[valid, best_li, best_si]

    print(f"\nOptimal: L{opt_layer}, s={opt_strength}")
    print(f"  Median Δrank: {np.median(opt_drank):+.0f}")
    print(f"  % rank improved: {(opt_drank > 0).mean()*100:.1f}%")
    print(f"  % p improved:    {(opt_dp > 0).mean()*100:.1f}%")
    print(f"  % top-1 flip:    {opt_top1_flip.mean()*100:.1f}%")

    # ── Part B Figure ──
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    ax = axes[0, 0]
    vmax = max(abs(median_delta_rank).max(), 1)
    im = ax.imshow(median_delta_rank.T, aspect='auto', cmap='RdYlGn', vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax.set_xticks(range(n_layers_int)); ax.set_xticklabels([f'L{l}' for l in intervention_layers], fontsize=8)
    ax.set_yticks(range(n_strengths)); ax.set_yticklabels([f'{s:.1f}' for s in strengths_arr])
    ax.set_xlabel('Intervention Layer'); ax.set_ylabel('Strength')
    ax.set_title('(a) Median Rank Improvement', fontweight='bold'); plt.colorbar(im, ax=ax, shrink=0.8)
    ax.plot(best_li, best_si, 'k*', ms=15)

    ax = axes[0, 1]
    best_per_layer = frac_rank_improved.max(axis=1)
    best_s_per_layer = [strengths_arr[frac_rank_improved[li].argmax()] for li in range(n_layers_int)]
    clrs = ['#4ECDC4' if r > 0.5 else '#FFE66D' if r > 0.3 else '#FF6B6B' for r in best_per_layer]
    ax.bar(range(n_layers_int), best_per_layer * 100, color=clrs, edgecolor='black', lw=0.5)
    ax.set_xticks(range(n_layers_int))
    ax.set_xticklabels([f'L{l}\n(s={s:.0f})' for l, s in zip(intervention_layers, best_s_per_layer)], fontsize=7)
    ax.axhline(50, color='green', ls='--', lw=1); ax.set_ylim(0, 100)
    ax.set_ylabel('% Rank Improved'); ax.set_title('(b) Best % Improved per Layer', fontweight='bold')

    ax = axes[0, 2]
    ax.hist(opt_drank, bins=50, color='#3498DB', edgecolor='black', lw=0.3, alpha=0.7)
    ax.axvline(0, color='grey', ls=':', lw=2)
    ax.axvline(np.median(opt_drank), color='red', ls='-', lw=2, label=f'Median={np.median(opt_drank):+.0f}')
    ax.axvline(opt_drank.mean(), color='orange', ls='--', lw=2, label=f'Mean={opt_drank.mean():+.0f}')
    ax.set_xlabel('Rank Change'); ax.set_ylabel('Count')
    ax.set_title(f'(c) Rank Change (L{opt_layer}, s={opt_strength})', fontweight='bold'); ax.legend(fontsize=8)

    ax = axes[1, 0]
    ds_stats = {}
    for ds_name in ['TruthfulQA', 'PopQA', 'TriviaQA', 'NeQA']:
        dm = (ds_labels[halluc_indices] == ds_name) & valid
        if dm.sum() > 0:
            dr = delta_rank[dm, best_li, best_si].astype(float)
            ds_stats[ds_name] = {'pct': (dr > 0).mean() * 100, 'med': np.median(dr), 'n': int(dm.sum())}
    if ds_stats:
        dn = sorted(ds_stats, key=lambda x: ds_stats[x]['pct'], reverse=True)
        ax.bar(range(len(dn)), [ds_stats[d]['pct'] for d in dn],
               color=[ds_colors_map.get(d, 'grey') for d in dn], edgecolor='black', lw=0.5)
        ax.set_xticks(range(len(dn))); ax.set_xticklabels(dn)
        for i, d in enumerate(dn):
            ax.text(i, ds_stats[d]['pct'] + 1.5, f"{ds_stats[d]['pct']:.0f}%\nn={ds_stats[d]['n']}", ha='center', fontsize=8)
    ax.axhline(50, color='green', ls='--', lw=1); ax.set_ylim(0, 110)
    ax.set_ylabel('% Rank Improved'); ax.set_title('(d) Per-Dataset Improvement', fontweight='bold')

    ax = axes[1, 1]
    for li_idx, layer in enumerate(intervention_layers[::3]):
        actual_li = intervention_layers.index(layer)
        ax.plot(strengths_arr, median_delta_rank[actual_li], '-o', ms=5, lw=2, label=f'L{layer}')
    ax.set_xlabel('Strength'); ax.set_ylabel('Median Δrank')
    ax.set_title('(e) Strength-Response Curve', fontweight='bold'); ax.legend(fontsize=8)
    ax.set_xscale('log'); ax.axhline(0, color='grey', ls=':', lw=1)

    ax = axes[1, 2]
    halluc_is_ambig = ambiguity_mask[halluc_indices] & valid
    halluc_is_misinfo = misinfo_mask[halluc_indices] & valid
    for li_idx, layer in enumerate(intervention_layers):
        best_si_l = frac_rank_improved[li_idx].argmax()
        dr_a = np.median(delta_rank[halluc_is_ambig, li_idx, best_si_l].astype(float)) if halluc_is_ambig.sum() > 0 else 0
        dr_m = np.median(delta_rank[halluc_is_misinfo, li_idx, best_si_l].astype(float)) if halluc_is_misinfo.sum() > 0 else 0
        ax.scatter(dr_a, dr_m, s=80, zorder=5)
        ax.annotate(f'L{layer}', (dr_a, dr_m), fontsize=7, ha='left', va='bottom')
    lim_v = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]), abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]), 10)
    ax.plot([-lim_v, lim_v], [-lim_v, lim_v], 'k--', lw=1)
    ax.axhline(0, color='grey', ls=':', lw=0.5); ax.axvline(0, color='grey', ls=':', lw=0.5)
    ax.set_xlabel('Ambiguity: Median Δrank'); ax.set_ylabel('Misinfo: Median Δrank')
    ax.set_title('(f) Ambiguity vs Misinfo', fontweight='bold')

    fig.suptitle('Phase 3B: Causal Blocking Intervention', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{C["output_dir"]}/causal_blocking.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved causal_blocking.png")

    # ══════════════════════════════════════════════════════════
    # Part C: Adaptive Pipeline
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Part C — ADAPTIVE PIPELINE")
    print(f"{'='*60}")

    detection_threshold = C['detection_threshold']
    sig_indices = np.where(sig_mask)[0]
    N_sig = len(sig_indices)

    detected_halluc = y_proba >= detection_threshold
    actual_halluc = y_sig == 1
    TP = int((detected_halluc & actual_halluc).sum())
    FP = int((detected_halluc & ~actual_halluc).sum())
    FN = int((~detected_halluc & actual_halluc).sum())
    TN = int((~detected_halluc & ~actual_halluc).sum())

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    pct_rank_improved = (opt_drank > 0).mean()
    pct_p_improved = (opt_dp > 0).mean()
    pct_top1_flip = float(opt_top1_flip.mean())

    baseline_accuracy = mask_f[sig_indices].sum() / N_sig
    corrected = int(TP * pct_top1_flip)
    pipeline_accuracy_conservative = (TN + corrected) / N_sig
    corrected_partial = int(TP * pct_rank_improved)
    pipeline_accuracy_optimistic = (TN + corrected_partial) / N_sig

    print(f"Detector: {best_model} (CV AUC={best_auc:.3f})")
    print(f"Intervention: L{opt_layer}, s={opt_strength}")
    print(f"Threshold: {detection_threshold}")
    print(f"\nDetection: TP={TP}, FP={FP}, FN={FN}, TN={TN}")
    print(f"  Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}")
    print(f"\nBlocking (L{opt_layer}, s={opt_strength}):")
    print(f"  % rank improved: {pct_rank_improved*100:.1f}%")
    print(f"  % top-1 flip:    {pct_top1_flip*100:.1f}%")
    print(f"\nPipeline:")
    print(f"  Baseline accuracy:                {baseline_accuracy:.3f}")
    print(f"  Conservative (top-1 flip):        {pipeline_accuracy_conservative:.3f}")
    print(f"  Optimistic (rank improved):       {pipeline_accuracy_optimistic:.3f}")

    # Threshold sweep
    print(f"\n{'Threshold':>10s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'#Trig':>6s}  {'Acc(flip)':>10s}  {'Acc(rank)':>10s}")
    print("-" * 70)
    for thr in C['threshold_sweep']:
        det = y_proba >= thr
        tp = (det & actual_halluc).sum()
        fp = (det & ~actual_halluc).sum()
        tn = (~det & ~actual_halluc).sum()
        fn = (actual_halluc & ~det).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_t = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        acc_flip = (tn + int(tp * pct_top1_flip)) / N_sig
        acc_rank = (tn + int(tp * pct_rank_improved)) / N_sig
        print(f"  {thr:8.1f}  {prec:6.3f}  {rec:6.3f}  {f1_t:6.3f}  {det.sum():6d}  {acc_flip:10.3f}  {acc_rank:10.3f}")

    # ── Save everything ──
    np.savez(f'{C["output_dir"]}/phase3_results.npz',
        X_features=X, y_labels=y, feature_names=np.array(feature_names),
        y_proba=y_proba, early_acts=early_acts, all_acts_norm=all_acts_norm,
        mean_acts_faithful=mean_acts_faithful, mean_acts_halluc=mean_acts_halluc,
        correction_dir_normed=correction_dir_normed,
        target_token_ids=target_token_ids,
        p_correct_clean=p_correct_clean, p_correct_steer=p_correct_steer,
        rank_correct_clean=rank_correct_clean, rank_correct_steer=rank_correct_steer,
        delta_p=delta_p, delta_rank=delta_rank, top1_became_correct=top1_became_correct,
        valid_mask=valid,
        intervention_layers=np.array(intervention_layers), strengths=np.array(strengths_arr),
        halluc_indices=halluc_indices,
        best_model=best_model, best_auc=best_auc, opt_layer=opt_layer, opt_strength=opt_strength,
    )

    summary = {
        'phase': '3',
        'part_a': {
            'best_model': best_model, 'cv_auc': float(best_auc),
            'n_features': len(feature_names),
            'top_features': [feature_names[i] for i in np.argsort(importances)[-5:][::-1]],
            'all_results': results,
        },
        'part_b': {
            'optimal_layer': int(opt_layer), 'optimal_strength': float(opt_strength),
            'n_valid': N_valid, 'n_total': int(N_halluc),
            'pct_rank_improved': float(pct_rank_improved * 100),
            'pct_p_improved': float(pct_p_improved * 100),
            'pct_top1_flipped': float(pct_top1_flip * 100),
            'median_delta_rank': float(np.median(opt_drank)),
            'mean_delta_p': float(opt_dp.mean()),
        },
        'part_c': {
            'baseline_accuracy': float(baseline_accuracy),
            'pipeline_accuracy_conservative': float(pipeline_accuracy_conservative),
            'pipeline_accuracy_optimistic': float(pipeline_accuracy_optimistic),
            'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
        },
    }
    with open(f'{C["output_dir"]}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("PHASE 3 COMPLETE")
    print(f"{'='*60}")
    print(f"Part A: {best_model} CV AUC={best_auc:.3f}")
    print(f"Part B: L{opt_layer}, s={opt_strength}, {pct_rank_improved*100:.1f}% rank improved")
    print(f"Part C: baseline={baseline_accuracy:.3f}, conservative={pipeline_accuracy_conservative:.3f}")
    print(f"Saved to {C['output_dir']}/")


if __name__ == '__main__':
    main()
