"""Shared modeling utilities: feature definitions, metrics, bootstrap CIs,
and a DeLong-style paired AUROC comparison, used by every evaluation script
so the same definitions are used everywhere."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                              precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

FEATURE_FAMILIES = {
    "volume": ["n_client_queries", "n_client_responses", "n_upstream_queries",
               "n_upstream_responses", "total_wire_bytes"],
    "aggregation": ["upstream_client_ratio", "response_multiplicity", "max_active_upstream"],
    "ecs": ["distinct_client_ecs", "distinct_upstream_ecs", "ecs_fanout",
            "missing_ecs_rate", "ecs_mismatch_rate"],
    "entropy": ["txid_entropy", "sport_entropy"],
    "answer_cache": ["answer_uniqueness", "answer_disagreement", "ttl_mean", "ttl_cv",
                      "rcode_diversity", "servfail_rate", "state_reuse_rate"],
    "timing": ["client_latency_mean", "client_latency_p95", "client_latency_cv",
               "upstream_latency_mean", "response_race_gap_mean"],
}
ALL_FEATURES = sum(FEATURE_FAMILIES.values(), [])
BASIC_PACKET_FEATURES = ["n_client_queries", "n_client_responses", "total_wire_bytes",
                          "client_latency_mean", "client_latency_cv"]

CLASSIFIERS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=2000, C=1.0),
    "random_forest": lambda: RandomForestClassifier(n_estimators=300, max_depth=None, random_state=0),
    "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(random_state=0),
}


def load_data(path):
    return pd.read_csv(path)


def fit_predict(model_name, X_train, y_train, X_test, feature_cols):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train[feature_cols].values)
    Xte = scaler.transform(X_test[feature_cols].values)
    clf = CLASSIFIERS[model_name]()
    clf.fit(Xtr, y_train)
    proba = clf.predict_proba(Xte)[:, 1]
    return clf, scaler, proba


def benign_quantile_threshold(train_proba, train_labels, q=0.995):
    benign_scores = train_proba[train_labels == 0]
    if len(benign_scores) == 0:
        return 0.5
    return float(np.quantile(benign_scores, q))


def metrics_at_threshold(y_true, proba, threshold):
    y_true = np.asarray(y_true)
    pred = (proba >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fam = 1e6 * fp / (fp + tn) if (fp + tn) > 0 else 0.0
    n_benign = fp + tn
    rule_of_three = (3.0 / n_benign) if (fp == 0 and n_benign > 0) else None
    try:
        auroc = roc_auc_score(y_true, proba) if len(set(y_true)) > 1 else float("nan")
    except ValueError:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y_true, proba) if len(set(y_true)) > 1 else float("nan")
    except ValueError:
        auprc = float("nan")
    return {"precision": precision, "recall": recall, "f1": f1, "auroc": auroc, "auprc": auprc,
            "fpr": fpr, "fam": fam, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_benign": n_benign, "rule_of_three_fpr_upper95": rule_of_three}


def bootstrap_ci(y_true, proba, metric_fn, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    n = len(y_true)
    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]
    vals = []
    for _ in range(n_boot):
        bi_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True) if len(idx_pos) else np.array([], dtype=int)
        bi_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True) if len(idx_neg) else np.array([], dtype=int)
        bi = np.concatenate([bi_pos, bi_neg])
        try:
            v = metric_fn(y_true[bi], proba[bi])
        except Exception:
            continue
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"), float("nan"))
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return (float(np.mean(vals)), lo, hi)


def _delong_auc_var(y_true, scores):
    """Fast DeLong AUC variance/covariance components (Sun & Xu 2014 style)."""
    order = np.argsort(scores)
    y_true = y_true[order]
    scores = scores[order]
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]

    def midrank(x):
        srt_idx = np.argsort(x)
        x_sorted = x[srt_idx]
        n = len(x)
        ranks = np.zeros(n)
        i = 0
        while i < n:
            j = i
            while j < n - 1 and x_sorted[j + 1] == x_sorted[i]:
                j += 1
            ranks[i:j + 1] = 0.5 * (i + j) + 1
            i = j + 1
        out = np.empty(n)
        out[srt_idx] = ranks
        return out

    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return float("nan"), None, None
    tx = midrank(pos)
    ty = midrank(neg)
    tz = midrank(np.concatenate([pos, neg]))
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    auc = float(np.mean(v01))
    return auc, v01, v10


def delong_paired_test(y_true, scores_a, scores_b):
    """Paired DeLong test for AUROC(a) == AUROC(b) on the same held-out set.
    Returns (auc_a, auc_b, z_stat, p_value_two_sided)."""
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    auc_a, v01_a, v10_a = _delong_auc_var(y_true, scores_a)
    auc_b, v01_b, v10_b = _delong_auc_var(y_true, scores_b)
    if v01_a is None or v01_b is None:
        return auc_a, auc_b, float("nan"), float("nan")
    v_a = np.concatenate([v01_a, v10_a])
    v_b = np.concatenate([v01_b, v10_b])
    cov = np.cov(np.vstack([v_a, v_b]))
    var = cov[0, 0] / len(v01_a) + cov[1, 1] - 2 * cov[0, 1]
    # standard DeLong combines within-class covariances; use conservative
    # pooled variance across positive/negative components separately.
    m = len(v01_a)
    n = len(v10_a)
    s01 = np.cov(np.vstack([v01_a, v01_b])) / m
    s10 = np.cov(np.vstack([v10_a, v10_b])) / n
    var_diff = s01[0, 0] + s10[0, 0] + s01[1, 1] + s10[1, 1] - 2 * s01[0, 1] - 2 * s10[0, 1]
    if var_diff <= 0:
        return auc_a, auc_b, float("nan"), float("nan")
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))
    return auc_a, auc_b, float(z), float(p)


def mcnemar_test(y_true, pred_a, pred_b):
    """McNemar's test (with continuity correction) comparing two paired
    binary classifiers' correctness on the same held-out episodes."""
    y_true = np.asarray(y_true)
    correct_a = (np.asarray(pred_a) == y_true)
    correct_b = (np.asarray(pred_b) == y_true)
    b = int(np.sum(correct_a & ~correct_b))  # a right, b wrong
    c = int(np.sum(~correct_a & correct_b))  # a wrong, b right
    if b + c == 0:
        return 0.0, 1.0, b, c
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    from scipy.stats import chi2
    p = 1 - chi2.cdf(stat, df=1)
    return float(stat), float(p), b, c
