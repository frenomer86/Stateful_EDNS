"""Primary detection evaluation: random split, LORO, LASO, seed stability,
threshold sensitivity, bootstrap confidence intervals, and a DeLong paired
significance test against the BasicPacket baseline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import modeling_common as mc

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "combined_features.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "tables" / "model_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POLICIES = ["aggregate", "ecs-keyed", "ecs-strict"]
ADVERSE_CONDITIONS = ["adverse-ecs-omit", "adverse-ecs-mismatch", "adverse-conflict", "adverse-ecs-fanout"]


def run_random_split(df, seed=0):
    rows = []
    train, test = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=seed)
    for model_name in mc.CLASSIFIERS:
        clf, scaler, proba = mc.fit_predict(model_name, train, train["label"].values, test, mc.ALL_FEATURES)
        m = mc.metrics_at_threshold(test["label"].values, proba, 0.5)
        rows.append({"model": model_name, **m})
    return pd.DataFrame(rows)


def run_loro(df, model_name="logistic_regression", feature_cols=None, seed_filter=None, threshold_mode="benign_q995"):
    feature_cols = feature_cols or mc.ALL_FEATURES
    results = []
    for held_out in POLICIES:
        sub = df if seed_filter is None else df[df["seed"] == seed_filter]
        train = sub[sub["policy"] != held_out]
        test = sub[sub["policy"] == held_out]
        if train.empty or test.empty:
            continue
        clf, scaler, train_proba = mc.fit_predict(model_name, train, train["label"].values, train, feature_cols)
        test_proba = clf.predict_proba(scaler.transform(test[feature_cols].values))[:, 1]
        if threshold_mode == "benign_q995":
            thr = mc.benign_quantile_threshold(train_proba, train["label"].values, q=0.995)
        else:
            thr = 0.5
        m = mc.metrics_at_threshold(test["label"].values, test_proba, thr)
        results.append({"held_out_policy": held_out, "threshold": thr, "n_train": len(train), "n_test": len(test),
                         **m, "y_true": test["label"].values, "proba": test_proba})
    return results


def run_laso(df, model_name="logistic_regression", feature_cols=None, seed_filter=None, threshold_mode="benign_q995",
             benign_test_frac=0.3, split_seed=0):
    feature_cols = feature_cols or mc.ALL_FEATURES
    results = []
    sub = df if seed_filter is None else df[df["seed"] == seed_filter]
    benign = sub[sub["label"] == 0]
    adverse = sub[sub["label"] == 1]
    benign_train, benign_test = train_test_split(benign, test_size=benign_test_frac, random_state=split_seed)
    for held_out in ADVERSE_CONDITIONS:
        adverse_train = adverse[adverse["condition"] != held_out]
        adverse_test = adverse[adverse["condition"] == held_out]
        train = pd.concat([benign_train, adverse_train], ignore_index=True)
        test = pd.concat([benign_test, adverse_test], ignore_index=True)
        clf, scaler, train_proba = mc.fit_predict(model_name, train, train["label"].values, train, feature_cols)
        test_proba = clf.predict_proba(scaler.transform(test[feature_cols].values))[:, 1]
        if threshold_mode == "benign_q995":
            thr = mc.benign_quantile_threshold(train_proba, train["label"].values, q=0.995)
        else:
            thr = 0.5
        m = mc.metrics_at_threshold(test["label"].values, test_proba, thr)
        results.append({"held_out_condition": held_out, "threshold": thr, "n_train": len(train), "n_test": len(test),
                         **m, "y_true": test["label"].values, "proba": test_proba})
    return results


def pooled_from_folds(fold_results):
    y_true = np.concatenate([r["y_true"] for r in fold_results])
    proba = np.concatenate([r["proba"] for r in fold_results])
    thr = float(np.mean([r["threshold"] for r in fold_results]))
    m = mc.metrics_at_threshold(y_true, proba, thr)
    return y_true, proba, m


def main():
    df = mc.load_data(DATA_PATH)
    seeds = sorted(df["seed"].unique().tolist())
    summary = {}

    # RQ1: random split, all classifiers
    rs = run_random_split(df)
    rs.to_csv(OUT_DIR / "random_split_metrics.csv", index=False)
    summary["random_split"] = rs.to_dict(orient="records")

    # RQ2: LORO, pooled + per-seed, logistic regression primary
    loro_folds = run_loro(df, "logistic_regression")
    loro_fixed_folds = run_loro(df, "logistic_regression", threshold_mode="fixed")
    y_true, proba, loro_pooled = pooled_from_folds(loro_folds)
    loro_rows = [{k: v for k, v in r.items() if k not in ("y_true", "proba")} for r in loro_folds]
    pd.DataFrame(loro_rows).to_csv(OUT_DIR / "loro_metrics.csv", index=False)
    pd.DataFrame([{"episode_idx": i, "y_true": int(yt), "proba": float(p)} for i, (yt, p) in enumerate(zip(y_true, proba))]).to_csv(
        OUT_DIR / "loro_predictions.csv", index=False)

    loro_seed_rows = []
    for s in seeds:
        folds = run_loro(df, "logistic_regression", seed_filter=s)
        _, _, m = pooled_from_folds(folds)
        loro_seed_rows.append({"seed": int(s), **m})
    pd.DataFrame(loro_seed_rows).to_csv(OUT_DIR / "loro_seed_stability.csv", index=False)

    # RQ3: LASO, pooled + per-seed
    laso_folds = run_laso(df, "logistic_regression")
    y_true_l, proba_l, laso_pooled = pooled_from_folds(laso_folds)
    laso_rows = [{k: v for k, v in r.items() if k not in ("y_true", "proba")} for r in laso_folds]
    pd.DataFrame(laso_rows).to_csv(OUT_DIR / "laso_metrics.csv", index=False)

    laso_seed_rows = []
    for s in seeds:
        folds = run_laso(df, "logistic_regression", seed_filter=s)
        _, _, m = pooled_from_folds(folds)
        laso_seed_rows.append({"seed": int(s), **m})
    pd.DataFrame(laso_seed_rows).to_csv(OUT_DIR / "laso_seed_stability.csv", index=False)

    # Bootstrap CIs on pooled LORO / LASO F1 and AUROC
    def f1_at_thr(y, p, thr):
        return mc.metrics_at_threshold(y, p, thr)["f1"]

    loro_thr_mean = float(np.mean([r["threshold"] for r in loro_folds]))
    laso_thr_mean = float(np.mean([r["threshold"] for r in laso_folds]))
    boot_ci = {
        "loro_f1": mc.bootstrap_ci(y_true, proba, lambda y, p: f1_at_thr(y, p, loro_thr_mean)),
        "loro_auroc": mc.bootstrap_ci(y_true, proba, lambda y, p: mc.metrics_at_threshold(y, p, loro_thr_mean)["auroc"]),
        "laso_f1": mc.bootstrap_ci(y_true_l, proba_l, lambda y, p: f1_at_thr(y, p, laso_thr_mean)),
        "laso_auroc": mc.bootstrap_ci(y_true_l, proba_l, lambda y, p: mc.metrics_at_threshold(y, p, laso_thr_mean)["auroc"]),
    }
    with open(OUT_DIR / "bootstrap_ci.json", "w") as fh:
        json.dump({k: {"mean": v[0], "ci95_lo": v[1], "ci95_hi": v[2]} for k, v in boot_ci.items()}, fh, indent=2)

    # DeLong paired AUROC test: StateDNS (full features) vs BasicPacket, under LORO, pooled
    basic_folds = run_loro(df, "logistic_regression", feature_cols=mc.BASIC_PACKET_FEATURES, threshold_mode="fixed")
    y_true_b, proba_b, basic_pooled = pooled_from_folds(basic_folds)
    # align on the same held-out episodes: both use the same policy folds and pooling order
    y_true_full_fixed, proba_full_fixed, full_fixed_pooled = pooled_from_folds(loro_fixed_folds)
    if len(y_true_full_fixed) == len(y_true_b) and np.array_equal(y_true_full_fixed, y_true_b):
        auc_a, auc_b, z, p = mc.delong_paired_test(y_true_full_fixed, proba_full_fixed, proba_b)
        pred_a = (proba_full_fixed >= 0.5).astype(int)
        pred_b = (proba_b >= 0.5).astype(int)
        mcnemar_stat, mcnemar_p, b_only, c_only = mc.mcnemar_test(y_true_full_fixed, pred_a, pred_b)
        significance = {"auroc_full": auc_a, "auroc_basic_packet": auc_b, "delong_z": z, "delong_p_value": p,
                         "mcnemar_stat": mcnemar_stat, "mcnemar_p_value": mcnemar_p,
                         "full_correct_basic_wrong": b_only, "basic_correct_full_wrong": c_only}
    else:
        significance = {"error": "pooled episode sets did not align; see per-fold CSVs"}
    with open(OUT_DIR / "significance_vs_basicpacket.json", "w") as fh:
        json.dump(significance, fh, indent=2)

    # Threshold sensitivity sweep for LASO
    sens_rows = []
    for q in [0.90, 0.95, 0.975, 0.99, 0.995, 0.999]:
        folds = []
        for held_out in ADVERSE_CONDITIONS:
            benign = df[df["label"] == 0]
            adverse = df[df["label"] == 1]
            benign_train, benign_test = train_test_split(benign, test_size=0.3, random_state=0)
            adverse_train = adverse[adverse["condition"] != held_out]
            adverse_test = adverse[adverse["condition"] == held_out]
            train = pd.concat([benign_train, adverse_train], ignore_index=True)
            test = pd.concat([benign_test, adverse_test], ignore_index=True)
            clf, scaler, train_proba = mc.fit_predict("logistic_regression", train, train["label"].values, train, mc.ALL_FEATURES)
            test_proba = clf.predict_proba(scaler.transform(test[mc.ALL_FEATURES].values))[:, 1]
            thr = mc.benign_quantile_threshold(train_proba, train["label"].values, q=q)
            m = mc.metrics_at_threshold(test["label"].values, test_proba, thr)
            folds.append({"y_true": test["label"].values, "proba": test_proba, "threshold": thr})
        _, _, pooled_m = pooled_from_folds(folds)
        sens_rows.append({"quantile": q, **pooled_m})
    pd.DataFrame(sens_rows).to_csv(OUT_DIR / "threshold_sensitivity.csv", index=False)

    # Feature importance: standardized logistic regression on full pooled data
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(df[mc.ALL_FEATURES].values)
    clf = LogisticRegression(max_iter=2000).fit(X, df["label"].values)
    importance = sorted(zip(mc.ALL_FEATURES, clf.coef_[0]), key=lambda kv: -abs(kv[1]))
    pd.DataFrame(importance, columns=["feature", "standardized_coefficient"]).to_csv(
        OUT_DIR / "feature_importance.csv", index=False)

    summary_out = {
        "n_episodes": len(df), "n_features": len(mc.ALL_FEATURES),
        "random_split": rs.to_dict(orient="records"),
        "loro_pooled": {k: v for k, v in loro_pooled.items()},
        "laso_pooled": {k: v for k, v in laso_pooled.items()},
        "loro_fixed_threshold_pooled": {k: v for k, v in full_fixed_pooled.items()},
        "basicpacket_loro_pooled": {k: v for k, v in basic_pooled.items()},
    }
    with open(OUT_DIR / "analysis_summary.json", "w") as fh:
        json.dump(summary_out, fh, indent=2, default=str)

    print("LORO pooled:", loro_pooled)
    print("LASO pooled:", laso_pooled)
    print("BasicPacket LORO pooled (fixed thr):", basic_pooled)
    print("Significance vs BasicPacket:", significance)


if __name__ == "__main__":
    main()
