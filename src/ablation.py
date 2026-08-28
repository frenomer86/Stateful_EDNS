"""RQ4: feature-family ablation under fixed-threshold LORO, plus the
BasicPacket baseline, mirroring evaluate_main.run_loro but sweeping which
feature families are included."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import modeling_common as mc
from evaluate_main import run_loro, pooled_from_folds, DATA_PATH

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "tables" / "model_eval"


def main():
    df = mc.load_data(DATA_PATH)
    rows = []

    folds = run_loro(df, "logistic_regression", feature_cols=mc.ALL_FEATURES, threshold_mode="fixed")
    _, _, m = pooled_from_folds(folds)
    rows.append({"variant": "full_29_feature" if len(mc.ALL_FEATURES) == 29 else f"full_{len(mc.ALL_FEATURES)}_feature",
                 "n_features": len(mc.ALL_FEATURES), **m})

    for family in mc.FEATURE_FAMILIES:
        remaining = [f for fam, feats in mc.FEATURE_FAMILIES.items() if fam != family for f in feats]
        folds = run_loro(df, "logistic_regression", feature_cols=remaining, threshold_mode="fixed")
        _, _, m = pooled_from_folds(folds)
        rows.append({"variant": f"minus_{family}", "n_features": len(remaining), **m})

    folds = run_loro(df, "logistic_regression", feature_cols=mc.BASIC_PACKET_FEATURES, threshold_mode="fixed")
    _, _, m = pooled_from_folds(folds)
    rows.append({"variant": "basic_packet_baseline", "n_features": len(mc.BASIC_PACKET_FEATURES), **m})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "ablation.csv", index=False)
    print(out[["variant", "n_features", "precision", "recall", "f1", "auroc", "auprc", "fpr"]])


if __name__ == "__main__":
    main()
