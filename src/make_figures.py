"""Generates every figure in the manuscript from the actual computed
results tables -- nothing here is drawn from hand-entered numbers."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

import modeling_common as mc

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"
FIGDIR = ROOT / "results" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colorblind-safe palette, standard in scientific publishing.
C = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
    "purple": "#CC79A7", "yellow": "#F0E442", "skyblue": "#56B4E9", "grey": "#666666",
    "black": "#111111",
}

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "font.family": "serif",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#333333",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "figure.dpi": 150,
})


def savefig(fig, name):
    fig.savefig(FIGDIR / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGDIR / name)


def fig1_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=8.2, textcolor="white"):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.08",
                               linewidth=0, facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
                 color=textcolor, wrap=True)

    def arrow(x0, y0, x1, y1, color=C["black"], style="-|>", lw=1.3, ls="-"):
        a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=10,
                             color=color, lw=lw, linestyle=ls)
        ax.add_patch(a)

    box(0.3, 1.9, 1.9, 1.2, "Client\n(loopback\nUDP)", C["grey"])
    box(3.1, 1.9, 2.4, 1.2, "Instrumented\nResolver\n(policy logic +\nepisode capture)", C["blue"])

    box(6.5, 3.15, 2.75, 1.55, "Real internet DNS\ninfrastructure\n(Google, Cloudflare,\nQuad9, OpenDNS,\nreal authoritative\nservers)", C["green"], fontsize=6.9)
    box(6.5, 0.35, 2.75, 1.55, "Local faulty\nauthority\n(127.0.0.1 only,\nseeded from a real\nlookup; faults injected\nwith calibrated timing)", C["red"], fontsize=6.7)

    box(9.55, 3.15, 2.15, 1.55, "BENIGN class\n(live query/\nanswer, real\nECS steering)", "#e8f4ea", textcolor="#1a1a1a", fontsize=7.2)
    box(9.55, 0.35, 2.15, 1.55, "ADVERSE class\n(ECS omit/\nmismatch, conflict,\nfan-out)", "#fbe9e2", textcolor="#1a1a1a", fontsize=7.2)

    arrow(2.2, 2.5, 3.1, 2.5)
    arrow(5.5, 2.7, 6.5, 3.7)
    arrow(5.5, 2.3, 6.5, 1.3)
    arrow(9.25, 3.92, 9.55, 3.92)
    arrow(9.25, 1.12, 9.55, 1.12)
    arrow(3.1, 2.1, 2.2, 2.1, color=C["grey"], style="-|>")

    ax.text(6.3, 4.95, "Benign traffic: real, live, unreplayed", fontsize=8, color=C["green"], ha="center")
    ax.text(6.3, 0.12, "Adverse traffic: locally injected, 127.0.0.1-bound, never sent to the internet", fontsize=8,
            color=C["red"], ha="center")
    savefig(fig, "Fig1_architecture.pdf")


def fig2_loro():
    df = pd.read_csv(TABLES / "model_eval" / "loro_metrics.csv")
    metrics = ["precision", "recall", "f1"]
    labels = df["held_out_policy"].tolist()
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    colors = [C["blue"], C["orange"], C["green"]]
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1) * width, df[m], width, label=m.capitalize(), color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Leave-one-resolver-policy-out (training-benign $Q_{0.995}$ threshold)")
    ax.legend(loc="lower right", ncol=3, frameon=False)
    savefig(fig, "Fig2_loro_generalization.pdf")


def fig3_laso():
    df = pd.read_csv(TABLES / "model_eval" / "laso_metrics.csv")
    labels = [c.replace("adverse-", "") for c in df["held_out_condition"]]
    metrics = ["precision", "recall", "f1", "auroc"]
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    colors = [C["blue"], C["orange"], C["green"], C["purple"]]
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, df[m], width, label=m.upper() if m == "auroc" else m.capitalize(), color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Leave-one-adverse-condition-out (training-benign $Q_{0.995}$ threshold)")
    ax.legend(loc="lower right", ncol=4, frameon=False, fontsize=7)
    savefig(fig, "Fig3_laso_generalization.pdf")


def fig4_ablation():
    df = pd.read_csv(TABLES / "model_eval" / "ablation.csv")
    order = df["variant"].tolist()
    labels = [v.replace("minus_", "− ").replace("_", " ") for v in order]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    x = np.arange(len(order))
    ax.bar(x - 0.18, df["f1"], 0.36, label="F1 (fixed 0.5 threshold)", color=C["blue"])
    ax.bar(x + 0.18, df["auroc"], 0.36, label="AUROC", color=C["orange"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Feature-family ablation under fixed-threshold LORO")
    ax.legend(loc="lower left", frameon=False)
    savefig(fig, "Fig4_ablation.pdf")


def fig5_signatures():
    df = pd.read_csv(ROOT / "data" / "processed" / "combined_features.csv")
    feats = ["missing_ecs_rate", "ecs_mismatch_rate", "ecs_fanout", "answer_disagreement"]
    conds = ["benign-plain", "benign-ecs-stable", "benign-ecs-geo", "benign-burst",
             "adverse-ecs-omit", "adverse-ecs-mismatch", "adverse-conflict", "adverse-ecs-fanout"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
    for ax, feat in zip(axes.flat, feats):
        data = [df[df["condition"] == c][feat].values for c in conds]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(C["green"] if conds[i].startswith("benign") else C["red"])
            patch.set_alpha(0.65)
        ax.set_xticks(range(1, len(conds) + 1))
        ax.set_xticklabels([c.replace("benign-", "b:").replace("adverse-", "a:") for c in conds],
                            rotation=55, ha="right", fontsize=6.6)
        ax.set_title(feat.replace("_", " "), fontsize=8.6)
    fig.suptitle("Raw behavioral signatures by condition (real benign traffic vs. locally injected adverse traffic)",
                 fontsize=8.8, y=1.01)
    fig.tight_layout()
    savefig(fig, "Fig5_behavioral_signatures.pdf")


def fig6_latency_ecdf():
    df = pd.read_csv(ROOT / "data" / "processed" / "combined_features.csv")
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for label, name, color in [(0, "Benign (real internet)", C["green"]), (1, "Adverse (local, calibrated)", C["red"])]:
        vals = np.sort(df[df["label"] == label]["client_latency_mean"].values * 1000)
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, label=name, color=color, lw=1.4)
    ax.set_xscale("log")
    ax.set_xlabel("Client-facing latency (ms, log scale)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Client latency: benign vs. adverse episodes")
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    savefig(fig, "Fig6_latency_ecdf.pdf")


def fig7_feature_importance():
    df = pd.read_csv(TABLES / "model_eval" / "feature_importance.csv").head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    colors = [C["blue"] if c >= 0 else C["red"] for c in df["standardized_coefficient"]]
    ax.barh(df["feature"].str.replace("_", " "), df["standardized_coefficient"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Standardized logistic-regression coefficient")
    ax.set_title("Largest-magnitude feature coefficients (pooled corpus)")
    savefig(fig, "Fig7_feature_importance.pdf")


def fig8_threshold_sensitivity():
    df = pd.read_csv(TABLES / "model_eval" / "threshold_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(df["quantile"], df["recall"], "-o", label="Recall", color=C["blue"], ms=3.5)
    ax.plot(df["quantile"], df["fpr"], "-s", label="FPR", color=C["red"], ms=3.5)
    ax.plot(df["quantile"], df["f1"], "-^", label="F1", color=C["green"], ms=3.5)
    ax.set_xlabel(r"Training-benign threshold quantile $Q$")
    ax.set_ylabel("Score")
    ax.set_title("LASO threshold sensitivity")
    ax.legend(loc="center left", frameon=False)
    savefig(fig, "Fig8_threshold_sensitivity.pdf")


def fig9_real_ecs_validation():
    pool = json.loads((ROOT / "data" / "processed" / "domain_pool" / "domain_pool.json").read_text())
    live = pool["live_domains"]
    n_answers = [d["n_answers"] for d in live]
    ecs_sensitive = sum(1 for d in live if d["ecs_sensitive"])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    axes[0].hist(n_answers, bins=range(1, max(n_answers) + 2), color=C["blue"], edgecolor="white", align="left")
    axes[0].set_xlabel("Distinct A-records returned (one real probe subnet)")
    axes[0].set_ylabel("Number of real domains")
    axes[0].set_title("Real answer-set cardinality")

    sizes = [ecs_sensitive, len(live) - ecs_sensitive]
    axes[1].bar(["ECS-sensitive\n(answer changes\nby client subnet)", "ECS-insensitive"], sizes,
                color=[C["orange"], C["grey"]])
    axes[1].set_title(f"{ecs_sensitive}/{len(live)} live real domains show\nreal ECS-driven answer steering")
    fig.suptitle("Live real-internet probe of the benign domain pool (measured on the collection date)", fontsize=8.6)
    fig.tight_layout()
    savefig(fig, "Fig9_real_ecs_validation.pdf")


if __name__ == "__main__":
    fig1_architecture()
    fig2_loro()
    fig3_laso()
    fig4_ablation()
    fig5_signatures()
    fig6_latency_ecdf()
    fig7_feature_importance()
    fig8_threshold_sensitivity()
    fig9_real_ecs_validation()
