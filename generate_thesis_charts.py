# File: generate_thesis_charts.py
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

SEED = 42
np.random.seed(SEED)
os.makedirs("thesis_charts", exist_ok=True)

print("--- 1. Loading Saved Models, Vectorizer, and Metrics Tables ---")
rf_model = joblib.load("exported_models/rf_model.pkl")
tfidf_vectorizer = joblib.load("exported_models/tfidf_vectorizer.pkl")
bilstm_model = tf.keras.models.load_model("exported_models/bilstm_model.keras")

with open("exported_models/lstm_tokenizer.json", "r", encoding="utf-8") as f:
    lstm_tokenizer = tokenizer_from_json(f.read())

with open("exported_models/top50_phishing_terms.json", "r", encoding="utf-8") as f:
    top50_terms = json.load(f)
top50_set = set(top50_terms)

metrics_df = pd.read_csv("exported_models/stage5_evaluation_metrics.csv")

# ==============================================================================
# FIGURE 1: THREE-PHASE METRICS BAR CHART (ACCURACY, PRECISION, RECALL, F1, AUC)
# ==============================================================================
print("Generating Figure 1: Three-Phase Grouped Bar Chart...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
phases = metrics_df["Phase"].unique()
metric_cols = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
palette = {
    "Random Forest": "#ef4444",
    "BiLSTM": "#2563eb",
    "Hybrid Ensemble (RF+BiLSTM)": "#10b981"
}

for idx, phase in enumerate(phases):
    ax = axes[idx]
    sub = metrics_df[metrics_df["Phase"] == phase]
    x = np.arange(len(metric_cols))
    n_models = len(sub)
    width = 0.8 / n_models

    for m_i, (_, row) in enumerate(sub.iterrows()):
        m_name = row["Model"]
        vals = [float(row[c]) * 100 for c in metric_cols]
        offset = (m_i - (n_models - 1) / 2) * width
        bars = ax.bar(
            x + offset, vals, width,
            label=m_name,
            color=palette.get(m_name, "#64748b"),
            edgecolor="black", linewidth=0.6
        )
        for b in bars:
            h = b.get_height()
            ax.annotate(
                f"{h:.1f}",
                xy=(b.get_x() + b.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold", rotation=90
            )

    ax.set_title(phase, fontsize=11, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=9.5, fontweight="bold")
    ax.set_ylim(40, 110)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    if idx == 0:
        ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    if idx == len(phases) - 1:
        ax.legend(loc="lower right", frameon=True, fontsize=8.5)

plt.suptitle("Empirical Performance Comparison Across Evaluation Phases", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("thesis_charts/fig1_three_phase_metrics_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

# ==============================================================================
# FIGURE 2: CONFUSION MATRIX HEATMAP GRID (RF vs. BiLSTM ACROSS PHASES 1, 2, 3)
# ==============================================================================
print("Generating Figure 2: Confusion Matrix Heatmaps...")
rf_rows = metrics_df[metrics_df["Model"] == "Random Forest"].reset_index(drop=True)
lstm_rows = metrics_df[metrics_df["Model"] == "BiLSTM"].reset_index(drop=True)
combined_rows = pd.concat([rf_rows, lstm_rows], ignore_index=True)

fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
axes = axes.ravel()

for i, (_, row) in enumerate(combined_rows.iterrows()):
    ax = axes[i]
    cm = np.array([[int(row["TN"]), int(row["FP"])],
                   [int(row["FN"]), int(row["TP"])]])
    tot = cm.sum()
    annot = np.array([
        [f"TN\n{cm[0,0]:,}\n({cm[0,0]/tot*100:.1f}%)", f"FP\n{cm[0,1]:,}\n({cm[0,1]/tot*100:.1f}%)"],
        [f"FN\n{cm[1,0]:,}\n({cm[1,0]/tot*100:.1f}%)", f"TP\n{cm[1,1]:,}\n({cm[1,1]/tot*100:.1f}%)"]
    ])
    cmap = "Blues" if row["Model"] == "BiLSTM" else "OrRd"
    sns.heatmap(
        cm, annot=annot, fmt="", cmap=cmap, cbar=False, ax=ax,
        xticklabels=["Pred: Legitimate (0)", "Pred: Phishing (1)"],
        yticklabels=["Actual: Legitimate (0)", "Actual: Phishing (1)"],
        annot_kws={"size": 10, "weight": "bold"}, linewidths=1, linecolor="black"
    )
    ax.set_title(f"{row['Phase']}\n({row['Model']})", fontsize=10.5, fontweight="bold", pad=8)

plt.suptitle("Confusion Matrix Breakdown Across All Three Evaluation Phases", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("thesis_charts/fig2_confusion_matrices_grid.png", dpi=300, bbox_inches="tight")
plt.close()

# ==============================================================================
# FIGURE 3: EXACT EMPIRICAL ROC-AUC CURVES ACROSS ALL 3 PHASES
# ==============================================================================
print("Generating Figure 3: Exact Empirical ROC Curves...")
ceas_df = pd.read_csv("CEAS_08_cleaned.csv")
enron_df = pd.read_csv("En_Enron_cleaned.csv")
trec_df = pd.read_csv("TREC_cleaned.csv")
ephish_df = pd.read_csv("ephish_blindtest_2000.csv")

legacy_df = pd.concat([ceas_df, enron_df, trec_df], ignore_index=True)
legacy_df["body"] = legacy_df["body"].fillna("").astype(str)
legacy_df = legacy_df[legacy_df["body"].str.strip() != ""].reset_index(drop=True)

_, X_leg_test, _, y_leg_test = train_test_split(
    legacy_df["body"].values, legacy_df["label"].values.astype(int),
    test_size=0.20, stratify=legacy_df["label"].values, random_state=SEED
)

ephish_df["body"] = ephish_df["body"].fillna("").astype(str)
_, X_mod_blind, _, y_mod_blind = train_test_split(
    ephish_df["body"].values, ephish_df["label"].values.astype(int),
    test_size=0.50, stratify=ephish_df["label"].values, random_state=SEED
)

low_ind_mask = np.array([len(set(str(t).lower().split()).intersection(top50_set)) == 0 for t in X_mod_blind])

eval_sets = [
    ("Phase 1: Classic 20% Test Set", X_leg_test, y_leg_test),
    ("Phase 2: Modern Blind Test (1,000)", X_mod_blind, y_mod_blind),
    ("Phase 3: Low-Indicator Subset (249)", X_mod_blind[low_ind_mask], y_mod_blind[low_ind_mask])
]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

for idx, (title, X_raw, y_true) in enumerate(eval_sets):
    ax = axes[idx]
    X_rf = tfidf_vectorizer.transform(X_raw)
    X_seq = pad_sequences(lstm_tokenizer.texts_to_sequences(X_raw), maxlen=200, padding="post", truncating="post")

    rf_prob = rf_model.predict_proba(X_rf)[:, 1]
    lstm_prob = bilstm_model.predict(X_seq, batch_size=256, verbose=0).ravel()
    hyb_prob = (rf_prob + lstm_prob) / 2.0

    fpr_rf, tpr_rf, _ = roc_curve(y_true, rf_prob)
    fpr_lstm, tpr_lstm, _ = roc_curve(y_true, lstm_prob)
    fpr_hyb, tpr_hyb, _ = roc_curve(y_true, hyb_prob)

    ax.plot(fpr_rf, tpr_rf, color="#ef4444", lw=2.2, label=f"Random Forest (AUC = {roc_auc_score(y_true, rf_prob):.4f})")
    ax.plot(fpr_lstm, tpr_lstm, color="#2563eb", lw=2.2, ls="--", label=f"Bi-LSTM (AUC = {roc_auc_score(y_true, lstm_prob):.4f})")
    ax.plot(fpr_hyb, tpr_hyb, color="#10b981", lw=2.0, ls="-.", label=f"Hybrid Ensemble (AUC = {roc_auc_score(y_true, hyb_prob):.4f})")
    ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1.2, label="Random Baseline (0.5000)")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10, fontweight="bold")
    if idx == 0:
        ax.set_ylabel("True Positive Rate (Recall)", fontsize=10, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    ax.grid(alpha=0.3, linestyle="--")

plt.suptitle("Empirical ROC Curves & AUC Comparison Across Evaluation Phases", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("thesis_charts/fig3_roc_auc_all_phases.png", dpi=300, bbox_inches="tight")
plt.close()

# ==============================================================================
# FIGURE 4: TOP 20 RANDOM FOREST GINI FEATURE IMPORTANCES
# ==============================================================================
print("Generating Figure 4: Top 20 Random Forest Feature Importances...")
feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
importances = rf_model.feature_importances_
top20_idx = np.argsort(importances)[::-1][:20]

fig, ax = plt.subplots(figsize=(9, 6))
y_words = feature_names[top20_idx][::-1]
x_vals = importances[top20_idx][::-1]

bars = ax.barh(y_words, x_vals, color="#0284c7", edgecolor="black", height=0.7)
for b in bars:
    w = b.get_width()
    ax.annotate(
        f"{w:.4f}",
        xy=(w, b.get_y() + b.get_height() / 2),
        xytext=(4, 0), textcoords="offset points",
        ha="left", va="center", fontsize=8.5, fontweight="bold"
    )

ax.set_xlim(0, max(x_vals) * 1.18)
ax.set_title("Top 20 Random Forest TF-IDF Feature Importances (Gini Impurity Reduction)", fontsize=12, fontweight="bold", pad=10)
ax.set_xlabel("Relative Gini Feature Importance", fontsize=10, fontweight="bold")
ax.set_ylabel("TF-IDF Vocabulary Term", fontsize=10, fontweight="bold")
ax.grid(axis="x", alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig("thesis_charts/fig4_top20_rf_gini_importances.png", dpi=300, bbox_inches="tight")
plt.close()

# ==============================================================================
# FIGURE 5: 2021-2025 CONCEPT DRIFT RECALL & F1 DEGRADATION CURVES
# ==============================================================================
if os.path.exists("exported_models/concept_drift_2021_2025_metrics.csv"):
    print("Generating Figure 5: 2021-2025 Concept Drift Recall & F1 Curves...")
    drift_df = pd.read_csv("exported_models/concept_drift_2021_2025_metrics.csv")
    years = sorted(drift_df["Year"].unique())
    rf_d = drift_df[drift_df["Model"] == "Random Forest"]
    lstm_d = drift_df[drift_df["Model"] == "BiLSTM"]
    hyb_d = drift_df[drift_df["Model"] == "Hybrid Ensemble"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    # Left panel: Recall Curve + Vocabulary Overlap
    ax1.plot(years, rf_d["Recall"] * 100, marker="o", lw=2.2, color="#ef4444", label="Random Forest Recall (%)")
    ax1.plot(years, lstm_d["Recall"] * 100, marker="s", lw=2.2, color="#2563eb", label="Bi-LSTM Recall (%)")
    ax1.plot(years, hyb_d["Recall"] * 100, marker="^", lw=2.0, ls="--", color="#10b981", label="Hybrid Ensemble Recall (%)")
    ax1.set_title("A. Phishing Recall vs. Lexical Drift (2021–2025)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Chronological Threat Partition", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Recall (%)", fontsize=10, fontweight="bold")
    ax1.set_xticks(years)
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.legend(loc="lower left", fontsize=8.5)

    # Right panel: F1-Score + Jaccard Similarity
    ax2.plot(years, rf_d["F1_Score"] * 100, marker="o", lw=2.2, color="#ef4444", label="Random Forest F1-Score (%)")
    ax2.plot(years, lstm_d["F1_Score"] * 100, marker="s", lw=2.2, color="#2563eb", label="Bi-LSTM F1-Score (%)")
    ax2.plot(years, hyb_d["F1_Score"] * 100, marker="^", lw=2.0, ls="--", color="#10b981", label="Hybrid Ensemble F1-Score (%)")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(years, rf_d["Jaccard_Sim"] * 100, marker="d", lw=1.8, ls=":", color="#475569", label="Jaccard Similarity (%)")
    ax2_twin.set_ylabel("Jaccard Similarity (%)", fontsize=10, color="#475569", fontweight="bold")

    ax2.set_title("B. F1-Score Stability vs. Jaccard Similarity (2021–2025)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Chronological Threat Partition", fontsize=10, fontweight="bold")
    ax2.set_ylabel("F1-Score (%)", fontsize=10, fontweight="bold")
    ax2.set_xticks(years)
    ax2.grid(alpha=0.3, linestyle="--")

    l1, lb1 = ax2.get_legend_handles_labels()
    l2, lb2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(l1 + l2, lb1 + lb2, loc="lower left", fontsize=8.5)

    plt.suptitle("Objective 6: Sequential Concept Drift Simulation (2021–2025)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("thesis_charts/fig5_concept_drift_2021_2025_dual.png", dpi=300, bbox_inches="tight")
    plt.close()

print("\nAll publication-ready thesis charts saved in ./thesis_charts/!")