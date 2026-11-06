# File: evaluate_ephish_2000.py
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# 1. Load the new 2,000-email unique blind test set
blind_df = pd.read_csv("ephish_blindtest_2000.csv")
blind_df["body"] = blind_df["body"].fillna("").astype(str)
X_blind_raw = blind_df["body"].values
y_blind = blind_df["label"].values.astype(int)

print(f"Loaded ephish_blindtest_2000.csv: {len(blind_df):,} unique emails "
      f"(Legitimate=0: {(y_blind == 0).sum():,}, Phishing=1: {(y_blind == 1).sum():,})")

# 2. Load trained models & vocabulary artifacts from Stage 6
rf_model = joblib.load("exported_models/rf_model.pkl")
tfidf_vectorizer = joblib.load("exported_models/tfidf_vectorizer.pkl")
bilstm_model = tf.keras.models.load_model("exported_models/bilstm_model.keras")

with open("exported_models/lstm_tokenizer.json", "r", encoding="utf-8") as f:
    lstm_tokenizer = tokenizer_from_json(f.read())

with open("exported_models/top50_phishing_terms.json", "r", encoding="utf-8") as f:
    top50_set = set(json.load(f))

# 3. Transform the 2,000 unseen emails for both branches
X_blind_tfidf = tfidf_vectorizer.transform(X_blind_raw)
X_blind_seq = pad_sequences(
    lstm_tokenizer.texts_to_sequences(X_blind_raw),
    maxlen=200, padding="post", truncating="post"
)

# 4. Predict probabilities without retraining
rf_prob_blind = rf_model.predict_proba(X_blind_tfidf)[:, 1]
lstm_prob_blind = bilstm_model.predict(X_blind_seq, batch_size=256, verbose=0).ravel()

# 5. Isolate Low-Indicator Phishing Subset (Objective 5)
def is_low_indicator_email(text):
    tokens = set(str(text).lower().split())
    has_url = any(u in tokens for u in ["http", "https", "www", "href"])
    has_top50_word = len(tokens.intersection(top50_set)) > 0
    return (not has_url) and (not has_top50_word)

low_ind_flags = np.array([is_low_indicator_email(t) for t in X_blind_raw])
low_ind_phish_mask = (y_blind == 1) & low_ind_flags
# Pair low-indicator phishing emails with low-indicator legitimate emails for balanced evaluation
low_ind_subset_mask = low_ind_flags

print(f"Strict Low-Indicator Subset: {low_ind_subset_mask.sum()} total emails "
      f"(Legitimate=0: {((y_blind == 0) & low_ind_flags).sum()}, "
      f"Phishing=1: {low_ind_phish_mask.sum()})")

def compute_metrics(y_true, y_prob, model_name, phase_name):
    y_pred = (y_prob >= 0.50).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Phase": phase_name,
        "Model": model_name,
        "Total_Samples": int(len(y_true)),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1_Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC_AUC": round(roc_auc_score(y_true, y_prob), 4) if len(np.unique(y_true)) > 1 else np.nan,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    }

# Keep Phase 1 (17,257 internal test set) and update Phase 2 & Phase 3
old_metrics = pd.read_csv("exported_models/stage5_evaluation_metrics.csv")
phase1_df = old_metrics[old_metrics["Phase"].str.startswith("Phase 1")].copy()

new_rows = pd.DataFrame([
    compute_metrics(y_blind, rf_prob_blind, "Random Forest", "Phase 2: External Blind Test (2,000)"),
    compute_metrics(y_blind, lstm_prob_blind, "BiLSTM", "Phase 2: External Blind Test (2,000)"),
    compute_metrics(y_blind[low_ind_subset_mask], rf_prob_blind[low_ind_subset_mask],
                    "Random Forest", "Phase 3: Low-Indicator Subset"),
    compute_metrics(y_blind[low_ind_subset_mask], lstm_prob_blind[low_ind_subset_mask],
                    "BiLSTM", "Phase 3: Low-Indicator Subset")
])

updated_df = pd.concat([phase1_df, new_rows], ignore_index=True)
print("\n================ UPDATED THESIS EVALUATION METRICS (ePhish 2,000) ================")
print(updated_df.to_string(index=False))
updated_df.to_csv("exported_models/stage5_evaluation_metrics.csv", index=False)