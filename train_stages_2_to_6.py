import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, Bidirectional, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Set random seeds for reproducible thesis benchmarks
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("exported_models", exist_ok=True)

# ==============================================================================
# STAGE 2: DATA INGESTION & LEAKAGE-FREE 80/20 SPLIT (Objectives 1 & 2)
# ==============================================================================
print("\n--- STAGE 2: Loading Datasets & Performing Stratified 80/20 Split ---")

ceas_df = pd.read_csv("CEAS_08_cleaned.csv")
enron_df = pd.read_csv("En_Enron_cleaned.csv")
trec_df = pd.read_csv("TREC_cleaned.csv")
blind_df = pd.read_csv("cleaned_emails-BlindTest.csv")

# Merge primary training/testing corpus (86,282 total rows across the 3 files)
corpus_df = pd.concat([ceas_df, enron_df, trec_df], ignore_index=True)
corpus_df["body"] = corpus_df["body"].fillna("").astype(str)
corpus_df = corpus_df[corpus_df["body"].str.strip() != ""].reset_index(drop=True)

# Clean external blind test set (kept completely isolated from training)
blind_df["body"] = blind_df["body"].fillna("").astype(str)
X_blind_raw = blind_df["body"].values
y_blind = blind_df["label"].values.astype(int)

print(f"Combined Primary Corpus: {len(corpus_df):,} emails")
print(f"  -> Legitimate (0): {(corpus_df['label'] == 0).sum():,}")
print(f"  -> Phishing   (1): {(corpus_df['label'] == 1).sum():,}")
print(f"External Blind Test Set: {len(blind_df):,} emails "
      f"(0: {(y_blind == 0).sum()}, 1: {(y_blind == 1).sum()})")

# Perform stratified 80/20 split BEFORE any oversampling to prevent data leakage
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    corpus_df["body"].values,
    corpus_df["label"].values.astype(int),
    test_size=0.20,
    stratify=corpus_df["label"].values,
    random_state=SEED
)

print(f"80% Training Partition: {len(X_train_raw):,} emails "
      f"(0: {(y_train == 0).sum():,}, 1: {(y_train == 1).sum():,})")
print(f"20% Held-Out Test Set:  {len(X_test_raw):,} emails "
      f"(0: {(y_test == 0).sum():,}, 1: {(y_test == 1).sum():,})")

# ==============================================================================
# STAGE 3: DUAL-BRANCH FEATURE EXTRACTION & SMOTE BALANCING (Objectives 1, 2, 3, 4)
# ==============================================================================
print("\n--- STAGE 3A: TF-IDF Vectorization (5,000 Terms) & SMOTE for Random Forest ---")

# Fit TF-IDF strictly on the 80% training partition
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_raw)
X_test_tfidf = tfidf_vectorizer.transform(X_test_raw)
X_blind_tfidf = tfidf_vectorizer.transform(X_blind_raw)

# Extract Top 50 Highest-Weighted Phishing TF-IDF Terms (Objective 1.4)
feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
phish_train_mask = (y_train == 1)
mean_phish_tfidf = np.asarray(X_train_tfidf[phish_train_mask].mean(axis=0)).ravel()
top50_indices = np.argsort(mean_phish_tfidf)[::-1][:50]
top50_phishing_terms = feature_names[top50_indices].tolist()

print("Top 15 Phishing TF-IDF Terms:", top50_phishing_terms[:15])

# Apply SMOTE strictly inside the 80% TF-IDF training partition
smote_rf = SMOTE(random_state=SEED)
X_train_rf_smote, y_train_rf_smote = smote_rf.fit_resample(X_train_tfidf, y_train)
print(f"RF Training Set after SMOTE: {X_train_rf_smote.shape[0]:,} samples "
      f"(0: {(y_train_rf_smote == 0).sum():,}, 1: {(y_train_rf_smote == 1).sum():,})")

print("\n--- STAGE 3B: Sequence Tokenization (10,000 Vocab, 200 Length) & SMOTE for BiLSTM ---")
VOCAB_SIZE = 10000
MAX_SEQ_LEN = 200

# Fit Tokenizer strictly on the 80% training partition
lstm_tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
lstm_tokenizer.fit_on_texts(X_train_raw)

X_train_seq = pad_sequences(
    lstm_tokenizer.texts_to_sequences(X_train_raw),
    maxlen=MAX_SEQ_LEN, padding="post", truncating="post"
)
X_test_seq = pad_sequences(
    lstm_tokenizer.texts_to_sequences(X_test_raw),
    maxlen=MAX_SEQ_LEN, padding="post", truncating="post"
)
X_blind_seq = pad_sequences(
    lstm_tokenizer.texts_to_sequences(X_blind_raw),
    maxlen=MAX_SEQ_LEN, padding="post", truncating="post"
)

# Apply SMOTE strictly inside the 80% integer sequence training partition
smote_lstm = SMOTE(random_state=SEED)
X_train_seq_smote, y_train_seq_smote = smote_lstm.fit_resample(X_train_seq, y_train)
# Ensure interpolated values remain valid integer token indices [0, VOCAB_SIZE - 1]
X_train_seq_smote = np.clip(np.round(X_train_seq_smote), 0, VOCAB_SIZE - 1).astype(np.int32)

print(f"BiLSTM Training Set after SMOTE: {X_train_seq_smote.shape[0]:,} samples "
      f"(0: {(y_train_seq_smote == 0).sum():,}, 1: {(y_train_seq_smote == 1).sum():,})")

# ==============================================================================
# STAGE 4: MODEL TRAINING (Objectives 3 & 4)
# ==============================================================================
print("\n--- STAGE 4A: Training Random Forest (100 Estimators, Unlimited Depth) ---")
t0 = time.time()
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=SEED,
    n_jobs=-1
)
rf_model.fit(X_train_rf_smote, y_train_rf_smote)
rf_train_seconds = time.time() - t0
print(f"Random Forest training completed in {rf_train_seconds:.2f} seconds.")

print("\n--- STAGE 4B: Training Bidirectional LSTM (64 Units, 30% Spatial Dropout) ---")
bilstm_model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=128, input_length=MAX_SEQ_LEN),
    SpatialDropout1D(0.30),
    Bidirectional(LSTM(64)),
    Dense(1, activation="sigmoid")
])

bilstm_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

t1 = time.time()
history = bilstm_model.fit(
    X_train_seq_smote,
    y_train_seq_smote,
    validation_split=0.10,
    epochs=10,
    batch_size=128,
    callbacks=[early_stopping],
    verbose=1
)
lstm_train_seconds = time.time() - t1
print(f"Bidirectional LSTM training completed in {lstm_train_seconds:.2f} seconds.")

# ==============================================================================
# STAGE 5: MULTI-PHASE EVALUATION & METRICS CALCULATION (Objectives 3, 4, 5, 7)
# ==============================================================================
print("\n--- STAGE 5: Computing Accuracy, Precision, Recall, F1-Score, and ROC-AUC ---")

# Programmatically isolate Low-Indicator Phishing Subset from Blind Test (Objective 5)
# Criteria: No URLs/links AND zero overlap with Top 50 Phishing TF-IDF trigger words
top50_set = set(top50_phishing_terms)

def is_low_indicator_email(text):
    tokens = set(str(text).lower().split())
    has_url = any(u in tokens for u in ["http", "https", "www", "href"])
    has_top50_word = len(tokens.intersection(top50_set)) > 0
    return (not has_url) and (not has_top50_word)

low_ind_flags = np.array([is_low_indicator_email(t) for t in X_blind_raw])
# Evaluate Low-Indicator Phishing emails alongside Legitimate emails so Precision/AUC are defined
low_ind_eval_mask = (y_blind == 0) | ((y_blind == 1) & low_ind_flags)
low_ind_phish_count = ((y_blind == 1) & low_ind_flags).sum()
print(f"Isolated {low_ind_phish_count} strict Low-Indicator Phishing emails out of {(y_blind == 1).sum()} blind phishing emails.")

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
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }

# Generate probabilities on all unseen test sets without retraining
rf_prob_test = rf_model.predict_proba(X_test_tfidf)[:, 1]
lstm_prob_test = bilstm_model.predict(X_test_seq, batch_size=256, verbose=0).ravel()

rf_prob_blind = rf_model.predict_proba(X_blind_tfidf)[:, 1]
lstm_prob_blind = bilstm_model.predict(X_blind_seq, batch_size=64, verbose=0).ravel()

metrics_list = [
    # Phase 1: 20% Held-Out Standard Test Set
    compute_metrics(y_test, rf_prob_test, "Random Forest", "Phase 1: 20% Held-Out Test Set"),
    compute_metrics(y_test, lstm_prob_test, "BiLSTM", "Phase 1: 20% Held-Out Test Set"),
    # Phase 2: Full External Blind Test Set
    compute_metrics(y_blind, rf_prob_blind, "Random Forest", "Phase 2: External Blind Test Set"),
    compute_metrics(y_blind, lstm_prob_blind, "BiLSTM", "Phase 2: External Blind Test Set"),
]

# Phase 3: Low-Indicator Evasion Subset
if low_ind_phish_count > 0:
    metrics_list.append(
        compute_metrics(y_blind[low_ind_eval_mask], rf_prob_blind[low_ind_eval_mask],
                        "Random Forest", "Phase 3: Low-Indicator Subset")
    )
    metrics_list.append(
        compute_metrics(y_blind[low_ind_eval_mask], lstm_prob_blind[low_ind_eval_mask],
                        "BiLSTM", "Phase 3: Low-Indicator Subset")
    )

metrics_df = pd.DataFrame(metrics_list)
print("\n================ FINAL THESIS EVALUATION METRICS ================")
print(metrics_df.to_string(index=False))

# Plot and save ROC Curve comparison for Chapter IV
plt.figure(figsize=(8, 6))
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob_test)
fpr_lstm, tpr_lstm, _ = roc_curve(y_test, lstm_prob_test)
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_score(y_test, rf_prob_test):.4f})", linewidth=2)
plt.plot(fpr_lstm, tpr_lstm, label=f"BiLSTM (AUC = {roc_auc_score(y_test, lstm_prob_test):.4f})", linewidth=2, linestyle="--")
plt.plot([0, 1], [0, 1], color="gray", linestyle=":")
plt.title("ROC Curve Comparison — 20% Held-Out Test Set")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Recall)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("exported_models/roc_curve_comparison.png", dpi=300)
plt.close()

# ==============================================================================
# STAGE 6: SERIALIZE & EXPORT ARTIFACTS FOR STAGE 7 EXTENSION
# ==============================================================================
print("\n--- STAGE 6: Saving Trained Models & Vocabulary Artifacts ---")

# 1. Save Random Forest & TF-IDF Vectorizer
joblib.dump(rf_model, "exported_models/rf_model.pkl")
joblib.dump(tfidf_vectorizer, "exported_models/tfidf_vectorizer.pkl")

# 2. Save BiLSTM Model & Keras Tokenizer
bilstm_model.save("exported_models/bilstm_model.keras")
with open("exported_models/lstm_tokenizer.json", "w", encoding="utf-8") as f:
    f.write(lstm_tokenizer.to_json())

# 3. Save Top 50 Phishing Trigger Words & Evaluation Tables
with open("exported_models/top50_phishing_terms.json", "w", encoding="utf-8") as f:
    json.dump(top50_phishing_terms, f, indent=2)

rf_feature_table = pd.DataFrame({
    "term": feature_names,
    "rf_gini_importance": rf_model.feature_importances_,
    "mean_phishing_tfidf": mean_phish_tfidf
}).sort_values(by="rf_gini_importance", ascending=False)
rf_feature_table.to_csv("exported_models/rf_feature_importances.csv", index=False)

metrics_df.to_csv("exported_models/stage5_evaluation_metrics.csv", index=False)

print("All Stage 1–6 artifacts successfully saved in ./exported_models/:")
for fname in sorted(os.listdir("exported_models")):
    print(f"  - exported_models/{fname}")