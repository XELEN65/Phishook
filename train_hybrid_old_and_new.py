# File: train_hybrid_old_and_new.py
import os
import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, Bidirectional, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs("exported_models", exist_ok=True)

print("\n--- 1. Loading Classic & Modern Datasets ---")
ceas_df = pd.read_csv("CEAS_08_cleaned.csv")
enron_df = pd.read_csv("En_Enron_cleaned.csv")
trec_df = pd.read_csv("TREC_cleaned.csv")
ephish_df = pd.read_csv("ephish_blindtest_2000.csv")
twente_df = pd.read_csv("cleaned_emails-BlindTest.csv")

# Combine 86,282 classic emails and split 80/20 BEFORE any oversampling
legacy_df = pd.concat([ceas_df, enron_df, trec_df], ignore_index=True)
legacy_df["body"] = legacy_df["body"].fillna("").astype(str)
legacy_df = legacy_df[legacy_df["body"].str.strip() != ""].reset_index(drop=True)

X_leg_train, X_leg_test, y_leg_train, y_leg_test = train_test_split(
    legacy_df["body"].values,
    legacy_df["label"].values.astype(int),
    test_size=0.20,
    stratify=legacy_df["label"].values,
    random_state=SEED
)

# Split 2,000-row modern ePhish dataset 50/50:
# - 1,000 emails enter the training partition (weighted 5x so modern patterns aren't drowned out by 69k old emails)
# - 1,000 emails remain 100% UNTOUCHED as the Modern Blind Test Set
ephish_df["body"] = ephish_df["body"].fillna("").astype(str)
X_mod_train, X_mod_blind, y_mod_train, y_mod_blind = train_test_split(
    ephish_df["body"].values,
    ephish_df["label"].values.astype(int),
    test_size=0.50,
    stratify=ephish_df["label"].values,
    random_state=SEED
)

# Save the 1,000 untouched modern blind test emails for reference
pd.DataFrame({"body": X_mod_blind, "label": y_mod_blind}).to_csv(
    "modern_blindtest_1000_untouched.csv", index=False
)

# Combine 80% legacy train + 5x weighted modern train (strictly inside training partition!)
X_train_raw = np.concatenate([X_leg_train, np.tile(X_mod_train, 5)])
y_train = np.concatenate([y_leg_train, np.tile(y_mod_train, 5)])

# Shuffle training data so Keras validation_split=0.10 gets a uniform mix of old, new, and SMOTE samples
shuffle_idx = np.random.permutation(len(X_train_raw))
X_train_raw = X_train_raw[shuffle_idx]
y_train = y_train[shuffle_idx]

print(f"Hybrid Training Set (Pre-SMOTE): {len(X_train_raw):,} emails")
print(f"Held-Out Classic Test Set (20%): {len(X_leg_test):,} emails")
print(f"Untouched Modern Blind Test Set: {len(X_mod_blind):,} emails")

# --- 2. TF-IDF (5,000 Terms) + Differential Top-50 Triggers + SMOTE ---
print("\n--- 2. Extracting TF-IDF & Differential Top-50 Phishing Triggers ---")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_raw)
X_test_leg_tfidf = tfidf_vectorizer.transform(X_leg_test)
X_blind_mod_tfidf = tfidf_vectorizer.transform(X_mod_blind)

# Rank Top 50 Phishing Terms by Differential TF-IDF (Phishing Mean minus Legitimate Mean)
# This prevents generic words like 'please', 'time', 'want' from being flagged as phishing triggers!
feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
mean_phish = np.asarray(X_train_tfidf[y_train == 1].mean(axis=0)).ravel()
mean_legit = np.asarray(X_train_tfidf[y_train == 0].mean(axis=0)).ravel()
diff_score = mean_phish - mean_legit
top50_indices = np.argsort(diff_score)[::-1][:50]
top50_phishing_terms = feature_names[top50_indices].tolist()

print("Clean Top 15 Differential Phishing Triggers:", top50_phishing_terms[:15])

smote_rf = SMOTE(random_state=SEED)
X_train_rf_smote, y_train_rf_smote = smote_rf.fit_resample(X_train_tfidf, y_train)

print("Training Hybrid Random Forest (100 trees)...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=SEED, n_jobs=-1)
rf_model.fit(X_train_rf_smote, y_train_rf_smote)

# --- 3. Sequence Tokenization (10,000 Vocab, 200 Length) + BiLSTM ---
print("\n--- 3. Tokenizing & Training Hybrid Bidirectional LSTM ---")
VOCAB_SIZE = 10000
MAX_SEQ_LEN = 200

lstm_tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
lstm_tokenizer.fit_on_texts(X_train_raw)

X_train_seq = pad_sequences(lstm_tokenizer.texts_to_sequences(X_train_raw), maxlen=MAX_SEQ_LEN, padding="post", truncating="post")
X_test_leg_seq = pad_sequences(lstm_tokenizer.texts_to_sequences(X_leg_test), maxlen=MAX_SEQ_LEN, padding="post", truncating="post")
X_blind_mod_seq = pad_sequences(lstm_tokenizer.texts_to_sequences(X_mod_blind), maxlen=MAX_SEQ_LEN, padding="post", truncating="post")

smote_lstm = SMOTE(random_state=SEED)
X_train_seq_smote, y_train_seq_smote = smote_lstm.fit_resample(X_train_seq, y_train)
X_train_seq_smote = np.clip(np.round(X_train_seq_smote), 0, VOCAB_SIZE - 1).astype(np.int32)

# Shuffle post-SMOTE sequence array so validation_split=0.10 is balanced
perm = np.random.permutation(len(X_train_seq_smote))
X_train_seq_smote = X_train_seq_smote[perm]
y_train_seq_smote = y_train_seq_smote[perm]

bilstm_model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=128),
    SpatialDropout1D(0.30),
    Bidirectional(LSTM(64)),
    Dense(1, activation="sigmoid")
])
bilstm_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

bilstm_model.fit(
    X_train_seq_smote, y_train_seq_smote,
    validation_split=0.10, epochs=6, batch_size=128,
    callbacks=[early_stop], verbose=1
)

# --- 4. Evaluate Across Classic, Modern, and Low-Indicator Benchmarks ---
top50_set = set(top50_phishing_terms)
low_ind_mask = np.array([len(set(str(t).lower().split()).intersection(top50_set)) == 0 for t in X_mod_blind])

def eval_row(y_true, y_prob, model_name, phase_name):
    y_pred = (y_prob >= 0.50).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Phase": phase_name, "Model": model_name, "Samples": int(len(y_true)),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1_Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC_AUC": round(roc_auc_score(y_true, y_prob), 4),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    }

rf_p_leg = rf_model.predict_proba(X_test_leg_tfidf)[:, 1]
lstm_p_leg = bilstm_model.predict(X_test_leg_seq, batch_size=256, verbose=0).ravel()

rf_p_mod = rf_model.predict_proba(X_blind_mod_tfidf)[:, 1]
lstm_p_mod = bilstm_model.predict(X_blind_mod_seq, batch_size=256, verbose=0).ravel()
ensemble_p_mod = (rf_p_mod + lstm_p_mod) / 2.0

metrics_df = pd.DataFrame([
    eval_row(y_leg_test, rf_p_leg, "Random Forest", "Phase 1: Classic 20% Test Set"),
    eval_row(y_leg_test, lstm_p_leg, "BiLSTM", "Phase 1: Classic 20% Test Set"),
    eval_row(y_mod_blind, rf_p_mod, "Random Forest", "Phase 2: Modern Blind Test (1,000)"),
    eval_row(y_mod_blind, lstm_p_mod, "BiLSTM", "Phase 2: Modern Blind Test (1,000)"),
    eval_row(y_mod_blind, ensemble_p_mod, "Hybrid Ensemble (RF+BiLSTM)", "Phase 2: Modern Blind Test (1,000)"),
    eval_row(y_mod_blind[low_ind_mask], rf_p_mod[low_ind_mask], "Random Forest", "Phase 3: Low-Indicator Subset"),
    eval_row(y_mod_blind[low_ind_mask], lstm_p_mod[low_ind_mask], "BiLSTM", "Phase 3: Low-Indicator Subset"),
    eval_row(y_mod_blind[low_ind_mask], ensemble_p_mod[low_ind_mask], "Hybrid Ensemble (RF+BiLSTM)", "Phase 3: Low-Indicator Subset")
])

print("\n================ UPGRADED HYBRID BENCHMARK METRICS ================")
print(metrics_df.to_string(index=False))
metrics_df.to_csv("exported_models/stage5_evaluation_metrics.csv", index=False)

# Overwrite exported models with the upgraded hybrid models
joblib.dump(rf_model, "exported_models/rf_model.pkl")
joblib.dump(tfidf_vectorizer, "exported_models/tfidf_vectorizer.pkl")
bilstm_model.save("exported_models/bilstm_model.keras")
with open("exported_models/lstm_tokenizer.json", "w", encoding="utf-8") as f:
    f.write(lstm_tokenizer.to_json())
with open("exported_models/top50_phishing_terms.json", "w", encoding="utf-8") as f:
    json.dump(top50_phishing_terms, f, indent=2)
print("\nSaved upgraded hybrid models to ./exported_models/!")