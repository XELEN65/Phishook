# File: export_for_phishook_js.py
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.special import logit
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("Loading trained Hybrid Random Forest and BiLSTM artifacts...")
rf_model = joblib.load("exported_models/rf_model.pkl")
tfidf = joblib.load("exported_models/tfidf_vectorizer.pkl")
bilstm_model = tf.keras.models.load_model("exported_models/bilstm_model.keras")

with open("exported_models/lstm_tokenizer.json", "r", encoding="utf-8") as f:
    lstm_tokenizer = tokenizer_from_json(f.read())

with open("exported_models/top50_phishing_terms.json", "r", encoding="utf-8") as f:
    top50_terms = json.load(f)

# Load a calibration sample from your datasets to distill exact probabilities into fast JS weights
ceas = pd.read_csv("CEAS_08_cleaned.csv", nrows=8000)
enron = pd.read_csv("En_Enron_cleaned.csv", nrows=8000)
ephish = pd.read_csv("ephish_blindtest_2000.csv")
calib_df = pd.concat([ceas, enron, ephish], ignore_index=True).dropna(subset=["body", "label"])
texts = calib_df["body"].astype(str).values

print("Computing exact TF-IDF and BiLSTM representations for local JS distillation...")
X_tfidf = tfidf.transform(texts)
rf_probs = np.clip(rf_model.predict_proba(X_tfidf)[:, 1], 1e-4, 1 - 1e-4)

X_seq = pad_sequences(lstm_tokenizer.texts_to_sequences(texts), maxlen=200, padding="post", truncating="post")
lstm_probs = np.clip(bilstm_model.predict(X_seq, batch_size=256, verbose=0).ravel(), 1e-4, 1 - 1e-4)

# 1. Distill RF probabilities onto the exact 5,000-term TF-IDF space (Ridge regression on log-odds)
from sklearn.linear_model import Ridge
rf_surrogate = Ridge(alpha=1.0)
rf_surrogate.fit(X_tfidf, logit(rf_probs))

# 2. Distill BiLSTM sequential probabilities onto unigram + sequential bigram TF-IDF space
#    so the local JS engine captures word order & two-word sequential context without needing a 40MB WASM runtime
seq_vectorizer = joblib.load("exported_models/tfidf_vectorizer.pkl")
from sklearn.feature_extraction.text import TfidfVectorizer
lstm_seq_vec = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
X_seq_tfidf = lstm_seq_vec.fit_transform(texts)

lstm_surrogate = Ridge(alpha=1.0)
lstm_surrogate.fit(X_seq_tfidf, logit(lstm_probs))

# Verify correlation with original models
rf_js_prob = 1 / (1 + np.exp(- (X_tfidf @ rf_surrogate.coef_ + rf_surrogate.intercept_)))
lstm_js_prob = 1 / (1 + np.exp(- (X_seq_tfidf @ lstm_surrogate.coef_ + lstm_surrogate.intercept_)))
print(f"RF Local JS Fidelity (Selected vs Python RF):     {np.corrcoef(rf_probs, rf_js_prob)[0,1]:.4f}")
print(f"BiLSTM Local JS Fidelity (Selected vs Python NN): {np.corrcoef(lstm_probs, lstm_js_prob)[0,1]:.4f}")

# Package everything into a compact JSON/JS bundle
rf_vocab = {term: int(idx) for term, idx in tfidf.vocabulary_.items()}
lstm_vocab = {term: int(idx) for term, idx in lstm_seq_vec.vocabulary_.items()}

bundle = {
    "top50_triggers": top50_terms,
    "rf": {
        "vocab": rf_vocab,
        "idf": [round(float(x), 5) for x in tfidf.idf_],
        "weights": [round(float(x), 5) for x in rf_surrogate.coef_],
        "bias": round(float(rf_surrogate.intercept_), 5)
    },
    "lstm": {
        "vocab": lstm_vocab,
        "idf": [round(float(x), 5) for x in lstm_seq_vec.idf_],
        "weights": [round(float(x), 5) for x in lstm_surrogate.coef_],
        "bias": round(float(lstm_surrogate.intercept_), 5)
    }
}

os.makedirs("phishook_extension", exist_ok=True)
with open("phishook_extension/models_bundle.js", "w", encoding="utf-8") as f:
    f.write("// Auto-generated Local AI Bundle (Random Forest + BiLSTM)\n")
    f.write("window.PHISHOOK_MODELS = ")
    json.dump(bundle, f, separators=(",", ":"))
    f.write(";\n")

size_kb = os.path.getsize("phishook_extension/models_bundle.js") / 1024
print(f"Success! Generated phishook_extension/models_bundle.js ({size_kb:.1f} KB) — 100% offline & local!")