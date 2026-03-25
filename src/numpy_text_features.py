import re
import math
from collections import Counter

import numpy as np


def tokenize_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return [tok for tok in text.split() if tok]


def build_vocab_and_idf(texts, max_features=5000, min_df=2):
    tokenized_docs = [tokenize_text(t) for t in texts]
    doc_freq = Counter()

    for tokens in tokenized_docs:
        doc_freq.update(set(tokens))

    filtered = [(tok, df) for tok, df in doc_freq.items() if df >= min_df]
    filtered.sort(key=lambda x: x[1], reverse=True)
    filtered = filtered[:max_features]

    vocab = {tok: i for i, (tok, _) in enumerate(filtered)}
    n_docs = len(tokenized_docs)
    idf = np.zeros(len(vocab), dtype=np.float32)

    for tok, idx in vocab.items():
        df = doc_freq[tok]
        idf[idx] = math.log((1 + n_docs) / (1 + df)) + 1.0

    return vocab, idf


def transform_tfidf(texts, vocab, idf):
    n_features = len(vocab)
    X = np.zeros((len(texts), n_features), dtype=np.float32)

    for i, text in enumerate(texts):
        tokens = tokenize_text(text)
        if not tokens:
            continue
        counts = Counter(tok for tok in tokens if tok in vocab)
        doc_len = len(tokens)
        if doc_len == 0:
            continue

        for tok, cnt in counts.items():
            j = vocab[tok]
            tf = cnt / doc_len
            X[i, j] = tf * idf[j]

    return X


def stratified_train_val_test_split(X, y, train_size=0.70, val_size=0.15, seed=42):
    if train_size <= 0 or val_size <= 0 or (train_size + val_size) >= 1:
        raise ValueError("Use valid train/val sizes where train_size + val_size < 1.")

    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    classes = np.unique(y)

    train_idx, val_idx, test_idx = [], [], []
    for c in classes:
        c_idx = np.where(y == c)[0]
        rng.shuffle(c_idx)

        n = len(c_idx)
        n_train = int(round(n * train_size))
        n_val = int(round(n * val_size))

        # Guarantee at least one sample in each split when class is large enough.
        if n >= 3:
            n_train = min(max(n_train, 1), n - 2)
            n_val = min(max(n_val, 1), n - n_train - 1)

        split_train = c_idx[:n_train]
        split_val = c_idx[n_train:n_train + n_val]
        split_test = c_idx[n_train + n_val:]

        train_idx.extend(split_train.tolist())
        val_idx.extend(split_val.tolist())
        test_idx.extend(split_test.tolist())

    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx = np.array(val_idx, dtype=np.int64)
    test_idx = np.array(test_idx, dtype=np.int64)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return (
        X[train_idx], X[val_idx], X[test_idx],
        y[train_idx], y[val_idx], y[test_idx],
    )
