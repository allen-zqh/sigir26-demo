# -*- coding: utf-8 -*-

"""
Detect semantic/usage change between two corpora using a covariance-spectrum
statistic computed from contextual embeddings (demo, review-stage).

Pipeline (per token, per corpus):
  1) Collect contextual embeddings and L2-normalize them (unit-sphere projection).
  2) (Optional multi-sense) Fit a GMM; choose K via silhouette (heuristic), K ∈ [1, kmax].
  3) For each cluster k, compute sample covariance Σ_k (with a small ridge),
     then compute a spectral-gap statistic:
         ζ_k = λ_max(Σ_k) - λ_min(Σ_k)
  4) Aggregate with cluster proportions π_k:
         ζ̂ = Σ_k π_k ζ_k
  5) Change score:
         C(S, T) = log( ζ̂_T / ζ̂_S )

Interpretation (consistent with this demo statistic):
  - C > 0: larger spectral gap in T (covariance spectrum more spread / more anisotropic)
  - C < 0: smaller spectral gap in T

NOTE:
  This is a lightweight demo scaffold. The full reproducible implementation and
  experiment configs described in the paper will be released after acceptance.

Inputs:
  -source_corpus / -target_corpus: one sentence per line (or batched per util)

Outputs:
  results.txt: token  C(S,T)  freq_S  freq_T
"""

import argparse
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, BertModel

import util  # expected to provide: load_sentences, to_batches, tokenize_and_vectorize
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


# -----------------------------
# Argument parsing
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-source_corpus', default='source.txt')
    parser.add_argument('-target_corpus', default='target.txt')
    parser.add_argument('-m', '--bert_model', default='bert-large-cased',
                        help='PLM used to produce contextual embeddings')
    parser.add_argument('-c', '--cased', action='store_true',
                        help='Preserve case distinctions when tokenizing')
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('-f', '--freq_threshold', default=10, type=int,
                        help='Minimum per-corpus frequency to keep a word')
    parser.add_argument('--kmax', default=5, type=int,
                        help='Upper bound on clusters per word per corpus')
    parser.add_argument('--out', default='results.txt', type=str,
                        help='Output file path')
    return parser.parse_args()


# -----------------------------
# Linear algebra utilities
# -----------------------------
_EPS = 1e-9

def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization."""
    if X.ndim == 1:
        X = X.reshape(1, -1)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < _EPS] = 1.0
    return X / norms

def safe_cov(X: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """
    Compute unbiased sample covariance of row-wise samples with a small
    diagonal ridge to improve numerical stability.
    """
    n, d = X.shape
    if n <= 1:
        return ridge * np.eye(d, dtype=X.dtype)
    C = np.cov(X, rowvar=False, bias=False)
    return C + ridge * np.eye(C.shape[0], dtype=C.dtype)

def eigengap(C: np.ndarray) -> float:
    """Return λ_max(C) - λ_min(C) for a symmetric covariance matrix C."""
    w = np.linalg.eigvalsh(C)  # sorted eigenvalues
    return float(w[-1] - w[0])


# -----------------------------
# Clustering (GMM + silhouette to select K)
# -----------------------------
def cluster_vectors(vecs: np.ndarray, max_clusters: int = 5, random_state: int = 0):
    """
    Fit a GMM with K ∈ [1, max_clusters] chosen via silhouette score.
    Returns the best-fitted GMM and labels.
    """
    vecs = np.asarray(vecs)
    n_samples = len(vecs)

    # Fallback to single cluster if too few samples
    if n_samples < 3:
        gmm = GaussianMixture(
            n_components=1, covariance_type='full',
            random_state=random_state, reg_covar=1e-6
        )
        gmm.fit(vecs)
        labels = gmm.predict(vecs)
        return gmm, labels

    best_gmm, best_labels, best_score = None, None, -np.inf

    # Try K from 2 to min(max_clusters, n_samples)
    for k in range(2, min(max_clusters, n_samples) + 1):
        gmm = GaussianMixture(
            n_components=k, covariance_type='full',
            random_state=random_state, reg_covar=1e-6
        )
        gmm.fit(vecs)
        labels = gmm.predict(vecs)

        # Silhouette requires at least 2 distinct labels
        if len(set(labels)) < 2:
            continue

        score = silhouette_score(vecs, labels, metric='euclidean')
        if score > best_score:
            best_gmm, best_labels, best_score = gmm, labels, score

    # If no multi-cluster solution is valid, fall back to K=1
    if best_gmm is None:
        gmm = GaussianMixture(
            n_components=1, covariance_type='full',
            random_state=random_state, reg_covar=1e-6
        ).fit(vecs)
        labels = gmm.predict(vecs)
        return gmm, labels

    return best_gmm, best_labels


# -----------------------------
# PG concentration: cluster-weighted eigengap
# -----------------------------
def estimate_pg_concentration_mixture(gmm, vecs, labels) -> float:
    """
    Compute the cluster-weighted PG concentration:
        ζ_hat = Σ_k π_k * (λ_max(Σ_k) - λ_min(Σ_k))
    where Σ_k is the sample covariance of cluster k and π_k is its sample proportion.
    If only one cluster exists, compute the eigengap on all samples.
    """
    X = l2_normalize(np.asarray(vecs))
    n = X.shape[0]

    # Single-cluster case
    if gmm is None or getattr(gmm, "n_components", 1) == 1 or len(np.unique(labels)) == 1:
        C = safe_cov(X)
        return eigengap(C)

    # Multi-cluster case
    zetas, pis = [], []
    for k in range(gmm.n_components):
        Xk = X[labels == k]
        nk = Xk.shape[0]
        if nk == 0:
            continue
        Ck = safe_cov(Xk)
        z_k = eigengap(Ck)
        zetas.append(z_k)
        pis.append(nk / n)

    if not zetas:  # rare safeguard
        C = safe_cov(X)
        return eigengap(C)

    zetas = np.array(zetas, dtype=float)
    pis = np.array(pis, dtype=float)
    return float((pis * zetas).sum())


# -----------------------------
# Embeddings extraction wrapper
# -----------------------------
def get_contextual_embeddings_with_freqs(vectorizer, tokenizer, sentences, device='cpu'):
    """
    Extract contextual embeddings and produce:
      - vector_size: embedding dimensionality
      - token2freq: token -> frequency in the corpus slice
      - token2vecs: token -> np.ndarray of contextual vectors (row-wise, L2-normalized)
    """
    token2vecs = util.tokenize_and_vectorize(vectorizer, tokenizer, sentences, device=device)

    token2freq = {}
    vector_size = None
    for token, vecs in token2vecs.items():
        vecs = l2_normalize(np.asarray(vecs))
        token2vecs[token] = vecs
        token2freq[token] = vecs.shape[0]
        if vector_size is None and vecs.shape[0] > 0:
            vector_size = vecs.shape[1]
    return vector_size, token2freq, token2vecs


# -----------------------------
# Compute C(S,T) over the intersection vocabulary
# -----------------------------
def cal_change_score(source_token2freq,
                     target_token2freq,
                     source_token2vecs,
                     target_token2vecs,
                     freq_threshold=10,
                     kmax=5):

    vocab_S = set(source_token2freq.keys())
    vocab_T = set(target_token2freq.keys())
    common_vocab = vocab_S & vocab_T

    token2score = {}
    for token in tqdm(common_vocab):
        fs = source_token2freq[token]
        ft = target_token2freq[token]
        if fs <= freq_threshold or ft <= freq_threshold:
            continue

        Xs = source_token2vecs[token]
        Xt = target_token2vecs[token]

        # Cluster and compute weighted concentration per corpus
        gmm_s, lab_s = cluster_vectors(Xs, max_clusters=kmax)
        gmm_t, lab_t = cluster_vectors(Xt, max_clusters=kmax)

        zeta_S = estimate_pg_concentration_mixture(gmm_s, Xs, lab_s)
        zeta_T = estimate_pg_concentration_mixture(gmm_t, Xt, lab_t)

        if zeta_S > 0 and zeta_T > 0:
            token2score[token] = float(np.log(zeta_T / zeta_S))

    return token2score


# -----------------------------
# Output
# -----------------------------
def output(token2score, source_token2freq, target_token2freq, output_path: str,
           delim: str = '\t', digit: int = 3):
    """
    Write TSV with columns: token, score, freq_S, freq_T
    Sorted by score descending.
    """
    results = sorted(token2score.items(), key=lambda x: x[1], reverse=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for token, score in tqdm(results):
            f.write(delim.join((
                token,
                str(round(score, digit)),
                str(source_token2freq.get(token, 0)),
                str(target_token2freq.get(token, 0))
            )) + '\n')


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    # Load corpora and batch sentences
    source_sentences = util.load_sentences(args.source_corpus, args.cased)
    target_sentences = util.load_sentences(args.target_corpus, args.cased)
    batched_source = util.to_batches(source_sentences, batch_size=args.batch_size)
    batched_target = util.to_batches(target_sentences, batch_size=args.batch_size)

    # Prepare PLM and tokenizer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    vectorizer = BertModel.from_pretrained(args.bert_model)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # Extract contextual embeddings
    _, src_freq, src_vecs = get_contextual_embeddings_with_freqs(
        vectorizer, tokenizer, batched_source, device=device
    )
    _, tgt_freq, tgt_vecs = get_contextual_embeddings_with_freqs(
        vectorizer, tokenizer, batched_target, device=device
    )

    # Compute PG-based change scores
    token2score = cal_change_score(
        src_freq, tgt_freq,
        src_vecs, tgt_vecs,
        freq_threshold=args.freq_threshold,
        kmax=args.kmax
    )

    output(token2score, src_freq, tgt_freq, output_path=args.out)


if __name__ == '__main__':
    main()