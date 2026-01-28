# -*- coding: utf-8 -*-
"""
Prototypical instance extraction for semantic change (Projected Gaussian)

Given a target word/phrase, this script finds representative contextual usages
that exemplify the detected change from source corpus S to target corpus T.

Method (as in the paper):
  1) For the target word, collect contextual embeddings from S and T.
  2) Estimate clustering-weighted covariances Σ_S and Σ_T:
        - cluster embeddings (GMM; K chosen by silhouette in [1..max_k])
        - compute per-cluster sample covariance and weight by cluster proportions
  3) Compute principal change vector v_diff as the top eigenvector of (Σ_T - Σ_S).
  4) Score each target embedding x in T by |x^T v_diff|.
  5) Return top-N target sentences as prototypical instances.

Notes:
  * We use single-subword vectors per token (fixed dimensionality).
  * Embeddings are L2-normalized (spherical geometry).
  * If you suspect narrowing instead (C>0), you can invert the order (Σ_S - Σ_T)
    to highlight source-side prototypes; here we default to T prototypes.
"""

import argparse
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, BertModel
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

import util


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-source_corpus", default="source.txt", help="Path to source corpus (S)")
    parser.add_argument("-target_corpus", default="target.txt", help="Path to target corpus (T)")
    parser.add_argument(
        "-t", "--target_phrase", default="aaa", help="Target word/phrase to collect instances for"
    )
    parser.add_argument(
        "-m", "--bert_model", default="bert-large-cased", help="HF model name for contextual embeddings"
    )
    parser.add_argument(
        "-c", "--cased", action="store_true", help="If set, keep case (do not lowercase corpora)"
    )
    parser.add_argument("-b", "--batch_size", default=32, type=int, help="Batch size for encoding")
    parser.add_argument(
        "-n", "--topN", default=10, type=int, help="Number of top prototypical instances to print"
    )
    parser.add_argument(
        "--max_k", default=5, type=int, help="Max number of clusters to try for GMM (1..max_k)"
    )
    parser.add_argument(
        "--ridge", default=1e-6, type=float, help="Small diagonal ridge added to covariances"
    )
    parser.add_argument(
        "--focus", choices=["T", "S"], default="T",
        help="Which side to rank/print prototypes from: T (broadening) or S (narrowing)"
    )
    # For PG, we require fixed-dim vectors per occurrence; do not concatenate all subwords.
    # (We keep the flag for API compatibility but ignore it.)
    parser.add_argument(
        "--all_subwords", action="store_true",
        help="(Ignored in PG mode) Incompatible with covariance; we always use first-subword vectors."
    )
    return parser.parse_args()


# -----------------------------
# Clustering & covariance
# -----------------------------
def _gmm_silhouette(vecs: np.ndarray, max_k: int) -> Tuple[GaussianMixture, np.ndarray]:
    """
    Fit GMM with K chosen by silhouette over K in [2..min(max_k, n)].
    Fallback to K=1 if silhouette is invalid or n<2.
    """
    n = len(vecs)
    if n < 2:
        gmm = GaussianMixture(n_components=1, covariance_type="full", random_state=0).fit(vecs)
        labels = gmm.predict(vecs)
        return gmm, labels

    best = None
    best_score = -1.0
    upper = min(max_k, n)
    # Try K from 2..upper; fall back to 1 if none valid
    for k in range(2, upper + 1):
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=0)
        gmm.fit(vecs)
        labels = gmm.predict(vecs)
        uniq = np.unique(labels)
        if len(uniq) < 2 or len(uniq) >= n:
            continue
        try:
            score = silhouette_score(vecs, labels)
        except Exception:
            score = -1.0
        if score > best_score:
            best_score = score
            best = (gmm, labels)

    if best is None:
        gmm = GaussianMixture(n_components=1, covariance_type="full", random_state=0).fit(vecs)
        labels = gmm.predict(vecs)
        return gmm, labels
    return best


def _sample_cov(x: np.ndarray) -> np.ndarray:
    """
    Sample covariance with row-wise observations. Safe for n<2.
    """
    n, d = x.shape
    if n <= 1:
        return np.zeros((d, d), dtype=np.float32)
    # Center then compute (n-1) denominator covariance
    mu = x.mean(axis=0, keepdims=True)
    xc = x - mu
    return (xc.T @ xc) / float(n - 1)


def estimate_weighted_covariance(
    vecs: np.ndarray, max_k: int = 5, ridge: float = 1e-6
) -> np.ndarray:
    """
    Clustering-weighted covariance:
      Σ_hat = sum_k π_k * Σ_k, where Σ_k is sample covariance within cluster k.

    Args:
        vecs: [n, d] L2-normalized contextual vectors.
        max_k: upper bound for GMM components.
        ridge: small diagonal added for numerical stability.

    Returns:
        Σ_hat: [d, d] positive semi-definite matrix.
    """
    # Ensure 2D
    vecs = np.asarray(vecs, dtype=np.float32)
    if vecs.ndim != 2:
        raise ValueError("vecs must be 2D [n, d]")

    # If too few samples, return small diagonal
    n, d = vecs.shape
    if n < 2:
        return ridge * np.eye(d, dtype=np.float32)

    gmm, labels = _gmm_silhouette(vecs, max_k=max_k)
    k = gmm.n_components

    # If K==1, just return sample covariance
    if k == 1:
        cov = _sample_cov(vecs)
        cov.flat[:: d + 1] += ridge
        return cov

    # Weighted sum of per-cluster covariances
    Sigma = np.zeros((d, d), dtype=np.float32)
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        xk = vecs[idx]
        cov_k = _sample_cov(xk)
        Sigma += (len(idx) / float(n)) * cov_k

    Sigma.flat[:: d + 1] += ridge
    return Sigma


# -----------------------------
# PG prototypical direction
# -----------------------------
def principal_change_direction(Sigma_T: np.ndarray, Sigma_S: np.ndarray) -> np.ndarray:
    """
    v_diff := top eigenvector of (Sigma_T - Sigma_S).
    """
    Delta = Sigma_T - Sigma_S
    # symmetric eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(Delta)
    idx = np.argmax(eigvals)
    v = eigvecs[:, idx].astype(np.float32)
    # Normalize direction (optional; scores use dot magnitude anyway)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v


def score_prototypes(vecs: np.ndarray, v_diff: np.ndarray) -> np.ndarray:
    """
    Score each vector x by |x^T v_diff|.
    Assumes vecs are L2-normalized.
    """
    return np.abs(vecs @ v_diff)


# -----------------------------
# Pretty printing
# -----------------------------
def format_context(words: List[str], span: Tuple[int, int], marker: str = "*") -> Tuple[str, str, str]:
    """
    Split a tokenized sentence into 3 fields (before, target, after) for display.
    """
    start, end = span
    w = ["[BOS]"] + words + ["[EOS]"]
    # shift because of [BOS]
    start += 1
    end += 1
    before = " ".join(w[: start])
    target = marker + " " + " ".join(w[start : end]) + " " + marker
    after = " ".join(w[end :])
    return before, target.strip(), after


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    # Load sentences and spans for target phrase in S and T
    src_sents, src_spans = util.load_sentences_with_target_spans(
        args.source_corpus, args.target_phrase, cased=args.cased
    )
    tgt_sents, tgt_spans = util.load_sentences_with_target_spans(
        args.target_corpus, args.target_phrase, cased=args.cased
    )
    if len(src_spans) < 1 or len(tgt_spans) < 1:
        return

    # Batch the data
    bs = args.batch_size
    batched_src_sents = util.to_batches(src_sents, batch_size=bs)
    batched_src_spans = util.to_batches(src_spans, batch_size=bs)
    batched_tgt_sents = util.to_batches(tgt_sents, batch_size=bs)
    batched_tgt_spans = util.to_batches(tgt_spans, batch_size=bs)

    # Encoder + tokenizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = BertModel.from_pretrained(args.bert_model)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # Extract contextual vectors for the target phrase in S and T
    # NOTE: In PG mode we require fixed-dim vectors, so we always take the first subword.
    src_vecs = util.tokenize_and_vectorize_with_spans(
        batched_src_sents, batched_src_spans, encoder, tokenizer, all_subwords=False, device=device
    )
    tgt_vecs = util.tokenize_and_vectorize_with_spans(
        batched_tgt_sents, batched_tgt_spans, encoder, tokenizer, all_subwords=False, device=device
    )

    # Convert to arrays and L2-normalize per vector
    def _to_unit(xlist: List[np.ndarray]) -> np.ndarray:
        X = np.asarray(xlist, dtype=object)  # possible ragged if any bad items
        # Force to 2D float array
        X = np.vstack([np.asarray(v, dtype=np.float32) for v in xlist])
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    Xs = _to_unit(src_vecs)
    Xt = _to_unit(tgt_vecs)

    # Estimate clustering-weighted covariances
    Sigma_S = estimate_weighted_covariance(Xs, max_k=args.max_k, ridge=args.ridge)
    Sigma_T = estimate_weighted_covariance(Xt, max_k=args.max_k, ridge=args.ridge)

    # Principal change direction
    v_diff = principal_change_direction(Sigma_T, Sigma_S)

    # Decide which side to score/print
    if args.focus == "T":
        scores = score_prototypes(Xt, v_diff)
        sentences = tgt_sents
        spans = tgt_spans
    else:
        # If focusing on S (e.g., narrowing), use the opposite difference
        v_src = principal_change_direction(Sigma_S, Sigma_T)
        scores = score_prototypes(Xs, v_src)
        sentences = src_sents
        spans = src_spans

    # Rank and print top-N
    ranked = sorted(zip(scores.tolist(), sentences, spans), key=lambda z: z[0], reverse=True)
    topN = min(args.topN, len(ranked))
    for s, words, span in ranked[:topN]:
        before, target, after = format_context(words, span, marker="*")
        print(f"{s:.6f}\t{before}\t{target}\t{after}")


if __name__ == "__main__":
    main()