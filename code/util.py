# -*- coding: utf-8 -*-
"""
Utility helpers for extracting contextual embeddings from BERT-like models.

Used by:
  i)  detect_semantic_change.py
  ii) extract_typical_instance.py
"""

import codecs
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm


# -----------------------------
# I/O: sentence loaders
# -----------------------------
def load_sentences(corpus_file: str, cased: bool = False) -> List[List[str]]:
    """
    Load a corpus as tokenized sentences (space-separated tokens per line).

    Args:
        corpus_file: Path to a text file; one sentence per line.
        cased: If False, lowercases all text.

    Returns:
        sentences: List of sentences; each sentence is a list of tokens.
    """
    sentences: List[List[str]] = []
    with codecs.open(corpus_file, "r", "utf-8", "ignore") as fp:
        for line in fp:
            line = line.rstrip("\n")
            if not cased:
                line = line.lower()
            tokens = line.split(" ")
            sentences.append(tokens)
    return sentences


def load_sentences_with_target_spans(
    corpus_file: str, target_phrase: str, cased: bool = False
) -> Tuple[List[List[str]], List[Tuple[int, int]]]:
    """
    Load sentences and collect spans where a target phrase occurs.

    Args:
        corpus_file: Path to text file (one sentence per line).
        target_phrase: Space-separated phrase to find.
        cased: If False, lowercases both corpus and phrase.

    Returns:
        sentences: List of tokenized sentences.
        spans: List of (start, end) indices *per kept sentence* where the phrase occurs.
               One entry per match; sentences with multiple matches are duplicated
               once per match (aligned with `spans` length).
    """
    if not cased:
        target_phrase = target_phrase.lower()
    phrase_tokens = target_phrase.split(" ")

    sentences: List[List[str]] = []
    spans: List[Tuple[int, int]] = []

    with codecs.open(corpus_file, "r", "utf-8", "ignore") as fp:
        for line in fp:
            line = line.rstrip("\n")
            if not cased:
                line = line.lower()
            tokens = line.split(" ")
            hit_spans = _find_spans(tokens, phrase_tokens)
            for s in hit_spans:
                sentences.append(tokens)
                spans.append(s)
    return sentences, spans


def _find_spans(words: List[str], phrase: List[str]) -> List[Tuple[int, int]]:
    """Return all (start, end) spans where `phrase` occurs in `words`."""
    spans: List[Tuple[int, int]] = []
    m = len(phrase)
    if m == 0:
        return spans
    for i in range(0, len(words) - m + 1):
        if words[i : i + m] == phrase:
            spans.append((i, i + m))
    return spans


# -----------------------------
# Batching
# -----------------------------
def to_batches(instances: List, batch_size: int = 32) -> List[List]:
    """
    Split a list into contiguous batches.

    Args:
        instances: Any list-like sequence.
        batch_size: Positive integer.

    Returns:
        List of batches (each a slice of `instances`).
    """
    if batch_size <= 0:
        return [instances]
    n = len(instances)
    num_full = n // batch_size
    batches = [instances[i * batch_size : (i + 1) * batch_size] for i in range(num_full)]
    if n % batch_size:
        batches.append(instances[num_full * batch_size :])
    return batches


# -----------------------------
# Tokenization + Embeddings
# -----------------------------
def tokenize_with_ids(tokenizer, sentences: List[List[str]], device: str = "cpu"):
    """
    Tokenize a batch of *tokenized* sentences with a HuggingFace tokenizer.

    Args:
        tokenizer: HF tokenizer (compatible with the model).
        sentences: List of token lists; pass is_split_into_words=True.
        device: Torch device string.

    Returns:
        tokens: HF BatchEncoding with word alignment access (word_ids).
        token_ids: Tensor [B, L]
        mask_ids:  Tensor [B, L]
    """
    tokens = tokenizer(
        sentences,
        return_tensors="pt",
        is_split_into_words=True,
        padding=True,
        truncation=True,
    )
    token_ids = tokens["input_ids"].to(device)
    mask_ids = tokens["attention_mask"].to(device)
    return tokens, token_ids, mask_ids


def to_ids_with_token_idx(tokenizer, sentences: List[List[str]], device: str = "cpu"):
    """
    Map original token indices to single-subword positions in the subword sequence.

    We keep only tokens that are not split (i.e., exactly one subword), so we can
    directly take the corresponding hidden state without pooling.

    Args:
        tokenizer: HF tokenizer.
        sentences: List of token lists.
        device: Torch device.

    Returns:
        token_ids: Tensor [B, L]
        mask_ids: Tensor [B, L]
        token_idx2subword_idx: List[Dict[int, int]] of length B.
            For each batch item, maps original token index -> subword index (if single-subword).
    """
    tokens = tokenizer(
        sentences,
        return_tensors="pt",
        is_split_into_words=True,
        padding=True,
        truncation=True,
    )
    token_ids = tokens["input_ids"].to(device)
    mask_ids = tokens["attention_mask"].to(device)

    token_idx2subword_idx: List[Dict[int, int]] = [{} for _ in sentences]
    for b in range(len(sentences)):
        sub2tok = tokens.word_ids(b)
        tok2subs: Dict[int, List[int]] = {}
        for sub_i, tok_i in enumerate(sub2tok):
            if tok_i is None:
                continue
            tok2subs.setdefault(tok_i, []).append(sub_i)
        for tok_i, subs in tok2subs.items():
            if len(subs) == 1:  # keep only tokens mapping to exactly one subword
                token_idx2subword_idx[b][tok_i] = subs[0]

    return token_ids, mask_ids, token_idx2subword_idx


def tokenize_and_vectorize(
    vectorizer,
    tokenizer,
    batched_sentences: List[List[List[str]]],
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """
    Produce contextual embeddings per token across the corpus (L2-normalized).

    For each sentence batch:
      - tokenize with alignment
      - run the encoder (no grad, eval mode)
      - for tokens that map to a single subword, collect their hidden state
      - L2-normalize vectors and accumulate by token string

    Args:
        vectorizer: HF encoder model (e.g., BertModel).
        tokenizer: Matching HF tokenizer.
        batched_sentences: List of batches; each batch is a list of token lists.
        device: Torch device.

    Returns:
        token2vecs: dict token -> np.ndarray of contextual embeddings (row-wise).
                    Note: each row is L2-normalized.
    """
    vectorizer.to(device)
    vectorizer.eval()

    token2vecs: Dict[str, List[np.ndarray]] = {}

    with torch.no_grad():
        for sentences in tqdm(batched_sentences):
            token_ids, mask_ids, tokidx2subidx = to_ids_with_token_idx(tokenizer, sentences, device=device)
            output = vectorizer(token_ids, mask_ids)
            H = output.last_hidden_state  # [B, L, D]

            for b, sent in enumerate(sentences):
                for tok_i, sub_i in tokidx2subidx[b].items():
                    vec = H[b, sub_i]  # [D]
                    v = vec.detach().cpu().numpy().astype(np.float32)
                    # L2 normalize (row-wise) for spherical geometry
                    n = np.linalg.norm(v)
                    if n > 0:
                        v = v / n
                    token = sent[tok_i]
                    token2vecs.setdefault(token, []).append(v)

    # pack lists -> arrays
    token2vecs_np: Dict[str, np.ndarray] = {}
    for tok, lst in token2vecs.items():
        if len(lst) == 0:
            continue
        token2vecs_np[tok] = np.vstack(lst)
    return token2vecs_np


# -----------------------------
# Span-based extraction (for prototype module)
# -----------------------------
def tokenize_and_vectorize_with_spans(
    batched_sentences: List[List[List[str]]],
    batched_target_spans: List[List[Tuple[int, int]]],
    vectorizer,
    tokenizer,
    all_subwords: bool = False,
    device: str = "cpu",
) -> List[np.ndarray]:
    """
    Extract contextual vectors for specific spans (per batch-aligned lists).

    For each (sentence, span):
      - tokenize with alignment
      - align original token span to subword span
      - take either the first subword vector or concatenate all subword vectors

    Args:
        batched_sentences: List of sentence batches.
        batched_target_spans: Same batching as sentences, list of span lists.
        vectorizer: HF encoder.
        tokenizer: HF tokenizer.
        all_subwords: If True, concatenate all subword vectors; else take first subword.
        device: Torch device.

    Returns:
        target_vecs: List of np.ndarray vectors (one per span).
    """
    target_vecs: List[np.ndarray] = []
    vectorizer.to(device)
    vectorizer.eval()

    with torch.no_grad():
        for sentences, spans in zip(batched_sentences, batched_target_spans):
            tokens, token_ids, mask_ids = tokenize_with_ids(tokenizer, sentences, device=device)
            H = vectorizer(token_ids, mask_ids).last_hidden_state  # [B, L, D]

            for b, span in enumerate(spans):
                word_ids = tokens.word_ids(b)  # alignment from subwords to word indices
                sub_start, sub_end = _align_span_to_subwords(span, word_ids)
                sub_vecs = [H[b, j] for j in range(sub_start, sub_end)]
                if all_subwords:
                    v = torch.cat(sub_vecs, dim=0)
                else:
                    v = sub_vecs[0]
                vec = v.detach().cpu().numpy().astype(np.float32)
                # (no normalization here; caller can decide depending on use)
                target_vecs.append(vec)

    return target_vecs


def _align_span_to_subwords(span: Tuple[int, int], word_ids: List[int]) -> Tuple[int, int]:
    """
    Map an original token span [start, end) to (start_subword_idx, end_subword_idx).

    We take the first subword index for `start` and the first subword index of `end`
    (effectively producing a half-open subword range).
    """
    start, end = span

    # First occurrence of the `start` token index in subwords
    sub_start = word_ids.index(start)

    # First subword whose word_id equals `end` gives us the half-open end.
    try:
        sub_end = word_ids.index(end)
    except ValueError:
        # If not present (e.g., `end` token collapsed), fallback to sub_start + 1
        sub_end = sub_start + 1

    return sub_start, sub_end


# -----------------------------
# Optional: vectors + positions (for retrieving instances)
# -----------------------------
def tokenize_and_vectorize_with_indices(
    vectorizer,
    tokenizer,
    batched_sentences: List[List[List[str]]],
    device: str = "cpu",
):
    """
    Like `tokenize_and_vectorize`, but also return (sentence_idx, token_idx) pairs
    for each collected vector. This is handy for retrieving the original text
    of top-ranked prototypical instances.

    Returns:
        token2vecs: dict token -> np.ndarray of vectors (L2-normalized, row-wise)
        token2indices: dict token -> List[(sent_idx_in_stream, token_idx_in_sent)]
    """
    vectorizer.to(device)
    vectorizer.eval()

    token2vecs: Dict[str, List[np.ndarray]] = {}
    token2indices: Dict[str, List[Tuple[int, int]]] = {}

    sent_counter = 0
    with torch.no_grad():
        for sentences in batched_sentences:
            token_ids, mask_ids, tokidx2subidx = to_ids_with_token_idx(tokenizer, sentences, device=device)
            H = vectorizer(token_ids, mask_ids).last_hidden_state  # [B, L, D]

            for b, sent in enumerate(sentences):
                for tok_i, sub_i in tokidx2subidx[b].items():
                    v = H[b, sub_i].detach().cpu().numpy().astype(np.float32)
                    n = np.linalg.norm(v)
                    if n > 0:
                        v = v / n
                    tok = sent[tok_i]
                    token2vecs.setdefault(tok, []).append(v)
                    token2indices.setdefault(tok, []).append((sent_counter + b, tok_i))
            sent_counter += len(sentences)

    # pack to arrays
    token2vecs_np: Dict[str, np.ndarray] = {}
    for tok, lst in token2vecs.items():
        if len(lst) == 0:
            continue
        token2vecs_np[tok] = np.vstack(lst)
    return token2vecs_np, token2indices