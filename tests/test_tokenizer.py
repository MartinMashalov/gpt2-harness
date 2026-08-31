"""Byte-level BPE: round-trip losslessness and agreement with the reference."""

from __future__ import annotations

import json

import pytest

from transformer_internals.tokenizer import BPETokenizer, bytes_to_unicode, get_pairs

SAMPLES = [
    "Hello, world!",
    "  leading and trailing spaces  ",
    "multiple   internal    spaces",
    "newlines\nand\ttabs\r\n",
    "emoji 🌍🚀 and CJK 你好世界 and Cyrillic Привет",
    "punctuation!?...,;:'\"()[]{}<>/\\|@#$%^&*",
    "numbers 0123456789 and 3.14159 and 1,000,000",
    "def f(x):\n    return x ** 2  # comment",
    "",
    "a",
    " ",
    "\x00\x01\x02 control bytes \x7f",
    "MixedCASE camelCase snake_case SCREAMING_SNAKE",
    "ünïcödé àccênts and ß ligatures œ æ",
]


def test_byte_map_is_a_bijection_over_256_values() -> None:
    m = bytes_to_unicode()
    assert len(m) == 256
    assert len(set(m.values())) == 256
    # Every mapped character must be printable and non-space, which is the whole
    # point of the map: it keeps the BPE alphabet safe for whitespace splitting.
    assert all(len(v) == 1 and not v.isspace() for v in m.values())


def test_get_pairs() -> None:
    assert get_pairs(("a", "b", "c")) == {("a", "b"), ("b", "c")}
    assert get_pairs(("a",)) == set()
    # Repeated bigrams collapse: multiplicity is irrelevant to merge selection.
    assert get_pairs(("a", "b", "a", "b")) == {("a", "b"), ("b", "a")}


def test_tokenizer_works_without_the_published_vocab() -> None:
    """A byte-only tokenizer with no merges must still round-trip anything.

    This runs in CI with no network: the vocabulary is the 256 byte symbols and
    the merge list is empty, so every string encodes to its raw bytes.
    """
    byte_map = bytes_to_unicode()
    encoder = {ch: i for i, ch in enumerate(byte_map.values())}
    tok = BPETokenizer(encoder, [])
    for s in SAMPLES:
        assert tok.decode(tok.encode(s)) == s, s


def test_merges_are_applied_in_rank_order() -> None:
    """A hand-built two-merge vocabulary, checked against the merges by hand."""
    byte_map = bytes_to_unicode()
    encoder = {ch: i for i, ch in enumerate(byte_map.values())}
    encoder["ab"] = 300
    encoder["abc"] = 301
    tok = BPETokenizer(encoder, [("a", "b"), ("ab", "c")])
    # "abc" -> merge (a,b) first (rank 0) -> "ab c" -> merge (ab,c) -> "abc"
    assert tok.encode("abc") == [301]
    assert tok.decode([301]) == "abc"
    # "acb" has no applicable merge, so it stays as three byte tokens.
    assert len(tok.encode("acb")) == 3


@pytest.mark.weights
def test_roundtrip_against_published_vocab(gpt2_available: bool) -> None:
    if not gpt2_available:
        pytest.skip("GPT-2 vocab not available offline")
    from transformer_internals.weights import resolve_checkpoint_dir

    tok = BPETokenizer.from_pretrained(resolve_checkpoint_dir(local_files_only=True))
    assert tok.vocab_size == 50257
    assert tok.eot_token_id == 50256
    for s in SAMPLES:
        assert tok.decode(tok.encode(s)) == s, s


@pytest.mark.weights
def test_matches_reference_tokenizer_exactly(gpt2_available: bool) -> None:
    """Our ids must equal HuggingFace's ids, token for token.

    Round-tripping only proves the tokenizer is *lossless*; it could still be
    losslessly wrong -- a different segmentation would round-trip perfectly and
    produce entirely different model inputs. Only agreement with the reference
    segmentation proves the merge algorithm is right.
    """
    if not gpt2_available:
        pytest.skip("GPT-2 vocab not available offline")
    transformers = pytest.importorskip("transformers")
    from transformer_internals.weights import resolve_checkpoint_dir

    ckpt = resolve_checkpoint_dir(local_files_only=True)
    ours = BPETokenizer.from_pretrained(ckpt)
    ref = transformers.GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    for s in SAMPLES:
        assert ours.encode(s) == ref.encode(s), s


@pytest.mark.weights
def test_vocabulary_agrees_with_the_published_file(gpt2_available: bool) -> None:
    if not gpt2_available:
        pytest.skip("GPT-2 vocab not available offline")
    from transformer_internals.weights import resolve_checkpoint_dir

    ckpt = resolve_checkpoint_dir(local_files_only=True)
    tok = BPETokenizer.from_pretrained(ckpt)
    published = json.loads((ckpt / "vocab.json").read_text(encoding="utf-8"))
    assert tok.encoder == published
