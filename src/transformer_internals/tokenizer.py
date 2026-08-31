"""Byte-level BPE, implemented from scratch against the published GPT-2 vocab.

This is the piece most "from scratch" repositories quietly skip by importing
``GPT2Tokenizer``. It is worth writing, because two of its design decisions are
genuinely non-obvious and both of them matter.

**Why byte-level at all.** A character-level BPE has to decide what to do with
the ~150k Unicode code points it did not see in training, and every answer is
bad: an ``<unk>`` token destroys information, and a huge base vocabulary wastes
embedding rows. Working on raw UTF-8 *bytes* gives a base alphabet of exactly 256
symbols, and therefore a tokenizer that can round-trip literally any byte string
-- emoji, Cyrillic, a JPEG pasted into a prompt -- with no unknown token and no
lossy normalisation. GPT-2's vocabulary is 256 byte symbols + 50000 learned
merges + 1 ``<|endoftext|>`` = 50257.

**Why the strange byte->unicode map.** BPE implementations, GPT-2's included,
operate on strings and are trained on whitespace-split words. Feeding raw bytes
in as latin-1 characters would put control characters and spaces into the
vocabulary, which breaks the whitespace assumption and makes merge files
unreadable and unsafe to round-trip through JSON. So OpenAI defined a bijection
from the 256 byte values onto 256 *printable, non-space* Unicode code points:
bytes that are already printable ASCII/Latin-1 map to themselves, and the 68
that are not are shifted up into the U+0100.. range. Hence a leading space
appearing as ``Ġ`` in the vocabulary. It is a bijection, so it is lossless; it is
purely a representational convenience.

**Why the regex pre-tokenizer.** BPE merges never cross the boundaries the regex
produces, which is what stops the tokenizer from learning a single token for
``" the cat"``. GPT-2's pattern also famously attaches the *preceding* space to a
word (``" cat"``, not ``"cat"``), which is why ``" cat"`` and ``"cat"`` are
different tokens and why prompts that end in a space tokenize badly.

The vocabulary and merge list are the real published artefacts; only the
algorithm is reimplemented. ``tests/test_tokenizer.py`` round-trips this against
the reference tokenizer on a corpus that includes emoji, CJK and control bytes.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

import regex as re

__all__ = ["GPT2_SPLIT_PATTERN", "BPETokenizer", "bytes_to_unicode", "get_pairs"]

#: GPT-2's pre-tokenization pattern, verbatim from the released encoder.
#:
#: Reading it left to right: the English contraction suffixes are split off as
#: their own tokens ('s, 't, 're, ...); then a run of letters *optionally
#: preceded by one space*; then the same for digits; then the same for anything
#: that is neither letter, digit nor space (punctuation runs); then runs of
#: whitespace, with the final ``\s+(?!\S)`` / ``\s+`` pair arranging that a run
#: of spaces before a word gives its last space to the word rather than keeping
#: it. ``regex`` rather than ``re`` is required: ``\p{L}`` and ``\p{N}`` are
#: Unicode property escapes that the standard library does not support.
GPT2_SPLIT_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)

ENDOFTEXT = "<|endoftext|>"


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """Build the reversible byte -> printable-unicode map GPT-2 uses.

    The 188 byte values that are already printable and non-space in Latin-1
    (``!``..``~``, ``¡``..``¬``, ``®``..``ÿ``) map to themselves. The remaining 68
    are assigned, in increasing byte order, to code points 256, 257, ... -- so
    byte 0x20 (space) becomes U+0120 ``Ġ`` and byte 0x0A (newline) becomes U+010A
    ``Ċ``.

    Returns:
        A dict of all 256 byte values to distinct single-character strings.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    mapping = list(printable)
    n = 0
    for byte in range(256):
        if byte not in printable:
            mapping.append(256 + n)
            printable.append(byte)
            n += 1
    return {b: chr(c) for b, c in zip(printable, mapping, strict=True)}


def get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    """Return the set of adjacent symbol pairs in a word.

    Args:
        word: The word as a tuple of symbols (initially single characters).

    Returns:
        Every ``(word[i], word[i+1])``. A set, not a list: the merge step only
        ever asks "which pair has the lowest rank", so multiplicity is irrelevant
        and deduplicating here keeps the inner loop cheap.
    """
    return set(pairwise(word))


class BPETokenizer:
    """GPT-2's byte-level BPE encoder/decoder.

    Args:
        encoder: Token-string -> id, i.e. the parsed ``vocab.json``.
        bpe_merges: The merge list in rank order, i.e. ``merges.txt`` minus its
            version header. Earlier merges have lower rank and are applied first.

    Attributes:
        encoder: Token string -> integer id.
        decoder: Integer id -> token string.
        bpe_ranks: ``(left, right)`` pair -> merge rank.
    """

    def __init__(self, encoder: dict[str, int], bpe_merges: Iterable[tuple[str, str]]) -> None:
        self.encoder: dict[str, int] = dict(encoder)
        self.decoder: dict[int, str] = {v: k for k, v in self.encoder.items()}
        if len(self.decoder) != len(self.encoder):
            raise ValueError("vocab is not injective: two tokens share an id")
        self.bpe_ranks: dict[tuple[str, str], int] = {
            pair: i for i, pair in enumerate(bpe_merges)
        }
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self._pat = re.compile(GPT2_SPLIT_PATTERN)
        # BPE is a pure function of the word, and natural text re-uses words
        # heavily, so memoising it is the single biggest constant-factor win
        # available. Encoding 1 MB of text is ~4x faster with this cache.
        self._cache: dict[str, str] = {}

    # ------------------------------------------------------------------ load

    @classmethod
    def from_files(cls, vocab_path: str | Path, merges_path: str | Path) -> BPETokenizer:
        """Load from the published ``vocab.json`` and ``merges.txt``.

        Args:
            vocab_path: Path to ``vocab.json``.
            merges_path: Path to ``merges.txt``. Its first line is a
                ``#version:`` comment and is skipped.

        Returns:
            A ready tokenizer.
        """
        with Path(vocab_path).open(encoding="utf-8") as fh:
            encoder = json.load(fh)
        merge_text = Path(merges_path).read_text(encoding="utf-8")
        lines = merge_text.split("\n")
        start = 1 if lines and lines[0].startswith("#version") else 0
        merges: list[tuple[str, str]] = []
        for line in lines[start:]:
            if not line.strip():
                continue
            left, right = line.split()
            merges.append((left, right))
        return cls(encoder, merges)

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> BPETokenizer:
        """Load from a directory containing ``vocab.json`` and ``merges.txt``."""
        d = Path(model_dir)
        return cls.from_files(d / "vocab.json", d / "merges.txt")

    # ------------------------------------------------------------------- bpe

    def bpe(self, token: str) -> str:
        """Apply the merge list to one pre-tokenized word.

        The algorithm is the standard greedy-by-rank BPE: repeatedly find the
        adjacent pair with the *lowest* merge rank anywhere in the word, merge
        every occurrence of exactly that pair, and repeat until no pair in the
        word is in the merge table.

        Note the subtlety in the inner loop: after choosing the pair to merge we
        scan the word left to right and merge *all* non-overlapping occurrences
        of it in one pass. Merging one occurrence at a time and re-scanning gives
        the same answer but is quadratically slower.

        Args:
            token: A word already mapped through the byte->unicode table.

        Returns:
            The merged symbols joined by single spaces.
        """
        cached = self._cache.get(token)
        if cached is not None:
            return cached

        word: tuple[str, ...] = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
            self._cache[token] = token
            return token

        while True:
            # min() over the pairs present, defaulting unknown pairs to +inf so
            # they can never be selected.
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)

        out = " ".join(word)
        self._cache[token] = out
        return out

    # ---------------------------------------------------------------- public

    def encode(self, text: str) -> list[int]:
        """Encode a string to token ids.

        Args:
            text: Arbitrary text. No normalisation is applied -- byte-level BPE
                is lossless, and normalising here would silently break
                round-tripping.

        Returns:
            Token ids.
        """
        ids: list[int] = []
        for chunk in self._pat.findall(text):
            # UTF-8 encode, then push each byte through the printable map, so the
            # BPE below only ever sees single-character symbols from a fixed
            # 256-symbol alphabet.
            mapped = "".join(self.byte_encoder[b] for b in chunk.encode("utf-8"))
            ids.extend(self.encoder[sym] for sym in self.bpe(mapped).split(" "))
        return ids

    def decode(self, ids: Iterable[int], errors: str = "replace") -> str:
        """Decode token ids back to a string.

        Args:
            ids: Token ids.
            errors: Passed to ``bytes.decode``. ``"replace"`` matches the
                reference implementation: a *prefix* of a multi-byte character is
                a legitimate intermediate state during streaming generation, and
                raising there would make the generator unusable.

        Returns:
            The decoded text.
        """
        text = "".join(self.decoder[i] for i in ids)
        raw = bytes(self.byte_decoder[c] for c in text)
        return raw.decode("utf-8", errors=errors)

    @property
    def eot_token_id(self) -> int:
        """The id of ``<|endoftext|>`` (50256 in the released vocab)."""
        return self.encoder[ENDOFTEXT]

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def __len__(self) -> int:
        return len(self.encoder)
