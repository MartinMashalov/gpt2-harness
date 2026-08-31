"""Part 2: numerical equivalence against HuggingFace GPT-2.

Every test here is marked ``weights`` and is deselected in CI, which has neither
the checkpoint nor the network. Locally they are the tests that matter most.
"""

from __future__ import annotations

import pytest
import torch

from transformer_internals.verify import (
    LOGIT_TOLERANCE,
    compare_activations,
    compare_greedy_generation,
    compare_perplexity,
)

pytestmark = pytest.mark.weights


def test_parameter_count_matches_the_published_model(gpt2_pair) -> None:
    model, ref, _ = gpt2_pair
    ours = model.num_parameters()
    # HF stores lm_head untied in its parameter list; count unique tensors.
    theirs = sum(p.numel() for p in {id(p): p for p in ref.parameters()}.values())
    assert ours == 124_439_808
    assert ours == theirs


def test_final_logits_match_within_tolerance(gpt2_pair) -> None:
    model, ref, tok = gpt2_pair
    x = torch.tensor([tok.encode("The quick brown fox jumps over the lazy dog.")])
    with torch.no_grad():
        ours = model(x)["logits"]
        theirs = ref(x).logits
    max_abs = (ours - theirs).abs().max().item()
    assert max_abs < LOGIT_TOLERANCE, f"max |delta| = {max_abs:.3e}"


def test_every_activation_matches(gpt2_pair) -> None:
    """Localises any discrepancy to a specific sub-module of a specific block."""
    model, ref, tok = gpt2_pair
    ids = tok.encode("In a shocking finding, scientists discovered a herd of unicorns")
    x = torch.tensor([ids])
    rows, _, _ = compare_activations(model, ref, x)
    # embed + 6 sub-modules per block + ln_f + logits
    assert len(rows) == 1 + 6 * model.config.n_layer + 2
    for row in rows:
        assert row.max_abs < 1e-3, f"{row.name}: {row.max_abs:.3e}"


def test_greedy_generation_is_token_exact(gpt2_pair) -> None:
    """The most sensitive check in the suite: 200 consecutive argmax agreements."""
    model, ref, tok = gpt2_pair
    records = compare_greedy_generation(
        model, ref, tok,
        prompts=["The capital of France is", "Once upon a time, there was a"],
        max_new_tokens=200,
    )
    for r in records:
        assert r["match"], (
            f"diverged at token {r['first_divergence']} for prompt {r['prompt']!r}"
        )


def test_perplexity_matches(gpt2_pair) -> None:
    model, ref, tok = gpt2_pair
    from transformer_internals.data import TokenDataset, encode_corpus

    tokens = encode_corpus(tok, max_chars=200_000, local_files_only=True)
    ds = TokenDataset(tokens, block_size=128)
    out = compare_perplexity(model, ref, ds, n_batches=2, batch_size=2)
    assert out["abs_ppl_diff"] < 1e-3
    assert 5.0 < out["ours_ppl"] < 200.0


def test_a_transposed_projection_is_detected(gpt2_pair) -> None:
    """A negative control: the suite must FAIL on a model that is subtly wrong.

    A verification suite that has never been shown to reject anything is not
    evidence. Transposing one square projection -- the exact bug that survives
    because 768x768 still multiplies -- must be caught.
    """
    import copy

    model, ref, tok = gpt2_pair
    broken = copy.deepcopy(model)
    with torch.no_grad():
        w = broken.h[3].attn.c_proj.weight
        w.copy_(w.t().contiguous())

    x = torch.tensor([tok.encode("The capital of France is")])
    with torch.no_grad():
        max_abs = (broken(x)["logits"] - ref(x).logits).abs().max().item()
    assert max_abs > LOGIT_TOLERANCE

    records = compare_greedy_generation(
        broken, ref, tok, prompts=["The capital of France is"], max_new_tokens=40
    )
    assert not records[0]["match"]
