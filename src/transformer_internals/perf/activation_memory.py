"""Activation memory: what the backward pass is holding, measured and predicted.

Parameters, gradients and optimizer state are easy to account for: they are a
fixed multiple of the parameter count, they do not depend on the batch, and
:mod:`transformer_internals.parallel` already reports them per rank. Activations
are none of those things. They scale with tokens rather than parameters, they
are what a longer context makes expensive, and on a real run they are usually
what actually runs the device out of memory. This repository had no number for
them at all.

What is measured
----------------
The **activation stash**: the distinct tensor storages that the autograd graph
is holding for the backward pass. This is the quantity the phrase "activation
memory" means. It is measured with ``torch.autograd.graph.saved_tensors_hooks``,
which sees every tensor saved for backward as it is saved, so the count is of
the actual graph and not of a model of it.

Two properties make this the right instrument rather than an allocator reading:

* It works on any device. ``torch.cuda.max_memory_allocated`` is the usual way
  to get this on a GPU and does not exist on CPU or MPS, so a CPU-only
  measurement would be impossible and the CUDA path would be the only tested
  one. Here the number means the same thing on all three.
* It isolates activations from everything else. An allocator peak includes the
  parameters, the gradients, the optimizer state and every transient workspace,
  so getting activations out of it means subtracting three other numbers and
  hoping nothing was fragmented. Saved tensors are activations by definition.

Storages are counted once, not once per tensor, so a view saved twice is charged
once. That is the same rule
:func:`transformer_internals.parallel.common.state_bytes` uses for parameters,
and it is what keeps the two columns comparable.

**Parameters and buffers are excluded, and that matters by 22%.** The autograd
graph saves weights as well as activations: ``addmm`` keeps the weight matrix
because the input gradient is ``grad @ W``. On the tested model 922,112 of the
5,197,572 bytes the hooks see are parameter storages and 4,096 are the causal
mask buffer. Charging those to activations would double-count the parameter
column, which is exactly the mistake this column exists to avoid, so the meter
is given the model's parameters and buffers and skips their storages.

Where CUDA is present, the allocator peak is reported **as well**, labelled
separately. It is the larger number and it is the one that decides whether a run
OOMs, because it includes the transient buffers that the stash does not.

What is predicted
-----------------
:func:`analytic_activation_bytes` counts the same thing from the architecture,
term by term, with every term named. It is not a fit and it has no free
parameters: it enumerates what this implementation's forward pass saves.

The two are reported side by side and their ratio is part of the result. On this
implementation the ratio is **1.00000**, byte for byte, across every
architecture variant in ``tests/test_activation_memory.py`` -- 1, 3 and 4
layers, pre-LN and post-LN, all three activations, tied and untied heads,
learned and sinusoidal positions. That is not luck and it is not a fit; the
count was derived by reading the operators and then corrected four times by the
measurement, and each correction is a fact about autograd worth knowing:

* the tanh GELU leaves **four** extra ``4C``-wide tensors in the graph, not one,
  because it is written out of primitive operators;
* ``relu`` leaves **none**, because it saves its output and its output is
  already charged as the next projection's input;
* the cross entropy saves **one** ``tokens x vocab`` tensor, not two, because
  the log-softmax backward is written in terms of its own output;
* ``masked_fill`` keeps the **inverted** causal mask, one byte per element and
  quadratic in the sequence length like the probabilities are.

A term-by-term count that is exact is worth more than an approximate one,
because it can be evaluated at a shape nobody has run: it says how much
activation memory a configuration will need before the box is rented. Where the
count would be wrong it refuses to produce a number at all, rather than being
quietly wrong; see :func:`analytic_activation_bytes`.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch

from transformer_internals.config import GPTConfig

__all__ = [
    "GELU_TANH_SAVED_TENSORS",
    "ActivationMeter",
    "ActivationReport",
    "analytic_activation_bytes",
    "measure_activation_bytes",
]


#: How many extra ``4 * n_embd``-wide tensors the tanh GELU leaves in the graph.
#:
#: ``0.5 * x * (1 + tanh(k * (x + 0.044715 * x^3)))``, walked operator by
#: operator, saves four of them:
#:
#: 1. ``x`` itself, saved by ``pow`` for ``d/dx x^3 = 3x^2``.
#: 2. ``tanh``'s output, because ``d/dx tanh = 1 - tanh^2`` is written in terms
#:    of the output rather than the input.
#: 3. ``0.5 * x``, saved as the left operand of the final multiply.
#: 4. ``1 + tanh(...)``, saved as its right operand.
#:
#: Every scalar multiply and add in between saves nothing, because the
#: derivative of ``a * x`` with respect to ``x`` is the constant ``a``. This is
#: the single largest term in a GPT-2 block's activation memory and it is a
#: property of how the activation is *written*: ``F.gelu`` saves one tensor,
#: because it is one fused operator with a hand-written backward.
GELU_TANH_SAVED_TENSORS = 4


class ActivationMeter:
    """Counts the distinct storage bytes the autograd graph is holding.

    Use as a context manager around a forward pass. ``peak_bytes`` is the high
    water mark of live saved-tensor storage; ``stash_bytes()`` is what is live
    right now, which is the interesting reading immediately after the forward
    pass and before the backward has freed anything.

    The pack hook returns the tensor unchanged, so nothing about the run is
    altered: this measures the graph, it does not perturb it.

    Args:
        device: Device to read allocator statistics from. CUDA statistics are
            collected when it is a CUDA device and skipped everywhere else.
        exclude: Tensors whose storages are *not* activations. Pass the model's
            parameters and buffers: the graph saves weights too, and counting
            them here would double-count the parameter column.

    Example:
        >>> meter = ActivationMeter(exclude=model.parameters())
        >>> with meter:
        ...     loss = model(x, targets=y)["loss"]
        >>> stash = meter.stash_bytes()   # everything the backward will need
        >>> loss.backward()
    """

    def __init__(
        self,
        device: torch.device | str | None = None,
        exclude: Iterable[torch.Tensor] | None = None,
    ) -> None:
        self.device = torch.device(device) if device is not None else None
        self._excluded: set[int] = set()
        self.excluded_bytes = 0
        for t in exclude or ():
            try:
                storage = t.untyped_storage()
            except (RuntimeError, NotImplementedError, AttributeError):
                continue
            if storage.data_ptr() not in self._excluded:
                self._excluded.add(storage.data_ptr())
                self.excluded_bytes += storage.nbytes()
        # storage data_ptr -> [bytes, how many saved tensors reference it]
        self._live: dict[int, list[int]] = {}
        self._current = 0
        self.peak_bytes = 0
        self.tensors_saved = 0
        self.cuda_peak_allocated_bytes: int | None = None
        self._cuda_baseline: int | None = None
        self._ctx: Any = None

    # -- the hooks ---------------------------------------------------------

    def _pack(self, tensor: torch.Tensor) -> torch.Tensor:
        try:
            storage = tensor.untyped_storage()
            key = storage.data_ptr()
            nbytes = storage.nbytes()
        except (RuntimeError, NotImplementedError):
            # Sparse, meta or otherwise storage-less tensors. Counting nothing
            # is the honest answer; raising here would break the run being
            # measured, which is never worth it.
            return tensor
        if nbytes == 0:
            return tensor
        if key in self._excluded:
            # A weight or a buffer the graph kept for its own backward. Real
            # memory, already counted in the parameter column.
            return tensor
        self.tensors_saved += 1
        entry = self._live.get(key)
        if entry is None:
            self._live[key] = [nbytes, 1]
            self._current += nbytes
            self.peak_bytes = max(self.peak_bytes, self._current)
        else:
            entry[1] += 1
        # Fires when the graph drops this tensor, which is what makes the
        # high-water mark a mark of *live* memory rather than of cumulative
        # traffic.
        weakref.finalize(tensor, self._release, key)
        return tensor

    def _release(self, key: int) -> None:
        entry = self._live.get(key)
        if entry is None:
            return
        entry[1] -= 1
        if entry[1] <= 0:
            self._current -= entry[0]
            del self._live[key]

    @staticmethod
    def _unpack(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> ActivationMeter:
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            self._cuda_baseline = int(torch.cuda.memory_allocated(self.device))
        self._ctx = torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._ctx.__exit__(*exc)
        self._ctx = None
        if self.device is not None and self.device.type == "cuda":
            peak = int(torch.cuda.max_memory_allocated(self.device))
            self.cuda_peak_allocated_bytes = peak - (self._cuda_baseline or 0)

    # -- readings ----------------------------------------------------------

    def stash_bytes(self) -> int:
        """Distinct storage bytes the graph is holding at this instant."""
        return self._current

    def snapshot(self) -> dict[str, Any]:
        """The reading at this instant, before the backward pass frees anything.

        Taken as a snapshot rather than read off the meter later, because the
        backward pass releases the graph and a reading taken afterwards would
        report almost nothing.
        """
        return {
            "activation_bytes": self._current,
            "distinct_storages": len(self._live),
            "saved_tensor_count": self.tensors_saved,
        }

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "saved_for_backward_peak_bytes": self.peak_bytes,
            "saved_for_backward_live_bytes": self._current,
            "distinct_storages": len(self._live),
            "saved_tensor_count": self.tensors_saved,
            "excluded_parameter_and_buffer_bytes": self.excluded_bytes,
            "method": (
                "torch.autograd.graph.saved_tensors_hooks; distinct storages, "
                "counted once each, parameters and buffers excluded"
            ),
        }
        if self.cuda_peak_allocated_bytes is not None:
            payload["cuda_peak_allocated_bytes"] = self.cuda_peak_allocated_bytes
            payload["cuda_peak_note"] = (
                "torch.cuda.max_memory_allocated minus the baseline at entry. "
                "Larger than the stash, because it includes transient workspaces "
                "and any gradient buffers allocated inside the window. This is "
                "the number that decides whether a step OOMs."
            )
        return payload


@dataclass
class ActivationReport:
    """A measured activation figure beside the analytic count of the same thing."""

    measured_bytes: int
    analytic_bytes: float
    batch: int
    seq: int
    dtype_bytes: int
    terms: dict[str, float] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def ratio_measured_to_analytic(self) -> float:
        return self.measured_bytes / self.analytic_bytes if self.analytic_bytes else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "measured_bytes": self.measured_bytes,
            "analytic_bytes": self.analytic_bytes,
            "ratio_measured_to_analytic": self.ratio_measured_to_analytic,
            "bytes_per_token_measured": self.measured_bytes / max(self.batch * self.seq, 1),
            "batch": self.batch,
            "seq": self.seq,
            "dtype_bytes": self.dtype_bytes,
            "analytic_terms": self.terms,
            **self.detail,
        }


def analytic_activation_bytes(
    cfg: GPTConfig,
    batch: int,
    seq: int,
    dtype_bytes: int = 4,
    with_loss: bool = True,
) -> dict[str, float]:
    """Bytes this implementation's forward pass saves for backward, term by term.

    Counted from the operators in :mod:`transformer_internals.model`, not from a
    published formula, because a published formula describes a different
    implementation's operator choices. It is comparable with Korthikanti et al.
    (arXiv:2205.05198) in shape but not term for term.

    Every term is an operator's saved tensor, named after the operator. There
    are no fitted constants. Three of the terms are worth pointing at because
    they are the ones a formula written from a paper gets wrong:

    * ``mlp_activation_intermediates`` is four ``4C``-wide tensors, not one,
      because the tanh GELU here is written out of primitive operators rather
      than being one fused kernel. See :data:`GELU_TANH_SAVED_TENSORS`. Under
      ``activation="relu"`` the same term is zero, because ReLU saves its output
      and its output is already charged as the down projection's input.
    * The cross entropy saves **one** ``tokens x vocab`` tensor, the log-softmax
      output, not two. The raw logits are not saved for backward, because the
      log-softmax backward is written in terms of its own output.
    * There is no separate embedding-output term: with dropout at zero, block 0's
      first LayerNorm saves the very tensor the embedding produced, so charging
      both would count one storage twice.

    Args:
        cfg: Architecture.
        batch / seq: Step shape.
        dtype_bytes: Element size of the activations. 4 for fp32, 2 under bf16
            autocast, except that LayerNorm's saved mean and inverse standard
            deviation stay fp32 and the token indices stay int64, and both are
            counted at their own widths.
        with_loss: Include the loss terms. The log-softmax output is a
            ``tokens x vocab`` tensor, which for GPT-2 at vocab 50257 is larger
            than an entire block's activations, so an accounting that leaves it
            out is describing a model that never computes a loss.

    Returns:
        A dict of named terms plus ``total``. Every value is bytes.

    Raises:
        ValueError: On a configuration whose saved set this count does not
            enumerate. Refusing is deliberate: an analytic number that is
            quietly wrong for some configurations is worse than no analytic
            number, because it would be compared against a measurement and the
            gap would be read as a property of the hardware.
    """
    if cfg.dropout > 0:
        raise ValueError(
            "analytic_activation_bytes covers dropout=0 only; a non-zero dropout "
            "saves a boolean mask per dropout site and this count does not "
            "enumerate them. Every measurement in this repository uses dropout=0."
        )
    if cfg.use_sdpa:
        raise ValueError(
            "analytic_activation_bytes covers the explicit attention path only. "
            "scaled_dot_product_attention is a fused kernel that saves its own "
            "chosen set, which is the whole point of it and is not enumerable "
            "from the model source."
        )
    if cfg.n_kv_head is not None:
        raise ValueError(
            "analytic_activation_bytes covers multi-head attention only; GQA and "
            "MQA insert a repeat_interleave whose output is an extra saved "
            "tensor of a different width."
        )

    n = float(batch * seq)  # tokens in the step
    c = float(cfg.n_embd)
    v = float(cfg.vocab_size)
    layers = float(cfg.n_layer)
    heads = float(cfg.n_head)
    e = float(dtype_bytes)
    act = n * c * e  # one full-width activation
    stats = 2.0 * n * 4.0  # LayerNorm mean and rstd, fp32 whatever e is

    gelu_tensors = {
        "gelu_tanh": float(GELU_TANH_SAVED_TENSORS),
        # F.gelu is one fused operator with a hand-written backward: it saves
        # its input and nothing else.
        "gelu_exact": 1.0,
        # ReLU saves its output, and its output is the very tensor the down
        # projection saves as its input, so it costs nothing beyond that term.
        "relu": 0.0,
    }[cfg.activation]

    per_layer = {
        "residual_into_the_first_sublayer": act,
        "layernorm_1_stats": stats,
        "qkv_projection_input": act,
        # q and the transposed k, both made contiguous for the batched score
        # matmul, which saves both operands.
        "score_matmul_operands_q_and_k": 2.0 * act,
        # The softmax output, which both the softmax backward and the
        # attn @ V backward need. Quadratic in the sequence length, which is why
        # long context is a memory problem before it is a compute problem.
        "attention_probabilities": float(batch) * heads * float(seq) * float(seq) * e,
        # masked_fill keeps the inverted mask for its backward. One byte per
        # element, one per layer, and it is quadratic in the sequence length like
        # the probabilities are.
        "inverted_causal_mask": float(seq) * float(seq),
        "value_matmul_operand": act,
        "attention_output_projection_input": act,
        "residual_into_the_second_sublayer": act,
        "layernorm_2_stats": stats,
        "mlp_up_projection_input": act,
        # Four times the width of the residual stream, and saved several times
        # over by the way the activation is written.
        "mlp_activation_intermediates": gelu_tensors * 4.0 * act,
        "mlp_down_projection_input": 4.0 * act,
    }
    per_layer_total = sum(per_layer.values())

    root = {
        "final_layernorm_input": act,
        "final_layernorm_stats": stats,
        "lm_head_input": act,
        # int64 indices the embedding and the loss keep.
        "token_ids": n * 8.0,
    }
    if cfg.pos_embedding == "learned":
        # nn.Embedding saves its indices, because the weight gradient is a
        # scatter back through them. A sinusoidal table is a fixed buffer with
        # no gradient, so indexing it saves nothing.
        root["position_ids"] = float(seq) * 8.0
    if with_loss:
        root["cross_entropy_log_softmax_output"] = n * v * e
        root["cross_entropy_target_ids"] = n * 8.0
        # nll_loss keeps a scalar total weight for the mean reduction.
        root["cross_entropy_total_weight"] = 4.0

    out: dict[str, float] = {f"per_layer.{k}": val for k, val in per_layer.items()}
    out["per_layer_subtotal"] = per_layer_total
    out["all_layers"] = layers * per_layer_total
    out.update({f"root.{k}": val for k, val in root.items()})
    out["root_subtotal"] = sum(root.values())
    out["total"] = out["all_layers"] + out["root_subtotal"]
    out["bytes_per_token"] = out["total"] / max(n, 1.0)
    return out


def measure_activation_bytes(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor | None = None,
    cfg: GPTConfig | None = None,
    dtype_bytes: int = 4,
) -> ActivationReport:
    """Run one forward pass under an :class:`ActivationMeter` and predict the same.

    The backward pass is run afterwards so the graph is released and the caller
    is not left holding a step's worth of activations. The reading is taken
    before it, which is where the stash is at its largest.

    Args:
        model: Anything callable as ``model(inputs, targets=targets)`` returning
            a dict with ``loss``.
        inputs / targets: One batch.
        cfg: Architecture for the analytic count. Taken from ``model.config``
            when omitted.
        dtype_bytes: Element size to predict with; 4 unless the forward ran
            under bf16 autocast.

    Returns:
        An :class:`ActivationReport` holding both numbers and their ratio.
    """
    cfg = cfg if cfg is not None else model.config
    batch, seq = int(inputs.shape[0]), int(inputs.shape[1])
    meter = ActivationMeter(
        device=inputs.device,
        exclude=[*model.parameters(), *model.buffers()],
    )
    with meter:
        out = model(inputs, targets=targets)
        loss = out["loss"] if targets is not None else out["logits"].float().mean()
    # Read before the backward pass, which is where the stash is at its largest
    # and after which almost nothing is left to read.
    snapshot = meter.snapshot()
    loss.backward()

    terms = analytic_activation_bytes(cfg, batch, seq, dtype_bytes, with_loss=targets is not None)
    detail = meter.to_dict()
    detail.update(snapshot)
    return ActivationReport(
        measured_bytes=snapshot["activation_bytes"],
        analytic_bytes=terms["total"],
        batch=batch,
        seq=seq,
        dtype_bytes=dtype_bytes,
        terms=terms,
        detail=detail,
    )
