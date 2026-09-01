"""The CUDA decisions, exercised on a machine that has no CUDA.

There is no GPU here, so the CUDA *code* cannot be run. What can be run, and is
run below, is every decision the CUDA path makes: which backend to open, which
device each rank gets, whether a world size is possible on a given machine, and
what happens when the answer is no. Those decisions are pure functions of a
:class:`~transformer_internals.hardware.Capabilities`, and
:meth:`Capabilities.stub` fabricates an eight-GPU node to feed them.

This is the difference between "the CUDA branch is untested" and "the CUDA
branch's logic is tested and only its two torch calls are not".
"""

from __future__ import annotations

import pytest

from transformer_internals.hardware import (
    Capabilities,
    HardwareError,
    check_placement,
    describe,
    environment_payload,
    select_backend,
    select_device,
)

CPU = Capabilities(cuda_available=False, device_count=0, nccl_available=False)
GPU8 = Capabilities.stub(device_count=8)
GPU2 = Capabilities.stub(device_count=2)


# --------------------------------------------------------------------------- #
# backend selection
# --------------------------------------------------------------------------- #


def test_auto_picks_nccl_on_a_cuda_box_and_gloo_otherwise():
    assert select_backend(GPU8) == "nccl"
    assert select_backend(GPU8, "auto") == "nccl"
    assert select_backend(CPU) == "gloo"
    assert select_backend(CPU, "auto") == "gloo"


def test_this_machine_selects_gloo():
    """Whatever this machine is, the detected capabilities must resolve."""
    caps = Capabilities.detect()
    backend = select_backend(caps)
    assert backend in ("nccl", "gloo")
    # No CUDA here, so it had better be gloo. If this ever fires on a GPU box
    # that is a signal, not a failure: rerun the suite and read the new number.
    if not caps.cuda_available:
        assert backend == "gloo"


def test_a_torch_build_without_nccl_refuses_rather_than_downgrading():
    """Silently falling back to gloo would produce fake collective numbers."""
    cuda_no_nccl = Capabilities(cuda_available=True, device_count=8, nccl_available=False)
    with pytest.raises(HardwareError, match="is_nccl_available"):
        select_backend(cuda_no_nccl, "nccl")


def test_requesting_nccl_without_cuda_says_what_to_do():
    with pytest.raises(HardwareError, match="no CUDA device"):
        select_backend(CPU, "nccl")


def test_unknown_backend_is_named_in_the_error():
    with pytest.raises(HardwareError, match="mpi"):
        select_backend(GPU8, "mpi")


def test_gloo_can_be_forced_on_a_cuda_box():
    """Running the equivalence proofs on both backends on one machine is useful."""
    assert select_backend(GPU8, "gloo") == "gloo"


# --------------------------------------------------------------------------- #
# device placement
# --------------------------------------------------------------------------- #


def test_one_gpu_per_rank_in_order():
    assert check_placement(GPU8, 8, "nccl") == [f"cuda:{i}" for i in range(8)]
    assert select_device(GPU8, 3, "nccl") == "cuda:3"


def test_gloo_ranks_all_land_on_cpu():
    assert check_placement(CPU, 4, "gloo") == ["cpu"] * 4
    assert select_device(GPU8, 3, "gloo") == "cpu"


def test_more_ranks_than_gpus_is_refused_by_default():
    with pytest.raises(HardwareError, match="exceeds the 2 visible"):
        check_placement(GPU2, 4, "nccl")


def test_oversubscription_is_allowed_when_asked_for_and_wraps_round():
    devices = check_placement(GPU2, 4, "nccl", allow_oversubscribe=True)
    assert devices == ["cuda:0", "cuda:1", "cuda:0", "cuda:1"]


def test_nccl_with_no_visible_devices_mentions_cuda_visible_devices():
    hidden = Capabilities(cuda_available=True, device_count=0, nccl_available=True)
    with pytest.raises(HardwareError, match="CUDA_VISIBLE_DEVICES"):
        check_placement(hidden, 2, "nccl")


def test_world_size_zero_is_refused():
    with pytest.raises(HardwareError, match="at least 1"):
        check_placement(GPU8, 0, "nccl")


# --------------------------------------------------------------------------- #
# capabilities
# --------------------------------------------------------------------------- #


def test_bf16_needs_every_device_to_be_ampere_or_newer():
    assert Capabilities.stub(device_count=8, capability=(8, 0)).bf16_supported
    assert Capabilities.stub(device_count=8, capability=(9, 0)).bf16_supported
    # Volta: fp16 tensor cores, no bf16.
    assert not Capabilities.stub(device_count=8, capability=(7, 0)).bf16_supported
    assert not CPU.bf16_supported


def test_a_mixed_node_does_not_claim_bf16():
    """One pre-Ampere card in the node means the ranks would disagree numerically."""
    mixed = Capabilities(
        cuda_available=True,
        device_count=2,
        nccl_available=True,
        compute_capabilities=((8, 0), (7, 0)),
    )
    assert not mixed.bf16_supported


def test_detect_never_raises_and_reports_this_machine():
    caps = Capabilities.detect()
    assert caps.source == "detected"
    assert caps.torch_version
    assert caps.accelerator in ("cuda", "mps", "cpu")


def test_a_stub_is_labelled_as_fabricated_everywhere_it_appears():
    """A fabricated machine must never be mistaken for a measured one."""
    caps = Capabilities.stub(device_count=8)
    assert caps.source == "stub"
    assert caps.to_dict()["source"] == "stub"
    assert "STUBBED" in describe(caps)
    assert environment_payload(caps, topology=False)["source"] == "stub"


def test_describe_lists_every_device():
    text = describe(Capabilities.stub(device_count=4, name="NVIDIA H100 80GB HBM3"))
    assert text.count("NVIDIA H100 80GB HBM3") == 4
    assert "sm_80" in text


def test_environment_payload_is_json_safe():
    import json

    payload = environment_payload(Capabilities.stub(device_count=8), topology=False)
    json.dumps(payload)
    assert payload["device_count"] == 8
    assert payload["bf16_supported"] is True
