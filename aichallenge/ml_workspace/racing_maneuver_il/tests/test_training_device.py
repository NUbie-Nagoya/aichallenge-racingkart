import pytest
import torch

from racing_maneuver_il.train import resolve_device


def test_resolve_device_cpu_is_explicit_cpu():
    device = resolve_device("cpu")

    assert device.type == "cpu"


def test_resolve_device_cuda_fails_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda")


def test_resolve_device_auto_uses_cpu_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_device("auto").type == "cpu"


def test_resolve_device_rejects_unknown_value():
    with pytest.raises(ValueError, match="device must be one of"):
        resolve_device("metal")
