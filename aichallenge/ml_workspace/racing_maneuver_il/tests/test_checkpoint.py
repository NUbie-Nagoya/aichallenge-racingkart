import torch

from racing_maneuver_il.checkpoint import load_checkpoint, save_checkpoint
from racing_maneuver_il.model import TemporalPolicy


def test_checkpoint_round_trip_includes_contract_and_normalizer(tmp_path):
    model = TemporalPolicy(hidden_size=8)
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "best.pt"
    metadata = {
        "normalizer": {"x": 1},
        "split_manifest": {"seed": 4},
        "dataset_provenance": {"sha": "a"},
    }
    save_checkpoint(
        path,
        model,
        optimizer,
        epoch=2,
        metrics={"loss": 0.3},
        metadata=metadata,
        model_config={"hidden_size": 8},
    )
    restored, payload = load_checkpoint(path)
    assert isinstance(restored, TemporalPolicy) and payload["epoch"] == 2
    assert payload["metadata"] == metadata and payload["metrics"]["loss"] == 0.3
