import json

import numpy as np
import torch

from racing_maneuver_il.export_torchscript import export_torchscript, load_metadata
from racing_maneuver_il.model import ExportPolicy, TemporalPolicy
from racing_maneuver_il.normalization import Normalizer


def test_eager_and_torchscript_raw_input_parity_with_metadata(tmp_path):
    torch.manual_seed(7)
    norm = Normalizer.fit(
        np.random.default_rng(1).normal(size=(4, 360)),
        np.random.default_rng(2).normal(size=(4, 9)),
        np.arange(4),
    )
    wrapper = ExportPolicy.from_normalizer(
        TemporalPolicy(hidden_size=16).eval(), norm, (-0.45, -4), (0.45, 2)
    ).eval()
    lidar = torch.rand(1, 10, 360) * 30
    aux = torch.rand(1, 10, 9)
    artifact = tmp_path / "policy.ts"
    metadata = {
        "dataset_provenance": {"sha256": "abc"},
        "split_provenance": {"seed": 3},
    }
    export_torchscript(wrapper, artifact, metadata)
    loaded = torch.jit.load(str(artifact))
    np.testing.assert_allclose(
        wrapper(lidar, aux).detach().numpy(),
        loaded(lidar, aux).detach().numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
    embedded = load_metadata(artifact)
    assert embedded["schema_version"] == 3 and embedded["history_length"] == 10
    assert embedded["model_max_range_m"] == 30.0
    assert embedded["torch_version"] == torch.__version__
    assert json.loads((tmp_path / "policy.ts.metadata.json").read_text()) == embedded
    assert embedded["feature_ordering"][0] == "canonical_lidar_ranges_m"
    assert embedded["output_units"] == ["rad", "m/s"]
