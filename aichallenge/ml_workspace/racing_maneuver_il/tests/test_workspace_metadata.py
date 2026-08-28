from pathlib import Path

import tomllib
import yaml

ROOT = Path(__file__).parents[1]


def test_configs_and_project_metadata_match_schema():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["project"]["requires-python"] == ">=3.10"
    base = yaml.safe_load((ROOT / "config/base.yaml").read_text())
    assert (
        base["lidar"]["rays"] == 360
        and base["history"]["length"] == 10
        and base["history"]["rate_hz"] == 20.0
    )
    assert len(base["features"]["auxiliary_order"]) == 11
    train = yaml.safe_load((ROOT / "config/train_baseline.yaml").read_text())
    assert train["action_limits"] == {
        "low": [-1.00, -3.0],
        "high": [1.00, 3.0],
    }
