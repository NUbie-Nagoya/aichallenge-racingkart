# Racing Maneuver Imitation Learning

Standalone Python 3.10+ temporal behavioral-cloning pipeline for manual multi-car racing demonstrations. It does not import ROS controllers, `tiny_lidar_net`, or prior pose-BC code. The workspace includes a finalized MCAP recorder, bag inventory, source-stamped bag extraction, grouped training, evaluation, and verified TorchScript export.

## Contract

Each frame contains 360 raw-metre LiDAR rays on the deployed forward-centred `[-89.5°, +89.5°]` grid, then 11 ordered auxiliary values: ego longitudinal velocity, measured steering, measured longitudinal acceleration, previous **safe** steering/longitudinal commands, and opponent present/body-frame x/y/vx/vy/age. A sample is 10 causal frames at 20 Hz, oldest first. Targets are physical `[steering_tire_angle_rad, longitudinal_acceleration_mps2]`.

`canonicalize_scan` interpolates using physical angles and deterministically maps NaN/infinite returns to the configured maximum. `latest_at_or_before` and `align_causal` never select future data and reject stale sources. Opponent velocity uses only bounded prior V2X finite differences.

## Install and test

Create the environment with the deployment Python's system packages so `pip` preserves its compatible PyTorch rather than silently selecting a newer TorchScript producer:

```bash
python3.10 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

Only NumPy, PyTorch, PyYAML, rosbags, and pytest are required.

## Processed dataset

The immutable NPZ keys are:

- `lidar`: `[N, 360]` float raw metres
- `aux`: `[N, 11]` float features in `schema.AUX_FEATURE_NAMES` order
- `targets`: `[N, 2]` physical actions
- `recording_ids`, `episode_ids`, `maneuver_classes`: `[N]` strings
- optional `source_timestamps_s`, `source_ages_s`: audit arrays

The extractor deserializes each accepted bag with `rosbags`, uses ROS source stamps and latest-at-or-before causal joins, verifies manual control mode (default `1`, verified for `teleop_manager_node`), canonicalizes LiDAR, derives body-frame opponent features, and writes the dataset/report atomically:

```bash
racing-maneuver-extract rawdata/RECORDING data/processed/RECORDING.npz data/processed/RECORDING.report.json
```

Use repeated `--human-control-mode MODE` only after verifying the deployed control-mode contract. The extractor rejects rows with stale/missing required inputs and refuses an output with zero confirmed manual rows. Merge complete recording datasets without losing recording/episode group identity:

```bash
racing-maneuver-merge data/processed/recording-*.npz \
  --output data/processed/maneuvers.npz \
  --report data/processed/maneuvers.report.json
```

The merger rejects duplicate recording IDs, schema/shape mismatches, and inconsistent audit arrays.

Groups are `recording_id::episode_id`; complete groups belong to exactly one locked split. Histories cannot cross groups. Normalization statistics are fit from train frames only; targets stay physical.

## Train, evaluate, export

```bash
racing-maneuver-train config/train_baseline.yaml
racing-maneuver-evaluate runs/baseline/best.pt data/processed/maneuvers.npz runs/baseline/evaluation.json
racing-maneuver-export runs/baseline/best.pt runs/baseline/policy.ts
```

Training uses weighted Huber steering/acceleration losses plus action-change loss and writes `best.pt`, `last.pt`, resolved config, normalizer, split manifest, curves, validation metrics, and provenance. Evaluation reports teacher-forced and runtime-oriented autoregressive metrics, including maneuver, opponent, and closing-speed slices. Promotion decisions should use autoregressive results.

### Train the included MPC-expert smoke dataset on a remote machine

The branch includes a processed, causally aligned MPC-expert dataset at:

```text
data/processed/mpc/20260827-114404-d6cb0e3d.npz
```

It contains 49,742 frames partitioned into 56 causal lap groups. Clone the feature
branch and run the dedicated within-session experiment without downloading the raw
MCAP bag:

```bash
git clone --branch feat/racing_maneuver_il \
  https://github.com/NUbie-Nagoya/aichallenge-racingkart.git
cd aichallenge-racingkart/aichallenge/ml_workspace/racing_maneuver_il

python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python -m racing_maneuver_il.train config/train_mpc_within_session.yaml
```

Outputs are written to a new timestamp-and-UUID directory under
`runs/mpc-within-session-20260827-114404/` for every invocation. This split keeps
complete laps disjoint, but every lap comes from one recording, scenario, seed, and
MPC configuration. Treat its validation/test metrics only as **within-session MPC
imitation** metrics—not cross-scenario or deployment-quality evidence.

The TorchScript wrapper accepts **raw** `[B, 10, 360]` LiDAR and `[B, 10, 11]` auxiliary tensors. It owns normalization and maps bounded network outputs to serialized physical limits. Export writes both `policy.ts` and `policy.ts.metadata.json`; deploy both. Metadata includes the producing PyTorch version, schema/geometry/history/order/units/limits/normalizer, opponent preprocessing, and dataset/split provenance. **Train/export with the same PyTorch major.minor used by the Autoware runtime** (verify with `python3 -c 'import torch; print(torch.__version__)'` inside that runtime). The controller reads the sidecar and refuses incompatible artifacts before calling `torch.jit.load`.

## Scan contract inspection

Capture a real AWSIM `LaserScan` fixture and inspect it with `racing-maneuver-inspect-scan`. See [docs/scan_contract.md](docs/scan_contract.md). No source geometry in this repository is claimed as measured until that live capture is performed.
