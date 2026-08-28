import numpy as np
import torch

from racing_maneuver_il.model import ExportPolicy, TemporalPolicy
from racing_maneuver_il.normalization import Normalizer


def inputs(batch=3, time=10):
    return torch.rand(batch, time, 360), torch.rand(batch, time, 11)


def test_model_accepts_batch_time_inputs_and_returns_two_actions():
    model = TemporalPolicy(hidden_size=32)
    lidar, aux = inputs()
    actions, hidden = model(lidar, aux)
    assert actions.shape == (3, 2) and hidden.shape == (1, 3, 32)


def test_hidden_state_round_trip_is_causal_and_shape_stable():
    torch.manual_seed(3)
    model = TemporalPolicy(hidden_size=16).eval()
    lidar, aux = inputs(2, 5)
    whole, _ = model(lidar, aux)
    _, hidden = model(lidar[:, :3], aux[:, :3])
    tail, hidden2 = model(lidar[:, 3:], aux[:, 3:], hidden)
    np.testing.assert_allclose(
        whole.detach().numpy(), tail.detach().numpy(), rtol=1e-5, atol=1e-6
    )
    assert hidden2.shape == hidden.shape


def test_model_rejects_wrong_shapes():
    model = TemporalPolicy()
    try:
        model(torch.rand(2, 360), torch.rand(2, 11))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong rank accepted")


def test_model_has_no_nan_and_bounded_normalized_output():
    action, _ = TemporalPolicy()(torch.full((2, 10, 360), 30.0), torch.zeros(2, 10, 11))
    assert torch.isfinite(action).all() and action.abs().max() <= 1


def test_export_wrapper_owns_normalization_and_physical_scaling():
    norm = Normalizer.fit(np.ones((2, 360)) * 2, np.ones((2, 11)) * 4, np.array([0, 1]))
    base = TemporalPolicy().eval()
    wrapper = ExportPolicy.from_normalizer(base, norm, (-0.5, -3), (0.5, 2)).eval()
    action = wrapper(torch.ones(1, 10, 360) * 2, torch.ones(1, 10, 11) * 4)
    assert action.shape == (1, 2)
    assert -0.5 <= action[0, 0] <= 0.5 and -3 <= action[0, 1] <= 2
    assert "lidar_mean" in dict(wrapper.named_buffers())
