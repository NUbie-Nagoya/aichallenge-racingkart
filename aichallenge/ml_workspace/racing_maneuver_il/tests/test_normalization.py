import numpy as np

from racing_maneuver_il.normalization import Normalizer


def test_normalizer_fits_only_training_frames():
    lidar = np.vstack([np.zeros((2, 360)), np.full((2, 360), 100)]).astype(np.float32)
    aux = np.vstack([np.zeros((2, 11)), np.full((2, 11), 100)]).astype(np.float32)
    norm = Normalizer.fit(lidar, aux, np.array([0, 1]))
    assert np.all(norm.lidar_mean == 0) and np.all(norm.aux_mean == 0)


def test_normalizer_round_trip_and_constant_columns_are_safe():
    lidar = np.ones((3, 360), np.float32)
    aux = np.ones((3, 11), np.float32)
    norm = Normalizer.fit(lidar, aux, np.arange(3))
    ln, an = norm.transform(lidar, aux)
    assert np.isfinite(ln).all() and np.isfinite(an).all()
    lr, ar = norm.inverse_transform(ln, an)
    np.testing.assert_allclose(lr, lidar)
    np.testing.assert_allclose(ar, aux)


def test_normalizer_serialization_preserves_statistics():
    norm = Normalizer.fit(
        np.arange(720).reshape(2, 360), np.arange(22).reshape(2, 11), np.array([0, 1])
    )
    restored = Normalizer.from_dict(norm.to_dict())
    np.testing.assert_allclose(restored.aux_mean, norm.aux_mean)
