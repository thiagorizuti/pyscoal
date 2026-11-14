import numpy as np

import pyscoal
from pyscoal.algorithms import MSCOAL, SCOAL
from pyscoal.data import make_dataset


def test_top_level_api_exports_subpackage_objects():
    assert pyscoal.SCOAL is SCOAL
    assert pyscoal.MSCOAL is MSCOAL
    assert pyscoal.make_dataset is make_dataset
    assert set(pyscoal.__all__) == {'SCOAL', 'MSCOAL', 'make_dataset'}


def test_make_dataset_returns_consistent_feature_and_target_shapes():
    target, row_features, col_features, row_clusters, col_clusters, coefs = (
        make_dataset(
            n_rows=4,
            n_cols=3,
            n_row_clusters=2,
            n_col_clusters=2,
            n_row_features=2,
            n_col_features=1,
            seed=7,
        )
    )

    assert target.shape == (12, 3)
    assert row_features.shape == (4, 2)
    assert col_features.shape == (3, 1)
    assert row_clusters.shape == (4,)
    assert col_clusters.shape == (3,)
    assert coefs.shape == (2, 2, 3)
    assert np.isfinite(target).all()


def test_make_dataset_does_not_change_numpy_global_random_state():
    initial_state = np.random.get_state()
    try:
        np.random.seed(123)
        expected_next_value = np.random.random()

        np.random.seed(123)
        make_dataset(n_rows=4, n_cols=3, seed=7)
        actual_next_value = np.random.random()
    finally:
        np.random.set_state(initial_state)

    assert actual_next_value == expected_next_value
