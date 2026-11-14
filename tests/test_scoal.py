import numpy as np
import pytest

from pyscoal import MSCOAL, SCOAL


def _linear_target(row_features, col_features):
    return np.array(
        [
            [
                row,
                col,
                3 * row_features[row, 0] - 2 * col_features[col, 0] + 1,
            ]
            for row in range(row_features.shape[0])
            for col in range(col_features.shape[0])
        ],
        dtype=float,
    )


@pytest.mark.parametrize('matrix', ['sparse', 'dense'])
def test_scoal_predicts_known_values_in_each_matrix_mode(matrix):
    row_features = np.array([[0.0], [1.0], [2.0], [3.0]])
    col_features = np.array([[0.0], [1.0], [2.0]])
    target = _linear_target(row_features, col_features)
    model = SCOAL(
        n_row_clusters=1,
        n_col_clusters=1,
        max_iter=0,
        random_state=0,
        matrix=matrix,
    )

    model.fit(target.copy(), row_features, col_features)
    prediction_target = target[[5, 0, 8, 2, 11, 3], :2]
    predictions = model.predict(
        prediction_target,
        row_features,
        col_features,
    )
    score = model.score(target, row_features, col_features)
    expected = (
        3 * row_features[prediction_target[:, 0].astype(int), 0]
        - 2 * col_features[prediction_target[:, 1].astype(int), 0]
        + 1
    )

    np.testing.assert_allclose(predictions, expected)
    assert score == pytest.approx(0)


def test_mscoal_fits_and_predicts_sparse_triplet_target():
    row_features = np.array([[0.0], [1.0], [2.0], [3.0]])
    col_features = np.array([[0.0], [1.0], [2.0]])
    target = _linear_target(row_features, col_features)
    target[:, 2] += target[:, 0] * target[:, 1]
    model = MSCOAL(
        max_split=0,
        validation_size=0.25,
        random_state=0,
    )

    model.fit(target.copy(), row_features, col_features)
    predictions = model.predict(target[:, :2], row_features, col_features)
    score = model.score(target, row_features, col_features)

    assert predictions.shape == (target.shape[0],)
    assert np.isfinite(predictions).all()
    assert np.isfinite(score)
