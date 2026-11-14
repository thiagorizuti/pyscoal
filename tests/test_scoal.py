import numpy as np

from pyscoal import SCOAL


def test_scoal_fit_predict_and_score_on_small_dense_target():
    row_features = np.array([[0.0], [1.0], [2.0], [3.0]])
    col_features = np.array([[0.0], [1.0], [2.0]])
    target = np.array(
        [
            [row, col, row_features[row, 0] + 2 * col_features[col, 0] + row * col]
            for row in range(row_features.shape[0])
            for col in range(col_features.shape[0])
        ],
        dtype=float,
    )
    model = SCOAL(
        n_row_clusters=1,
        n_col_clusters=1,
        max_iter=1,
        random_state=0,
    )

    model.fit(target.copy(), row_features, col_features)
    predictions = model.predict(target[:, :2], row_features, col_features)
    score = model.score(target, row_features, col_features)

    assert predictions.shape == (target.shape[0],)
    assert np.isfinite(predictions).all()
    assert np.isfinite(score)
