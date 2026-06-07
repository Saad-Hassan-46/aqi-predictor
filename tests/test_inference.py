import numpy as np

from pipelines.inference_pipeline import run_inference


class DummyModel:
    def predict(self, X):
        assert X.shape == (1, 2)
        return np.array([[101.234, 130.5, 155.9]])


def test_run_inference_formats_three_horizons():
    predictions = run_inference(DummyModel(), np.array([[1.0, 2.0]]), {})

    assert predictions == {
        "aqi_pred_24h": 101.23,
        "aqi_pred_48h": 130.5,
        "aqi_pred_72h": 155.9,
    }
