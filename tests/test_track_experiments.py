import mlflow
import pytest

from track_experiments import CONFIGS, train_and_log


def test_configs_match_the_spec():
    assert list(CONFIGS) == [
        "linreg_baseline", "rf_100", "rf_300_depth10", "gb_100_lr01", "gb_200_lr005",
    ]


def test_train_and_log_records_params_metrics_and_model(local_mlflow, sample_splits):
    run_id, metrics = train_and_log("linreg_baseline", *sample_splits)

    run = mlflow.get_run(run_id)
    assert run.info.run_name == "linreg_baseline"
    assert run.data.params["model_type"] == "LinearRegression"
    assert run.data.params["target_transform"] == "log1p/expm1"
    assert set(metrics) == {"rmse", "mae", "r2"}
    assert run.data.metrics["rmse"] == pytest.approx(metrics["rmse"])

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    assert (model.predict(sample_splits[1]) > 0).all()
