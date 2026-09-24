import mlflow
import pytest
from mlflow import MlflowClient

from registry import CHAMPION_ALIAS, MAX_MODEL_SIZE_MB, MODEL_NAME, pick_best, register_and_promote
from track_experiments import CONFIGS, train_and_log


def test_configs_are_the_five_planned_runs():
    assert list(CONFIGS) == [
        "linreg_baseline", "rf_100", "rf_300_depth10", "gb_100_lr01", "gb_200_lr005",
    ]


def test_train_and_log_records_params_metrics_and_model(local_mlflow, sample_splits):
    run_id, metrics = train_and_log("linreg_baseline", *sample_splits)

    run = mlflow.get_run(run_id)
    assert run.info.run_name == "linreg_baseline"
    assert run.data.params["model_type"] == "LinearRegression"
    assert run.data.params["target_transform"] == "log1p/expm1"
    assert set(metrics) == {"rmse", "mae", "r2", "model_size_mb"}
    assert run.data.metrics["rmse"] == pytest.approx(metrics["rmse"])
    assert 0 < run.data.metrics["model_size_mb"] < 5  # LinearRegression is tiny

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    assert (model.predict(sample_splits[1]) > 0).all()


@pytest.mark.parametrize("run_name", list(CONFIGS))
def test_every_config_can_be_logged_and_loaded_back(local_mlflow, sample_splits, run_name):
    # MLflow 3 saves sklearn models with skops, which refuses unfamiliar types
    # (e.g. the Tree objects inside RandomForest/GradientBoosting) unless trusted.
    run_id, _ = train_and_log(run_name, *sample_splits)
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    assert (model.predict(sample_splits[1]) > 0).all()


def test_pick_best_uses_lowest_rmse_within_size_budget():
    results = {
        "huge": {"rmse": 77.5, "mae": 43.4, "r2": 0.48, "model_size_mb": 326.0},
        "mid": {"rmse": 78.7, "mae": 43.8, "r2": 0.46, "model_size_mb": 39.3},
        "tiny": {"rmse": 83.5, "mae": 47.1, "r2": 0.40, "model_size_mb": 0.3},
    }
    assert MAX_MODEL_SIZE_MB == 100
    assert pick_best(results) == "mid"  # "huge" wins on RMSE but is over budget


def test_pick_best_prefers_rmse_when_mae_disagrees():
    results = {
        "a": {"rmse": 70.0, "mae": 45.0, "r2": 0.50, "model_size_mb": 1.0},
        "b": {"rmse": 80.0, "mae": 40.0, "r2": 0.45, "model_size_mb": 1.0},
    }
    assert pick_best(results) == "a"


def test_pick_best_refuses_when_no_model_fits_the_budget():
    with pytest.raises(ValueError, match="No model within"):
        pick_best({"huge": {"rmse": 70.0, "mae": 40.0, "r2": 0.5, "model_size_mb": 500.0}})


def test_register_and_promote_sets_champion_alias(local_mlflow, sample_splits):
    run_id, _ = train_and_log("linreg_baseline", *sample_splits)
    version = register_and_promote(run_id)

    champion = MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    assert champion.version == version
    assert champion.run_id == run_id

    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    assert (model.predict(sample_splits[1]) > 0).all()


def test_promoting_again_moves_the_alias(local_mlflow, sample_splits):
    first = register_and_promote(train_and_log("linreg_baseline", *sample_splits)[0])
    second = register_and_promote(train_and_log("linreg_baseline", *sample_splits)[0])
    assert int(second) == int(first) + 1
    assert MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS).version == second


def test_register_uses_the_runs_logged_model_not_a_fallback(local_mlflow, sample_splits, caplog):
    # MLflow 3 stores a run's model as its own LoggedModel (models:/m-...).
    # Registering "runs:/<id>/model" only works via a fallback that logs a warning.
    run_id, _ = train_and_log("linreg_baseline", *sample_splits)
    register_and_promote(run_id)
    assert "has no artifacts at artifact path" not in caplog.text
    champion = MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    assert champion.source.startswith("models:/m-")
