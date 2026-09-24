"""CI only: train a quick baseline on the committed sample and make it @champion,
so models:/AirbnbPriceModel@champion resolves for tests/test_model_registry.py.

Run as: DATA_PATH=tests/fixtures/listings_sample.csv python -m scripts.ci_seed_model
"""
import mlflow

from features import clean_data, load_data, split_data
from registry import EXPERIMENT_NAME, register_and_promote, require_tracking_uri
from track_experiments import train_and_log


def main():
    require_tracking_uri()
    mlflow.set_experiment(EXPERIMENT_NAME)
    run_id, metrics = train_and_log("linreg_baseline", *split_data(clean_data(load_data())))
    version = register_and_promote(run_id)
    print(f"seeded AirbnbPriceModel v{version} @champion (rmse={metrics['rmse']:.2f})")


if __name__ == "__main__":
    main()
