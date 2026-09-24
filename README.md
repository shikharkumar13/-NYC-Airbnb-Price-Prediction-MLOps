# NYC Airbnb Price Prediction — MLOps practice project

Predicts a NYC Airbnb listing's nightly price (USD). See
`Implementation_Plan_NYC_Airbnb_Price_Prediction.md` for the spec and
`docs/superpowers/plans/` for the build plan.

## Setup

    uv venv --python 3.11 .venv
    source .venv/bin/activate
    uv pip install -r requirements.txt dvc
    dvc pull

## Tests

    pytest                                              # unit tests; registry tests skip without a server
    MLFLOW_TRACKING_URI=http://127.0.0.1:5001 pytest    # + tests against the real @champion

## Continuous integration

Every pull request runs `.github/workflows/ci.yml`:

1. **test** installs `requirements.txt` and starts a throwaway MLflow server. It then
   trains a quick baseline on the committed 2,000-row sample
   (`tests/fixtures/listings_sample.csv`), promotes it to `@champion`, and runs the
   full test suite.
2. **build-image** builds the API Docker image. It runs only if `test` passes, and
   doesn't push the image anywhere.

CI uses the sample because the real dataset lives in a DVC remote on a laptop,
which GitHub's runners can't reach.
