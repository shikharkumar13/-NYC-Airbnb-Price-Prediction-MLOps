"""Write the small, PII-free sample used by tests and CI.

CI can't `dvc pull` (the DVC remote lives on this laptop), so it trains on
this committed sample instead. Only model columns are kept: no names or ids.
"""
from pathlib import Path

from features import DEFAULT_DATA_PATH, FEATURES, RANDOM_STATE, TARGET, load_data

SAMPLE_PATH = Path("tests/fixtures/listings_sample.csv")
SAMPLE_SIZE = 2000


def main():
    df = load_data(DEFAULT_DATA_PATH)
    sample = df[FEATURES + [TARGET]].sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(SAMPLE_PATH, index=False)
    print(f"wrote {len(sample)} rows to {SAMPLE_PATH}")


if __name__ == "__main__":
    main()
