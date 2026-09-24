# MLOps from Scratch: NYC Airbnb Price Prediction

> **Project:** predict the nightly price (USD) of a New York City Airbnb listing
> **Dataset:** New York City Airbnb Open Data 2019 (`AB_NYC_2019.csv`, 48,895 listings, 16 columns, a regression problem)
> **Goal:** build a complete, working MLOps system by hand in the terminal, and understand how and why each step works
> **For:** anyone who can write basic Python and pandas and has trained a model in a notebook, but has never used Git for data, experiment trackers, Docker, orchestrators or CI/CD
> **Time:** about 3 to 5 days at a relaxed pace (each chapter takes 1 to 3 hours)
> **Platforms:** macOS (every command tested) and Windows through WSL2 (documented but untested; see [Setting up on Windows](#setting-up-on-windows-wsl2))

By the end you will have:

- a Git repository with versioned data (DVC) and 46 automated tests
- five tracked experiments, and a model registry whose `@champion` model is picked by a written rule
- a prediction API (FastAPI) packaged as a Docker image and published to Docker Hub for both Intel and Apple Silicon
- scheduled retraining (Prefect) that promotes a new champion and then triggers its own deployment
- CI that tests every pull request, and a local stack you start with one command (Docker Compose)

---

## Before we start: what MLOps is for

If you've only trained models in notebooks, names like DVC, MLflow, Prefect, Docker Compose and GitHub Actions can feel like a wall. So before typing any commands, it helps to see the problem these tools solve.

### How notebook work usually goes

1. Open `Untitled.ipynb`.
2. Load a CSV from your Downloads folder.
3. Clean it, train a model, evaluate.
4. See `RMSE: 78.7` and feel good.
5. Close the notebook.

That's fine for exploring. But it quietly depends on things that are only true today, on your laptop: that exact CSV, your pandas version, your Python version, the random seed, and the order you happened to run the cells. Six months later you usually can't reproduce your own result.

### What a model in production needs

Once other people rely on a model, it has to:

- run somewhere other than your laptop (a server, a container, the cloud)
- answer requests from other software, such as a website that sends a listing and gets a price back
- be reproducible, so anyone can retrain it and get the same numbers
- be updatable, so a retrained model can replace the old one safely
- be auditable, so you can answer "which model served prices last Tuesday, and what data trained it?"

A notebook can't do any of that. MLOps is the set of practices and tools that fills the gap.

### What MLOps is

MLOps is DevOps applied to machine learning. Some of it is ordinary software engineering. The rest exists because ML depends on data and produces models:

| Need | Tool in this guide | Specific to ML? |
|---|---|---|
| Version the code | Git and GitHub | No |
| Version the data | DVC | Yes. A model means little without the exact data it learned from |
| Record every training experiment | MLflow Tracking | Yes. You train many models and need to compare them |
| Manage model versions and which one is live | MLflow Model Registry | Yes |
| Validate inputs and serve predictions | Pydantic and FastAPI | No |
| Package the app so it runs anywhere | Docker | No |
| Automate and schedule the pipeline | Prefect | Partly |
| Test every change and ship automatically | GitHub Actions (CI/CD) and Docker Hub | Partly |
| Start the whole stack with one command | Docker Compose | No |

You'll add these one at a time, and each tool shows up at the point where you run into the problem it solves.

### Why this dataset makes a good teacher

It's a regression problem (you predict a number), and the data is messy in realistic ways:

- some prices are `$0` (data errors), and some go up to `$10,000`
- the target is heavily skewed, with most listings around $100
- some missing values carry meaning (no reviews means no review rate)
- one column, `neighbourhood`, has 221 different values

Real projects look like this, so you'll run into real problems. The guide includes the ones we hit while building the project, along with how to spot and fix them.

---

## How to use this guide

Read each step, then type it. Every step tells you what to do, what it does, and what you should see. If your output differs, stop and work out why before moving on, because the next step assumes this one worked.

These labels appear throughout:

| Label | Meaning |
|---|---|
| ```` ```bash ```` blocks | Commands to type in your terminal, from the project folder unless the step says otherwise |
| **File: `path`** | Create this file with exactly this content (copy the whole block) |
| **Append to: `path`** | Add the block to the end of an existing file |
| `Expected:` / output blocks | What you should see. IDs, timestamps and durations will differ, but numbers like RMSE should match |
| **Checkpoint** | Commands that prove the chapter worked. Don't move on until they pass |
| Tip | Why something is done a particular way |
| Warning | A trap that catches people |
| **See it fail** | An optional exercise where you break something on purpose and watch the safety net catch it |
| Note (Windows) | What's different on Windows (WSL2) |

About the code: every file in this guide is the final, working version from the finished project. When a line exists because of a real bug we hit, a tip explains the story, and a "See it fail" exercise often lets you remove the line and watch the bug come back. Some files mention `implementation.md` in a comment. That's the original project's build log, and you can ignore or delete those references in your copy.

About your numbers: the dataset is byte-for-byte the same and every model uses a fixed random seed, so your metrics should match the guide exactly (for example, a baseline RMSE of `83.545`). If they don't, something is different, usually a package version. That's why Chapter 1 pins versions.

Personal values: wherever you see `<your-github-username>`, `<your-repo>` or `<your-dockerhub-username>`, use your own. The examples show the original project's values (GitHub `shikharkumar13`, Docker Hub `krshikhar13`).

---

## Prerequisites

### What you need

| Item | Why | Cost |
|---|---|---|
| A Mac (Apple Silicon or Intel), or Windows 10/11 with WSL2 | Everything runs here | |
| About 10 GB of free disk space | Python packages, Docker images, saved models | |
| A GitHub account | Code hosting and CI/CD | Free |
| A Docker Hub account (Chapter 12) | Publishing the API image | Free |

You don't need a Kaggle account. The dataset downloads directly.

### Install the tools (macOS)

1. Homebrew, the macOS package manager. If `brew --version` fails, install it from https://brew.sh.
2. Git and uv:
   ```bash
   brew install git uv
   ```
   uv is a fast Python package and environment manager. It creates virtual environments and installs packages the way `pip` does, and it can also download the exact Python version you need.
3. Docker Desktop. Download it from https://www.docker.com/products/docker-desktop/, install it, and start it (you'll see a whale icon in the menu bar).

### Check that everything is installed

```bash
git --version        # Expected: git version 2.x
uv --version         # Expected: uv 0.x
docker --version     # Expected: Docker version 2x.x
docker info --format '{{.ServerVersion}}'   # Expected: a version number (Docker Desktop is running)
```

> [!WARNING]
> **Other Pythons on your machine.** If you have Anaconda (or another Python distribution), it may come with its own copies of tools like `mlflow` and `prefect`. Running the wrong copy gives confusing errors. The classic one is `ImportError: cannot import name 'service' from 'google.protobuf'`. The fix is always the same: use the project's virtual environment (Chapter 1), and run `which mlflow` to check that the path contains `.venv`. If your prompt shows `(base)`, run `conda deactivate` first.

### Setting up on Windows (WSL2)

> [!NOTE]
> These Windows instructions follow the documented WSL2 setup, but nobody tested them for this guide. The macOS path was tested.

On Windows you'll work inside WSL2, which runs a real Ubuntu Linux inside Windows. Every `bash` command in this guide then works almost unchanged.

1. Install WSL2 and Ubuntu. Open PowerShell as Administrator and run:
   ```powershell
   wsl --install
   ```
   Restart when asked, then open Ubuntu from the Start menu and create a Linux username and password.
2. Install the tools inside Ubuntu:
   ```bash
   sudo apt update && sudo apt install -y git curl lsof
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source ~/.bashrc     # or open a new Ubuntu terminal so `uv` is found
   ```
3. Install Docker Desktop for Windows, then open Settings → Resources → WSL integration and switch your Ubuntu distribution on. In Ubuntu, `docker info` should now work.
4. Keep the project inside Linux. Create it under your Linux home (`~/…`), not under `/mnt/c/…`. Files on the Windows drive are much slower to reach from Linux and cause permission problems with Docker (Chapter 13).
5. For an editor, install VS Code with the WSL extension. Running `code .` from the project folder in Ubuntu opens it.
6. URLs like `http://127.0.0.1:5001` work from your normal Windows browser.

Later sections marked "Windows" cover the differences: installing tools (`apt` and install scripts instead of Homebrew), port 5000 (normally free on Windows, but we still use 5001 so every file matches), and how `git push` logs in to GitHub.

---

## Roadmap

| Part | Chapter | Tools | You build | Time |
|---|---|---|---|---|
| 1 Foundations | 1. Project setup | Git, uv | Repository, virtual environment, pinned dependencies | 1 h |
| | 2. Data versioning | DVC | The dataset, versioned and restorable | 1 h |
| 2 Modelling code | 3. Features and pipeline | pandas, scikit-learn, pytest | One shared module for cleaning and model building, with tests | 2 to 3 h |
| | 4. Baseline | scikit-learn | `train.py`, the number every model must beat | 1 h |
| 3 Serving | 5. Input validation | Pydantic | Schemas that reject impossible listings | 1 h |
| | 6. Prediction API | FastAPI | `/health` and `/predict` | 1 to 2 h |
| 4 Experiments | 7. Experiment tracking | MLflow | A tracking server and five logged experiments | 2 h |
| | 8. Model registry | MLflow Registry | A `@champion` model picked by a written rule | 1 to 2 h |
| 5 Packaging and automation | 9. Container | Docker | The API as a portable image | 1 to 2 h |
| | 10. Orchestration | Prefect | Scheduled retraining with retries | 2 h |
| 6 CI/CD | 11. Continuous integration | GitHub, GitHub Actions | Tests on every pull request | 2 h |
| | 12. Continuous deployment | GitHub Actions, Docker Hub | A new champion publishes a new image by itself | 2 h |
| 7 Local stack | 13. Compose | Docker Compose | MLflow and the API with one command | 1 h |
| 8 Wrap-up | 14. Definition of done | | Proof that everything works, and a README | 1 h |

---

## The big picture

```mermaid
flowchart LR
    D["DVC\nversioned dataset"] --> P
    subgraph P ["Prefect flow (weekly)"]
        direction TB
        L["load data\n(retries)"] --> T["train 5 models"] --> R["promote best\n≤100 MB, lowest RMSE"] --> TD["trigger deploy"]
    end
    T -- "params, metrics, models" --> M["MLflow\ntracking + registry\n@champion"]
    R -- "move alias" --> M
    TD -- "GitHub API" --> G["GitHub Actions\ndeploy.yml"]
    G -- "amd64 + arm64" --> H["Docker Hub\nairbnb-price-api"]
    H --> A["API container\n/predict"]
    A -- "loads @champion\nat startup" --> M
    PR["Pull requests"] --> CI["ci.yml\ntests + build"]
```

Read it from left to right. Versioned data feeds a scheduled training flow, and MLflow records every experiment. The best model gets the `@champion` label, and that promotion triggers a build that publishes a new Docker image. Wherever the API runs, it loads whichever model holds `@champion`. Separately, every proposed code change gets tested automatically before anyone can merge it.

### Your terminals

Several programs have to keep running while you work, so give each one its own terminal window or tab. Every new terminal needs the virtual environment activated. Forgetting this is the most common beginner mistake.

| Terminal | Runs | From chapter | Needs |
|---|---|---|---|
| Work | Everything you type | 1 | `source .venv/bin/activate`, and from Chapter 7 `export MLFLOW_TRACKING_URI=http://127.0.0.1:5001` |
| MLflow | `mlflow server …` | 7 (Docker Compose replaces it in 13) | `source .venv/bin/activate` |
| Prefect | `prefect server start` | 10 | `source .venv/bin/activate` |

An environment variable set with `export NAME=value` only exists in the terminal where you set it. If something "can't find the server" in a new terminal, that's usually why.
