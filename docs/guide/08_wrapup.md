---

---

# PART 8 — Wrap-Up

---

## Chapter 14 — Definition of Done, and a README

### What Problem This Solves

"It worked when I built it" isn't the same as "it works now". Later chapters changed things that earlier ones depended on. So finish by **re-checking everything against the current state** — not from memory — and by writing the page strangers will read first.

### Step 1 — The definition-of-done checklist

Run each check. Every row should pass.

| Requirement | Check | Expected |
|---|---|---|
| Baseline works (Ch 4) | `python train.py` | `rmse: 83.545` |
| Five experiments tracked (Ch 7) | MLflow UI → `airbnb-price-prediction` | each of the 5 configs has finished runs with params and metrics |
| Champion loads in a fresh process (Ch 8) | the `python -c "…load_model('models:/AirbnbPriceModel@champion')…"` command from Chapter 8 | `TransformedTargetRegressor 244.25` |
| All tests pass | `pytest -q` (with `MLFLOW_TRACKING_URI` set) | `46 passed` |
| Tests really catch bugs | change a rule in `schemas.py` (e.g. `availability_365` `le=365` → `le=400`), run `pytest -q`, then undo | exactly one test fails, then all pass |
| CI runs on pull requests (Ch 11) | your repository's **Pull requests → Closed** | every merged PR shows `test` ✅ and `build-image` ✅ |
| Deploys are published (Ch 12) | Docker Hub tags | `latest` + `model-vN`, amd64 + arm64 |
| Local stack runs (Ch 13) | `docker compose ps` | MLflow healthy, API up |

### Step 2 — Rebuild from GitHub

The strongest proof that a project is complete: a fresh clone, nothing else.
```bash
cd /tmp
git clone https://github.com/<your-github-username>/<your-repo>.git airbnb-check
cd airbnb-check
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt dvc
dvc pull
python train.py | grep rmse
env -u MLFLOW_TRACKING_URI pytest -q
```
Expected:
```
rmse: 83.545
43 passed, 3 skipped
```
Then clean up: `deactivate; cd ~; rm -rf /tmp/airbnb-check`.

💡 `dvc pull` works here because your DVC remote is a folder on **this** machine. A teammate on another computer couldn't reach it — for that, the remote would move to shared storage (e.g. an S3 bucket). The clone *will* run CI anywhere, because CI uses the committed sample.

💡 If your repository's name starts with `-`, use `cd -- -name` — otherwise the shell reads it as an option.

### Step 3 — Write the README

The README is your repository's front page, written for someone who has never seen the project. Suggested sections:

1. **What it does** — one paragraph, and a diagram of the whole loop (GitHub renders ```` ```mermaid ```` blocks).
2. **Results** — the five-model table from Chapter 7, and the **champion rule** (≤ 100 MB, lowest RMSE) with its reason.
3. **Key decisions** — log target inside the model; cleaning thresholds; the high-cardinality `neighbourhood`; the model not baked into the image.
4. **Project layout** — one line per important file (Appendix A).
5. **Setup** — venv, install, `dvc pull`, and the "other Python" warning.
6. **Run it** — `docker compose up -d`, then running each part by hand; tests.
7. **CI/CD** — what each workflow does and which secrets it needs.
8. **Known limitations** — honestly: modest accuracy, growing artifact storage, local-only DVC remote and MLflow server.

Replace the one-line README from Chapter 11, then commit it through a pull request like any other change.

### Step 4 — Shut down cleanly

```bash
docker compose down                 # stop MLflow + API (data stays on disk)
unset GITHUB_TOKEN                  # or just close the work terminal
```
…and press Ctrl-C in the Prefect terminal. Everything is saved: `mlflow.db`, `mlartifacts/`, and Prefect's history in `~/.prefect`. `docker compose up -d` and `prefect server start` bring it all back.

---

## What You Built, and What You Learned

### The finished system

```mermaid
flowchart LR
    DVC["🗂 DVC\nversioned dataset"] --> PF
    subgraph PF ["🔁 Prefect flow (weekly, Mon 03:00 UTC)"]
        direction TB
        L["load (retries)"] --> T["train ×5"] --> PR["promote ≤100 MB\nlowest RMSE"] --> TD["trigger deploy"]
    end
    T -- "runs" --> MLF["📊 MLflow\n@champion"]
    PR -- "alias" --> MLF
    TD -- "workflow_dispatch" --> GH["⚙️ GitHub Actions\ndeploy.yml"]
    GH -- "amd64 + arm64" --> HUB["🐳 Docker Hub"]
    HUB --> API["⚡ API\n/predict"]
    API -- "loads @champion" --> MLF
    PRS["Pull requests"] --> CI["⚙️ ci.yml\ntests + build"]
    DC["🧩 docker compose up -d"] -.-> MLF
    DC -.-> API
```

### The real problems behind this guide's 💡 notes

Every one of these actually happened while this project was built. Each teaches something general.

| # | Problem | Chapter | Lesson |
|---|---|---|---|
| 1 | A test used a relative path and failed from another folder | 3 | Build paths from `__file__`, not the current directory |
| 2 | `httpx` deprecation warning | 1, 6 | Read warnings — they're future errors |
| 3 | The API froze ~4 minutes, silently, when MLflow was down | 6 | Log before slow calls; limit retries so failures are fast and clear |
| 4 | Anaconda's `mlflow`/`prefect` shadowed the project's | 7, 10 | Check `which <tool>` before trusting a command |
| 5 | Port 5000 taken by AirPlay; another port taken by another project | 7, 9 | Check who holds a port before binding; don't stop what isn't yours |
| 6 | Tree models refused by skops (`UntrustedTypesFoundException`) | 7 | Test *every* variant you use; trust only the exact type needed |
| 7 | A `grep` pipe hid a crash (exit code 0) | 7 | Check the real program's exit code |
| 8 | The best model was 326 MB | 8 | "Best metric" isn't the only criterion; write trade-offs down as rules |
| 9 | Registering through an MLflow 3 compatibility fallback | 8 | Treat fallback warnings as bugs; use the current API |
| 10 | `mlflow-skinny` lacked `skops` | 9 | Slim images need their dependencies checked explicitly |
| 11 | A personal email almost went public | 1, 11 | Set a noreply email first; review what you publish before the first push |
| 12 | The published image didn't run on Apple Silicon | 12 | Build for the platforms your users actually have |
| 13 | A rejected deploy (`403`) reported "Completed" | 10, 12 | In pipelines, failures must *raise*, not print |
| 14 | Moving MLflow into Compose risked two servers on one SQLite file | 13 | Back up first, stop the old server, pin the same version, mount the folder |

### Where to go next

- **Clean up artifacts automatically** (Appendix E), or cap/drop the 326 MB forest from the scheduled configs.
- **Better features** — size, bedrooms, amenities would lift accuracy well beyond R² ≈ 0.46.
- **Shared infrastructure** — DVC remote on S3, a hosted MLflow server, the Prefect flow on an always-on machine.
- **Monitoring** — log predictions and watch for **data drift** (new neighbourhoods, price shifts) to decide when retraining matters.
- **Cloud deployment** — run the published image on a cloud service, pointing `MLFLOW_TRACKING_URI` at a hosted MLflow.

---

---

# Appendices

---

## Appendix A — The Final Repository

```
NYC-Airbnb-Price-Prediction/
├── .github/workflows/
│   ├── ci.yml                     ← on every PR: throwaway MLflow → seed champion → tests → build image
│   └── deploy.yml                 ← on demand: build amd64 + arm64, push to Docker Hub
├── .dvc/config                    ← DVC remote settings
├── data/
│   ├── AB_NYC_2019.csv.dvc        ← pointer to the dataset (the CSV itself is in DVC)
│   └── .gitignore
├── scripts/
│   ├── __init__.py
│   ├── make_sample.py             ← writes the 2,000-row CI sample
│   ├── ci_seed_model.py           ← CI: train on the sample, promote @champion
│   └── trigger_deploy.py          ← asks GitHub to run deploy.yml
├── tests/
│   ├── conftest.py                ← shared fixtures (sample, throwaway MLflow)
│   ├── fixtures/listings_sample.csv
│   ├── test_features.py           ← 7
│   ├── test_schemas.py            ← 14
│   ├── test_api.py                ← 5
│   ├── test_track_experiments.py  ← 13
│   ├── test_model_registry.py     ← 3 (real champion; skip without a server)
│   └── test_trigger_deploy.py     ← 4
├── features.py                    ← load, clean, split, build model, evaluate
├── train.py                       ← baseline
├── track_experiments.py           ← five configs → MLflow runs (with model size)
├── registry.py                    ← champion rule, register, move @champion
├── orchestrate_training.py        ← Prefect flow + weekly schedule
├── schemas.py                     ← Pydantic validation
├── main.py                        ← FastAPI app
├── Dockerfile · .dockerignore · requirements-serve.txt   ← API image
├── docker-compose.yml             ← MLflow + API together
├── requirements.txt · pytest.ini · .gitignore · .dvcignore
└── README.md
```

Not in Git (and never should be): `.venv/`, `data/AB_NYC_2019.csv`, `models/`, `mlflow.db`, `mlartifacts/`.

---

## Appendix B — Troubleshooting Index

| Symptom | Cause | Fix | Ch |
|---|---|---|---|
| `command not found: pytest` / `No module named …` | venv not active in this terminal | `source .venv/bin/activate` from the project folder | 1 |
| `ImportError: cannot import name 'service' from 'google.protobuf'` | Another Python's (e.g. Anaconda's) `mlflow`/`prefect` is running | `which mlflow`; `conda deactivate`; `source .venv/bin/activate`; or call `.venv/bin/mlflow` | 7, 10 |
| `The following paths are ignored by one of your .gitignore files: data` | `data/` is in `.gitignore` | Remove that line; DVC's `data/.gitignore` handles the CSV | 2 |
| `ModuleNotFoundError: No module named 'features'` when running a script | Ran `python scripts/x.py` | Run as a module from the project root: `python -m scripts.x` | 3 |
| A test passes from the project root but fails from elsewhere | Relative path in the test | Build paths from `Path(__file__)` | 3 |
| Nothing appears in the MLflow UI | `MLFLOW_TRACKING_URI` not set in this terminal | `export MLFLOW_TRACKING_URI=http://127.0.0.1:5001` | 7 |
| MLflow won't start / port in use | macOS AirPlay Receiver holds 5000 | Use port 5001 (this guide) or switch AirPlay Receiver off | 7 |
| zsh: `no matches found: http://…?…` | zsh treats `?` as a wildcard | Put the URL in quotes | 7 |
| `UntrustedTypesFoundException: … sklearn.tree._tree.Tree` | skops blocks tree objects by default | `skops_trusted_types=["sklearn.tree._tree.Tree"]` in `log_model` | 7 |
| `Inferred schema contains integer column(s)` / `Failed to resolve installed pip version` | Informational MLflow warnings | Safe to ignore (explained in Ch 7) | 7 |
| Warning: `has no artifacts at artifact path 'model', registering … based on models:/m-…` | Registering the MLflow 2 way (`runs:/…`) | Register the logged model's own URI (`registry.logged_model_uri`) | 8 |
| API startup hangs for minutes | MLflow unreachable; default retries (~4 min) | Start MLflow; set `MLFLOW_HTTP_REQUEST_MAX_RETRIES=3`, `MLFLOW_HTTP_REQUEST_TIMEOUT=10` | 6, 9 |
| Ctrl-C doesn't stop a stuck server | Blocked inside startup retries | `pgrep -fl uvicorn`, then `kill -9 <pid>` | 6 |
| `port is already allocated` (Docker) / `address already in use` | Something else uses the port | `lsof -nP -iTCP:<port> -sTCP:LISTEN`; if `com.docker.backend`, check `docker ps`; choose another port | 9 |
| Container: `No module named 'skops'` | `mlflow-skinny` doesn't include skops | Add `skops==0.16.0` to `requirements-serve.txt`, rebuild | 9 |
| Container can't reach MLflow | Server bound to `127.0.0.1`, or host name not allowed | Start MLflow with `--host 0.0.0.0` and `host.docker.internal:*` in `--allowed-hosts` | 7, 9 |
| `Invalid Host header` / HTTP 403 from MLflow | Name not in `--allowed-hosts` | Add it (e.g. `mlflow-server:*`) | 7, 13 |
| `no matching manifest for linux/arm64/v8` | Image built only for amd64 | Build multi-arch (`setup-qemu-action`, `platforms: linux/amd64,linux/arm64`) | 12 |
| Deploy trigger `401 Bad credentials` | Token missing/expired/incomplete in this terminal | Re-export `GITHUB_TOKEN` | 12 |
| Deploy trigger `403 Resource not accessible by personal access token` | Token lacks **Actions: Read and write** | Edit the fine-grained token's permissions | 12 |
| Deploy trigger `404 Not Found` | Token not granted this repo, or wrong `GITHUB_REPO` | Fix repository access / the `owner/repo` value | 12 |
| Scheduled runs never happen | The `--serve` process isn't running; deployment paused | Keep `python orchestrate_training.py --serve` running | 10 |
| `cd -NYC-…: invalid option` | Name starts with `-` | `cd -- -NYC-…` | 14 |
| GitHub API `rate limit exceeded` (scripts polling GitHub) | 60 requests/hour without logging in | Poll less often, or authenticate | 12 |

### "My personal email is in my commits"

Only safe **before you've pushed** — it rewrites every commit's ID:
```bash
NR="12345678+yourname@users.noreply.github.com"
git config user.email "$NR"
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --env-filter \
  "export GIT_AUTHOR_EMAIL='$NR' GIT_COMMITTER_EMAIL='$NR'" -- --all
git update-ref -d refs/original/refs/heads/main          # remove filter-branch's backup of the old commits
git reflog expire --expire=now --all && git gc -q --prune=now
git log --format='%ae' | sort -u                         # only the noreply address
```
If you've already pushed, the old commits are public; rewriting would break everyone else's copy.

---

## Appendix C — Command Cheat Sheet

| Task | Command |
|---|---|
| Activate the venv | `source .venv/bin/activate` |
| Run all tests / quietly / one file / by name | `pytest -v` · `pytest -q` · `pytest tests/test_api.py` · `pytest -k skops` |
| Status, stage, commit | `git status --short` · `git add <files>` · `git commit -m "…"` |
| Branch → PR → back to main | `git switch -c <branch>` · `git push -u origin <branch>` · (merge on GitHub) · `git switch main && git pull && git fetch --prune && git branch -d <branch>` |
| Data | `dvc add <file>` · `dvc push` · `dvc pull` · `dvc status` |
| MLflow server (by hand) | `mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 0.0.0.0 --port 5001 --allowed-hosts "localhost:*,127.0.0.1:*,host.docker.internal:*,mlflow-server:*"` |
| Point tools at MLflow | `export MLFLOW_TRACKING_URI=http://127.0.0.1:5001` |
| Train five / promote best | `python track_experiments.py` · `python registry.py` |
| Full pipeline / scheduled | `python orchestrate_training.py` · `python orchestrate_training.py --serve` |
| Prefect | `prefect server start` · `prefect deployment run 'airbnb-price-training/weekly-retrain'` |
| API locally | `uvicorn main:app --port 8000` → http://127.0.0.1:8000/docs |
| Docker | `docker build -t airbnb-price-api:local .` · `docker ps` · `docker logs <name>` · `docker rm -f <name>` |
| Compose | `docker compose up -d` · `ps` · `logs -f api` · `restart api` · `down` |
| Who holds a port? | `lsof -nP -iTCP:<port> -sTCP:LISTEN` |
| Which tool will run? | `which mlflow` |

---

## Appendix D — Glossary

| Term | Meaning |
|---|---|
| **Alias** | A movable label on a registered model version (`@champion`) |
| **Artifact** | A file produced by a run — here, the saved model |
| **Backend store** | MLflow's database of runs, parameters and metrics (`mlflow.db`) |
| **Bind mount** | A host folder made visible inside a container |
| **CI / CD** | Continuous Integration (test every change) / Continuous Deployment (ship automatically) |
| **Container / image** | A running instance / the read-only package it runs from |
| **Cron** | A five-field schedule: minute hour day month weekday |
| **Deployment (Prefect)** | A named, schedulable version of a flow |
| **DVC remote** | Storage outside the project that holds versioned data |
| **Experiment / run** | A named group of training runs / one training attempt |
| **Fixture (pytest)** | A function that prepares something tests need |
| **Flow / task (Prefect)** | A pipeline / one step of it |
| **High cardinality** | A categorical column with many distinct values |
| **Layer (Docker)** | One cached step of an image build |
| **MAE / RMSE / R²** | Average miss / miss with big errors weighted more / share of variation explained |
| **MD5 hash** | A fingerprint of a file's bytes |
| **Multi-arch image** | One tag containing builds for several CPU types (amd64, arm64) |
| **Pinning** | Recording exact dependency versions (`package==1.2.3`) |
| **Pointer file** | DVC's small `.dvc` file that Git tracks in place of the data |
| **Pull request (PR)** | A proposed merge of a branch, with automated checks |
| **Registry (model)** | A catalogue of approved model versions |
| **Secret (GitHub)** | An encrypted repository setting, hidden in logs |
| **skops** | A safer model file format than pickle, with an allow-list of loadable types |
| **Virtual environment** | A project-private Python and set of libraries |
| **workflow_dispatch** | A GitHub Actions trigger that runs only when requested |

---

## Appendix E — Cleanup and Freeing Disk Space

**Stop everything:** `docker compose down`, Ctrl-C the Prefect server, `unset GITHUB_TOKEN`.

**Why disk grows:** every full retrain saves ~377 MB of models, mostly the 326 MB `rf_100` that the size rule never promotes. `du -sh mlartifacts` shows the total.

**Free it** — delete runs you don't need (they go to MLflow's *Deleted* view), then permanently remove deleted runs and their files.

⚠️ *This recipe wasn't exercised in the original project. Back up first (Chapter 13, Step 1), and stop the stack so nothing else is writing the database.*

```bash
docker compose down
python - <<'EOF'
import mlflow
from mlflow import MlflowClient
mlflow.set_tracking_uri("sqlite:///mlflow.db")      # talk to the database directly (server stopped)
client = MlflowClient()
champion_run = client.get_model_version_by_alias("AirbnbPriceModel", "champion").run_id
runs = mlflow.search_runs(experiment_names=["airbnb-price-prediction"],
                          filter_string="tags.mlflow.runName = 'rf_100'")
for run_id in runs["run_id"]:
    if run_id != champion_run:
        client.delete_run(run_id)
print("deleted", len(runs), "rf_100 runs")
EOF
mlflow gc --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts
du -sh mlartifacts
docker compose up -d
```

**Revoke tokens you no longer need:** GitHub → Settings → Developer settings → Personal access tokens; Docker Hub → Account settings → Personal access tokens.

---

*This guide was written from the finished project. Every file shown is the project's final code; every expected output comes from real runs. The original build log, with every detour, is `implementation.md`.*
