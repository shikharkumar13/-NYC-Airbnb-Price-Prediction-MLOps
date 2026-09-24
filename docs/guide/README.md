# Source of `airbnb_mlops_guide.md`

The beginner guide in the repo root is **generated**, in two formats: `airbnb_mlops_guide.md` and `airbnb_mlops_guide.html`. Don't edit either directly: edit the files here, then rebuild.

| File | Content |
|---|---|
| `00_front.md` | Title, why MLOps, how to use the guide, prerequisites (macOS + WSL2), roadmap |
| `01_foundations.md` | Chapters 1–2: project setup, DVC |
| `02_modelling.md` | Chapters 3–4: `features.py`, baseline |
| `03_serving.md` | Chapters 5–6: schemas, FastAPI |
| `04_experiments.md` | Chapters 7–8: MLflow tracking, registry |
| `05_packaging.md` | Chapters 9–10: Docker, Prefect |
| `06_cicd.md` | Chapters 11–12: CI, CD |
| `07_compose.md` | Chapter 13: Docker Compose |
| `08_wrapup.md` | Chapter 14, lessons, appendices |
| `template.html` | The HTML page: layout, styles and the script that renders the embedded Markdown in the browser |
| `build.py` | Joins the files in number order, fills in the code blocks, and writes both outputs |

## Code blocks come from the real files

A line like `<<<FILE:features.py>>>` in a chapter file becomes that file's **current contents** in the guide. Change `features.py`, rebuild, and the guide shows the new code. You never copy code into the guide by hand. For a file shown in pieces across chapters:

```
<<<FILE:tests/conftest.py|until:def local_mlflow>>>              top of the file, up to that function
<<<FILE:tests/conftest.py|from:def local_mlflow|title:append>>>  that function to the end, as "Append to"
```

The prose around a code block is **not** automatic. If you change what a file does, also update the explanation, expected output and test counts in the chapter.

## Rebuild and check

```bash
python docs/guide/build.py            # regenerate airbnb_mlops_guide.md and .html
python docs/guide/build.py --check    # are both up to date? (exit 1 if not)
```

`.github/workflows/docs.yml` runs `--check` on every pull request. If a code change alters any file the guide shows and the guide wasn't rebuilt, the check fails and says which command to run.

## The HTML version

`build.py` embeds the finished Markdown in `template.html`; the page renders it when opened, so building needs no extra Python package. Open `airbnb_mlops_guide.html` in a browser (it needs an internet connection for its fonts and libraries). It adds a chapter map, search (`/`), per-chapter progress saved in the browser, copy buttons and a dark theme.
