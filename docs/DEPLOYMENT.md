# Deployment and publishing

NeuroQuest is a desktop Python game: it runs locally from a clone of the repository. What can be *hosted* is the documentation and the self-contained AI dashboard. The repository is `https://github.com/Reetika-k/neuroquest`, and every link in this project's documents uses the path `neuroquest/reports/dashboard.html`: it is the path from the parent of a clone, it matches GitHub's file viewer (`…/neuroquest/blob/main/reports/dashboard.html`) and, once GitHub Pages is on, the public URL `https://reetika-k.github.io/neuroquest/reports/dashboard.html`.

## 1. Publish the repository

The project lives at the root of its own repository, `https://github.com/Reetika-k/neuroquest`. To publish further changes:

```bash
git add .
git commit -m "<what changed>"
git push
```

`.gitignore` excludes the virtual environment and `data/raw/` (episode logs are regenerable from their seeds). The processed dataset, the trained models, the reports and the dashboard HTML are tracked, so a fresh clone can play, predict, adapt and open the dashboard without re-running anything.

## 2. Host the dashboard with GitHub Pages

1. On GitHub: **Settings → Pages → Build and deployment → Source: Deploy from a branch → Branch: `main`, folder `/ (root)` → Save**.
2. After a minute the site is at `https://reetika-k.github.io/neuroquest/`, and the dashboard at `https://reetika-k.github.io/neuroquest/reports/dashboard.html`.
3. `README.md` already links that path; the animations in the README are relative paths inside `docs/screenshots/` and display on both GitHub and Pages.

The dashboard embeds its figures as base64, so it is one file with no dependencies and no server-side code; it can also be attached to a submission or emailed as is.

Before publishing, rebuild it from the recorded data so the page matches the repository: `cd neuroquest && .venv/bin/python -m neuroquest dashboard` (about 2 s).

## 3. Run the game from a clone

```bash
git clone https://github.com/Reetika-k/neuroquest.git
cd neuroquest
python3 -m venv .venv            # Python 3.12 or newer
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m neuroquest check     # headless smoke test
.venv/bin/python -m neuroquest play      # the game
.venv/bin/python -m neuroquest play --agent   # the Q-learning agent on its board
```

Windows: `py -3 -m venv .venv` and `.venv\Scripts\python -m neuroquest play`. Exact tested versions are in `requirements.txt`.

## 4. Regenerating everything (optional)

A clone has the processed dataset and models but not the raw logs. To rebuild from scratch:

```bash
.venv/bin/python -m neuroquest simulate        # 360 episode logs, ~13 s
.venv/bin/python -m neuroquest build-dataset
.venv/bin/python -m neuroquest train           # ~8 s; reproduces reports/metrics.json exactly (seeded)
.venv/bin/python -m neuroquest ab-test         # ~1 min
.venv/bin/python -m neuroquest rl-train        # ~15 s
.venv/bin/python -m neuroquest dashboard
```

## 5. Not done: a browser build of the game

A browser version of the game-only part would be possible with `pygbag` (Pygame to WebAssembly; listed as the optional `web` extra). It was not attempted in this project: the ML pipeline, the adaptive controller's model loading and the dashboard are desktop-side, and the brief required a local demonstration. If it is tried later, the game loop would need the classifier and controller stubbed out in the browser build, and the result should be tested in at least two browsers before being linked.
