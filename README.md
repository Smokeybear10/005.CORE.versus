# Versus | UFC Matchup Research

**Live demo: [versus.thomasou.com](https://versus.thomasou.com)**

Compare fighters, review their recorded bouts, and generate conditional win estimates from a versioned model.

## Quick Start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Requires access to the private runtime release in Smokeybear10/802-DATA.MMA
gh auth login
bash scripts/fetch-data.sh

python backend/app.py
```

Open http://localhost:2102

The fetch script downloads a pinned runtime release and verifies its checksums. Model and data files stay outside this repository. `GITHUB_TOKEN` can be used instead of GitHub CLI authentication.

## What It Does

**Matchup reports**
- Search fighters and compare recent form, striking, grappling, shared opponents, and earlier career records
- Follow source links and inspect missing data before drawing conclusions
- Share a matchup through its URL

**Estimates and prices**
- Set the fight date, division, rounds, and title status before requesting an estimate
- Show estimates only when the model has sufficient supported inputs
- Compare entered prices and probabilities; provider prices require a server-side `ODDS_API_KEY`

**Journal and results**
- Save research notes and picks in the current browser
- Inspect the model's historical replay, calibration, and limitations on the results page

The published replay includes outcomes examined during development. It is not an independent test or evidence that the model beats prediction markets. Missing records and unsupported matchups are reported explicitly.

## Tech Stack

| Layer | Tools |
|-------|-------|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python 3.12, Flask |
| Model and data | scikit-learn, pandas, NumPy, Parquet |
| Hosting | Vercel |
| Runtime artifacts | Versioned GitHub release with SHA-256 verification |

## Project Structure

```text
api/                  # Vercel entrypoint
backend/
  app.py              # Pages and API routes
  research.py         # Fighter identities and matchup evidence
  forecast.py         # Validated model inference
  model_engine.py     # Dataset loading and legacy model support
public/               # Home, research workspace, results, methodology
scripts/              # Data audits, evaluation, and runtime fetch
tests/               # API, model, data, and interface regressions
data/                 # Downloaded runtime; not committed
vercel.json
```

## Checks

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m unittest discover -s scripts -p 'test_repair_identities.py'
node --test tests/*.mjs tests/*.cjs
```

Some research tests need archived evaluation artifacts in addition to the deployed runtime.

Originally built for CIS 5450 at the University of Pennsylvania with Aakash Jha and Kevin Jiang.

Built by Thomas Ou
