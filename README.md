# Moral Argument Mining

A collection of datasets, tools, and models for philosophical argument mining and moral / ethical reasoning extraction from scholarly text. This repository contains data-processing scripts, snowball-sampling utilities, model artifacts, experimental results, and small web apps used across the project.

High-level goals
----------------
- Provide a reproducible pipeline to find, extract and process logical arguments from (moral) philosophy papers.
- Maintain intermediate datasets and processed outputs for reproducible experiments.

Quick start
-----------
Prerequisites

- Python 3.8+ (3.12 recommended)

Install Python dependencies into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run a script (example — show help for the snowball sampling phase-1 script):

```bash
python scripts/snowball_phase_1.py --help
```

Project layout
--------------

- datasets/ — raw, interim, and processed datasets. See subfolders for provenance.
- figures/ — plots and visualizations used in analysis and reports.
- models/ — location fo model checkpoints and tokenizers.
- results/ — exported evaluation, metrics and logs from experiments.
- scripts/ — data-processing, extraction, and sampling scripts. Examples:
	- `scripts/snowball_phase_1.py` — entrypoint for snowball sampling phase 1.
    - `scripts/snowball_phase_1_E2E_eval.py` — entrypoint for evaluating generated outputs for phase 1 of snowball sampling. 
- src/ — primary python package modules for the project.
- requirements.txt, pyproject.toml — dependency declarations for the repository.