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

Run a script (example — show help for the snowball sampling phase-1 prompt generation script):

```bash
python scripts/snowball_phase_1_generate.py --help
```

Project layout
--------------

- `config.yaml` — central repository configuration (paths, model names, logging, and snowball settings). Most library code loads settings via `src/moralkg/config.py`.
- `datasets/` — raw, interim, and processed datasets. See subfolders for provenance and processed artifacts.
- `figures/` — plots and visualizations used in analysis and reports.
- `models/` — model checkpoints, tokenizers, and model-specific config / metadata.
- `results/` — exported evaluation outputs, metrics, and experiment logs.
- `scripts/` — convenience CLI scripts and entrypoints used to run data-processing, extraction, and sampling tasks. Examples:
		- `scripts/snowball_phase_1_generate.py` — entrypoint for generating LLM prompts / prompt manifests for phase 1.
	- `scripts/snowball_phase_1_E2E_eval.py` — end-to-end evaluation entrypoint for phase 1 generated outputs.
- `src/` — primary Python package; see `src/moralkg` for core project APIs. Notable subpackages:
	- `src/moralkg/argmining/` — argument-mining model code, registry, prompts, and evaluation helpers used across experiments.
	- `src/moralkg/snowball/` — snowball-sampling utilities and orchestration for phase-1 and later phases; contains `phase_1/` with evaluation and prompt generation code.
	- `src/moralkg/preprocessing/` — text cleaning, document parsing and dataset preparation utilities.
	- `src/moralkg/figures/` — scripts and helpers to generate visualizations found under the top-level `figures/` directory.
	- `src/moralkg/config.py` and `src/moralkg/__readme__` — central config loader and package-level documentation.
- `requirements.txt`, `pyproject.toml` — dependency declarations for the repository.