# CAFA6 Protein Function Prediction

This repository contains solutions for the CAFA6 Kaggle challenge, predicting Gene Ontology (GO) terms for proteins based on amino acid sequences.

## Setup

1. Install uv (fast Python package manager):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create virtual environment and install dependencies:
   ```bash
   uv venv
   uv pip install -r requirements.txt
   ```

3. Activate the environment:
   ```bash
   source .venv/bin/activate
   ```

## Notebooks

- `cafa6_solution.ipynb`: Main prediction notebook (single Kaggle submission).
- `protein_embeddings.ipynb`: Modular notebook for generating protein embeddings (ESM2/ESM3).
- `go_embeddings.ipynb`: Modular notebook for generating GO term embeddings.

## Plans

- `plans/plan_1_initial_approach.md`: Initial approach using ESM2, GO embeddings, and cosine similarity.
- `plans/plan_2_modular_embeddings.md`: Modular embedding generation for efficiency.

## Data

Data is expected at `/kaggle/input/cafa-6-protein-function-prediction/` for Kaggle submissions.

## Usage

Run the notebooks in order: first generate embeddings (optional), then run the main prediction notebook.