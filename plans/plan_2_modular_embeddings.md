# Plan 2: Modular Embedding Generation for CAFA6

## Overview
Create two separate notebooks for generating embeddings: one for proteins (ESM2/ESM3) and one for GO terms. This allows pre-computing embeddings to speed up the main prediction notebook.

## Protein Embedding Notebook
- **Purpose**: Generate embeddings for protein sequences using ESM2 or ESM3.
- **Features**:
  - Model selection at the top (ESM2 or ESM3).
  - Load sequences from train_sequences.fasta and testsuperset.fasta.
  - Batch processing for efficiency.
  - Save embeddings as .npy or .pkl files.
- **Models Supported**:
  - ESM2: facebook/esm2_t6_8M_UR50D (default)
  - ESM3: Add support for ESM3 variants (e.g., esm3_sm_open_v0 if available).

## GO Embedding Notebook
- **Purpose**: Generate embeddings for GO terms.
- **Features**:
  - Load GO graph from go-basic.obo.
  - Compute embeddings using graph features (degree, depth) + PCA.
  - Save embeddings as dict or array.
- **Improvements**: Explore better methods like node2vec if time allows.

## Integration
- Main prediction notebook loads pre-computed embeddings from these notebooks.
- Reduces runtime in main notebook.

## Todo List
1. Create protein embedding notebook with model selection
2. Add ESM3 support
3. Create GO embedding notebook
4. Test integration with main notebook