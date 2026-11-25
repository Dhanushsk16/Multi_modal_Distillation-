---
{}
---

# PeptideMTR: MLM Variant

This model is part of the **PeptideMTR** suite, to be released in a forthcoming paper. It is designed for multi-target biochemical property prediction.

## Overview
- **Architecture:** BERT-based Transformer
- **Pretraining:** MLM
- **Input:** SMILES string

All models in this suite are designed for simplicity. For inference, the model accepts a SMILES string directly.

## Usage
Example scripts can be found at https://github.com/AaronFeller/PeptideMTR

## Dataset
The training data and pre-computed 99 RDKit descriptors are available at [PeptideMTR_training_data](https://huggingface.co/datasets/aaronfeller/PeptideMTR_training_data).

## Citation
Please cite the forthcoming **PeptideMTR** paper for methodology and benchmark results.