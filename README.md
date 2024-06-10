# MLP Namnes Generator

A small LLM to generate names using an MLP, based on the method in [this paper](./bengio03a.pdf). It differs by using a 3-letter context window as input and predicts the next letter instead of the next word.

## Getting Started

**Install dependencies**:

```bash
conda env create -f torch-conda.yml
```

**Activate virtual environment**:

```bash
conda activate torch
```
