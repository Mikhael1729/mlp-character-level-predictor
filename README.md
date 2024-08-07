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

## Run the program

Run the program while keeping track of the logs:

```bash
python3 main.py | tee logs/<log-file-name>.txt
```

## Documentation

To display the different configurations to the execution, run:

```bash
python3 main.py --help
```
