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

Example:

```bash
python main.py -d "samples"
```

Output:

```txt
Training settings: 

- Training steps (default 100): 50000
- Learning rate (default 0.1): 0.1
- Minibatch size (default 32): 32

Parameters info:

Parameters: 3481
W1: torch.Size([6, 100])
b1: torch.Size([100])
W2: torch.Size([100, 27])
b2: torch.Size([27])


Train loss: 2.413003921508789
Dev (test) loss: 2.3668856620788574
---

Samples:

- careah.
- amelle.
- khkimrex.
- tatlannan.
- saler.
- hetn.
- deliyat.
- kaqui.
- nellara.
- chaiiv.
- kaleigphh.
- mandin.
- quibt.
- sroilei.
- jadbi.
- wazelo.
- dearyxi.
- jacen.
- dusa.
- jed.

Continue training? (y/n/r): 
---
