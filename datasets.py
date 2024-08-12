import torch
import random
from typing import Dict, Tuple

random.seed(42)

class Dataset:
  def __init__(self, name: str, X: torch.Tensor, Y: torch.Tensor):
    self.X = X
    self.Y = Y
    self.name = name


class Datasets:
  def __init__(self, names: list[str], stoi: Dict[int, str], block_size: int):
    random.shuffle(names)

    n = len(names)
    eighty_percent_index = int(0.8 * n)
    ninety_percent_index = int(0.9 * n)

    X_train, Y_train = build_dataset(names[:eighty_percent_index], stoi, block_size)
    X_dev, Y_dev = build_dataset(names[eighty_percent_index:ninety_percent_index], stoi, block_size)
    X_test, Y_test = build_dataset(names[ninety_percent_index:], stoi, block_size)
    X_all, Y_all = build_dataset(names, stoi, block_size)

    self.train = Dataset("train", X_train, Y_train)
    self.dev = Dataset("dev", X_dev, Y_dev)
    self.test = Dataset("test", X_test, Y_test)
    self.all = Dataset("all", X_all, Y_all)


def build_dataset(names: list[str], stoi: Dict[str, int], block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
  """
  Builds dataset for training from a list of names.

  Returns:
    Tuple[torch.Tensor, torch.Tensor]:
      - X: A tensor of shape (m, n) of type torch.int64 where m is the number of samples and n is the context length.
      - Y: A tensor of shape (m,) of type torch.int64 , containing encoded target characters.
  """
  X, Y, = [], []

  for word in names:
    context = [0] * block_size # Empty letters, represented by '.', '.', '.' which encoded would be 0, 0, 0

    for character in word + '.': # The '.' uses "allucination" to encode meta information about the end of names
      encoded_character = stoi[character]

      X.append(context)
      Y.append(encoded_character)

      context = context[1:] + [encoded_character] # Update the context to use the next letter. i.e.: 0,0,0 -> 0,0,5 = '.', '.', 'e'

  return torch.tensor(X), torch.tensor(Y)