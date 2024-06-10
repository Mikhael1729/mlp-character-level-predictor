import torch
from typing import Dict, Tuple

BLOCK_SIZE = 3 # Context lenght. How many characters are needed to predict the next one
END_START_CHARACTER = '.'
DATASET_PATH = "./names.txt"


def main():
  names = get_dataset(DATASET_PATH)
  _, itos, stoi = build_vocabulary_of_characters(names)
  X, Y = build_dataset(names, itos, stoi)


def build_dataset(names: list[str], itos: Dict[int, str], stoi: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
  """
  Builds dataset for training from a list of names.

  Returns:
    Tuple[torch.Tensor, torch.Tensor]: 
      - X: A tensor of shape (m, n) of type torch.int64 where m is the number of samples and n is the context length.
      - Y: A tensor of shape (m,) of type torch.int64 , containing encoded target characters. 
  """
  X, Y, = [], []

  for word in names[:5]:
    context = [0] * BLOCK_SIZE # Empty letters, represented by '.', '.', '.' which encoded would be 0, 0, 0

    for character in word + '.': # The '.' uses "allucination" to encode meta information about the end of names
      encoded_character = stoi[character]

      X.append(context)
      Y.append(encoded_character)

      context = context[1:] + [encoded_character] # Update the context to use the next letter. i.e.: 0,0,0 -> 0,0,5 = '.', '.', 'e'
    
  return torch.tensor(X), torch.tensor(Y)


def build_vocabulary_of_characters(names: list[str]) -> Tuple[list[str], Dict[str, int], Dict[int, str]]:
  # Get the set of unique characters in the dataset
  characters = sorted(list(set(''.join(names))))

  # Create dictionary to map integers to characters
  stoi = {c: i + 1 for i, c in enumerate(characters)}
  stoi[END_START_CHARACTER] = 0

  # Create dictionary to map encoded characters to their corresponding symbols
  itos = {i: c for c,i in stoi.items()}

  return characters, itos, stoi


def get_dataset(path: str) -> list[str]:
  with open(path, 'r') as file:
    lines = file.read().splitlines()

  return lines


if __name__ == "__main__":
  main()