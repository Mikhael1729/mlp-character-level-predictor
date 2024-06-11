import torch
from typing import Dict, Tuple
import torch.nn.functional as F


BLOCK_SIZE = 3 # Context lenght. How many characters are needed to predict the next one
CHARACTER_FEATURES_SIZE = 2
CHARACTERS_NUMBER = 27

END_START_CHARACTER = '.'
DATASET_PATH = "./names.txt"


def main():
  names = get_dataset(DATASET_PATH)
  _, itos, stoi = build_vocabulary_of_characters(names)
  X, Y = build_dataset(names, itos, stoi)
  g = torch.Generator().manual_seed(2147483647)
  embeddings, characters_features = create_lookup_table(X, g)
  loss, parameters = forward(embeddings, characters_features, Y, g)

  print(sum(p.nelement() for p in parameters))


def forward(embeddings: torch.Tensor, characters_features: torch.Tensor, Y: torch.Tensor, g: torch.Generator, size = 100) -> Tuple[torch.Tensor, list[torch.Tensor]]:
  neurons_per_sample = embeddings.shape[1] # 6
  W1 = torch.randn((neurons_per_sample, size), generator=g)
  b1 = torch.randn(size, generator=g)
  a1 = embeddings @ W1 + b1
  z1 = torch.tanh(a1)

  W2 = torch.randn((size, CHARACTERS_NUMBER), generator=g)
  b2 = torch.randn(CHARACTERS_NUMBER, generator=g)
  a1 = z1 @ W2 + b2

  loss = F.cross_entropy(a1, Y)

  return loss, [characters_features, W1, b1, W2, b2]

def create_lookup_table(X: torch.tensor, g: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
  """
  Builds the features table associated with the given input X

  Returns:
    embedding: A tensor of shape (m, l) where m is the number of samples and l is total of features for each character in a given sample (6 in total)
  """
  

  # Randomly generates a two dimensional vector for each character
  characters_features = torch.randn((CHARACTERS_NUMBER, CHARACTER_FEATURES_SIZE), generator=g) 
  
  # Index each character in the samples with its corresponding features. Its shape is (m, n, k) where m is the number of samples, n the context lenght and k the features size
  embedding = characters_features[X]; s = embedding.shape
  embedding = embedding.view(s[0], s[1] * s[2]) # Flatten the resulting embedding to have all the features of all characters in a single row

  return embedding, characters_features


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