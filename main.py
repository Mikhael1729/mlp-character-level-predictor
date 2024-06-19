from typing import Dict, Tuple
import torch
import torch.nn.functional as F
import random

random.seed(42)

BLOCK_SIZE = 3 # Context lenght. How many characters are needed to predict the next one
CHARACTER_FEATURES_SIZE = 2
CHARACTERS_NUMBER = 27

END_START_CHARACTER = '.'
DATASET_PATH = "./names.txt"


class Parameters:
  def __init__(self, features: torch.Tensor, W1: torch.Tensor, b1: torch.Tensor, W2: torch.Tensor, b2: torch.Tensor):
    self.features = features
    self.W1 = W1
    self.b1 = b1
    self.W2 = W2
    self.b2 = b2
    self.parameters_list = [self.features, self.W1, self.b1, self.W2, self.b2]

  def print_count(self):
    print(sum(p.nelement() for p in self.get_parameters()))

  def prepare_for_backward(self):
    parameters = self.get_parameters()
    
    for parameter in parameters:
      parameter.requires_grad = True

  def reset_gradients(self):
    parameters = self.get_parameters()

    for parameter in parameters:
      parameter.grad = None
  
  def get_parameters(self) -> list[torch.Tensor]:
    return self.parameters_list

class Dataset:
  def __init__(self, X: torch.Tensor, Y: torch.Tensor):
    self.X = X
    self.Y = Y

class Datasets:
  def __init__(self, names: list[str], stoi: Dict[int, str]):
    random.shuffle(names)

    n = len(names)
    eighty_percent_index = int(0.8 * n)
    ninety_percent_index = int(0.9 * n)

    X_train, Y_train = build_dataset(names[:eighty_percent_index], stoi)
    X_dev, Y_dev = build_dataset(names[eighty_percent_index:ninety_percent_index], stoi)
    X_test, Y_test = build_dataset(names[ninety_percent_index:], stoi)

    self.train = Dataset(X_train, Y_train)
    self.dev = Dataset(X_dev, Y_dev)
    self.test = Dataset(X_test, Y_test)


def main():
  g = torch.Generator().manual_seed(2147483647)

  # Get training raw data
  names = get_dataset(DATASET_PATH)

  # Create the matrix with the features associated with each character
  characters_features = create_lookup_table(g)

  # Initialize the network parameters
  parameters = initialize_network_parameters(
    generator = g,
    characters_features=characters_features,
    first_layer_size=CHARACTER_FEATURES_SIZE * BLOCK_SIZE, # 6, because of the 3-lenght context size
    second_layer_size=100
  )

  # Create mapping to encode the tokens into numbers (that the network actually process)
  _, _, stoi = generate_token_mappings(names)

  # Get trainining dataset
  datasets = Datasets(names, stoi)

  # Train the network
  while True:
    trainin_steps = int(input("Trainig steps: "))

    gradient_descent(
      datasets.train.X,
      datasets.train.Y,
      parameters,
      training_steps=trainin_steps,
      debug=True
    )

    continue_training = input("Continue training? (y/n)")
    print("---")

    if continue_training == "n":
      break


def gradient_descent(X: torch.Tensor, Y: torch.Tensor, p: Parameters, debug=True, training_steps=1000):
  p.print_count()
  p.prepare_for_backward()

  for _ in range(training_steps):
    embeddings = p.features[X].view(-1, 6) # (32, 6)

    # Forward
    a1 = embeddings @ p.W1 + p.b1
    z1 = torch.tanh(a1)

    a2 = z1 @ p.W2 + p.b2

    loss = F.cross_entropy(a2, Y)

    # Backward
    p.reset_gradients()
    loss.backward()

    # Update
    for parameter in p.get_parameters():
      parameter.data += -0.1 * parameter.grad

  if debug:
    print(f"loss: {loss.item()}")

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

def initialize_network_parameters(generator: torch.Generator, characters_features: torch.Tensor, first_layer_size: int, second_layer_size: int):
  W1 = torch.randn((first_layer_size, second_layer_size), generator=generator)
  b1 = torch.randn(second_layer_size, generator=generator)
  W2 = torch.randn((second_layer_size, CHARACTERS_NUMBER), generator=generator)
  b2 = torch.randn(CHARACTERS_NUMBER, generator=generator)

  return Parameters(
    features= characters_features,
    W1= W1,
    b1= b1,
    W2= W2,
    b2= b2,
  )

def create_lookup_table(g: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
  """
  Builds the features table associated with the given input X

  Returns:
    embedding: A tensor of shape (c, l) where c is the number of unique characters and
               l is total of features for a character. In this case c is 27 and l is 2

  """
  # Randomly generate a two dimensional vector for each character
  characters_features = torch.randn((CHARACTERS_NUMBER, CHARACTER_FEATURES_SIZE), generator=g) 
  
  return characters_features


def build_dataset(names: list[str], stoi: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
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


def generate_token_mappings(names: list[str]) -> Tuple[list[str], Dict[str, int], Dict[int, str]]:
  """
  Extracts the tokens for the character-level model and then return the character mappings
  for the dataset of names

  Args:
    names: A list of names
  
  Returns:
    characters: The list of tokens (characters) used by the model
    itos: A mapping form integers to characters
    stoi: A mapping form characters to integers
  """

  # Get the set of unique characters in the dataset
  characters = sorted(list(set(''.join(names))))

  # Create dictionary to map integers to characters
  stoi = {c: i + 1 for i, c in enumerate(characters)}
  stoi[END_START_CHARACTER] = 0

  # Create dictionary to map encoded characters to their corresponding symbols
  itos = {i: c for c,i in stoi.items()}

  return characters, itos, stoi


def get_dataset(path: str) -> list[str]:
  """
  Obtains the dataset of names in the format of a list of strings 
  """
  with open(path, 'r') as file:
    lines = file.read().splitlines()

  return lines


if __name__ == "__main__":
  main()