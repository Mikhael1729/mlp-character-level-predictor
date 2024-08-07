import argparse
import os
from typing import Dict, Tuple
import torch
import torch.nn.functional as F
import random
from exploration_helpers import LearningRateStatistics, StepLossesStatistics

random.seed(42)

BLOCK_SIZE = 3 # Context lenght. How many characters are needed to predict the next one
CHARACTER_FEATURES_SIZE = 2
CHARACTERS_NUMBER = 27

END_START_CHARACTER = '.'
DATASET_PATH = "./names.txt"


class Hyperparameters:
  def __init__(self, learning_rate: float, training_steps: int, mini_batch_size: int):
    self.learning_rate = learning_rate
    self.training_steps = training_steps
    self.minibatch_size = mini_batch_size
  
  def __str__(self):
    return f"Training steps: {self.training_steps}\nLearning rate: {self.learning_rate}\nMinibatch size: {self.minibatch_size}"


class Parameters:
  def __init__(self, features: torch.Tensor, W1: torch.Tensor, b1: torch.Tensor, W2: torch.Tensor, b2: torch.Tensor):
    self.features = features
    self.W1 = W1
    self.b1 = b1
    self.W2 = W2
    self.b2 = b2
    self.parameters_list = [self.features, self.W1, self.b1, self.W2, self.b2]

  def print_count(self):
    print(f"Parameters: {sum(p.nelement() for p in self.get_parameters())}")
    print(f"W1: {self.W1.shape}")
    print(f"b1: {self.b1.shape}")
    print(f"W2: {self.W2.shape}")
    print(f"b2: {self.b2.shape}")

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
  def __init__(self, name: str, X: torch.Tensor, Y: torch.Tensor):
    self.X = X
    self.Y = Y
    self.name = name

class Datasets:
  def __init__(self, names: list[str], stoi: Dict[int, str]):
    random.shuffle(names)

    n = len(names)
    eighty_percent_index = int(0.8 * n)
    ninety_percent_index = int(0.9 * n)

    X_train, Y_train = build_dataset(names[:eighty_percent_index], stoi)
    X_dev, Y_dev = build_dataset(names[eighty_percent_index:ninety_percent_index], stoi)
    X_test, Y_test = build_dataset(names[ninety_percent_index:], stoi)
    X_all, Y_all = build_dataset(names, stoi)

    self.train = Dataset("train", X_train, Y_train)
    self.dev = Dataset("dev", X_dev, Y_dev)
    self.test = Dataset("test", X_test, Y_test)
    self.all = Dataset("all", X_all, Y_all)

ARG_LEARNING_RATE = "learning-rate"
ARG_TRAINING_LOSS = "training-loss"

def main():
  parser = argparse.ArgumentParser(description="Names Generator (MLP approach)")
  parser.add_argument("-d", "--display-statistics", choices=[ARG_LEARNING_RATE, ARG_TRAINING_LOSS], help='Plots the chosen graphic after training')


  args = parser.parse_args()
  explore_learning_rates = 'Y' if args.display_statistics == ARG_LEARNING_RATE else 'N'
  explore_training_loss = True if args.display_statistics == ARG_TRAINING_LOSS else False

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

  repeated_hyper_parameters = None
  continue_training = None

  # Train loop

  while True:
    # Define the hyperparameters of the network
    if explore_learning_rates == "Y":
      clear_console()
      print("Exploration settings: \n")

      hyperparameters = repeated_hyper_parameters or Hyperparameters(
        training_steps=None,
        learning_rate=None,
        mini_batch_size=int(input("- Minibatch size (default 32): ") or 32),
      )
    else:
      if continue_training == None:
        clear_console()
        print("Training settings: \n")

      hyperparameters = repeated_hyper_parameters or Hyperparameters(
        training_steps=int(input("- Training steps (default 100): ") or 100),
        learning_rate=float(input("- Learning rate (default 0.1): ") or 0.1),
        mini_batch_size=int(input("- Minibatch size (default 32): ") or 32)
      )

      print("")

    repeated_hyper_parameters = None

    # Train network
    gradient_descent(
      train_set=datasets.all if explore_learning_rates == "Y" else datasets.train,
      p=parameters,
      hyperparameters=hyperparameters,
      debug=True,
      display_learning_rate_stats=True if explore_learning_rates == "Y" else False,
      display_steps_stats=explore_training_loss
    )

    # Test network with dev set
    loss = forward2(datasets.dev, parameters)
    print(f"Test loss: {loss}\n---\n")

    if explore_learning_rates == 'Y':
      continue

    continue_training = input("Continue training? (y/n/r): ")
    print("")
    print("---")

    if continue_training == "n":
      break

    if continue_training == "r":
      repeated_hyper_parameters = hyperparameters
      print(repeated_hyper_parameters)

def gradient_descent(train_set: Dataset, p: Parameters, hyperparameters: Hyperparameters, debug=True, display_learning_rate_stats=False, display_steps_stats=False):
  # Prepare for gradient descent
  lr_stats, steps_stats = init_statistics(p, display_learning_rate_stats, display_steps_stats, debug)
  steps = hyperparameters.training_steps if lr_stats is None else lr_stats.steps

  p.prepare_for_backward()

  for i in range(steps):
    # Create minibatches for training
    minibatch_indices = get_indices_mini_batch(train_set.X.shape[0], hyperparameters.minibatch_size)
    minibatch = Dataset(f"mini_of_{hyperparameters.minibatch_size}_from_{train_set.name}", train_set.X[minibatch_indices], train_set.Y[minibatch_indices])

    # Perform forward pass
    loss = forward2(minibatch, p)

    # Perform backward pass
    p.reset_gradients()
    loss.backward()

    # Optimize the network
    learning_rate = hyperparameters.learning_rate if lr_stats is None else lr_stats.learning_rate_space[i]

    for parameter in p.get_parameters():
      parameter.data += -learning_rate * parameter.grad

    handle_statistics_during_iterations(steps_stats, lr_stats, i, loss)

  plot_statistics(steps_stats, lr_stats, debug, loss)

def init_statistics(p: Parameters, display_learning_rate_stats: bool, display_step_stats: bool, debug: bool) -> Tuple[LearningRateStatistics | None, StepLossesStatistics | None]:
  learning_rate_stats: LearningRateStatistics = None
  steps_stats: LearningRateStatistics = None

  if display_learning_rate_stats:
    learning_rate_stats = LearningRateStatistics(
      lower_bound=int(input("- Lower bound (default -3): ") or -3),
      upper_bound=int(input("- Upper bound (default 1): ") or 1),
      steps=int(input("- Number of steps (default 1000): ") or 1000),
    )

    print("\n")

  if display_step_stats:
    steps_stats = StepLossesStatistics()

  if debug:
    print("Parameters info:\n")
    p.print_count()
    print("\n")

  return learning_rate_stats, steps_stats

def handle_statistics_during_iterations(steps_statistics: StepLossesStatistics | None, learning_rate_statistics: LearningRateStatistics | None, step: int, loss: torch.Tensor):
  if learning_rate_statistics is not None:
    learning_rate_statistics.add_record(step, loss)

  if steps_statistics is not None:
    steps_statistics.add_record(step, loss)

def plot_statistics(steps_statistics: StepLossesStatistics | None, learning_rate_statistics: LearningRateStatistics | None, debug: bool, loss: torch.Tensor):
  if debug:
    print(f"Train loss: {loss.item()}")

  if learning_rate_statistics is not None:
    learning_rate_statistics.plot_exponents_stats()

  if steps_statistics is not None:
    steps_statistics.plot()

def get_indices_mini_batch(samples_size: int, mini_batch_size: int):
  return torch.randint(0, samples_size, (mini_batch_size,))

def forward2(dataset: Dataset, p: Parameters) -> torch.Tensor:
  embeddings = p.features[dataset.X].view(-1, 6) # (m, 6)

  a1 = embeddings @ p.W1 + p.b1
  z1 = torch.tanh(a1)

  a2 = z1 @ p.W2 + p.b2

  loss = F.cross_entropy(a2, dataset.Y)

  return loss

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

  for word in names:
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
    itos: A mapping of integers to characters
    stoi: A mapping of characters to integers
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

def clear_console():
    os.system('cls' if os.name=='nt' else 'clear')

if __name__ == "__main__":
  main()
