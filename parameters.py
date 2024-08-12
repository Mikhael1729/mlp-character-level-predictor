import torch

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
