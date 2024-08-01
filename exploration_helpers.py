import torch
import matplotlib.pyplot as plt

class LearningRateStatistics:
  def __init__(self, lower_bound: int, upper_bound: int, steps=1000):
    self.steps = steps
    self.exponents_space = torch.linspace(lower_bound, upper_bound, steps)
    self.learning_rate_space = 10**self.exponents_space
    self.learning_rates = []
    self.exponents = []
    self.losses = []

  def add_record(self, step: int, loss: torch.Tensor):
    self.learning_rates.append(self.learning_rate_space[step])
    self.exponents.append(self.exponents_space[step])
    self.losses.append(loss.item())

  def plot_exponents_stats(self):
    plt.plot(self.exponents_space, self.losses)
    plt.show()

  def plot_learning_rates_stats(self):
    plt.plot(self.learning_rates, self.losses)
    plt.show()