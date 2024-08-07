import torch
import matplotlib.pyplot as plt

class StepLossesStatistics:
  def __init__(self):
    self.steps = []
    self.losses = []

  def add_record(self, step: int, loss: torch.Tensor):
    self.steps.append(step)
    self.losses.append(loss.item())

  def plot(self):
    plt.title("Steps Stats (step, loss)")
    plt.plot(self.steps, self.losses)
    plt.show()

class LearningRateStatistics:
  def __init__(self, lower_bound: int, upper_bound: int, steps=1000):
    self.steps = steps
    self.exponents_space = torch.linspace(lower_bound, upper_bound, steps)
    self.learning_rate_space = 10**self.exponents_space
    self.learning_rates = []
    self.exponents = []
    self.losses = []
    self.steps = []

  def add_record(self, step: int, loss: torch.Tensor):
    self.learning_rates.append(self.learning_rate_space[step])
    self.exponents.append(self.exponents_space[step])
    self.losses.append(loss.item())
    self.steps.append(step)

  def plot_exponents_stats(self):
    plt.title("Exponent Stats (exponent, loss)")
    plt.plot(self.exponents_space, self.losses)
    plt.show()

  def plot_learning_rates_stats(self):
    plt.title("Learning rate Stats (learning, loss)")
    plt.plot(self.learning_rates, self.losses)
    plt.show()