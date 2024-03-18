import random

from micrograd import Value

class Neuron:
  def __init__(self, in_dim):
    self.w = [Value(random.uniform(-1, 1)) for _ in range(in_dim)]
    self.b = [Value(random.uniform(-1, 1))]

  def __call__(self, x):
    act = sum(wi*xi for wi, xi in zip(self.w, x)) + self.b
    out = act.tanh()
    return out

class Linear:
  def __init__(self, in_dim, out_dim):
    self.neurons = [Neuron(in_dim) for _ in range(out_dim)]
  
  def __call__(self, x):
    # x should be an array of length `in_dim`.
    return [self.neurons(x) for _ in self.neurons]

class Net:
  def __init__(self, layers):
    self.layers = layers
    pass