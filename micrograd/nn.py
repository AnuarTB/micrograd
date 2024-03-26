import abc
import random

from value import Value


class Neuron:
  def __init__(self, in_dim):
    self.w = [Value(random.uniform(-1, 1)) for _ in range(in_dim)]
    self.b = Value(random.uniform(-1, 1))

  def __call__(self, x):
    act = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
    out = act.tanh()
    return out

  def parameters(self):
    return self.w + [self.b]


class Layer(abc.ABC):
  @abc.abstractmethod
  def parameters(self):
    pass


class Linear(Layer):
  def __init__(self, in_dim, out_dim):
    self.neurons = [Neuron(in_dim) for _ in range(out_dim)]

  def __call__(self, x):
    # x should be an array of length `in_dim`.
    out = [n(x) for n in self.neurons]
    return out[0] if len(out) == 1 else out

  def parameters(self):
    params = []
    for n in self.neurons:
      params += n.parameters()
    return params


class Net(abc.ABC):
  @abc.abstractmethod
  def __init__(self):
    pass

  @abc.abstractmethod
  def forward(self, x):
    pass

  def __call__(self, x):
    return self.forward(x)

  def parameters(self):
    params = []
    for prop in self.__dict__.values():
      if isinstance(prop, Layer):
        params += prop.parameters()
    return params
