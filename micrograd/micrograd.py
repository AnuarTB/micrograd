import math
from dataclasses import dataclass
import queue

def scalar_to_value(func):
    """All Value operations can accept scalar values.
    
    Casts any scalar values that are used in operations with
    Value to Value type.
    """
    def wrapper(*args):
        if isinstance(args[1], (float, int)):
            return func(args[0], Value(args[1]))
        else:
            return func(*args)
    return wrapper

class Value(object):
  def __init__(self, data):
    self.data = data
    self.operands = [] # list[Operand]
    self.grad = 0
    self.is_enqueued = False
    self.op = ''

  def __repr__(self):
    return f'val={self.data}'

  @scalar_to_value
  def __add__(self, other):
    out = Value(self.data + other.data)
    out.operands = [
        Operand(self, 1),
        Operand(other, 1)
    ]
    out.op = '+'
    return out

  @scalar_to_value
  def __sub__(self, other):
    out = Value(self.data - other.data)
    out.operands = [
      Operand(self, 1),
      Operand(self, -1)
    ]
    out.op = '-'
    return out

  @scalar_to_value
  def __mul__(self, other):
    out = Value(self.data * other.data)
    out.operands = [
        Operand(self, other.data),
        Operand(other, self.data)
    ]
    out.op = '*'
    return out

  @scalar_to_value
  def __truediv__(self, other):
    out = Value(self.data / other.data)
    out.operands = [
      Operand(self, 1 / other.data),
      Operand(other, -self.data / (other.data ** 2))
    ]
    out.op = '/'
    return out

  @scalar_to_value
  def __pow__(self, other):
    out = Value(self.data ** other)
    out.operands = [
        Operand(self, other * (self.data ** (other - 1))),
        Operand(other, out.data * math.log(self.data))
    ]
    out.op = '^'
    return out

  def __rmul__(self, other):
    return self * other
 
  def __radd__(self, other):
    return self + other
  
  def __rsub__(self, other):
    return Value(other) - self

  def __rtruediv__(self, other):
    return Value(other) / self

  def backward(self):
    self.grad = 1
    self.is_enqueue = True

    q = queue.Queue()
    q.put(self)

    while not q.empty():
      v = q.get()

      for operand in v.operands:
        operand.val.grad += v.grad * operand.deriv
        if not operand.val.is_enqueued:
          operand.val.is_enqueued = True
          q.put(operand.val)

  def zero_grad(self):
    self.grad = 0
    self.is_enqueued = False
    for operand in self.operands:
      operand.val.zero_grad()

@dataclass
class Operand:
  val: Value
  deriv: float