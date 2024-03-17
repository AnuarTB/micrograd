import queue
from dataclasses import dataclass

class Value(object):
  def __init__(self, data):
    self.data = data
    self.operands = [] # list[Operand]
    self.grad = 0
    self.is_enqueued = False
    self.op = ''

  def __repr__(self):
    return f'val={self.data}'

  def __add__(self, other):
    out = Value(self.data + other.data)
    out.operands = [
        Operand(self, 1),
        Operand(other, 1)
    ]
    out.op = '+'
    return out

  def __mul__(self, other):
    out = Value(self.data * other.data)
    out.operands = [
        Operand(self, other.data),
        Operand(other, self.data)
    ]
    out.op = '*'
    return out

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