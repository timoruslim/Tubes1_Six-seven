from .engine import Tensor, unbroadcast
import numpy as np

class Optimizer:

   def __init__(self, parameters, learning_rate=0.01, params=None):
      self.parameters = parameters
      self.alpha = learning_rate
      self.params = params if params is not None else {}

   def step(self):
      raise NotImplementedError("Optimizer step method must be implemented by subclasses")

   def zero_grad(self):
      for p in self.parameters:
         p.grad = np.zeros_like(p.data)


class SGD(Optimizer):

   def step(self):
      for p in self.parameters:
         p.data -= self.alpha * p.grad

OPTIMIZER = {
   'sgd': SGD
}