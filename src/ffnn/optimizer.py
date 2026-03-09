from .engine import Tensor, unbroadcast
import numpy as np

class Optimizer:

   def __init__(self, parameters, alpha=0.01):
      self.parameters = parameters
      self.alpha = alpha

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