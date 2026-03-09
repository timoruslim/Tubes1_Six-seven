from .engine import Tensor
from .activation import ACTIVATIONS
from .initialize import INITIALIZATIONS
from .optimizer import OPTIMIZER
from .loss import LOSSES

import numpy as np

class Module:

   def parameters(self):
      return []

   def zero_grad(self):
      for p in self.parameters():
         p.grad = np.zeros_like(p.data)

class Layer(Module):

   def __init__(self, n_neurons, input_size=None, activation='relu', weight_init='xavier', bias_init='zero', seed=None):

      self.n_neurons = n_neurons
      self.input_size = input_size
      self.weight_init = weight_init
      self.bias_init = bias_init
      self.seed = seed

      rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
      self.weights = self._initialize_parameters(input_size, n_neurons, weight_init, rng) if input_size is not None else None
      self.bias = self._initialize_parameters(1, n_neurons, bias_init, rng)
      
      self.activation = ACTIVATIONS.get(activation) if activation in ACTIVATIONS else None
      if self.activation is None:
         raise ValueError(f"Unknown activation function: {activation}")

   def _initialize_parameters(self, rows, cols, init_method, rng): 

      method = init_method.pop('method', init_method) if isinstance(init_method, dict) else init_method

      if isinstance(init_method, dict):
         method = init_method.get('method')
         params = {key : value for key, value in init_method.items() if key != 'method'}
      else:
         method = init_method
         params = {}
      
      initializer = INITIALIZATIONS.get(method) if method in INITIALIZATIONS else None
      if initializer is None:
         raise ValueError(f"Unknown initialization method: {init_method}")
      
      return initializer(rows, cols, params, rng)

   def parameters(self):
      return [self.weights, self.bias] if self.weights is not None else [self.bias]

   def __call__(self, input):

      if self.weights is not None:
         Z = input @ self.weights + self.bias
      else:
         raise ValueError("Layer input size is not defined. ")
      
      if self.activation is not None:
         return self.activation(Z)
      else:
         raise ValueError("Layer activation is not defined. ") 

class MLP(Module):

   def __init__(self, layers=None, input_size=None, seed=None):
      
      self.seed = seed
      self.input_size = input_size
      self.layers = list(layers) if layers is not None else []
      self._cascade_inputs(self.layers, input_size)

   def parameters(self):
      params = []
      for layer in self.layers:
         params.extend(layer.parameters())
      return params
   
   def add(self, layers):
      layers = list(layers) if isinstance(layers, list) else [layers]
      if len(self.layers) > 0:
         self._push_input(self.layers[-1], layers[0])
      self._cascade_inputs(layers, layers[0].input_size)
      self.layers.extend(layers)

   def _add_input(self, layer, input_size):

      if layer.input_size is not None and layer.input_size != input_size:
         raise ValueError(f"Input size mismatch: expected {layer.input_size}, got {input_size}")
      
      else:
         if input_size is None:
            raise ValueError("Input size must be specified for the first layer")
         rng = np.random.default_rng(layer.seed if layer.seed is not None else self.seed)
         layer.weights = layer._initialize_parameters(input_size, layer.n_neurons, layer.weight_init, rng)
         layer.bias = layer._initialize_parameters(1, layer.n_neurons, layer.bias_init, rng)
         layer.input_size = input_size
   
   def _push_input(self, prev_layer, layer):
      self._add_input(layer, prev_layer.n_neurons)

   def _cascade_inputs(self, layers, initial_input_size):
      if len(layers) == 0:
         return
      self._add_input(layers[0], initial_input_size)
      for i in range(1, len(layers)):
         self._push_input(layers[i-1], layers[i])

   def compile(self, optimizer, loss):
      self.loss_fn = LOSSES.get(loss) if loss in LOSSES else None

      method = optimizer.pop('optimizer', optimizer) if isinstance(optimizer, dict) else optimizer

      if isinstance(optimizer, dict):
         method = optimizer.get('method')
         params = {key : value for key, value in optimizer.items() if key != 'optimizer'}
      else:
         method = optimizer
         params = {}
      
      self.optimizer = OPTIMIZER.get(method) if method in OPTIMIZER else None
      if self.optimizer is None:
         raise ValueError(f"Unknown optimizer: {optimizer}")

   def __call__(self, input):
      output = input if isinstance(input, Tensor) else Tensor(input)
      for layer in self.layers:
         output = layer(output)
      return output