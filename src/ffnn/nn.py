from .engine import Tensor
from .activation import ACTIVATIONS
from .initialize import INITIALIZATIONS
from .optimizer import OPTIMIZER
from .loss import LOSSES

import pickle
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np

class Module:

   def parameters(self):
      return []

   def zero_grad(self):
      for p in self.parameters():
         p.grad = np.zeros_like(p.data)

class Layer(Module):

   def __init__(self, n_neurons, input_size=None, activation='relu', weight_init='xavier', bias_init='zero', seed=None, l1=0.0, l2=0.0):

      self.n_neurons = n_neurons
      self.input_size = input_size
      self.weight_init = weight_init
      self.bias_init = bias_init
      self.seed = seed
      self.l1 = l1
      self.l2 = l2

      rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
      self.weights = self._initialize_parameters(input_size, n_neurons, weight_init, rng) if input_size is not None else None
      self.bias = self._initialize_parameters(1, n_neurons, bias_init, rng)
      
      self.activation = ACTIVATIONS.get(activation) if activation in ACTIVATIONS else None
      if self.activation is None:
         raise ValueError(f"Unknown activation function: {activation}")

   def _initialize_parameters(self, rows, cols, init_method, rng): 

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
   
   def regularization_loss(self):
      reg_loss = Tensor(0.0) 
      
      if self.weights is not None:
         if self.l1 > 0:
            reg_loss = reg_loss + self.l1 * abs(self.weights).sum()
         if self.l2 > 0:
            reg_loss = reg_loss + self.l2 * (self.weights ** 2).sum()
            
      return reg_loss

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
      self.loss_fn = None
      self.optimizer = None

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
      
      # First layer 
      if initial_input_size is None and layers[0].input_size is not None:
         initial_input_size = layers[0].input_size
         self.input_size = initial_input_size 
         
      self._add_input(layers[0], initial_input_size)
      
      # Next layers
      for i in range(1, len(layers)):
         self._push_input(layers[i-1], layers[i])

   def compile(self, optimizer, loss):
      self.loss_fn = LOSSES.get(loss) 
      if self.loss_fn is None:
         raise ValueError(f"Unknown loss function: {loss}")

      if isinstance(optimizer, dict):
         method = optimizer.get('method', optimizer.get('optimizer'))
         params = {key : value for key, value in optimizer.items() if key not in ('method', 'optimizer')}
      else:
         method = optimizer
         params = {}
      
      optimizer_class = OPTIMIZER.get(method) if method in OPTIMIZER else None
      if optimizer_class is None:
         raise ValueError(f"Unknown optimizer: {optimizer}")
      
      self.optimizer = optimizer_class(self.parameters(), **params)

   def _regularization_loss(self):
      total_reg = Tensor(0.0)
      for layer in self.layers:
         total_reg = total_reg + layer.regularization_loss()
      return total_reg

   def fit(self, X, y, batch_size=32, epochs=100, validation_data=None, verbose=1):

      if self.optimizer is None or self.loss_fn is None:
         raise ValueError("Model must be compiled with an optimizer and loss function before training.")
      
      n_samples = X.shape[0]
      rng = np.random.default_rng(self.seed)

      history = {'train_loss': [], 'val_loss': []}
      
      epoch_iterator = tqdm(range(epochs), disable=(verbose == 0), desc="Training FFNN")
      for _ in epoch_iterator:

         indices = rng.permutation(n_samples)
         X_shuffled = X[indices]
         y_shuffled = y[indices]
         
         for i in range(0, n_samples, batch_size):

            # Forward Pass
            pred = self(X_shuffled[i:i+batch_size])
            data_loss = self.loss_fn(pred, y_shuffled[i:i+batch_size])
            reg_loss = self._regularization_loss()
            total_loss = data_loss + reg_loss

            # Backward Pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
         
         # Record loss
         train_loss = self.loss_fn(self(X), y).data.item()
         history['train_loss'].append(train_loss)
         metrics = {'train_loss': f"{train_loss:.4f}"}

         if validation_data is not None:
            X_val, y_val = validation_data
            val_loss = self.loss_fn(self(X_val), y_val).data.item()
            history['val_loss'].append(val_loss)
            metrics['val_loss'] = f"{val_loss:.4f}"
   
         epoch_iterator.set_postfix(metrics)
         
      return history

   def plot_weights(self, layer_indices):
      
      plt.figure(figsize=(10, 5))
      for idx in layer_indices:
         if 0 <= idx < len(self.layers) and self.layers[idx].weights is not None:
            weights_flat = self.layers[idx].weights.data.flatten()
            plt.hist(weights_flat, bins=50, alpha=0.5, label=f'Layer {idx+1}')
            
      plt.title('Weight Distributions')
      plt.xlabel('Value')
      plt.ylabel('Frequency')
      plt.legend()
      plt.show()

   def plot_gradients(self, layer_indices):
      import matplotlib.pyplot as plt
      
      plt.figure(figsize=(10, 5))
      for idx in layer_indices:
         if 0 <= idx < len(self.layers) and self.layers[idx].weights is not None:
            grads_flat = self.layers[idx].weights.grad.flatten()
            plt.hist(grads_flat, bins=50, alpha=0.5, label=f'Layer {idx+1}')
            
      plt.title('Gradient Distributions')
      plt.xlabel('Value')
      plt.ylabel('Frequency')
      plt.legend()
      plt.show()

   def save(self, filepath):
      model_state = []
      for layer in self.layers:
         layer_state = {
            'weights': layer.weights.data if layer.weights is not None else None,
            'bias': layer.bias.data if layer.bias is not None else None
         }
         model_state.append(layer_state)
         
      with open(filepath, 'wb') as f:
         pickle.dump(model_state, f)
      print(f"Model successfully saved to {filepath}")

   def load(self, filepath):
      with open(filepath, 'rb') as f:
         model_state = pickle.load(f)
         
      if len(model_state) != len(self.layers):
         raise ValueError("The saved model has a different number of layers.")
         
      for i, layer_state in enumerate(model_state):
         if self.layers[i].weights is not None and layer_state['weights'] is not None:
            self.layers[i].weights.data = layer_state['weights']
         if self.layers[i].bias is not None and layer_state['bias'] is not None:
            self.layers[i].bias.data = layer_state['bias']
            
      print(f"Model successfully loaded from {filepath}")

   def summary(self):
      print("\n" + "=" * 67)
      print(f"{'Layer':<17} {'Matrix Shape':<17} {'Activation':<17} {'Param #':<16}")
      print("=" * 67)
      
      total_params = 0
      
      for idx, layer in enumerate(self.layers):

         # Get shape 
         output_shape = f"({layer.input_size}, {layer.n_neurons})"
         
         # Get activation 
         if hasattr(layer.activation, '__name__'):
            activation_name = layer.activation.__name__
         else:
            activation_name = str(layer.activation)
         
         # Get parameter count
         layer_params = 0
         if layer.weights is not None:
            layer_params += layer.weights.data.numel() if hasattr(layer.weights.data, 'numel') else np.prod(layer.weights.data.shape)
         if layer.bias is not None:
            layer_params += layer.bias.data.numel() if hasattr(layer.bias.data, 'numel') else np.prod(layer.bias.data.shape)
         
         total_params += layer_params
         
         # Print info
         layer_name = f"layer_{idx+1}"
         print(f"{layer_name:<17} {output_shape:<17} {activation_name:<17} {layer_params:<16,}")
      
      print("=" * 67)
      print(f"Total parameters: {total_params:,}")
      print("=" * 67 + "\n")

   def __call__(self, input):
      output = input if isinstance(input, Tensor) else Tensor(input)
      for layer in self.layers:
         output = layer(output)
      return output