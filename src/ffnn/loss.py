from .engine import Tensor, unbroadcast
import numpy as np

def mse(pred, target):
   mse_loss = Tensor(np.mean((pred.data - target.data) ** 2), (pred, target))
   def _backward():
      pred.grad += unbroadcast(2 * (pred.data - target.data) / pred.data.size * mse_loss.grad, pred.data.shape) # dL/dP = dL/dMSE * dMSE/dP = 2 * (P - T) / N
      target.grad -= unbroadcast(2 * (pred.data - target.data) / target.data.size * mse_loss.grad, target.data.shape) # dL/dT = dL/dMSE * dMSE/dT = -2 * (P - T) / N
   mse_loss._backward = _backward
   return mse_loss

def bce(pred, target):
   bce_loss = Tensor(-np.mean(target.data * np.log(pred.data + 1e-15) + (1 - target.data) * np.log(1 - pred.data + 1e-15)), (pred, target))
   def _backward():
      pred.grad -= unbroadcast(((target.data / (pred.data + 1e-15)) - ((1 - target.data) / (1 - pred.data + 1e-15))) / pred.data.size * bce_loss.grad, pred.data.shape) # dL/dP = dL/dBCE * dBCE/dP = - (T / P - (1 - T) / (1 - P)) / N
      target.grad -= unbroadcast((np.log(pred.data + 1e-15) - np.log(1 - pred.data + 1e-15)) / target.data.size * bce_loss.grad, target.data.shape) # dL/dT = dL/dBCE * dBCE/dT = -(log(P) - log(1 - P)) / N
   bce_loss._backward = _backward
   return bce_loss

def cce(pred, target):
   cce_loss = Tensor(-np.mean(np.sum(target.data * np.log(pred.data + 1e-15), axis=-1)), (pred, target)) 
   def _backward():
      pred.grad -= unbroadcast((target.data / (pred.data + 1e-15)) / pred.data.shape[0] * cce_loss.grad, pred.data.shape) # dL/dP = dL/dCCE * dCCE/dP = -T / P / N
      target.grad -= unbroadcast((np.log(pred.data + 1e-15)) / target.data.shape[0] * cce_loss.grad, target.data.shape) # dL/dT = dL/dCCE * dCCE/dT = -log(P) / N
   cce_loss._backward = _backward
   return cce_loss