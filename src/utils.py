#!/usr/bin/env python3
import numpy as np
from scipy.special import expit
from scipy.stats import truncnorm

''' Truncated normal distribution '''
def truncated_norm(mean=0, sd=1, low=0, upp=10):
    a = (low - mean) / sd
    b = (upp - mean) / sd
    return truncnorm(a, b, loc=mean, scale=sd)

''' Mixin for vector-izing neural network activations '''
class Activator:
    def activation(self, X):
        return 0 if X <= 0 else 1
    
    def activation_gradient(self, X):
        return 0

    def activate(self, X):
        return np.vectorize(lambda z:  self.activation(z))(X)

    def gradient(self, X):
        return np.vectorize(lambda z: self.activation_gradient(z))(X)

class Sigmoid(Activator):
    def activation(self, X):
        return expit(X)
        #return 1 / (1 + np.e ** -X) # <- manual implementation
    
    def activation_gradient(self, X):
        z = self.activation(X)
        return z * (1.0 - z)

class ReLU(Activator):
    def activation(self, X):
        return np.maximum(0.0, X)
    
    def activation_gradient(self, X):
        return 0 if X <= 0 else 1

    