#!/usr/bin/env python3
import numpy as np
from src.utils import Sigmoid, ReLU, truncated_norm

activation = Sigmoid()

class NeuralNetwork:
    def __init__(self, topology, learning_rate, bias=None):
        input_size, hidden_size, output_size = topology 
        self.no_of_in_nodes = input_size 
        self.no_of_hidden_nodes = hidden_size
        self.no_of_out_nodes = output_size
        self.learning_rate = learning_rate
        self.bias = bias
        self.create_weight_matrices()

    def create_weight_matrices(self):
        bias_node = 1 if self.bias else 0
        # Weights between input and hidden
        r = 1 / np.sqrt(self.no_of_in_nodes + bias_node)
        X = truncated_norm(mean=0, sd=1, low=-r, upp=r)
        self.weights_in_hidden = X.rvs(
            ( self.no_of_hidden_nodes, self.no_of_in_nodes + bias_node )
        )
        # Weights between hidden and output
        r = 1 / np.sqrt(self.no_of_hidden_nodes + bias_node)
        X = truncated_norm(mean=0, sd=1, low=-r, upp=r)
        self.weights_hidden_out = X.rvs(
            ( self.no_of_out_nodes, self.no_of_hidden_nodes + bias_node )
        )

    def train(self, input_vector, target_vector, objective=None):
        bias_node = 1 if self.bias else 0
        if self.bias:
            # adding bias node to the end of the input->hidden layer
            input_vector = np.concatenate( (input_vector, [ bias_node ]) )
        """
         Feedforward input and decompose each layer to compute error incrementally
        """
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T
        # Decompose input layer
        output_vector1 = np.dot(self.weights_in_hidden, input_vector)
        output_vector_hidden = activation.activate(output_vector1)
        
        if self.bias:
            # add bias node to end of hidden->output layer
            output_vector_hidden = np.concatenate( (output_vector_hidden, [[ bias_node ]]) )
        # Decompose hidden layer
        output_vector2 = np.dot(self.weights_hidden_out, output_vector_hidden)
        output_vector_network = activation.activate(output_vector2)
        # Compute error
        if objective:
            output_errors = objective(target_vector, output_vector_network)
        else:
            output_errors = target_vector - output_vector_network
        """
         Update weights via gradient descent
         Uses derivative of activation (i.e gradient of activation layer)
         To determine how much to adjust weights, with the objective to 
         minimize error
        """
        # Update: Hidden <- Output <- Errors
        gradient = output_errors * activation.gradient(output_vector_network)
        delta = -self.learning_rate * np.dot(gradient, output_vector_hidden.T)
        self.weights_hidden_out += delta
        # Update:  Input <- Hidden <- Errors
        hidden_errors = np.dot(self.weights_hidden_out.T, output_errors)
        gradient = hidden_errors * activation.gradient(output_vector_hidden)
        if self.bias:
            gradient = np.dot(gradient, input_vector.T)[:-1,:] # last element cut off
        else:
            gradient = np.dot(gradient, input_vector.T)
        delta = -self.learning_rate * gradient
        self.weights_in_hidden += delta

    def run(self, input_vector):
        """
        running the network with an input vector input_vector. 
        input_vector can be tuple, list or ndarray
        """
        bias_node = 1 if self.bias else 0
        if self.bias:
            # adding bias node to the end of the inpuy_vector
            input_vector = np.concatenate( (input_vector, [ bias_node ]) )
        input_vector = np.array(input_vector, ndmin=2).T
        # Input -> Hidden
        output_vector = np.dot(self.weights_in_hidden, input_vector)
        output_vector = activation.activate(output_vector)
        # Hidden -> Output
        if self.bias:
            output_vector = np.concatenate( (output_vector, [[ bias_node ]]) )
        output_vector = np.dot(self.weights_hidden_out, output_vector)
        output_vector = activation.activate(output_vector)
        return output_vector
