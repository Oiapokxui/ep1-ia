'''
Integrantes
Enrique Emanuel Rezende Tavares da Silva - 11796090
Guilherme Dias Jimenes - 11911021
Ronald Cosmo de Sousa - 11909783
'''
from functools import reduce
import math
import random


def multi_layer_perceptron_iter(input_layer, responses, num_of_hidden_layers, hidden_layer_dims, output_layer_dims, activation_function):

    output_layer, weights_matrix = feed_forward_network(
        input_layer,
        num_of_hidden_layers,
        hidden_layer_dims,
        output_layer_dims,
        activation_function
    )

    backpropagation()


def backpropagation(derivative_activation_function):

    def error(expected_response, obtained_response):
    pass


"""
Initializes weights for the whole network and feeds them forward in order to get the output layer of the network.
Returns the output_layer and the weights obtained.
"""


def feed_forward_network(input_layer, num_of_hidden_layers, hidden_layer_dims, output_layer_dims, activation_function):
    current_input = input_layer
    weights_dim = len(input_layer)
    weights = []

    for _ in range(num_of_hidden_layers):
        layer_weights = initialize_weights(hidden_layer_dims, weights_dim)
        hidden_layer = feed_forward_layer(
            current_input, layer_weights, activation_function)
        weights_dim = len(hidden_layer)
        current_input = hidden_layer
        weights.append(layer_weights)

    final_weights = initialize_weights(output_layer_dims, weights_dim)
    output_layer = feed_forward_layer(
        current_input, final_weights, activation_function)
    weights.append(final_weights)

    return output_layer, weights


def update_weights(inputs, weights, error, learning_rate):
    def update_weight(component, weight):
        return weight + (learning_rate * component * error)
    return list(map(update_weight, inputs, weights))


"""
Feeds forward, creating a new layer.
This works by applying a weights vector to the inputs, obtaining a response for each weights vector
"""


def feed_forward_layer(vector, weights, activation_function):

    def feed_forward_neuron(vector, weights, activation_function):
        dims_range = range(len(vector))
        linear_combination = map(
            lambda dim: vector[dim] * weights[dim], dims_range)
        net = reduce(lambda item, acc: item + acc, linear_combination)
        return activation_function(net)

    # This expects that weights is a list of weight's vector (another list)
    return list(map(lambda wghts: feed_forward_neuron(vector, wghts, activation_function), weights))


"""
Randomly initializes a matrix of weights which has `num_of_weights`
weight's vectors, each with `weights_len` dimensions
"""


def initialize_weights(num_of_weights, weights_len):
    return [[random.random() for _ in range(weights_len) for _ in range(num_of_weights)]]


def bipolar_degree(max, min, net_value: float, threshold: float):
    return max if net_value >= threshold else min


def hyperbolic_tangent(net_value: float):
    return (2 / (1 + (math.e ** (- 2 * net_value)))) - 1


def derivative_hyperbolic_tangent(net_value):
    return (1 - hyperbolic_tangent(net_value)) ** 2


def main():
    num_of_hidden_layers = 1
    dims_of_hidden_layer = 1
    dims_of_output_layer = 1
    # List of bias vectors
    bias = [[]]
    threshold = 0.0001
    max_iter = 100
    learning_rate = 3


if (__name__ == "__main__"):
    main()
