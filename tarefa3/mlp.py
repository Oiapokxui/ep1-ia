'''
Integrantes
Enrique Emanuel Rezende Tavares da Silva - 11796090
Guilherme Dias Jimenes - 11911021
Ronald Cosmo de Sousa - 11909783
'''
from functools import reduce
import math
import random


def multi_layer_perceptron_iter(
        input_layer,
        responses,
        num_of_hidden_layers,
        hidden_layer_dims,
        output_layer_dims,
        activation_function,
        learning_rate
):

    weights = initialize_network_weights(
        len(input_layer),
        num_of_hidden_layers,
        hidden_layer_dims,
        output_layer_dims
    )

    network = feed_forward_network(
        input_layer,
        weights,
        weights,
        num_of_hidden_layers + 2,
        activation_function["self"],
    )

    errors = make_backpropagation_errors(
        network,
        responses,
        activation_function["derivative"],
    )

    new_weights = update_weights(

    )


def make_backpropagation_errors(
        network,
        weights,
        expected_responses,
        activation_function_derivative,
):
    number_of_layers = len(network)
    # The first element of this list corresponds to the last layer's error.
    errors = [[]]
    errors.append(error_output_layer(
        network[-1],
        expected_responses,
        activation_function_derivative
    ))

    # Excluding output layer
    for layer_index in reversed(range(0, number_of_layers - 1)):
        errors.append(error_hidden_layer(
            weights[layer_index],
            errors[-1],
            len(network[layer_index + 1]),
            len(network(layer_index))
        ))

    return errors.reverse()


def error_output_layer(
        output_responses,       # List of floats
        expected_responses,     # List of floats
        gradient_function,      # Function of type (float, float) -> float
):
    def gradient_output_layer(obtained, response):
        return (obtained - response) * gradient_function(obtained)

    return list(map(
        lambda obtained, response: gradient_output_layer(obtained, response),
        output_responses,
        expected_responses
    ))


def error_hidden_layer(
        # List of Lists of floats (dimensions: `next_layer_size`  x  `current_layer_size`)
        next_layer_weights,
        # List of floats (size: `next_layer_size`)
        next_layer_gradients,
        # An integer
        next_layer_size,
        # A list of floats
        current_layer,
        # Function of type (float, float) -> float
        gradient_function
):
    def net_error(next_layer_index, weights_matrix, next_layer_gradients, current_layer_neuron_index):
        next_layer_index_range = range(next_layer_index)
        neuron_weights = map(
            lambda next_layer_neuron_index:
                weights_matrix[next_layer_neuron_index][current_layer_neuron_index],
            next_layer_index_range
        )
        neuron_gradients = map(
            lambda next_layer_neuron_index: next_layer_gradients[next_layer_neuron_index],
            next_layer_index_range
        )
        individual_errors = map(
            lambda x, y: x * y,
            neuron_weights,
            neuron_gradients
        )

        net_error = reduce(
            lambda curr, acc: curr + acc,
            individual_errors
        )
        return net_error * gradient_function(current_layer(current_layer_neuron_index))

    return list(map(
        lambda current_neuron_index: net_error(
            next_layer_size,
            next_layer_weights,
            next_layer_gradients,
            current_neuron_index,
        ),
        range(len(current_layer))
    ))


def feed_forward_network(
        input_layer,
        weights,
        biases,
        total_layers,
        activation_function,
):
    """
    Initializes weights for the whole network and feeds them forward in order to get the output layer of the network.
    Returns the output_layer and the weights obtained.
    """
    network = [input_layer]

    for layer_index in range(total_layers):
        next_layer = feed_forward_layer(
            network[-1],    # Previous layer obtained of the network
            weights[layer_index],
            biases[layer_index],
            activation_function
        )
        network.append(next_layer)

    return network


def initialize_network_weights(
        input_layer_dims,
        num_of_hidden_layers,
        hidden_layer_dims,
        output_layer_dims,

):
    weights = [initialize_weights(hidden_layer_dims, input_layer_dims)]
    for _ in range(num_of_hidden_layers):
        weights.append(initialize_weights(
            hidden_layer_dims, hidden_layer_dims))
    weights.append(initialize_weights(output_layer_dims, hidden_layer_dims))
    return weights


def update_layer_weights(inputs, weights, errors, learning_rate):
    def update_weight(component, weight, error):
        return weight + (learning_rate * component * error)

    inputs_range = range(len(inputs))
    # Assuming that all entries on `weights` have the same length
    weight_vector_range = range(len(weights[0]))

    for input_index in inputs_range:
        for weight_index in weight_vector_range:
            # TODO
            pass

    return list(map(update_weight, inputs, weights, errors))


def feed_forward_layer(vector, weights_matrix, biases, activation_function):
    """
    Feeds forward, creating a new layer.
    This works by applying a weights vector to the inputs, obtaining a response for each weights vector
    """

    def feed_forward_neuron(vector, weights, activation_function):
        dims_range = range(len(vector))
        linear_combination = map(
            lambda dim: vector[dim] * weights[dim], dims_range)
        net = reduce(lambda item, acc: item + acc, linear_combination)
        return activation_function(net)

    # This expects that weights is a list of weight's vector (another list)
    return list(map(lambda weights_vector: feed_forward_neuron(vector, weights_vector, activation_function), weights_matrix))


def initialize_weights(num_of_weights, weights_len):
    return [[random.random() for _ in range(weights_len) for _ in range(num_of_weights)]]


def sigmoid(net_value: float):
    return (1 / (1 + (math.e ** (- 1 * net_value))))


def sigmoid_derivative(net_value: float):
    return sigmoid(net_value) * (1 - sigmoid(net_value))


def hyperbolic_tangent(net_value: float):
    return (2 / (1 + (math.e ** (- 2 * net_value)))) - 1


def hyperbolic_tangent_derivative(net_value: float):
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
