'''
Integrantes
Enrique Emanuel Rezende Tavares da Silva - 11796090
Guilherme Dias Jimenes - 11911021
Ronald Cosmo de Sousa - 11909783
'''
from functools import reduce
import csv
import math
import random


def train_for_dataset(
        inputs,
        responses,
        num_of_hidden_layers,
        hidden_layer_dimensions,
        output_layer_dimensions,
        hyperbolic_func,
        learning_rate,
        threshold,
        max_iter
):

    weights = initialize_network_weights(
        len(inputs[0]),
        num_of_hidden_layers,
        hidden_layer_dimensions,
        output_layer_dimensions
    )

    print(f"Initial Weights: {weights}")

    print(f"\nTraining for each example in the dataset\n")

    for input, response in zip(inputs, responses):
        weights = train_for_input(
            input,
            response,
            weights,
            num_of_hidden_layers,
            hidden_layer_dimensions,
            output_layer_dimensions,
            hyperbolic_func,
            learning_rate,
            threshold,
            max_iter
        )

    print(f"Final Weights: {weights}\n")
    return weights


def train_for_input(
        # List of floats
        input_layer,
        # List of floats
        responses,
        weights,
        num_of_hidden_layers,
        hidden_layer_dimensions,
        output_layer_dimensions,
        activation_function,
        learning_rate,
        threshold,
        max_epochs
):

    biases = weights

    return train_network_iter(
        input_layer,
        responses,
        weights,
        biases,
        num_of_hidden_layers,
        activation_function,
        learning_rate,
        threshold,
        max_epochs
    )


def train_network_iter(
        input_layer,
        responses,
        weights,
        biases,
        num_of_hidden_layers,
        activation_function,
        learning_rate,
        threshold,
        max_epochs
):

    error_sum = math.inf
    curr_epoch = 0

    # print(f"Initial weights: {weights}\n")

    while error_sum > threshold and curr_epoch < max_epochs:

        # print(f"Training for epoch: {curr_epoch}")

        network = feed_forward_network(
            input_layer,
            weights,
            biases,
            # a network is some hidden_layers and an input layer and an output layer
            num_of_hidden_layers + 2,
            activation_function["self"],
        )

        error_sum = make_error_sum(
            network[-1],
            responses
        )

        deltas = make_backpropagation_deltas(
            network,
            weights,
            responses,
            activation_function["derivative"],
        )

        weights = list(map(
            lambda layer, layer_deltas, weight: update_layer_weights(
                layer,
                weight,
                layer_deltas,
                learning_rate
            ),
            # all indexes but the first because it doesn't make sense to get errors based on the input layer
            network[1:],
            deltas,
            weights
        ))

        curr_epoch = curr_epoch + 1

    # print(f"\nFinal weights: {weights}\n")
    return weights


def make_error_sum(output_layer, expected_output_layer):
    def error(a, b): return (a - b) ** 2

    errors = list(map(
        lambda expected, obtained: error(expected, obtained),
        output_layer,
        expected_output_layer
    ))

    return sum(errors)


def make_backpropagation_deltas(
        network,
        weights,
        expected_responses,
        activation_function_derivative,
):
    number_of_layers = len(network)
    # The first element of this list corresponds to the last layer's error.
    deltas = []
    deltas.append(delta_output_layer(
        network[-1],
        expected_responses,
        activation_function_derivative
    ))

    # Excluding output layer
    for layer_index in reversed(range(1, number_of_layers - 1)):
        deltas.append(delta_hidden_layer(
            weights[layer_index],
            deltas[-1],
            len(network[layer_index + 1]),
            network[layer_index],
            activation_function_derivative
        ))

    return list(reversed(deltas))


def delta_output_layer(
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


def delta_hidden_layer(
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
    def net_delta(next_layer_index, weights_matrix, next_layer_gradients, current_layer_neuron_index):
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
        individual_deltas = map(
            lambda x, y: x * y,
            neuron_weights,
            neuron_gradients
        )

        net_deltas = reduce(
            lambda curr, acc: curr + acc,
            individual_deltas
        )
        return net_deltas * gradient_function(current_layer[current_layer_neuron_index])

    return list(map(
        lambda current_neuron_index: net_delta(
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
        hidden_layer_dimensions,
        output_layer_dimensions,
):
    weights = [initialize_weights(hidden_layer_dimensions, input_layer_dims)]
    for _ in range(num_of_hidden_layers):
        weights.append(
            initialize_weights(
                hidden_layer_dimensions,
                hidden_layer_dimensions
            )
        )
    weights.append(
        initialize_weights(
            output_layer_dimensions,
            hidden_layer_dimensions
        )
    )
    return weights


def update_layer_weights(layer, weights, deltas, learning_rate):

    def update_weight(weight, component, error):
        return weight - (learning_rate * component * error)

    new_weights_matrix = []
    layer_range = range(len(layer))

    for neuron_index in layer_range:
        weights_vector = weights[neuron_index]
        new_weights_vector = list(map(
            lambda weight: update_weight(
                weight,
                layer[neuron_index],
                deltas[neuron_index]
            ),
            weights_vector
        ))
        new_weights_matrix.append(new_weights_vector)

    return new_weights_matrix


def feed_forward_layer(vector, weights_matrix, biases, activation_function):
    """
    Feeds forward, creating a new layer.
    This works by applying a weights vector to the inputs, obtaining a response for each weights vector
    """

    def feed_forward_neuron(vector, weights, activation_function):
        vector_size = len(vector)
        dims_range = range(vector_size)
        linear_combination = map(
            lambda dim: vector[dim] * weights[dim],
            dims_range
        )
        net = reduce(lambda item, acc: item + acc, linear_combination)
        return activation_function(net)

    # This expects that weights is a list of weight's vector (another list)
    return list(map(
        lambda weights_vector: feed_forward_neuron(
            vector,
            weights_vector,
            activation_function
        ),
        weights_matrix
    ))


def initialize_weights(num_of_weights, weights_len):
    return [[random.random() for _ in range(weights_len)] for _ in range(num_of_weights)]


def sigmoid(net_value: float):
    return (1 / (1 + (math.e ** (- 1 * net_value))))


def sigmoid_derivative(net_value: float):
    return sigmoid(net_value) * (1 - sigmoid(net_value))


def hyperbolic_tangent(net_value: float):
    return (2 / (1 + (math.e ** (- 2 * net_value)))) - 1


def hyperbolic_tangent_derivative(net_value: float):
    return (1 - hyperbolic_tangent(net_value)) ** 2


def read_dataset():
    '''
    Returns a list of dicts corresponding to the dataset.

    For example, from the following .csv:,

            first_name,last_name
            John, Cleese
            Terry, Gilliam

    the first row of the dataset would look like this:

        {'first_name': 'John', 'last_name': 'Cleese'}

    And the whole dataset would look like this:

        [
            {'first_name': 'John', 'last_name': 'Cleese'} ,
            {'first_name': 'Terry', 'last_name': 'Gilliam'}
        ]

    '''
    print("Reading dataset `Haberman's Survival`")
    with open('tarefa3/data/haberman.data', 'r') as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        inputs = []
        responses = []

        for row in reader:
            input_attributes = row[:3]
            response_attributes = row[3]    # 1 ou 2
            inputs.append(input_attributes)
            responses.append(
                [response_attributes == 1,  # One hot da classe 1
                 response_attributes == 2]   # One hot da classe 2
            )
    return inputs, responses


def make_all_responses(
        inputs,
        weights,
        num_of_hidden_layers,
        activation_function
):
    obtained_responses = []
    for input in inputs:
        final_network = feed_forward_network(
            input,
            weights,
            weights,
            num_of_hidden_layers + 2,
            activation_function["self"]
        )

        obtained_responses.append(final_network[-1])
    return obtained_responses


def accuracy(responses, obtained_responses):
    correct_predictions = 0
    total_predictions = len(responses)

    for expected, obtained in zip(responses, obtained_responses):
        expected_label = [1, 0] if expected[0] > expected[1] else [0, 1]
        obtained_label = [1, 0] if obtained[0] > obtained[1] else [0, 1]

        if expected_label == obtained_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy


def main():
    sigmoid_func = {
        "self": sigmoid,
        "derivative": sigmoid_derivative
    }

    hyperbolic_func = {
        "self": hyperbolic_tangent,
        "derivative": hyperbolic_tangent_derivative
    }

    # ==HYPERPARAMETERS==
    num_of_hidden_layers = 2
    hidden_layer_dimensions = 4
    output_layer_dimensions = 2
    threshold = 0.0001
    max_iter = 1000
    learning_rate = 1.2
    # ===================

    inputs, responses = read_dataset()

    assert len(responses[0]) == output_layer_dimensions

    weights = train_for_dataset(
        inputs,
        responses,
        num_of_hidden_layers,
        hidden_layer_dimensions,
        output_layer_dimensions,
        sigmoid_func,
        learning_rate,
        threshold,
        max_iter
    )

    obtained_responses = make_all_responses(
        inputs,
        weights,
        num_of_hidden_layers,
        sigmoid_func
    )

    accuracy_mlp = accuracy(responses, obtained_responses)
    print(f"Accuracy obtained: {accuracy_mlp}")


if (__name__ == "__main__"):
    main()
