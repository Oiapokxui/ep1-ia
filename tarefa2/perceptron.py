'''
Integrantes
Enrique Emanuel Rezende Tavares da Silva - 11796090
Guilherme Dias Jimenes - 11911021
Ronald Cosmo de Sousa - 11909783
'''
from functools import reduce
import random


def train_perceptron(dataset, responses, threshold, max_iter, learning_rate, activation_function):
    input_len = len(dataset)
    weights = initialize_weights(input_len)
    print(f"Initial weights: {weights}\n")

    final_weights = perceptron_iter(
        dataset,
        responses,
        weights,
        max_iter,
        threshold,
        learning_rate,
        activation_function
    )
    final_responses = list(map(
        lambda vector: make_network_neuron_response(
            vector,
            final_weights,
            activation_function
        ),
        dataset
    ))
    print(f"Final weights: {final_weights}")
    print(f"Obtained responses {final_responses}")
    return final_weights


def perceptron_iter(dataset, responses, initial_weights, max_iter, threshold, learning_rate, activation_function):
    iter = 0
    current_error = 0
    current_weights = initial_weights
    current_response = 0

    while iter < max_iter and (current_error > threshold or iter == 0) :
        print(f"Epoch: {iter} out of {max_iter}")

        current_error = 0

        for vector, expected_response in zip(dataset, responses):
            current_response = make_network_neuron_response(
                vector,
                current_weights,
                activation_function
            )
            error = expected_response - current_response
            current_error = current_error + (error ** 2)
            current_weights = update_weights(vector, current_weights, error, learning_rate)

        iter = iter + 1

    print(f"Finished with {iter} epochs")

    return current_weights


def update_weights(inputs, weights, error, learning_rate):
    def update_weight(component, weight):
        return weight + (learning_rate * component * error)
    return list(map(update_weight, inputs, weights))


def make_network_neuron_response(vector, weights, activation_function):
    dims = range(len(vector))
    linear_combination = map(lambda dim: vector[dim] * weights[dim], dims)
    net = reduce(lambda item, acc: item + acc, linear_combination)
    return activation_function(net)


"""
Initializes an weight vector with `weights_len` dimensions of a real number between 0 and 1
"""


def initialize_weights(weights_len):
    return [random.random() for _ in range(weights_len)]


def bipolar_degree(max, min, net_value: float, threshold: float):
    return max if net_value >= threshold else min


def make_logical_or_dataset():
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    responses = list(map(lambda x: any(x), inputs))
    return inputs, responses


def make_image_dataset():
    inputs = [
        [-1, -1, -1, -1],
        [-1, -1, -1, 1],
        [-1, -1, 1, 1],
        [-1, 1, 1, 1],
        [1, 1, 1, 1],
        [-1, -1, 1, -1],
        [-1, -1, 1, 1],
        [-1, 1, -1, -1],
        [-1, 1, -1, 1],
        [1, -1, -1, -1],
        [1, -1, -1, 1],
        [1, -1, 1, 1],
        [1, 1, -1, -1],
        [1, 1, -1, 1],
        [1, -1, 1, -1],
        [1, -1, 1, 1]
    ]
    responses = list(map(lambda x: 1 if x.count(1) > 1 else -1, inputs))
    return inputs, responses


def train_logical_or_dataset(threshold, max_iter, learning_rate):
    print("\nTraining for logical OR dataset")
    or_dataset, or_responses = make_logical_or_dataset()

    return train_perceptron(
        or_dataset,
        or_responses,
        threshold,
        max_iter,
        learning_rate,
        lambda net: bipolar_degree(1, 0, net, threshold)
    )


def train_image_dataset(threshold, max_iter, learning_rate):
    print("\nTraining for 2x2 images dataset")
    image_dataset, image_responses = make_image_dataset()

    return train_perceptron(
        image_dataset,
        image_responses,
        threshold,
        max_iter,
        learning_rate,
        lambda net: bipolar_degree(1, -1, net, threshold)
    )


def main():
    threshold = 0.0001
    max_iter = 100
    learning_rate = 3

    train_logical_or_dataset(threshold, max_iter, learning_rate)

    train_image_dataset(threshold, max_iter, learning_rate)


if (__name__ == "__main__"):
    main()
