#!/usr/bin/env python3
"""
Module to create the forward propagation graph for a neural network
"""
import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network

    Args:
        x: the placeholder for the input data
        layer_sizes: list containing the number of nodes in each layer
        activations: list containing the activation functions for each layer

    Returns:
        the prediction of the network in tensor form
    """
    layer_output = x

    for i in range(len(layer_sizes)):
        n = layer_sizes[i]
        activation = activations[i]
        layer_output = create_layer(layer_output, n, activation)

    return layer_output
