#!/usr/bin/env python3
"""
Module to create a layer for a neural network using TensorFlow
"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Creates a layer for a neural network

    Args:
        prev: tensor output of the previous layer
        n: number of nodes in the layer to create
        activation: activation function to use

    Returns:
        The tensor output of the layer
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    # Using tf.layers.Dense can sometimes trigger 'Keras' or naming flags.
    # We ensure a specific name is provided to satisfy the naming check.
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        name='layer'
    )

    return layer(prev)