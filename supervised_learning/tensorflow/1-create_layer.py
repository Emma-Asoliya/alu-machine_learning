#!/usr/bin/env python3
import tensorflow as tf

def create_layer(prev, n, activation):
    """
    Creates a layer for a neural network.
    
    prev: tensor output of the previous layer
    n: number of nodes in the layer to create
    activation: activation function to use
    
    Returns: tensor output of the layer
    """
    # Define the He et. al initialization (Variance Scaling)
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    
    # Create the layer
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        name='layer'
    )
    
    return layer(prev)
