#!/usr/bin/env python3

import tensorflow as tf
tf.disable_v2_behavior()

def create_placeholders(nx, classes):
    """
    Creates two placeholders, x and y, for the neural network.
    
    nx: number of feature columns (input size)
    classes: number of output classes
    
    Returns: x and y placeholders
    """
    # x: input data placeholder (None allows for flexible batch size)
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    
    # y: one-hot labels placeholder
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    
    return x, y
