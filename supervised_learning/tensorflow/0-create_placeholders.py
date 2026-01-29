#!/usr/bin/env python3
"""
Module that defines a function to create placeholders for a neural network
"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Function that returns two placeholders, x and y, for the neural network

    Args:
        nx: number of feature units
        classes: number of classifier classes

    Returns:
        x: placeholder for the input data
        y: placeholder for the one-hot labels
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name="x")
    y = tf.placeholder(tf.float32, shape=[None, classes], name="y")
    return x, y