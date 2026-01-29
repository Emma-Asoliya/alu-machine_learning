#!/usr/bin/env python3
"""
Module to calculate the accuracy of a neural network prediction
"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction

    Args:
        y: placeholder for the labels of the input data
        y_pred: tensor containing the network's predictions

    Returns:
        A tensor containing the decimal accuracy of the prediction
    """
    actual_class = tf.argmax(y, axis=1)
    predicted_class = tf.argmax(y_pred, axis=1)

    is_correct = tf.equal(actual_class, predicted_class)

    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    return accuracy
