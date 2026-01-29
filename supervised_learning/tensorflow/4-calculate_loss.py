#!/usr/bin/env python3
"""
Module to calculate the loss of a neural network prediction
"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction

    Args:
        y: placeholder for the labels of the input data
        y_pred: tensor containing the network's predictions

    Returns:
        A tensor containing the loss of the prediction
    """
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)

    return loss
