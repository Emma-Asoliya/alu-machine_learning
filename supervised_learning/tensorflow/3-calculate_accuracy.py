#!/usr/bin/env python3
import tensorflow as tf

def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction.
    
    y: placeholder for the labels of the input data (one-hot)
    y_pred: tensor containing the network's predictions
    
    Returns: a tensor containing the decimal accuracy of the prediction
    """
    # Get the index of the max value (the predicted class) for each example
    # argmax returns the index of the largest value across an axis
    actual_class = tf.argmax(y, axis=1)
    predicted_class = tf.argmax(y_pred, axis=1)
    
    # Compare predictions to actual labels (returns a boolean tensor)
    is_correct = tf.equal(actual_class, predicted_class)
    
    # Cast booleans to floats (True -> 1.0, False -> 0.0) and take the mean
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    
    return accuracy