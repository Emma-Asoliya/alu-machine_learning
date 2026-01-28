#!/usr/bin/env python3
import tensorflow as tf

def create_train_op(loss, alpha):
    """
    Creates the training operation for the network.
    
    loss: the loss of the network's prediction
    alpha: the learning rate
    
    Returns: an operation that trains the network using gradient descent
    """
    # Initialize the Gradient Descent Optimizer with the learning rate alpha
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    
    # Create the training operation to minimize the loss
    train_op = optimizer.minimize(loss)
    
    return train_op
