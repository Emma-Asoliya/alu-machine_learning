#!/usr/bin/env python3
import tensorflow as tf

def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network.
    
    X: numpy.ndarray containing the input data to evaluate
    Y: numpy.ndarray containing the one-hot labels for X
    save_path: location to load the model from
    
    Returns: network's prediction, accuracy, and loss
    """
    # Import the meta graph to rebuild the structure
    saver = tf.train.import_meta_graph(save_path + ".meta")
    
    with tf.Session() as sess:
        # Restore the saved values (weights/biases) into the graph
        saver.restore(sess, save_path)
        
        # Retrieve the tensors from the collection
        # Note: get_collection returns a list, so we take the first element [0]
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        
        # Run the session to get the actual values
        prediction, acc, cost = sess.run(
            [y_pred, accuracy, loss],
            feed_dict={x: X, y: Y}
        )
        
    return prediction, acc, cost