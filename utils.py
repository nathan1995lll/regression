import tensorflow as tf


def add_layers(x, input_shape, output_shape, activation=None):
    weights = tf.Variable(tf.truncated_normal(shape=(input_shape,output_shape)), dtype=tf.float32)
    biases = tf.Variable(tf.zeros(output_shape), dtype=tf.float32)
    output = tf.add(tf.matmul(x, weights), biases)
    if activation is None:
        return output
    else:
        return activation(output)


def loss(y, y_pred):
    return tf.reduce_mean(tf.square(tf.subtract(y , y_pred)))
