import tensorflow as tf


def my_network(input):
    W_1 = tf.Variable(tf.random_uniform([784, 100], -1, 1), name='W_1')
    b_1 = tf.Variable(tf.zeros([100]), name='biases_1')
    output_1 = tf.matmul(input, W_1) + b_1

    W_2 = tf.Variable(tf.random_uniform([100, 50], -1, 1), name='W_2')
    b_2 = tf.Variable(tf.zeros([50]), name='biases_2')
    output_2 = tf.matmul(output_1, W_2) + b_2

    W_3 = tf.Variable(tf.random_uniform([50, 10], -1, 1), name='W_3')
    b_3 = tf.Variable(tf.zeros([10]), name='biases_3')
    output_3 = tf.matmul(output_2, W_3) + b_3

    print("Printing names of weight parameters")
    print(W_1.name, W_2.name, W_3.name)
    print("Printing names of bias parameters")
    print(b_1.name, b_2.name, b_3.name)

    return output_3


def layer(input, weight_shape, bias_shape):
    weight_init = tf.random_uniform_initializer(minval=-1, maxval=1)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable('W', weight_shape, initializer=weight_init)
    b = tf.get_variable('b', bias_shape, initializer=bias_init)

    return tf.matmul(input, W) + b


def refined_my_network(input):
    with tf.variable_scope('layer_1'):
        output_1 = layer(input, [784, 100], [100])
    with tf.variable_scope('layer_2'):
        output_2 = layer(output_1, [100, 50], [50])
    with tf.variable_scope('layer_3'):
        output_3 = layer(output_2, [50, 10], [10])

    return output_3


with tf.variable_scope('shared_variables') as scope:
    i_1 = tf.placeholder(tf.float32, [1000, 784], name='i_1')
    refined_my_network(i_1)
    scope.reuse_variables()
    i_2 = tf.placeholder(tf.float32, [1000, 784], name='i_2')
    refined_my_network(i_2)
