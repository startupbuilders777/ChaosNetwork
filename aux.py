# play around with tensorflow implementations here

import tensorflow as tf
import numpy as np


def fc_layer(input_, output_size, activation=None, bias=True, scope=None):
    '''
        fully connnected layer
        Args :
            input_  - 2D tensor
                general shape : [batch, input_size]
            output_size - int
                shape of output 2D tensor
            activation - activation function
                defaults to be None
            scope - string
                defaults to be None then scope becomes "fc"
    '''
    with tf.variable_scope(scope or "fc"):
        w = tf.get_variable(name="weight", shape=[input_.get_shape().as_list()[1], output_size],
                            initializer=tf.contrib.layers.xavier_initializer())
        output_ = tf.matmul(input_, w)
        if bias:
            b = tf.get_variable(name="bias", shape=[output_size], initializer=tf.constant_initializer(0.001))
            output_ += b
        return output_ if activation is None else activation(output_)


# Tensorflow stuff


sess = tf.InteractiveSession()

degree_size = 10
TensorArr = tf.TensorArray(dtype=tf.float32, size=10, dynamic_size=False)


sorted_nodes = TensorArr.unstack( np.array([-1] * degree_size, dtype=np.float32) )
x_elem0 = sorted_nodes.read(0) # first element

print_elem = tf.Print(x_elem0, [x_elem0], message="yee tensor array value is: ")

print_elem.eval()

