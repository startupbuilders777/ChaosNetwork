import tensorflow as tf
import numpy as np



def fc_layer(input_, output_size, activation=None, bias=True, scope=None):
    '''
    fully convlolution layer
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



class Node():
    def __init__(self, degree, name, dtype=tf.float64):
        self.activationList = []
        self.candidateField = []
        self.weights = tf.get_variable(name=name, 
                                       shape=[1, degree],     
                                       initializer=tf.random_normal_initializer(mean=0.0, 
                                                                                stddev=0.5),
                                       dtype=dtype) 
        
        


    def activate():
        

    def setCandidateField():
        return 2

    def getCandidateField(self):
        return self.candidateField

    

# ok so lets have degree 2 nodes, with each node having 4 candidate nodes.


class ChaosNetwork():
    def __init__(self, *args, **kwargs):
        return 4
    



node = Node(3, "A");

print(node.getCandidateField());