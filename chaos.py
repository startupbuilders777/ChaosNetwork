import tensorflow as tf
import numpy as np


class Node():
    def __init__(self, degree, candidate_degree, name, dtype=tf.float64):
        self.activation_list = []
        self.candidate_field_nodes = []


        self.weights = tf.get_variable(name=name, 
                                       shape=[1, degree],     
                                       initializer=tf.random_normal_initializer(mean=0.0, 
                                                                                stddev=0.5),
                                       dtype=dtype) 
        
        

    def set_candidate_field(candidate_field_nodes):
        self.candidate_field_nodes = candidate_field_nodes

    def get_candidate_field(self):
        return self.candidate_field_nodes

    

# ok so lets have degree 2 nodes, with each node having 4 candidate nodes.

# A, B, C, D, E, F, H, 



class ChaosNetwork():
    
    '''
    Degree Map specifies the number of nodes, the degrees for each type of node, and candidate degree for each type of node. 
    
    {2: (3,5), 5: (4,7) } => this is a degree map. so 2 nodes will have a degree 2 with candidate field size 5, and 5 nodes will have degree 4 with candidate field size 7
     this map can be used in the random chaos graph construction algorithm.

    We also want a way to save and load chaos graph structures, call random to 
    see a bunch of examples, and then choose one, save it, and use it for training.
    different instances on the same chaos graph

    '''
    def __init__(self, 
                 number_of_nodes, 
                 degree_map, 
                 input_size,
                 output_size, 
                 load_graph_structure=False):
        self.degree_map = degree_map 
        self.number_of_nodes = number_of_nodes
        self.nodes = []    
        self.input_size = input_size
        self.output_size = output_size


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


    # have to deal with batch size
    def create_chaos_graph(self, method="random"):
        if(method == "random"):
            self.create_random_chaos_graph(number_)
        
    def create_random_chaos_graph():
        number_of_nodes = 50;
        degree = 3;
        candidate_field_size = 6;

        # index position in array is the node name.
        for i in range(number_of_nodes): 
            self.nodes.push(Node(name=i, 
                                 degree=degree, 
                                 candidate_degree=candidate_field_size))

        
        # assign relantionships now
        # score 
        for i in range(number_of_nodes):
            '''
            >>> np.random.choice(5, 8, replace=True)
            array([0, 0, 3, 0, 0, 0, 0, 0])

            >>> np.random.choice(12, 8, replace=False)
            array([11,  4,  2,  0,  9,  7,  5,  8])
            '''
            '''
            Have to build relationships for node i
            '''

            nodes_to_be_in_candidate_field = np.random.choice(number_of_nodes, 
                                                              candidate_field_size, 
                                                              replace=False)

            self.nodes[i].set_candidate_field(nodes_to_be_in_candidate_field)

    def pass_through(self, inputs):
        if self._pass_through is None:  
            #just a projection layer, therefore, no bias
            activation_zero = self.fc_layer(inputs, 
                                            self.number_of_nodes, 
                                            activation=tf.tanh, 
                                            bias=False)
            # give activation 0 to each layer
            list_of_activation_zero_tensors = tf.unstack(activation_zero, 
                       num=self.number_of_nodes, 
                       axis=0)
            
            for i in list_of_activation_zero_tensors:
                print i 
            
            


    def infer(self, inputs):
        if self._infer is None:
            output = self.pass_through(inputs)
            output = tf.nn.softmax(output)
            self._infer = output
        
        return self._infer
    
    def train(self, learning_rate = 0.01, batch_x, batch_y):
        if self._train is None:
            logits = self._pass_through

            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logis(logits=logits, labels=batch_y))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(loss_op)
            self._train = train_op, loss_op
        
        return self._train
        



        

node = Node(3, "A");

print(node.getCandidateField());