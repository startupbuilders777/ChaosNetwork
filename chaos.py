import tensorflow as tf
import numpy as np
from aux import fc_layer

class Node():
    def __init__(self, degree, candidate_degree, name, dtype=tf.float64):
        self.activation_list = []
        self.candidate_field_nodes = []
        self._degree = degree
        self._dtype = dtype
        self._candidate_degree = candidate_degree


        self.weights = tf.get_variable(name=name, 
                                       shape=[1, degree],     
                                       initializer=tf.random_normal_initializer(mean=0.0, 
                                                                                stddev=0.5),
                                       dtype=dtype) 

    def add_activation(self, activation):
        self.activation_list.append(activation)
    
    def get_top_activation(self):
        if len(self.activation_list) == 0:
            return None
        else:
            return self.activation_list[-1]

   
    def get_degree(self):
        return self._degree
    

    def get_candidate_degree(self):
        return self._candidate_degree

    # there are 5 nodes in candidate field, and 3 weights
    # to match them to weights do the following: 
    # priority list building:
    # priority list algorithm:
    # => a will match with weight 1, otherwise go up by 1, it will match with weight2, 
    #              otherwise go up by one, it will match with weight weight3, go up by one, 
    #               if it reaches the end, start counting from weight0 again
    # => and weights will greedily be taken by highest score to lowest score.
    # => in other words, each node should store, the weight they assign to, check if the index 
    #                location in the array is taken for that weight, and if it is, go to the next empty spot in 
    #                this array which has degree k, where k is the degree of the node
    # { a: [1,2,3]   
    #   b: [2,3,1]
    #   c: [3,2,1]
    #   d: [1,2,3]
    #  }
    # in the node, the candidate node and weight should be stored as a tuple!!!!

    def set_candidate_field(self, candidate_field_nodes):
        '''
        >>> a = [(3,2), (4,6)]
        >>> b = np.array(a)
        >>> a
        [(3, 2), (4, 6)]
        >>> b
        array([[3, 2],
               [4, 6]])
        '''
        weight_counter = 0
        arr = []

        for i in candidate_field_nodes: 
            arr.append((i, weight_counter))
            weight_counter += 1
            
            if weight_counter == self.degree:
                weight_counter = 0

        self.candidate_field_nodes = np.array(arr)
    
  
    def get_candidate_field(self):
        return self.candidate_field_nodes
    



# ok so lets have degree 2 nodes, with each node having 4 candidate nodes.

# A, B, C, D, E, F, H, 


## ALGORIHTM OUTLINE:
'''
FOR EACH TEST EXAMPLE
GRAB TEST EXAMPLE (X, Y)
Input projection layer(can be convolutional nets, FC layers, ) takes in X creates activation 0, for each node in chaos graph
The output from input projection layer is pushed into the node activation list as the first element 
for each node (so activation at index 0, or activation 0)

    FOR i from 1 to X (So the iteration hyperparameter for the chaos graph is X)
        nodes are scored by controller graph (score is calculated based on activations at time i-1, 
                                              which in this case is the output from the input projection layer),
        each node finds top k nodes and then computes its dot product and tanh , and stores that value in its node 
                                                activation list. (use tie breaking rules for weight sharing in this step)


    Values in Node activation list are then fed into a convolutional net/feed forward net to compute output (output can be probabilities for a classiication problem)
    (The output projection layer can be whatever you want it to be)
'''




class ChaosNetwork():
    
    '''
    Degree Map specifies the number of nodes, the degrees for each type of node, 
                and candidate degree for each type of node. 
    
    {2: (3,5), 5: (4,7) } => this is a degree map. so 2 nodes will have a degree 2 
    with candidate field size 5, and 5 nodes will have degree 4 with candidate field size 7
     this map can be used in the random chaos graph construction algorithm.

    We also want a way to save and load chaos graph structures, call random to 
    see a bunch of examples, and then choose one, save it, and use it for training.
    different instances on the same chaos graph

    chaos_number is the number of iterations and activations completed in chaos graph before outputting result

    '''
    def __init__(self, 
                 number_of_nodes, 
                 input_size,
                 output_size, 
                 chaos_number, 
                 degree_map= None, 
                 load_graph_structure = False,
                 graph_structure = None):

        self.dtype = tf.float64
        self.degree_map = degree_map 
        self.number_of_nodes = number_of_nodes
        self.nodes = []    
        self.input_size = input_size
        self.output_size = output_size
        self.chaos_number = chaos_number


        # defined by chaos graph 
        self._train = None
        self._controller = None
        self._pass_through = None

        if load_graph_structure: 
            self.load_graph(graph_structure)
        else: 
            self.create_chaos_graph("random")

    def load_graph(self, graph_structure):
        return 3
    
    # have to deal with batch size
    def create_chaos_graph(self, method="random"):
        if(method == "random"):
            self.create_random_chaos_graph()
        
    def create_random_chaos_graph(self):
        number_of_nodes = self.number_of_nodes;
        degree = 3;
        candidate_field_size = 6;

        # index position in array is the node name.
        for i in range(number_of_nodes): 
            self.nodes.append(Node(name=("node" + str(i)), 
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

    def build_controller(self, activation_input, scope=None):
        
        return fc_layer(activation_input, 
                        self.number_of_nodes, 
                        activation=tf.tanh, 
                        bias=True, 
                        scope=scope)

    # Controller Scores each node, takes its previous activation 
    def evaluate_nodes(self, activation_input): 
        
        if self._controller is None:
            # can be anytime of graph 
            self._controller = self.build_controller(activation_input)

        return self._controller
        


    def pass_through(self, inputs):
        if self._pass_through is None:  
            #just a projection layer, therefore, no bias
            
            activation_zero = fc_layer(inputs, 
                                       self.number_of_nodes, 
                                       activation=tf.tanh, 
                                       bias=False,
                                       scope="input")

            # give activation 0 to each layer
            list_of_activation_zero_tensors = tf.unstack(activation_zero, 
                       num=self.number_of_nodes, 
                       axis=0)
            
            # TREAT STEP 0 SEPERATE FROM STEP 1 TO STEP <ChaosNumber>

            node_scores = None #This will contain scores from controller graph
            
            # could probablu do a tf map here and above if, since for loops dont work on computational graph construction
            for node_id, i in enumerate(list_of_activation_zero_tensors):
                self.nodes[node_id].add_activation(i)

                node_scores = self.evaluate_nodes(tf.stack(list_of_activation_zero_tensors))

            for i in range(self.chaos_number):
                # this probably has to be tf.while
                current_activations = self.chaos_iteration(node_scores) # current activations should be a tensor array of activations
                node_scores = self.evaluate_nodes(current_activations)
            
            # final output is a projection layer, so set bias to false
            _pass_through = fc_layer(current_activations, self.output_size, activation=tf.tanh, bias=False, scope="output")
        
        return _pass_through

    def get_previous_node_activations(self, nodes):
        # gets node activations
        # nodes is an np array of indexes 
        return [self.nodes[i].get_top_activation for i in nodes]
    


    def chaos_iteration(self, node_scores):
        
        chaos_activations = []
        for i in range(self.number_of_nodes):
            node = self.nodes[i]
            
            candidate_field_for_node = node.get_candidate_field()
            node_degree = node.get_degree()

            top_values, top_indices = tf.nn.top_k(tf.gather(node_scores, candidate_field_for_node[:, 0]), k=node_degree) # get the scores revelent to the particular node
            # top indices reflects index locations into the candidate_field_for_node array
            # which will retrieve the node id's for nodes that are in the top sore.
            activation_field_nodes = candidate_field_for_node[top_indices] # indexing into an np array

            # the field input has to be sorted based on which activation inputs will multiply with which weights...

            # sort out the activation_field_nodes and insert them into the array so that ties are broken and the weight matchings are correct

            sorted_nodes = np.array([None] * node_degree)

            for i in range(len(activation_field_nodes)):
                while True:
                    weight_match = activation_field_nodes[1]
                    if(sorted_nodes[weight_match] is None):
                        sorted_nodes[weight_match] = activation_field_nodes[0]
                        break
                    else: 
                        weight_match += 1
                        if weight_match == node_degree:
                            weight_match = 0    
                


            activation_field_input = self.get_previous_node_activations(sorted_nodes)
            node_evaluation = tf.reduce_sum(tf.matmul(activation_field_input, node.weights))

            node_activation = tf.tanh(node_evaluation)
            node.add_activation(node_activation)

            chaos_activations.push(node_activation)

        return chaos_activations

    def infer(self, inputs):
        if self._infer is None:
            output = self.pass_through(inputs)
            output = tf.nn.softmax(output)
            self._infer = output
        
        return self._infer
       
    
    def train(self, batch_x, batch_y, learning_rate = 0.01, ):
        if self._train is None:
            logits = self.pass_through(batch_x)

            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, 
                                                                             labels=batch_y))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(loss_op)
            self._train = train_op, loss_op
        
        return self._train
        