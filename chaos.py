import tensorflow as tf
import numpy as np
from util import fc_layer

class Node():
    def __init__(self, degree, chaos_number, candidate_degree, name, chaos_weight_scope, dtype=tf.float64):
        
        self.activation_list = []

        # tensor_array_name = "ta"  + name
        
        # self.ta_activation = tf.TensorArray(dtype=dtype, size=chaos_number, dynamic_size=False, name=tensor_array_name )
        self.currIndex = 0

        self.candidate_field_nodes = []
        self._degree = degree
        self._dtype = dtype
        self._candidate_degree = candidate_degree
        
        with tf.variable_scope(chaos_weight_scope):
            self.weights = tf.get_variable(name=name, 
                                       shape=[degree],     
                                       initializer=tf.random_normal_initializer(mean=0.0, 
                                                                                stddev=0.5),
                                       dtype=dtype) 
            print(self.weights)

            self.chaos_var_scope = chaos_weight_scope
    
    def get_weight_scope(self):
        return self.chaos_var_scope
    
    def clear_node(self): 
        # clears activations and activations. 
        self.activation_list.clear();

    
    def add_activation(self, activation):
        self.activation_list.append(activation)
        # self.ta_activation = self.ta_activation.write(self.currIndex, activation)
        # self.currIndex += 1

    def get_top_activation(self):
        if len(self.activation_list) == 0:
            return None
        else:
            return self.activation_list[-1]
            # return self.ta_activation.read(self.currIndex - 1)
   
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
            
            if weight_counter == self._degree:
                weight_counter = 0

        self.candidate_field_nodes = np.array(arr)
        # print("CANDIDATE FIELD IS: ", self.candidate_field_nodes)
  
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
        self.chaos_weights = None

        with tf.variable_scope("chaos_weight_scope") as vs: 
            self.chaos_weight_scope = vs

        # defined by chaos graph 
        self._train = None
        self._controller = None
        self._pass_through = None
        self._selected_field_activations = None

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

    def initialize_chaos_weights(self, total_degrees):
        self.chaos_weights = tf.get_variable(name="chaos_weights", 
                                       shape=[1, total_degrees],     
                                       initializer=tf.random_normal_initializer(mean=0.0, 
                                                                                stddev=0.5),
                                       dtype=tf.float64)

        # Each node has its weight in chaos_weights
        # the weights for node 3 is: self.chaos_weights[sumOfWeightsForNode1AndNode2 : sumOfWeightsForNode1AndNode2AndNode3]

    def create_random_chaos_graph(self):
        number_of_nodes = self.number_of_nodes;
        degree = 3;
        candidate_field_size = 6;

        # index position in array is the node name.
        for i in range(number_of_nodes): 
            self.nodes.append(Node(name=("node" + str(i)), 
                                 degree=degree, 
                                 candidate_degree=candidate_field_size,
                                 chaos_number=self.chaos_number, 
                                 chaos_weight_scope=self.chaos_weight_scope
                                 ))
        # create chaos weights
        total_degree = degree * number_of_nodes
        self.initialize_chaos_weights(total_degree)
      

          
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
        return fc_layer(input_= activation_input,
                        input_size = self.number_of_nodes, 
                        output_size=self.number_of_nodes, 
                        activation=tf.tanh, 
                        bias=True, 
                        scope=scope)

    # Controller Scores each node, takes its previous activation 
    def score_nodes(self, activation_input): 
        
        if self._controller is None:
            # can be anytime of graph 
            self._controller = self.build_controller(activation_input)

        return self._controller
        
    def pass_through(self, inputs):
        if self._pass_through is None:  
            #just a projection layer, therefore, no bias
            
            total_chaos_from_pass = tf.TensorArray(size=self.chaos_number, dtype=self.dtype)
            chaos_idx = tf.constant(0, tf.int32)
            
            print("inputs", inputs)
            #print("inputs shape", inputs.shape())

            activation_zero = fc_layer(input_=inputs,
                                       input_size = self.input_size, 
                                       output_size = self.number_of_nodes, 
                                       activation=tf.tanh, 
                                       bias=False,
                                       scope="input")
            
            print("activation zero, ", activation_zero)

            #print_activation_zero = tf.Print(activation_zero, [activation_zero], "Activation Zero: ")

            # give activation 0 to each layer
            list_of_activation_zero_tensors = activation_zero
            
            # TREAT STEP 0 SEPERATE FROM STEP 1 TO STEP <ChaosNumber>

            node_scores = None #This will contain scores from controller graph
            
            # could probablu do a tf map here and above if, since for loops dont work on computational graph construction

            # build a list up of activation tensors, have a tensor called => all_activation_tensors => keep appending to this tensor activation tensors at a certain timestep,
            # indexing in will give you the activations for a certain node !!!!!!
            # for node_id, i in enumerate(list_of_activation_zero_tensors):
            #    self.nodes[node_id].add_activation(i)
            
            #list_of_activation_zero_tensors = tf.stack(list_of_activation_zero_tensors)
            node_scores = self.score_nodes(list_of_activation_zero_tensors)
            
            prev_activations = list_of_activation_zero_tensors
            print("prev_activations, ", prev_activations)
            print("prev_activations_shape, ", prev_activations.get_shape())

            def pass_iteration(idx, cumulative_chaos, scores_for_nodes, prev_activations):
                # this probably has to be tf.while
                current_activations = self.chaos_iteration(scores_for_nodes, prev_activations) # current activations should be a tensor array of activations
                
                print("current_activations, ", current_activations)

                next_score_for_nodes = self.score_nodes(current_activations)

                cumulative_chaos = cumulative_chaos.write(idx, current_activations)
                #current_activations.set_shape((None, self.number_of_nodes))
                return (idx+1, cumulative_chaos, next_score_for_nodes, current_activations)
            

            _, total_chaos_from_pass_final, final_node_scores, activation_on_final_index = tf.while_loop(
                lambda idx, a, b, c : tf.less(idx, self.chaos_number), 
                pass_iteration, 
                [chaos_idx, total_chaos_from_pass, node_scores, prev_activations], 
                shape_invariants=[chaos_idx.get_shape(), tf.TensorShape(None), node_scores.get_shape(), tf.TensorShape([None, None])],
                parallel_iterations = 1, 
                back_prop=True)

            stacked_total_chaos = total_chaos_from_pass_final.stack()
            stacked_total_chaos_print = tf.Print(stacked_total_chaos, [stacked_total_chaos], "stacked_total_chaos: ")
          

            print("current activations: ", activation_on_final_index)
            # final output is a projection layer, so set bias to false
            _pass_through = fc_layer(input_=activation_on_final_index, 
                                     input_size=self.number_of_nodes, 
                                     output_size=self.output_size, 
                                     activation=tf.tanh, 
                                     bias=False, 
                                     scope="output")
        
        return _pass_through

    def get_previous_node_activations(self, nodes):
        # gets node activations
        # nodes is an np array of indexes 
        return [self.nodes[i].get_top_activation for i in nodes]
    '''
    def selected_field_activations(self, selected_field_nodes, prev_activations, node_degree):
            # reading from this
        if(self._selected_field_activations is None): 
            input_selected_field_nodes_ta = tf.TensorArray(size=node_degree, dtype=tf.float32, dynamic_size=False)
            input_selected_field_nodes_arr = input_selected_field_nodes_ta.unstack(selected_field_nodes) 

            #writing to this
            weight_matched_nodes_arr = tf.TensorArray(dtype=tf.float32, size=node_degree, dynamic_size=False)
            #weights_matched_nodes_arr = weight_matched_nodes_ta.unstack(np.array([-1] * node_degree, dtype=np.float32)) 
            
            # OK SO I GOT THIS ISSUE:  TensorArray TensorArray_1_1: Could not write to TensorArray index 2 because it has already been read.
            # this is because we cant read from an array and then write to it in the outer while loop, so just split to two output arrays. (MAYBE THATS THE issue not sure)

            weights_taken_table = tf.contrib.lookup.MutableHashTable(key_dtype=tf.int64, value_dtype=tf.float32, default_value=-1)


            def find_weight_matched_nodes_cond(index, output_arrm):
                return index < node_degree
            
            def find_weight_matched_nodes_body(index, output_arr):
                input_node_matching = input_selected_field_nodes_arr.read(index)
                
                node_id = tf.gather(input_node_matching, 0)
                weight_match = tf.cast(tf.gather(input_node_matching, 1), tf.int64) 
                
            
                def find_weight_match_cond(weight_match, keep_going): # THIS WHILE LOOP IS ACTUALLY an infinite while loop with a break in the body
                    return keep_going
                    
                def find_weight_match_body(weight_match, keep_going):
                    taken = weights_taken_table.lookup(weight_match)
                    taken_print = tf.Print(taken, [taken], "taken: ")

                    def not_taken(): 
                        insert_op = weights_taken_table.insert(weight_match, tf.constant(1.0, tf.float32)) # put 1.0 to indicate taken
                        #val_print = tf.Print(val, [val], message="fook")
                        #return val_print
                        with tf.control_dependencies([insert_op]):
                            # Now, we are under the dependency scope:
                            # All the operations happening here will only happens after 
                            # the "assign_op" has been computed first
                            # i think you can put if statements here maybe.

                            # THIS IS A NO-OP TENSOR. because we need some tensor to execute to cause the control dependency to work
                            identity_weight_match = tf.identity(weight_match) 

                        return (identity_weight_match, False)

                    def is_taken():
                        incremented_weight_match = tf.cond(tf.equal(weight_match, node_degree-1), lambda: tf.constant(0, dtype=tf.int64), lambda: (weight_match + 1))
                        return (incremented_weight_match, True)
                    
                    # its not taken if its -1, otherwise its taken, set 1 if its taken (cant do both these tasks in same array cause tf complains)                   
                    # kill while loop when its not taken    

                    return tf.cond(tf.equal(taken_print, -1.0), not_taken, is_taken)

                empty_index_to_write_to, _ = tf.while_loop( find_weight_match_cond, 
                                                            find_weight_match_body, 
                                                            parallel_iterations=1,
                                                            loop_vars=(weight_match, True))
                
                output_arr_changed = output_arr.write(tf.cast(empty_index_to_write_to, dtype=tf.int32), node_id)
                #output_arr.write(tf.cast(empty_index_to_write_to, dtype=tf.int32), node_id)
                
                return (index + 1, output_arr_changed)

            _,  final_weight_matched_nodes_arr = tf.while_loop(find_weight_matched_nodes_cond, 
                                                                        find_weight_matched_nodes_body, 
                                                                        loop_vars=(0, weight_matched_nodes_arr), 
                                                                                        shape_invariants=None,
                                                                                        parallel_iterations=1, 
                                                                                        back_prop=True,  #MAYBE NO BACKPROP TRAINING NEEDED HERE BECAUSE WE ARE JUST REORDERING RESULTS. 
                                                                                                        #WHICH WILL THEN MULTIPLY WITH WEIGHTS. YES DISABLE BACK PROP HERE!
                                                                                        swap_memory=False)
            
            final_weight_matched_nodes_tensor = tf.cast(final_weight_matched_nodes_arr.stack(), dtype=tf.int32)
            weight_matched_activations = tf.gather(prev_activations, final_weight_matched_nodes_tensor)
            print_weight_matched_activations = tf.Print(weight_matched_activations, [weight_matched_activations], "Weight matched activations is: ")
            self._selected_field_activations = print_weight_matched_activations
        
        return self._selected_field_activations
    '''
    

    def chaos_iteration(self, node_scores, prev_activations):
        
        # Build a tensor reflecting chaos graph relationships that can be used in the tf computation graph
        node_degrees = [node.get_degree() for node in self.nodes]
        node_candidate_fields = [node.get_candidate_field() for node in self.nodes]

        #print(all_node_weights)
        print(self.chaos_weights)

        node_degree_t = tf.constant(np.array(node_degrees, dtype=np.int32))
        node_candidate_fields_t = tf.convert_to_tensor(np.array(node_candidate_fields, dtype=np.int32))
        # node_weights_t = tf.stack(node_weights) => YOU CANT STACK TENSORFLOW VARIABLES, BECAUSE THEY ARE NOT TENSORS. TF VARIABLES CANNOT BE TREATED LIK TENSORS

        chaos_activations = tf.TensorArray(dtype=tf.float64, size=self.number_of_nodes)
        print("chaos_activations_tensor_array", chaos_activations)


        def chaos_iteration_body(i, activations, weight_index_begin):

            candidate_field_for_node = tf.gather(node_candidate_fields_t, i)
            node_degree = tf.gather(node_degree_t, i)
            print("node_deg,", node_degree)
            
            node_weights = self.chaos_weights[:, weight_index_begin: weight_index_begin + i]#tf.slice(self.chaos_weights, begin=[weight_index_begin], size=[node_degree]) 
            
       
            print("candidate_field_for_node", candidate_field_for_node)
            print("node weights: ", node_weights)

            # i is not a tensor, i think its just a python int. also 
            # I THINK IN A WHILE LOOP YOU CAN HAVE NORMAL PYTHON OBJECTS IN THE LOOP VARS!!! such as a {}, or a [], NOT EVERYTHING HAS TO BE A TENSOR rite...
            #  How to index a list with a TensorFlow tensor? -> "Simply run tf.gather(list, tf_look_up[index]), you'll get what you want."
            
            
            '''
            # THIS DOESNT WORK WHEN FETCHING TF VARIABLES. 
            # WHY? you cant have a name as a tensor. but everything is a tensor... #tfsucks

            node_name = tf.string_join(["node", tf.as_string(i)])
            print(node_name)
            print_node_name = tf.Print(node_name, [node_name], "node's weight name: ")
            with tf.name_scope(None): #EXIT THE WHILE SCOPE
                with tf.variable_scope(self.chaos_weight_scope, reuse=True):     
                    node_weights = tf.get_variable(name=print_node_name) #tf.gather(all_node_weights, index)
            '''
            

            top_values, top_indices = tf.nn.top_k(tf.gather(node_scores, candidate_field_for_node[:, 0]), k=node_degree) # get the scores revelent to the particular node
            # top indices reflects index locations into the candidate_field_for_node array
            # which will retrieve the node id's for nodes that are in the top sore.
            selected_field_nodes = tf.gather(tf.convert_to_tensor(candidate_field_for_node), top_indices) # indexing into an np array

            # the field input has to be sorted based on which activation inputs will multiply with which weights...

            # sort out the selected_field_nodes and insert them into the array so that ties are broken and the weight matchings are correct

            #selected_activations = self.selected_field_activations(selected_field_nodes, prev_activations, node_degree)
            selected_activations=tf.constant([[0.3],[0.3],[0.3]], dtype=tf.float64)
            
            node_dot_prod = tf.matmul(node_weights, selected_activations)
            print("node_mat_mult", node_dot_prod)

            #node_evaluation = tf.reduce_sum(node_mat_mult)

            node_activation = tf.reshape( tf.tanh(node_dot_prod), [-1])
            
            # node.add_activation(node_activation)
            # chaos_activations.append(node_activation)
            print("node_activation,", node_activation)

            activations = activations.write(i, node_activation)
            
            return (i+1, activations, weight_index_begin + node_degree)

        _, final_chaos_activations, _ = tf.while_loop(
            lambda idx, a, b: tf.less(idx, self.number_of_nodes),  
            chaos_iteration_body,
            (0, chaos_activations, 0),
            parallel_iterations=1
        )

        new_activations = tf.reshape(final_chaos_activations.stack(), (-1, 50))
        
        #new_activations.set_shape([None, self.number_of_nodes])

        #new_activations.set_shape([50,50])
        print("CHAOS ITERATION, NEW ACTIVATIONS CALCULATED: ", new_activations)
        return new_activations



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
        