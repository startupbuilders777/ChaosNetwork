import tensorflow as tf
import numpy as np
from util import fc_layer


# SHARE TF VARIABLES DECORATOR:

share_variables = lambda func: tf.make_template(
    func.__name__, func, create_scope_now_=True)

class Node():
    def __init__(self, degree, chaos_number, candidate_degree, name, chaos_weight_scope, dtype=tf.float32):
        
        self.activation_list = []
        self.currIndex = 0

        self.candidate_field_nodes = []
        self._degree = degree
        self._dtype = dtype
        self._candidate_degree = candidate_degree
        self.chaos_var_scope = chaos_weight_scope
    
    def get_weight_scope(self):
        return self.chaos_var_scope
    

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


# Pass in class Object that defines the structure for the graph

#class StructuredChaosNetworkGraph():
    

#class RandomChaosNetworkGraph():
    


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
    #TODO: REMOVE BATCH_SIZE FROM INIT. it can be inferred from input. 
    def __init__(self, 
                 number_of_nodes, 
                 input_size,
                 output_size, 
                 chaos_number,
                 batch_size,  
                 degree_map= None, 
                 load_graph_structure = False,
                 graph_structure = None):

        self.dtype = tf.float32
        self.degree_map = degree_map 
        self.number_of_nodes = number_of_nodes
        self.nodes = []    
        self.input_size = input_size
        self.output_size = output_size
        self.chaos_number = chaos_number
        self.batch_size = batch_size
        self.chaos_weights = None
        self.highest_degree_node = 3

        with tf.variable_scope("chaos_weight_scope") as vs: 
            self.chaos_weight_scope = vs

        # defined by chaos graph 
        self._train = None
        self._test = None
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
                                       dtype=tf.float32)

        # Each node has its weight in chaos_weights
        # the weights for node 3 is: self.chaos_weights[sumOfWeightsForNode1AndNode2 : sumOfWeightsForNode1AndNode2AndNode3]

    def create_random_chaos_graph(self):
        number_of_nodes = self.number_of_nodes;
        degree = 3;
        candidate_field_size = 6;
        self.node_degree = 3;
        
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
        #with tf.variable_scope("chaos"):
        return fc_layer(input_= activation_input,
                        input_size = self.number_of_nodes, 
                        output_size=self.number_of_nodes, 
                        activation=tf.nn.relu, 
                        bias=True, 
                        scope="chaos-controller")
    
    
    def build_rnn_controller(self, activation_input, relevant_index, scope=None, ):
            #with tf.variable_scope("chaos"):


        # activation input has timestep as chaos_number
        # 

        forward_cells = []
        backward_cells = []
        rnn_layers = 1
        cell_size = 10


       
        activation_input = tf.Print(activation_input, [activation_input], "activation_input: ", summarize=90)

        for i in range(rnn_layers):
            forward_cell = tf.contrib.rnn.LSTMCell(10, forget_bias=1.0)
            backward_cell = tf.contrib.rnn.LSTMCell(10, forget_bias=1.0)
            forward_cells.append(forward_cell)
            backward_cells.append(backward_cell)

        '''
        
        outputs: The RNN output Tensor.
        If time_major == False (default), this will be a Tensor shaped: [batch_size, max_time, cell.output_size].

        Note, if cell.output_size is a (possibly nested) tuple of integers or TensorShape objects, 
        then outputs will be a tuple having the same structure as cell.output_size, containing 
        Tensors having shapes corresponding to the shape data in cell.output_size.

        '''
        
        
        rnn_output, fw_state, bw_state = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(                                        
                                                cells_fw=forward_cells, 
                                                cells_bw=backward_cells, 
                                                inputs=activation_input,                   
                                  
                                                dtype=tf.float32)

        rnn_output = tf.Print(rnn_output, [rnn_output], "RNN_OUTPUT", summarize=500)

        relevant_output = rnn_output[relevant_index]

        relevant_output = tf.Print(relevant_output, [relevant_output], "relevant_output: ", summarize=500)

        return tf.layers.dense(relevant_output, self.number_of_nodes, use_bias=False)

        #
        #return fc_layer(input_= rnn_output,
        #                input_size = self.number_of_nodes, 
        #                output_size=self.number_of_nodes, 
        #                activation=tf.tanh, 
        #                bias=False, 
        #                scope="chaos-controller")
    
    
    # Controller Scores each node, takes its previous activation 
    
    def score_nodes(self, activation_input, type="fc", idx=0): 

        #if self._controller is None and type == "fc":
            # can be anytime of graph 
        #   self._controller = self.build_controller(activation_input)
        #   return self._controller
        #elif self._controller is None and type == "rnn":
        if self._controller is None:
            self._controller = self.build_rnn_controller(activation_input, idx)
            return self._controller
        else:
            with tf.variable_scope("chaos", reuse=True):
                return self._controller

    # RNN CONTROLLER NOTES => 
    # INPUT => [BATCH_SIZE, TIME_STEPS, FEATURES]


    # OK SO FOR RNN CONTROLLER => INPUT HAS TO BE CREATED LIKE THIS (assume chaos number = 3)
    # FOR CHAOS ITERATION 0 => INPUT WILL BE [activation0, zero_tensor, zero_tensor]    
    # FOR CHAOS ITERATION 1 => [activation0, activation1, zero_tensor] (sort of like apddig)
    # FOR CHAOS ITERATION 2 => [activation0, activation1, activation2]

    # ALSO SELECT THE LAST RELEVANT OUTPUT. SO FOR CHAOS ITERATION 0, LAST RELEVANT OUTPUT IS 

    
    def pass_through(self, inputs):
        if self._pass_through is None:  
            #just a projection layer, therefore, no bias
            #with tf.variable_scope("chaos"):
            total_chaos_from_pass = tf.TensorArray(size=1, dtype=self.dtype, dynamic_size=True)
            chaos_idx = tf.constant(0, tf.int32)
            
            print("inputs", inputs)
            #print("inputs shape", inputs.shape())
            activation_zero = fc_layer(input_=inputs,
                                    input_size = self.input_size, 
                                    output_size = self.number_of_nodes, 
                                    activation=tf.nn.relu, 
                                    bias=False,
                                    scope="input_projection")
            
            print("activation zero, ", activation_zero)
            activation_zero = tf.Print(activation_zero, [activation_zero], "Activation Zero: ", summarize=90)
            # give activation 0 to each layer

                
            prev_activations = activation_zero
            print("prev_activations, ", prev_activations)
            print("prev_activations_shape, ", prev_activations.get_shape())



            def pass_iteration(idx, cumulative_chaos,  prev_activations):
                # this probably has to be tf.while
                
                cumulative_chaos = cumulative_chaos.write(idx, prev_activations)

                # input has to be padded to chaos number, which is number of timesteps for rnn
                # ok so rn its (timestep, batch_size, features) 
                controller_chaos_input = cumulative_chaos.concat() #stack needs everything written in
                controller_chaos_input = tf.Print(controller_chaos_input, [controller_chaos_input], "controller_chaos_input: ", summarize=90)
                pad_timestep = tf.zeros_like(prev_activations)

                def timestep_pad(index):
                    return tf.cond(tf.less_equal(index, idx), lambda: controller_chaos_input[index], lambda: pad_timestep)

                pad_ids = tf.range(tf.convert_to_tensor(self.chaos_number))
                timestep_padded_controller_chaos = tf.map_fn(lambda id: timestep_pad(id), pad_ids, dtype=tf.float32,
                                                   back_prop=True, 
                                                   parallel_iterations=1)


                # timestep_padded_controller_chaos = timestep_padded_controller_chaos.stack()
                timestep_padded_controller_chaos = tf.Print(timestep_padded_controller_chaos, [timestep_padded_controller_chaos], 
                                                                            "timestep_padded_controller_chaos: ", summarize=90)

                print("timestep_padded_controller_chaos", timestep_padded_controller_chaos)
                timestep_padded_controller_chaos = tf.reshape(timestep_padded_controller_chaos, [-1, self.chaos_number, self.number_of_nodes])

                scores_for_nodes = self.score_nodes(timestep_padded_controller_chaos, type="rnn", idx=idx)

                scores_for_nodes = tf.Print(scores_for_nodes, [scores_for_nodes], "scores_for_nodes: ", summarize=90)


                


                current_activations = self.chaos_iteration(scores_for_nodes, prev_activations) # current activations should be a tensor array of activations
                
                current_activations = tf.Print(current_activations, [current_activations], "CURRENT_ACTIVATIONS IN CHAOS_ITERATION BODY", summarize=90)


                print("current_activations, ", current_activations)

                #current_activations.set_shape((None, self.number_of_nodes))
                return (idx+1, cumulative_chaos, current_activations)
            
            _, total_chaos_from_pass_final, activation_on_final_index = tf.while_loop(
                lambda idx, a, b : tf.less(idx, self.chaos_number), 
                pass_iteration, 
                [chaos_idx, total_chaos_from_pass, prev_activations], 
                shape_invariants=[chaos_idx.get_shape(), tf.TensorShape(None), tf.TensorShape([None, None])],
                parallel_iterations = 1, # must do first iteration before you can start seconnd one right...
                back_prop=True)
            
            stacked_total_chaos = total_chaos_from_pass_final.stack()
            # stacked_total_chaos = tf.Print(stacked_total_chaos, [stacked_total_chaos], "stacked_total_chaos: ")
        
            print("current activations: ", activation_on_final_index)
            #activation_on_final_index = tf.Print(activation_on_final_index, [activation_on_final_index], "activation_on_final_index")
            # final output is a projection layer, so set bias to false
            _pass_through = fc_layer(input_=activation_on_final_index, 
                                    input_size=self.number_of_nodes, 
                                    output_size=self.output_size, 
                                    activation=tf.nn.relu, 
                                    bias=False, 
                                    scope="output")
            return _pass_through
        else:
            #with tf.variable_scope("chaos", reuse=True) :
            return _pass_through

    def get_previous_node_activations(self, nodes):
        # gets node activations
        # nodes is an np array of indexes 
        return [self.nodes[i].get_top_activation for i in nodes]
    

    
    def hard_selected_field_activations_batch(self, candidate_field_for_node_with_weight_matching, node_scores, prev_activations, node_degree):
        # this is the selected field nodes. it is a tensor that indicates each node that was selected in this
        # chaos iteration from the candidate field by a node.
        # its a list of tuples [x,y], where x is the node id (which is used to retrieve the node from the chaos graph), 
        # and the weight it usually couples with. 
        # a greedy tie breaking algo is used, if the weight is already taken. 
        
        # the top k nodes (for batch 1) are 1 -> 3 -> 6 in that order (1, 3, and 6 were in the candidate field for this node, along with a bunch of other ones)     
        # selected_field_nodes_batched = tf.convert_to_tensor([ [[1, 2], [3,2], [6,0]], [[4,2] , [6,0],  [7,1]] ], dtype=tf.float32)
        # selected_field_nodes shape is (node_degree, batch_Size, 2)
        #selected_field_nodes_batched = tf.reshape(selected_field_nodes_batched, [-1, node_degree, 2])

        # so the weights should match like this: 
        # 1 -> weight2
        # 3 -> weight0
        # 6 -> weight1
        # (node1 gets 2, node3 cant get 2 so it gets 0 (its next favorite), node6 cant get 0 so it gets 1 (node6's next favorite))

        # in this algo the previous activations were: (so there were 11 nodes, and the index into the array is the node id):
        
        # in this algo the previous activations were: (so there were 11 nodes, and the index into the array is the node id):
        # activations have batch size 2
        # prev_activations = [[0.3, 0.6, 0.2, 0.5, 0.4, 0.3, 0.7, 0.8, 0.4, 0.6, 0.5], 
        #                    [0.2, 0.5, 0.6, 0.0, 0.74, 0.3, 0.4, 0.1, 0.1, 0.1, 0.2]]


        #selected_field_nodes_batched = tf.convert_to_tensor( [ [[1, 2], [3, 2], [6,0]] ], dtype=tf.float32)

        #prev_activations = [[0.3, 0.6, 0.2, 0.5, 0.4, 0.3, 0.7, 0.8, 0.4, 0.6, 0.5]]

        # the tie breaking algo should give us [3, 6, 1] and then we do prev_activations[3,6,1] to gather the activations for those nodes
        # and finally do the matrix multiple with the weights [0.5, 0.7, 0.6]

        #JUST MAP OVER THE BATHC!!!

        index = 0
        batch_size = self.batch_size


        candidate_field_for_node = candidate_field_for_node_with_weight_matching[:, 0]
        
        #can reshape nodes as [total_number_of_nodes, ?] => then grab it, then reshape back. orrrr use gather and axis argument
        node_scores_for_candidate_field = tf.gather(node_scores, candidate_field_for_node, axis=1)
        # node_scores_for_candidate_field = tf.Print(node_scores_for_candidate_field, [node_scores_for_candidate_field], "node_scodes_for_candidate_field: ", summarize=90)


        top_values, top_indices = tf.nn.top_k(node_scores_for_candidate_field, 
                                                  k=node_degree,
                                                  sorted = True) 
        # get the scores revelent to the particular node
        # top indices reflects index locations into the candidate_field_for_node array
        # which will retrieve the node id's for nodes that are in the top sore.
        print("top_values", top_values)
        print("top_indices", top_indices)
        # top_indices = tf.Print(top_indices, [top_indices], "TOP_INDICIES:  ", summarize=90)
        # top_indices = tf.Print(top_indices, [top_values], "TOP_VALUES:  ", summarize=90)
        
        selected_field_nodes_batched = tf.reshape(tf.gather(tf.convert_to_tensor(candidate_field_for_node_with_weight_matching), 
                                          top_indices), [-1, node_degree, 2]) # WRONG, TOP INDICES IS ONLY FOR NODE_sCORE_FOR_CANDIDATE_FIELD
        # selected_field_nodes = tf.Print(selected_field_nodes, [selected_field_nodes], "SELECTED_FIELD_NODES in chaos iteration body: ", summarize=90)
        # selected_field_nodes = tf.Print(selected_field_nodes, [selected_field_nodes ], "SELECTED FIELD NODES: ", summarize=90)
        # SELECTED FIELD NODES PRINTED ARE:
        #SELECTED FIELD NODES: [[[12 1][25 2][33 2]][[12 1][11 1][25 2]][[12 1][31 0][25 2]][[12 1][31 0][11 1]]]
        # the field input has to be sorted based on which activation inputs will multiply with which weights...
        # sort out the selected_field_nodes and insert them into the array so that ties are broken and the weight matchings are correct
        # actiivations has to be shaped like below => each val in the array is a single activation for a batch size of 2
        # selected_activations=tf.constant([[0.3, 0.4],[0.3, 0.5],[0.3, 0.7]], dtype=tf.float32) 
        # batch size 4 will be like this: 
        # selected_activations=tf.constant([[0.3, 0.4, 0.2, 0.6],[0.3, 0.5, 0.4, 0.5],[0.3, 0.7, 0.7, 0.8]], dtype=tf.float32)        


        def get_activation_from_selected_nodes(selected_field_nodes, batch_idx, debug=False):
            # reading from this
            input_selected_field_nodes_ta = tf.TensorArray(size=node_degree, dtype=tf.float32, dynamic_size=False)

            # selected_field_nodes = tf.Print(selected_field_nodes, [selected_field_nodes], "SELECTED FIELD NODES in selected_field_nodes_batch: ", summarize=90)

            input_selected_field_nodes_arr = input_selected_field_nodes_ta.unstack(tf.cast(selected_field_nodes, dtype=tf.float32)) 

            #writing to this
            weight_matched_nodes_arr = tf.TensorArray(dtype=tf.float32, size=node_degree, dynamic_size=False)
            
            # OK SO I GOT THIS ISSUE:  TensorArray TensorArray_1_1: Could not write to TensorArray index 2 because it has already been read.
            # this is because we cant read from an array and then write to it in the outer while loop, so just split to two output arrays. (MAYBE THATS THE issue not sure)
            #with tf.name_scope(str(calendar.timegm(time.gmtime()))):
            weights_taken_table = tf.contrib.lookup.MutableHashTable(key_dtype=tf.int64, 
                                                                    value_dtype=tf.float32, 
                                                                    default_value=-1,
                                                                     )

            
            # I HAD THIS ERROR FOR THE LONGEST TIME. BECAUSE PARALLEL ITERATIONS FOR MAP WAS 10, AND IT WASNT CLEARING, EVEN WITH THE CONTROL DEPENDENICES 
            # AND INFITIE LOOPING.
            # FIX BY TURNING PARALLEL ITERATIONS TO 1 AND USING THE FOLLOWING CONTROL DEPENDENICES.
            # OKAY TIME FOR SOME OPEN SOURCE WORK FOR THE TF LIBRARY. WHY ISNT THERE A CLEAR HASH TABLE FUNCTION ???
            # OK CODE TO CLEAR MAP HERE:
            # 
            
            map_clearing_ops = []
            for i in range(self.highest_degree_node): # CLEAR UP TO THE HIGHEST DEGREE NODE.
                op = weights_taken_table.insert(tf.constant(i, dtype=tf.int64), tf.constant(-1.0, tf.float32))
                map_clearing_ops.append(op)
                
            # CLEAR THE TABLE ??
            def find_weight_matched_nodes_cond(index, output_arrm):
                return index < node_degree
            
            def find_weight_matched_nodes_body(index, output_arr):
                input_node_matching = input_selected_field_nodes_arr.read(index)
                
                node_id = tf.gather(input_node_matching, 0)
                weight_match = tf.cast(tf.gather(input_node_matching, 1), tf.int64) 
                
            
                def find_weight_match_cond(weight_match, keep_going): # THIS WHILE LOOP IS ACTUALLY an infinite while loop with a break in the body
                    return keep_going
                    
                def find_weight_match_body(weight_match, keep_going):
                    # weight_match = tf.Print(weight_match, [weight_match], "weight_match: ") 
                    taken = weights_taken_table.lookup(weight_match)
                    # taken = tf.Print(taken, [taken], "taken: ")

                    def notTaken(): 
                        insert_op = weights_taken_table.insert(weight_match, tf.constant(1.0, tf.float32)) # put 1.0 to indicate taken
                        with tf.control_dependencies([insert_op]):
                            # Now, we are under the dependency scope:
                            # All the operations happening here will only happens after 
                            # the "assign_op" has been computed first
                            # i think you can put if statements here maybe.

                            # THIS IS A NO-OP TENSOR. because we need some tensor to execute to cause the control dependency to work
                            identity_weight_match = tf.identity(weight_match) 

                        return (identity_weight_match, False)

                    def isTaken():
                        incremented_weight_match = tf.cond(tf.equal(weight_match, tf.cast(node_degree-1, dtype=tf.int64) ), 
                                lambda: tf.constant(0, dtype=tf.int64), lambda: (weight_match + 1))
                        return (incremented_weight_match, True)
                    
                    # its not taken if its -1, otherwise its taken, set 1 if its taken (cant do both these tasks in same array cause tf complains)                   
                    # kill while loop when its not taken    

                    return tf.cond(tf.equal(taken, -1.0), notTaken, isTaken)

                empty_index_to_write_to, _ = tf.while_loop( find_weight_match_cond, 
                                                            find_weight_match_body, 
                                                            parallel_iterations=1,
                                                            back_prop=True, # MAKING THIS TRUE IS OK
                                                            loop_vars=(weight_match, True))
                
                output_arr_changed = output_arr.write(tf.cast(empty_index_to_write_to, dtype=tf.int32), node_id)
                
                return (index + 1, output_arr_changed)

            with tf.control_dependencies(map_clearing_ops):
                _,  final_weight_matched_nodes_arr = tf.while_loop(find_weight_matched_nodes_cond, 
                                                                            find_weight_matched_nodes_body, 
                                                                            loop_vars=(index, weight_matched_nodes_arr), 
                                                                                            shape_invariants=None,
                                                                                            parallel_iterations=1, 
                                                                                            back_prop=True,  #MAYBE NO BACKPROP TRAINING NEEDED HERE BECAUSE WE ARE JUST REORDERING RESULTS. 
                                                                                                            #WHICH WILL THEN MULTIPLY WITH WEIGHTS. YES DISABLE BACK PROP HERE!
                                                                                            swap_memory=False)
            
            final_weight_matched_nodes_tensor = tf.cast(final_weight_matched_nodes_arr.stack(), dtype=tf.int32)
            # final_weight_matched_nodes_tensor = tf.Print(final_weight_matched_nodes_tensor, [final_weight_matched_nodes_tensor], "final_weight_matched_nodes_tensor: ")
            op = final_weight_matched_nodes_arr.close()
            print("tensor array tingssss,", op)
            # prev_activations = tf.Print(prev_activations, [prev_activations], "prev_activations: ", summarize=90)

            weight_matched_activations = tf.gather(prev_activations[tf.cast(batch_idx, dtype=tf.int32)] , final_weight_matched_nodes_tensor) # fcking up here


            #print(result.eval())
            # weight_matched_activations = tf.Print(weight_matched_activations, [weight_matched_activations], "Weight matched activations is: ")
            

            return weight_matched_activations
        
        # ACTIVATIONS SHAPE HAS TO BE (?, DEGREE)

        
        print("selected_field_nodes_batch shape ", selected_field_nodes_batched.shape) # (2, 3, 2) => batchsize, degree size, weight_matching_
        batch_indexes = tf.convert_to_tensor(np.array( [ i for i in range(batch_size) ]  ))
        
        print("batch_indexes,", batch_indexes)
        
        elems = (selected_field_nodes_batched,  batch_indexes) #second array is just indexes to be used.
        batched_activations = tf.map_fn(lambda elem: get_activation_from_selected_nodes(elem[0], elem[1]), 
                                        elems, 
                                        dtype=tf.float32, 
                                        back_prop=True,
                                        parallel_iterations=1)
        
        return batched_activations

    




    # soft version of the hard version above => soft chaos net. does a dot product with chaos score.
    # todo: OK MAKE YOUR OWN DYNAMIC PARTITION WITH A TENSOR ARRAY BECAUSE TENSORFLOW'S DYNAMIC PARTIITON IS GARBAGE. IT requires a static int value for num_partitions 
    # which means each node has to have same degree. Not a requirement, so build your own that is better for testing cooler graphs
    # todo: TO FIX ABOVE PROBLEM, PARTITION THEM BEFORE HAND IN THE GRAPH CREATION, SO YOU CAN JUST IMPORT THE PARTITIONED NDOES INTO THE CHAOS_ITERATION FUNCTION
    #       SO DO THE PARITITIONING BEFOREHANDDDDDDD, BEFORE CREATING THE TF COMPUTATION GRAPH    
    
    def soft_selected_field_activations_batch(self, candidate_field_for_node_with_weight_matching, node_scores, prev_activations, node_degree = 3):
        candidate_field_for_node = candidate_field_for_node_with_weight_matching[:, 0]
        weight_matching_partitions = tf.cast(candidate_field_for_node_with_weight_matching[:, 1], dtype=tf.int32)
        
        
        # candidate_field_for_node = tf.Print( candidate_field_for_node, [ candidate_field_for_node], "CANDIDATE FIELD FOR NODE: ", summarize=90)
        # weight_matching_partitions = tf.Print(weight_matching_partitions, [weight_matching_partitions], "weight_matching_partitions: ", summarize=90)



        # ok just do a dot product over all the candidate field activations with the node score. 
        # they will dot product with other nodes allocated for the index. 
        # so for candidate fields in for weight match 0 -> say that is node 1 and 4 (dot product this with their node score)
        # # so for candidate fields in for weight match 1 -> node 2 and 5 (dot prod this with their node score)
        # # so for candidate fields in for weight match 2 ->  node 3 and 6
        # this imposes restriction that candidate field has to be divisible by degree
        # ok so node score will give weighted average => 

        # how many times can node degree fit into candidate field size.
        # assume candidate field size is 6
  
        # do a tf.dynamic_partition (to put it together, you have to use dynamic_stitch) 

        # number of partiitons is degree
        all_weight_matched_nodes = tf.dynamic_partition(candidate_field_for_node, weight_matching_partitions, num_partitions=self.node_degree)
        print("all weight_matched nodes", all_weight_matched_nodes)
        all_weight_matched_nodes = tf.Print(all_weight_matched_nodes, [all_weight_matched_nodes], "ALL WEIGHT MATCHED NODES: ", summarize=90) #TODO: UNCOMMENTING THIS BREAKS COMPUTATION GRAPH?

 
        # do a map on each element. 

        def soft_compute_selection(weight_matched_nodes):
            
            # say for weight 0, we identified, node a, b, and c to be soft dotted for it. 
            # prev activations => (?)
            # node scores, Tensor("while/Identity_2:0", shape=(?, number_of_nodes), dtype=float32)
            weight_matched_prev_activations = tf.gather(prev_activations, weight_matched_nodes, axis=1)
            weight_matched_scores = tf.gather(node_scores, weight_matched_nodes, axis=1)

            # OK SO THINK OF IT LIEK THIS => we have (batch_size, degree) shape activations and scores
            #  do multiply and reduce sum on axis=1 so we keep batch size.
            # weight_matched_prev_activations = tf.Print(weight_matched_prev_activations, [weight_matched_prev_activations], "weight_matched_prev_activations", summarize=90)
            # weight_matched_scores = tf.Print(weight_matched_scores, [weight_matched_scores], "weight_matched_scores", summarize=90)


            weight_matched_activation = tf.reduce_sum(tf.multiply(weight_matched_prev_activations, weight_matched_scores), axis=1)
            # weight_matched_activation = tf.Print(weight_matched_activation, [weight_matched_activation], "weigh_matched_activation: ", summarize=90)
            return weight_matched_activation

        computed_selection_activations = tf.map_fn(lambda elem: soft_compute_selection(elem), 
                                                   all_weight_matched_nodes, 
                                                   dtype=tf.float32,
                                                   back_prop=True, 
                                                   parallel_iterations=1)

        # computed_selection_activations = computed_selection_activations.stack() # stack it.
        print("computed_selection_activations: ", computed_selection_activations)
        
        # computed_selection_activations = tf.Print(computed_selection_activations, [computed_selection_activations], "computed_selection_activations: ", summarize=90)
        
        return computed_selection_activations


    
    def selected_field_activations(self, candidate_field_for_node_with_weight_matching, node_scores, prev_activations, node_degree, batch_type="SOFT"):
            #if(batch_type == "NO_BATCH"):
            #    return self.selected_field_activations_no_batch(selected_field_nodes, prev_activations, node_degree)
            #else if(batch_type == "BATCH"):
            if(batch_type == "HARD"):
                return self.hard_selected_field_activations_batch(candidate_field_for_node_with_weight_matching, node_scores, prev_activations, node_degree)
            elif(batch_type == "SOFT"): 
                return self.soft_selected_field_activations_batch(candidate_field_for_node_with_weight_matching, node_scores, prev_activations, node_degree)
            else:
                raise "NO BAD CHOICE"
    
    
    def chaos_iteration(self, node_scores, prev_activations):
        
        # Build a tensor reflecting chaos graph relationships that can be used in the tf computation graph
        node_degrees = [node.get_degree() for node in self.nodes]
        node_candidate_fields = [node.get_candidate_field() for node in self.nodes]

        #print(all_node_weights)
        print(self.chaos_weights)
        print(node_candidate_fields)

        node_degree_t = tf.constant(np.array(node_degrees, dtype=np.int32))
        node_candidate_fields_t = tf.convert_to_tensor(np.array(node_candidate_fields, dtype=np.int64))
        # node_weights_t = tf.stack(node_weights) => YOU CANT STACK TENSORFLOW VARIABLES, BECAUSE THEY ARE NOT TENSORS. TF VARIABLES CANNOT BE TREATED LIK TENSORS

        chaos_activations = tf.TensorArray(dtype=tf.float32, size=self.number_of_nodes)
        print("chaos_activations_tensor_array", chaos_activations)

        def chaos_iteration_body(i, activations, weight_index_begin):

            #candidate_field_for_node = tf.reshape(tf.gather(node_candidate_fields_t, i), [-1, 6, 2])
            candidate_field_for_node_with_weight_matching = tf.gather(node_candidate_fields_t, i)
            node_degree = tf.gather(node_degree_t, i)
            print("node_deg,", node_degree)
            print("self.chaos_weights, ", self.chaos_weights)
            node_weights = self.chaos_weights[:, weight_index_begin: weight_index_begin + node_degree]#tf.slice(self.chaos_weights, begin=[weight_index_begin], size=[node_degree]) 
            
       
            print("candidate_field_for_node with weight matching", candidate_field_for_node_with_weight_matching)
            print("node weights: ", node_weights)

            # i is not a tensor, i think its just a python int. also 
            # I THINK IN A WHILE LOOP YOU CAN HAVE NORMAL PYTHON OBJECTS IN THE LOOP VARS!!! such as a {}, or a [], NOT EVERYTHING HAS TO BE A TENSOR rite...
            #  How to index a list with a TensorFlow tensor? -> "Simply run tf.gather(list, tf_look_up[index]), you'll get what you want."
            
            # candidate_field_for_node_with_weight_matching = tf.Print(candidate_field_for_node_with_weight_matching, [candidate_field_for_node_with_weight_matching], "candidate_field_for_node_with_weight_matching", summarize=90)

            
            print("node scores,", node_scores) # node scores, Tensor("while/Identity_2:0", shape=(?, 50), dtype=float32)
                  
            selected_activations = tf.reshape(self.selected_field_activations(candidate_field_for_node_with_weight_matching, node_scores, prev_activations, node_degree, "SOFT"), [node_degree, -1])
            # selected_activations = tf.Print(selected_activations, [selected_activations], "SELECTED_ACTIVATIONS: ", summarize=90)
            # node_weights = tf.Print(node_weights, [node_weights], "NODE WEIGHTS IN CHAOS ITERATION BODY: ", summarize=90)

            node_dot_prod = tf.matmul(node_weights, selected_activations)
            #print("node_mat_mult", node_dot_prod)

            #node_evaluation = tf.reduce_sum(node_mat_mult)
            # node_dot_prod = tf.Print(node_dot_prod, [node_dot_prod], "NODE_DOT_PROD: ", summarize=90)
            node_activation = tf.reshape( tf.nn.relu(node_dot_prod), [-1])
            # node_activation = tf.Print(node_activation, [node_activation], "NODE_ACTIVATION: ", summarize=90)

            
            # node.add_activation(node_activation)
            # chaos_activations.append(node_activation)
            print("node_activation,", node_activation)

            activations = activations.write(i, node_activation)
            
            return (i+1, activations, weight_index_begin + node_degree)

        _, final_chaos_activations, _ = tf.while_loop(
            lambda idx, a, b: tf.less(idx, self.number_of_nodes),  
            chaos_iteration_body,
            (0, chaos_activations, 0),
            back_prop=True,
            parallel_iterations=1
        )

        new_activations = tf.reshape(final_chaos_activations.stack(), (-1, self.number_of_nodes))
        # new_activations = tf.Print(new_activations, [new_activations], "NEW ACTIVATIONS: ")

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
    
      
    def train(self, batch_x, batch_y, learning_rate = 0.01):
        if self._train is None:
            logits = self.pass_through(batch_x)

            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, 
                                                                             labels=batch_y))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
            compute_grads = optimizer.compute_gradients(loss_op)
            train_op = optimizer.apply_gradients(grads_and_vars=compute_grads)#optimizer.minimize(loss_op)
            self._train = train_op, loss_op, compute_grads
        
        return self._train

    
    def test(self, batch_x, batch_y):
        if self._test is None: 
            logits = self.pass_through(batch_x)
            correct_prediction = tf.equal(tf.argmax(logits, -1), tf.argmax(batch_y, -1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            error = (1 - accuracy) * 100
            loss_op = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=batch_y,
                                                                    logits=logits))
            self._test = error, loss_op, accuracy
        
        return self._test
