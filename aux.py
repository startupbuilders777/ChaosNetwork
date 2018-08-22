# play around with tensorflow implementations here

import tensorflow as tf
import numpy as np


# Tensorflow stuff

def example1(): 
    sess = tf.InteractiveSession()

    node_degree = 3
    TensorArr = tf.TensorArray(dtype=tf.float32, size=node_degree, dynamic_size=False)


    sorted_nodes = TensorArr.unstack( np.array([-1] * node_degree, dtype=np.float32) )
    x_elem0 = sorted_nodes.read(0) # first element

    print_elem = tf.Print(x_elem0, [x_elem0], message="yee tensor array value is: ")

    print(print_elem.eval()) # another way to print values

    # encode tie break rule attempt here:

    selected_field_nodes = tf.convert_to_tensor([[1, 2], [3, 2], [6,0]])
    #tf.gather(tf.convert_to_tensor(candidate_field_for_node), top_indices)

    stacked_tensor = sorted_nodes.stack() # GET BACK NORMAL TENSOR FROM TENSOR ARRAY LIKE SO
    print(stacked_tensor.eval())


    '''
    # what the algo would be if we could use normal to put the code into the tf computation graph:

    for i in tf.unstack(selected_field_nodes):
        while True:
            weight_match = tf.squeeze(tf.gather(i, [1]))
            if(sorted_nodes.read(weight_match) == -1):
                sorted_nodes = sorted_nodes.write(index=weight_match, value=tf.squeeze(tf.gather(i, [0])) )
                break
            else: 
                weight_match += 1
                if weight_match == node_degree:
                    weight_match = 0
    '''
def tie_breaking_algo_test():

    sess = tf.InteractiveSession()

    node_degree = 3
    index = 0
    
    # this is the selected field nodes. it is a tensor that indicates each node that was selected in this
    # chaos iteration from the candidate field by a node.
    # its a list of tuples [x,y], where x is the node id (which is used to retrieve the node from the chaos graph), 
    # and the weight it usually couples with. 
    # a greedy tie breaking algo is used, if the weight is already taken. 
     
    # the top k nodes are 1 -> 3 -> 6 in that order (1, 3, and 6 were in the candidate field for this node, along with a bunch of other ones)     
    selected_field_nodes = tf.convert_to_tensor([[1, 2], [3, 2], [6,0]], dtype=tf.float32)
    # so the weights should match like this: 
    # 1 -> weight2
    # 3 -> weight0
    # 6 -> weight1
    # (node1 gets 2, node3 cant get 2 so it gets 0 (its next favorite), node6 cant get 0 so it gets 1 (node6's next favorite))

    # in this algo the previous activations were: (so there were 11 nodes, and the index into the array is the node id):
    
    prev_activations = [0.3, 0.6, 0.2, 0.5, 0.4, 0.3, 0.7, 0.8, 0.4, 0.6, 0.5]


    # the tie breaking algo should give us [3, 6, 1] and then we do prev_activations[3,6,1] to gather the activations for those nodes
    # and finally do the matrix multiple with the weights [0.5, 0.7, 0.6]

    # reading from this
    input_selected_field_nodes_arr = tf.TensorArray(size=node_degree, dtype=tf.float32)
    input_selected_field_nodes_arr = input_selected_field_nodes_arr.unstack(selected_field_nodes) 

    #writing to this
    weight_matched_nodes_arr = tf.TensorArray(dtype=tf.float32, size=node_degree, dynamic_size=False)
    #weights_matched_nodes_arr = weight_matched_nodes_ta.unstack(np.array([-1] * node_degree, dtype=np.float32)) 
    
    # OK SO I GOT THIS ISSUE:  TensorArray TensorArray_1_1: Could not write to TensorArray index 2 because it has already been read.
    # this is because we cant read from an array and then write to it in the outer while loop, so just split to two output arrays.

    weights_taken = tf.TensorArray(dtype=tf.float32, size=node_degree, dynamic_size=False)
    weights_taken_arr = weights_taken.unstack(np.array([-1] * node_degree, dtype=np.float32)) 


    def find_weight_matched_nodes_cond(index, output_arr):
        return index < node_degree
    
    def find_weight_matched_nodes_body(index, output_arr):
        input_node_matching = input_selected_field_nodes_arr.read(index)
        node_id = tf.gather(input_node_matching, 0)
        
        weight_match = tf.cast(tf.gather(input_node_matching, 1), tf.int32) 

    



        def find_weight_match_cond(weight_match):
            # its not taken if its -1, otherwise its taken, set 1 if its taken (cant do both these tasks in same array cause tf complains) 
            taken = weights_taken_arr.read(weight_match)
            
            def notTaken(): 
                weights_taken_arr.write(weight_match, 1) # Take it. 
                return True

            def isTaken():
                return False
                        
            return tf.cond(tf.equal(taken, -1), notTaken, isTaken)     

        def find_weight_match_body(weight_match):
            return tf.cond(tf.equal(weight_match, node_degree), lambda: 0, lambda: (weight_match + 1))

        empty_index_to_write_to = tf.while_loop(find_weight_match_cond, 
                                                find_weight_match_body, 
                                                parallel_iterations=1,
                                                loop_vars=[weight_match])
        
        output_arr_changed = output_arr.write(empty_index_to_write_to, node_id)
                            
        return (index + 1, output_arr_changed)

    index_final, weight_matched_nodes_arr_final = tf.while_loop(find_weight_matched_nodes_cond, 
                                                                find_weight_matched_nodes_body, 
                                                                loop_vars=(index, weight_matched_nodes_arr), 
                                                                                shape_invariants=None,
                                                                                parallel_iterations=1, 
                                                                                back_prop=True,  #MAYBE NO BACKPROP TRAINING NEEDED HERE BECAUSE WE ARE JUST REORDERING RESULTS. 
                                                                                                #WHICH WILL THEN MULTIPLY WITH WEIGHTS. YES DISABLE BACK PROP HERE!
                                                                                swap_memory=False)
    
    result = weight_matched_nodes_arr_final.stack() 
    print_result = tf.Print(result, [result], "Result is: ")
    print_result.eval()
   

tie_breaking_algo_test()