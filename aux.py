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

    activation_field_nodes = tf.convert_to_tensor([[1, 2], [3, 2], [6,0]])
    #tf.gather(tf.convert_to_tensor(candidate_field_for_node), top_indices)

    stacked_tensor = sorted_nodes.stack() # GET BACK NORMAL TENSOR FROM TENSOR ARRAY LIKE SO
    print(stacked_tensor.eval())


    '''
    # what the algo would be if we could use normal to put the code into the tf computation graph:

    for i in tf.unstack(activation_field_nodes):
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
def example2():

    sess = tf.InteractiveSession()

    node_degree = 3
    index = 0

    activation_field_nodes = tf.convert_to_tensor([[1, 2], [3, 2], [6,0]])

    # reading from this
    input_activation_field_nodes_arr = tf.TensorArray(size=node_degree, dtype=tf.float32)
    input_activation_field_nodes_arr = input_activation_field_nodes_arr.unstack(activation_field_nodes) 

    #writing to this
    weight_matched_nodes_ta = tf.TensorArray(dtype=tf.float32, size=node_degree, dynamic_size=False)
    weight_matched_nodes_arr = weight_matched_nodes_ta.unstack(np.array([-1] * node_degree, dtype=np.float32)) 

    def find_weight_matched_nodes_cond(index, output_arr):
        return index < node_degree
    
    def find_weight_matched_nodes_body(index, output_arr):
        input_node_matching = input_activation_field_nodes_arr.read(index)
        node_activation = tf.gather(input_node_matching, 0)
        
        weight_match = tf.cast(tf.gather(input_node_matching, 1), tf.int32) #THIS MIGHT NOT WORK ACCTUALY!

        def find_weight_match_cond(weight_match):
            return tf.not_equal(output_arr.read(weight_match), -1)  

        def find_weight_match_body(weight_match):
            return tf.cond(tf.equal(weight_match, node_degree), lambda: 0, lambda: (weight_match + 1))

        empty_index_to_write_to = tf.while_loop(find_weight_match_cond, 
                                                find_weight_match_body, 
                                                loop_vars=[weight_match])
        
        output_arr_changed = output_arr.write(empty_index_to_write_to, node_activation)
          
        return (index + 1, output_arr_changed)

    index_final, weight_matched_nodes_arr_final = tf.while_loop(find_weight_matched_nodes_cond, 
                                                                find_weight_matched_nodes_body, 
                                                                loop_vars=(index, weight_matched_nodes_arr), 
                                                                                shape_invariants=None,
                                                                                parallel_iterations=1, 
                                                                                back_prop=True,  #MAYBE NO BACKPROP TRAINING NEEDED HERE BECAUSE WE ARE JUST REORDERING RESULTS. 
                                                                                                #WHICH WILL THEN MULTIPLY WITH WEIGHTS. YES DISABLE BACK PROP HERE!
                                                                                swap_memory=False, 
                                                                                name=None)
    
    result = weight_matched_nodes_arr_final.stack() 
    print_result = tf.Print(result, [result], "Result is: ")
    print_result.eval()

example2()