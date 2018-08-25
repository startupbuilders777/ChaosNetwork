# play around with tensorflow implementations here

import tensorflow as tf
import numpy as np
import calendar
import time

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
    close_op = sorted_nodes.close()

    print("close op ta", close_op) 
 
    stacked_tensor = sorted_nodes.stack() # GET BACK NORMAL TENSOR FROM TENSOR ARRAY LIKE SO
    
    
    print(stacked_tensor.eval())




example1()



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
# THIS WORKS UP TO AN EXTENT BUT BREAKS WHEN WE WRITE AND READ TO A TENSOR ARRAY. OK LETS USE A TENSORFLOW MUTABLE HASH TABLE
'''
InvalidArgumentError (see above for traceback): TensorArray TensorArray_2_1: Could not write to TensorArray index 2 because it has already been read.
	 [[Node: while/while/cond/TensorArrayWrite/TensorArrayWriteV3 = TensorArrayWriteV3[T=DT_FLOAT, _class=["loc:@TensorArrayUnstack_1/TensorArrayScatter/value"], _device="/job:localhost/replica:0/task:0/cpu:0"](while/while/cond/TensorArrayWrite/TensorArrayWriteV3/Switch:1, while/while/cond/TensorArrayWrite/TensorArrayWriteV3/Switch_1:1, while/while/cond/Cast, while/while/cond/TensorArrayWrite/TensorArrayWriteV3/Switch_2:1)]]

'''

def tie_breaking_algo_test_attempt_1():

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
    # this is because we cant read from an array and then write to it in the outer while loop, so just split to two output arrays. (MAYBE THATS THE issue not sure)

    weights_taken = tf.TensorArray(dtype=tf.float32, 
                                   size=node_degree, 
                                   dynamic_size=False, 
                                   clear_after_read=False) # HAVE TO SET THIS SO YOU CAN READ INDEXES MULTIPLE TIMES.

    weights_taken_arr = weights_taken.unstack(np.array([-1] * node_degree, dtype=np.float32)) 


    def find_weight_matched_nodes_cond(index, output_arrm, weights_taken_arr):
        return index < node_degree
    
    def find_weight_matched_nodes_body(index, output_arr, weights_taken_arr):
        input_node_matching = input_selected_field_nodes_arr.read(index)
        node_id = tf.gather(input_node_matching, 0)
        
        weight_match = tf.cast(tf.gather(input_node_matching, 1), tf.int32) 

    
        def find_weight_match_cond(weight_match, taken_arr, keep_going):
            return keep_going
             

        def find_weight_match_body(weight_match, taken_arr, keep_going):
            #keep_going is our variable, we just return it in the cond
            taken = taken_arr.read(weight_match)
            # taken_print = tf.Print(taken, [taken], "yee: ")
            # weight_match_print = tf.Print(weight_match, [weight_match], "weight match val: ")
            
            def notTaken(): 
                modified_taken_arr = taken_arr.write(weight_match, tf.cast(1, dtype=tf.float32)) # Take it. (PROBLEM WITH THIS IS YOU ARENT KEEPING THE NODE!!!)??????
                #stack and unstack it to get rid of error
                new_ta = tf.TensorArray(dtype=tf.float32, size=node_degree, dynamic_size=False).unstack(modified_taken_arr.stack())

                
                #val_print = tf.Print(val, [val], message="fook")
                #return val_print
                return (weight_match, new_ta, False)

            def isTaken():
                incremented_weight_match = tf.cond(tf.equal(weight_match, node_degree-1), lambda: 0, lambda: (weight_match + 1))
                #val_print = tf.Print(val, [val], message="poop")
                return (incremented_weight_match, taken_arr, True)
            
            # its not taken if its -1, otherwise its taken, set 1 if its taken (cant do both these tasks in same array cause tf complains)                   
            # kill while loop when its not taken    

           
            return tf.cond(tf.equal(taken, -1), notTaken, isTaken)



        empty_index_to_write_to, modified_weights_taken_arr, _ = tf.while_loop(find_weight_match_cond, 
                                                                            find_weight_match_body, 
                                                                            parallel_iterations=1,
                                                                            loop_vars=(weight_match, weights_taken_arr, True))
        
        output_arr_changed = output_arr.write(empty_index_to_write_to, node_id)
                            
        return (index + 1, output_arr_changed, modified_weights_taken_arr)

    index_final, final_weight_matched_nodes_arr, final_weights_taken_arr = tf.while_loop(find_weight_matched_nodes_cond, 
                                                                find_weight_matched_nodes_body, 
                                                                loop_vars=(index, weight_matched_nodes_arr, weights_taken_arr), 
                                                                                shape_invariants=None,
                                                                                parallel_iterations=1, 
                                                                                back_prop=True,  #MAYBE NO BACKPROP TRAINING NEEDED HERE BECAUSE WE ARE JUST REORDERING RESULTS. 
                                                                                                #WHICH WILL THEN MULTIPLY WITH WEIGHTS. YES DISABLE BACK PROP HERE!
                                                                                swap_memory=False)
    
    result = final_weight_matched_nodes_arr.stack() 
    print_result = tf.Print(result, [result], "Result is: ")
    print_result.eval()
   

#tie_breaking_algo_test_attempt_1()




## MUTABLE HASH ARRAY USED LIKE THIS:
def tf_mutable_hash_table_techniques(): 
    sess = tf.InteractiveSession()

    table = tf.contrib.lookup.MutableHashTable(key_dtype=tf.string, value_dtype=tf.float32, default_value=-1)
    key = tf.constant('hi', tf.string)
    val = tf.constant(1.1,tf.float32)
    table_insert_operation = table.insert(key, val) # THIS IS AN OPERATION. WE CANT JUST ADD IT TO THE COMPUTATION GRAPH BUT WE WANT
                                                    # IT EXECUTED BEFORE WE DO THE LOOKUP. SO NEED A TF.CONTROL_DEPENDENCY
    #sess.run(table_insert_operation) i want to make table_insert_operation run before the lookup so add a control dependency
    with tf.control_dependencies([table_insert_operation]):
        value = table.lookup(key)
        print_table = tf.Print(value, [value], "hash table value: ") 
    
    value2 = table.lookup(key)
    print_table.eval()
    print(value2.eval()) # value 2 prints 1.1 too because print table was evaluated before hand!

    value3 = table.lookup(tf.constant("foo", tf.string))
    print(value3.eval()) # this prints -1 which is the default value.
    '''
    BTW, these are the only data types supported by mutable has table

    tensorflow.python.framework.errors_impl.InvalidArgumentError: No OpKernel was registered to support Op 
            'MutableHashTableV2' with these attrs.  Registered devices: [CPU], Registered kernels:
    device='CPU'; key_dtype in [DT_INT64]; value_dtype in [DT_FLOAT]
    device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_BOOL]
    device='CPU'; key_dtype in [DT_INT64]; value_dtype in [DT_STRING]
    device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_INT64]
    device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_FLOAT]
    '''

# you cannot put tensor variables in hash map
def tf_mutable_hash_table_techniques_2():
    table2 = tf.contrib.lookup.MutableHashTable(key_dtype=tf.int64, value_dtype=tf.float32, default_value=-1)

    for i in range(12):
        var = tf.get_variable(name=str(i), shape=[1], dtype=tf.float32)
        insert_op = table2.insert(tf.constant(i, dtype=tf.int64), var)
    
    #insert_op.eval()
    result = table2.lookup(tf.constant(2, tf.int64))
    print(result.eval())

# tf_mutable_hash_table_techniques_2()

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
    # activations have batch size 2
    prev_activations = [0.3, 0.6, 0.2, 0.5, 0.4, 0.3, 0.7, 0.8, 0.4, 0.6, 0.5]


    # the tie breaking algo should give us [3, 6, 1] and then we do prev_activations[3,6,1] to gather the activations for those nodes
    # and finally do the matrix multiple with the weights [0.5, 0.7, 0.6]

    # reading from this
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

            def notTaken(): 
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

            def isTaken():
                incremented_weight_match = tf.cond(tf.equal(weight_match, node_degree-1), lambda: tf.constant(0, dtype=tf.int64), lambda: (weight_match + 1))
                return (incremented_weight_match, True)
            
            # its not taken if its -1, otherwise its taken, set 1 if its taken (cant do both these tasks in same array cause tf complains)                   
            # kill while loop when its not taken    

            return tf.cond(tf.equal(taken_print, -1.0), notTaken, isTaken)

        empty_index_to_write_to, _ = tf.while_loop( find_weight_match_cond, 
                                                    find_weight_match_body, 
                                                    parallel_iterations=1,
                                                    loop_vars=(weight_match, True))
        
        output_arr_changed = output_arr.write(tf.cast(empty_index_to_write_to, dtype=tf.int32), node_id)
        #output_arr.write(tf.cast(empty_index_to_write_to, dtype=tf.int32), node_id)
           
        return (index + 1, output_arr_changed)

    _,  final_weight_matched_nodes_arr = tf.while_loop(find_weight_matched_nodes_cond, 
                                                                find_weight_matched_nodes_body, 
                                                                loop_vars=(index, weight_matched_nodes_arr), 
                                                                                shape_invariants=None,
                                                                                parallel_iterations=1, 
                                                                                back_prop=True,  #MAYBE NO BACKPROP TRAINING NEEDED HERE BECAUSE WE ARE JUST REORDERING RESULTS. 
                                                                                                #WHICH WILL THEN MULTIPLY WITH WEIGHTS. YES DISABLE BACK PROP HERE!
                                                                                swap_memory=False)
    
    final_weight_matched_nodes_tensor = tf.cast(final_weight_matched_nodes_arr.stack(), dtype=tf.int32)
    weight_matched_activations = tf.gather(prev_activations, final_weight_matched_nodes_tensor)
    print_result = tf.Print(weight_matched_activations, [weight_matched_activations], "Weight matched activations is: ")
    print_result.eval()
    

tie_breaking_algo_test();


def tie_breaking_algo_with_batches_test():
    
    sess = tf.InteractiveSession()

    node_degree = 3
    index = 0
    batch_size=2;

    # this is the selected field nodes. it is a tensor that indicates each node that was selected in this
    # chaos iteration from the candidate field by a node.
    # its a list of tuples [x,y], where x is the node id (which is used to retrieve the node from the chaos graph), 
    # and the weight it usually couples with. 
    # a greedy tie breaking algo is used, if the weight is already taken. 
     
    # the top k nodes (for batch 1) are 1 -> 3 -> 6 in that order (1, 3, and 6 were in the candidate field for this node, along with a bunch of other ones)     
    selected_field_nodes_batched = tf.convert_to_tensor([ [[1, 2], [3,2], [6,0]], [[4,2] , [6,0],  [7,1]] ], dtype=tf.float32)
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
    prev_activations = [[0.3, 0.6, 0.2, 0.5, 0.4, 0.3, 0.7, 0.8, 0.4, 0.6, 0.5], 
                        [0.2, 0.5, 0.6, 0.0, 0.74, 0.3, 0.4, 0.1, 0.1, 0.1, 0.2]]


    #selected_field_nodes_batched = tf.convert_to_tensor( [ [[1, 2], [3, 2], [6,0]] ], dtype=tf.float32)

    #prev_activations = [[0.3, 0.6, 0.2, 0.5, 0.4, 0.3, 0.7, 0.8, 0.4, 0.6, 0.5]]

    # the tie breaking algo should give us [3, 6, 1] and then we do prev_activations[3,6,1] to gather the activations for those nodes
    # and finally do the matrix multiple with the weights [0.5, 0.7, 0.6]

    #JUST MAP OVER THE BATHC!!!

    def get_activation_from_selected_nodes(selected_field_nodes, batch_idx):
        # reading from this
        input_selected_field_nodes_ta = tf.TensorArray(size=node_degree, dtype=tf.float32, dynamic_size=False)
        input_selected_field_nodes_arr = input_selected_field_nodes_ta.unstack(selected_field_nodes) 

        #writing to this
        weight_matched_nodes_arr = tf.TensorArray(dtype=tf.float32, size=node_degree, dynamic_size=False)
        #weights_matched_nodes_arr = weight_matched_nodes_ta.unstack(np.array([-1] * node_degree, dtype=np.float32)) 
        
        # OK SO I GOT THIS ISSUE:  TensorArray TensorArray_1_1: Could not write to TensorArray index 2 because it has already been read.
        # this is because we cant read from an array and then write to it in the outer while loop, so just split to two output arrays. (MAYBE THATS THE issue not sure)
        #with tf.name_scope(str(calendar.timegm(time.gmtime()))):
        weights_taken_table = tf.contrib.lookup.MutableHashTable(key_dtype=tf.int64, 
                                                                 value_dtype=tf.float32, 
                                                                 default_value=-1
                                                                  ) # fetech new mutable map each time

        
        # I HAD THIS ERROR FOR THE LONGEST TIME. BECAUSE PARALLEL ITERATIONS FOR MAP WAS 10, AND IT WASNT CLEARING, EVEN WITH THE CONTROL DEPENDENICES 
        # AND INFITIE LOOPING.
        # FIX BY TURNING PARALLEL ITERATIONS TO 1 AND USING THE FOLLOWING CONTROL DEPENDENICES.
        # OKAY TIME FOR SOME OPEN SOURCE WORK FOR THE TF LIBRARY. WHY ISNT THERE A CLEAR HASH TABLE FUNCTION ???
        # OK CODE TO CLEAR MAP HERE:
        # 
        
        map_clearing_ops = []
        for i in range(node_degree):
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
                weight_match_print = tf.Print(weight_match, [weight_match], "weight_match: ") 
                taken = weights_taken_table.lookup(weight_match_print)
                taken_print = tf.Print(taken, [taken], "taken: ")

                def notTaken(): 
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

                def isTaken():
                    incremented_weight_match = tf.cond(tf.equal(weight_match, tf.cast(node_degree-1, dtype=tf.int64)), lambda: tf.constant(0, dtype=tf.int64), lambda: (weight_match + 1))
                    return (incremented_weight_match, True)
                
                # its not taken if its -1, otherwise its taken, set 1 if its taken (cant do both these tasks in same array cause tf complains)                   
                # kill while loop when its not taken    

                return tf.cond(tf.equal(taken_print, -1.0), notTaken, isTaken)

            empty_index_to_write_to, _ = tf.while_loop( find_weight_match_cond, 
                                                        find_weight_match_body, 
                                                        parallel_iterations=1,
                                                        loop_vars=(weight_match, True))
            
            output_arr_changed = output_arr.write(tf.cast(empty_index_to_write_to, dtype=tf.int32), node_id)
            #output_arr.write(tf.cast(empty_index_to_write_to, dtype=tf.int32), node_id)
            
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
        
        op = final_weight_matched_nodes_arr.close()
        print("tensor array tingssss,", op)
        print_prev_activations = tf.Print(prev_activations, [prev_activations], "prev_activations: ")

        weight_matched_activations = tf.gather(print_prev_activations[batch_idx ] , final_weight_matched_nodes_tensor) # fcking up here
        
        result = weights_taken_table.export()
        print("result", result)

        #print(result.eval())
        print_result = tf.Print(weight_matched_activations, [weight_matched_activations], "Weight matched activations is: ")
         

        return print_result
    
    # ACTIVATIONS SHAPE HAS TO BE (?, DEGREE)

    
    print("selected_field_nodes_batch shape ", selected_field_nodes_batched.shape) # (2, 3, 2) => batchsize, degree size, weight_matching_
    batch_indexes = tf.convert_to_tensor(np.array( [ i for i in range(batch_size) ]  ))
    
    print("batch_indexes,", batch_indexes)
    
    elems = (selected_field_nodes_batched,  batch_indexes) #second array is just indexes to be used.
    batched_activations = tf.map_fn(lambda elem: get_activation_from_selected_nodes(elem[0], elem[1]), 
                                    elems, 
                                    dtype=tf.float32, 
                                    parallel_iterations=1)

    
    print(batched_activations.eval())
    return batched_activations


tie_breaking_algo_with_batches_test();
