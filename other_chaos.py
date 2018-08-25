class OtherChaos():

        # NO_BATCH VERSION MIGHT BE BETTER THAN BATCH VERSION (gradients might not be going through the computation graph in the batch version properly, need to test gradient flow)
    # (HAVE TO CHECK IN THE TRAINING IF BATCH CAUSES IMPROVEMNTS OR NOT. USUALLY It does though for other ml models).
    def selected_field_activations_no_batch(self, selected_field_nodes, prev_activations, node_degree):
            # reading from this
        print("selected_field_nodes put into selected_field_activations,", selected_field_nodes)
        print("prev activations put into selected_field_activationsm, ", prev_activations)

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

    

