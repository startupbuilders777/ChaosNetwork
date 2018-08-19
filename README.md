# CHAOS NETWORKS design and implementation paper 
### Author: Harman Singh

## INTRODUCTION 
The chaos network is designed similar to the human brain. Relationships between neurons, and paths of neural activity formed
during pattern recognition is the basis for the "chaos" network.

The chaos network is a very malleable deep learning network since each neuron or node can speak to k other nodes
a set number of times and complex graphs can be formed
between these neurons for activity paths to travel.
Certain types of graphs shapes and arrangements will be better than others for certain problems (graph shapes will be discussed in a later investigation).

EACH NODE HAS A CERTAIN DEGREE, WHICH SPECIFIES THE NUMBER OF WEIGHTS IT HAS.
Each node does the following computation => A = tanh(w1*x1 + w2*x2 + w3*x3 + ... + wk*xk), where k is the degree for the node.
(tanh can be replaced by other nonlinearities such as relu, this will be further investigated later).

For example, each node does a dot product with its inputs. so a node with degree 3, will have 3 weights it will need to train, and will take 3 inputs which will be
outputs from 3 other nodes in the chaos graph (and these outputs were calculated in iteration i-1 if we are computing the activations in iteration i)




## Design Details of ChaosNetwork:

The chaos network is iteratiive, so it can have an iteration count of 10, which results in 10 activations created by each node in the graph, for each test sample.
(A possible thing that can be investigated later is backpropping 10 times, instead of once for each test example as a result)

a fully connected layer projects input into output, output is equal to number of nodes in the CHAOS GRAPH.
the chaos graph is a bunch of nodes conneceted to other nodes (connected as in, it takes its activations as inputs.)
iterations are run on the chaos network

Each node has a candidate field and an activation field.
A candidate field is all the nodes that the node can talk to. (sort of like in the brain, a neuron has neurons surrounding it that it can spark)
the activation field is the specific nodes chosen from the candidate field that will be spoken to in the current iteration.
("spoken to" means "take activations from to be used in its dot product calculation")

HARD CONSTRAINTS for this network
Each node computes a dot product, so each node outputs only 1 value each iteration.
Each node stores all its activations in a node list.(THIS node list of activations can be used for all sorts of things, for instance, an output projection layer can finally
aggregate all the activations computed from the chaos graph in 10 iterations to compute the final output probabilities for a classification problem)


A "Controller" neural network is used to decide which nodes the node should choose from its candidate field.
To decide, the controller network will take in the previous activation for that node, and activations for all the nodes in the candidate field, and it will score each
node in the candidate field. since the node has degree k, the top k nodes will be used for activation.

The controller neural network can be made in lots of ways...

DOES THIS MEAN EACH NODE HAS A NEURAL NETWORK since each node needs a controller? (SEEMS INTRACTABLE.). Not that bad idea of an idea(maybe something that can be tested later).
better solution is one controller or 2 controllers (sort of like the 2 hemispheres that control the human brain).
The controller can be a series of full connected neural network layers (or conv net layers, rnns) that will take in activation input from each node in the chaos graph.
Output size for the controller is the same as the number of nodes because each node will be a given a score.

After getting the score, the nodes will choose the top k scored nodes in their candidate fields to be in its active field for that iteration.

The chaos network should form paths in terms of nodes chosen.



## Other cool thoughts:
Since the iterations for each node are stored in the node activiation list, and there are X nodes, and K iterations, then we have data that is usable 
by an LSTM to make predictions based on timestepped data where the number of timesteps is K, and each timestep has X features. 
(Features can be organized based on the graph shape).




## ALGORIHTM OUTLINE:
```
FOR EACH TEST EXAMPLE
GRAB TEST EXAMPLE (X, Y)
Input projection layer(can be convolutional nets, FC layers, ) takes in X creates activation 0, for each node in chaos graph
The output from input projection layer is pushed into the node activation list as the first element for each node (so activation at index 0, or activation 0)

    FOR i from 1 to X (So the iteration hyperparameter for the chaos graph is X)
        nodes are scored by controller graph (score is calculated based on activations at time i-1, which in this case is the output from the input projection layer),
        each node finds top k nodes and then computes its dot product and tanh , and stores that value in its node activation list. (use tie breaking rules for weight sharing in this step)


    Values in Node activation list are then fed into a convolutional net/feed forward net to compute output (output can be probabilities for a classiication problem)
    (The output projection layer can be whatever you want it to be)
```

## DEEPER QUESTIONS: 
Can weights be shared between nodes? A weight for a node will multiply with inputs from different nodes instead of the same node each time which is done in traditional feedforward networks. 




## IMPORTANT DETAILS ON WEIGHT SHARING FOR CHAOS NETWORK: 

Each node in the candidate field will always be multiplied by the same weight (out of the k weights) when it is chosen for the activation field.
Node X's candidate field nodes will have a weight-match with a particular weight in node X! 
This means if (a, b, c, d, e) are nodes in the candidate field for node F, and node F has weights (w1, w2, w3) (Degree 3),
then a will always multiply with w1 if its chosen, (lets use the term weigth-match, so a's weight-match is w1 )
     b will always multiple with w2 if its chosen, 
     c will always multiple with w3 if its chosen, 
     d will always multiply with w1 if its chosen,  
     e will always multiply with w2 if its chosen as an example. 
    The reason for this constraint is so that these weights can then gain affinity with the nodes that can be chosen for those weights, leading to meaningful learning in the network, 
    without this constraint the weights will not learn ANYTHING because the weights are backpropping on values from too many different nodes inconsistently!! 
    IF WE ARE TO ALLOW WEIGHTS TO BE LEARNING ON DIFFERENT NODES (AKA THE QUESTION OF WEIGHT SHARING), then the weight should only be consistently shared between some nodes, and never any other
    nodes. (In this example, A AND D WILL ALWAYS SHARE WEIGHT 1). 

    Problem: WHAT IF the activation field is (a, d, e), then which weights are assigned to which nodes, since a and d are both assigned to weight 1?
    Here are the solutions we can investigate: 
    Solution 1: DO NOT ALLOW THE ACTIVATION FIELD TO CONTAIN Nodes with common weight-matches. 
    Solution 2: Allow each node to have a priority list of weights matches, so if a weight is taken by another node, then go to the second weight match in the priority list and check if thats
                taken and if it isnt, then take that weight. Therefore  every node in the candidate field can possibly be assigned to  any weight.
                The weights will be taken based on highest score to lowest score greedily, by the node's in the activation field, and preferences based on the priority list. 

    Solution 3: Use simpler tie breaking rules than solution 2. (however simpler rules than solution 2 will reduce weight node affinity, which is the basis for fast, viable, and accurate learing)
    









## Advantages of Chaos Network 
- Can memorize lots of details due to the high number of permutations and states for "memories" to be in
- Sort of like the human brain.

## Disadvantages of Chaos Network
- Very Indeterministic and a lot of configuration compared to other networks
