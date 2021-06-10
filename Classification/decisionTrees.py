'''
It is a Supervised Algorithm.
A Decision tree is a flowchart like tree structure, where each internal node denotes a test 
on an attribute, each branch represents an outcome of the test, 
and each leaf node (terminal node) holds a class label.

****
Assumptions:
    1. At the beginning, we consider the whole training set as the root.
    2. Feature values are preferred to be categorical. 
       If the values are continuous then they are discretized prior to building the model.
    3. On the basis of attribute values records are distributed recursively.
    4. We use statistical methods for ordering attributes as root or the internal node.

****
In Decision Tree the major challenge is to identification of the attribute for the root node in each level. 
This process is known as attribute selection. 
    1. Information Gain
       Information gain is a measure of this change in entropy.
       
       ***** Entropy ****
       Entropy is the measure of uncertainty of a random variable, it characterizes the impurity of an arbitrary collection of examples.
       
        Building Decision tree using Information Gain:
           1. Start with all training instances associated with the root node
           2. Use info gain to choose which attribute to label each node with
            Note: No root-to-leaf path should contain the same discrete attribute twice
           3. Recursively construct each subtree on the subset of training instances that would be classified down that path in the tree.

        The border cases:
           1. If all positive or all negative training instances remain, label that node “yes” or “no” accordingly
           2. If no attributes remain, label with a majority vote of training instances left at that node
           3. If no instances remain, label with a majority vote of the parent’s training instances
     
    2. Ginnie Index
        1. Gini Index is a metric to measure how often a randomly chosen element would be incorrectly identified.
        2. It means an attribute with lower Gini index should be preferred.
        3. Sklearn supports “Gini” criteria for Gini Index and by default, it takes “gini” value. 
        
****
Advantages: 
    1. Decision trees are able to generate understandable rules.
    2. Decision trees perform classification without requiring much computation.
    3. Decision trees are able to handle both continuous and categorical variables.
    4. Decision trees provide a clear indication of which fields are most important for prediction or classification.
    
****
Disadvantages:
    1. Decision trees are less appropriate for estimation tasks where the goal is to predict the value of a continuous attribute.
    2. Decision trees are prone to errors in classification problems with many class and relatively small number of training examples.
    3. Decision tree can be computationally expensive to train. The process of growing a decision tree is computationally expensive. 
    At each node, each candidate splitting field must be sorted before its best split can be found. 
    In some algorithms, combinations of fields are used and a search must be made for optimal combining weights. 
    Pruning algorithms can also be expensive since many candidate sub-trees must be formed and compared.
'''