'''
A Hierarchical clustering method works via grouping data into a tree of clusters. 
Hierarchical clustering begins by treating every data points as a separate cluster. Then, it repeatedly executes the subsequent steps:
    1. Identify the 2 clusters which can be closest together, and
    2. Merge the 2 maximum comparable clusters. We need to continue these steps until all the clusters are merged together.
    
Dendrogram is a tree-like diagram that statistics the sequences of merges or splits) graphically represents this hierarchy and 
is an inverted tree that describes the order in which factors are merged (bottom-up view) or cluster are break up (top-down view).

The basic method to generate hierarchical clustering are:

****
1. Agglomerative:
    It is a bottom-up approach.
    1.1 Calculate the similarity of one cluster with all the other clusters (calculate proximity matrix)
    1.2 Consider every data point as a individual cluster
    1.3 Merge the clusters which are highly similar or close to each other.
    1.5 Recalculate the proximity matrix for each cluster
    1.6 Repeat Step 3 and 4 until only a single cluster remains.
    
    Time complexity of a naive agglomerative clustering is O(n3) 
    because we exhaustively scan the N x N matrix dist_mat for the lowest distance in each of N-1 iterations. 
    Using priority queue data structure we can reduce this complexity to O(n2logn). 
    By using some more optimizations it can be brought down to O(n2)
    
    Agglomerative clustering makes decisions by considering the local patterns or neighbor points.
    It does not take into account the global distribution of data.
    
****
2. Divisive
    It is a top-down approach.
    2.1 given a dataset (d1, d2, d3, ....dN) of size Nat the top we have all data in one cluster
        the cluster is split using a flat clustering method eg. K-Means etc.
    2.2 repeat
    2.3 choose the best cluster among all the clusters to split
        split that cluster by the flat clustering algorithm
        untill each data is in its own singleton cluster

    Divisive clustering given a fixed number of top levels, using an efficient flat algorithm 
    like K-Means, divisive algorithms are linear in the number of patterns and clusters.
    
    Divisive algorithm is more accurate becuase takes into consideration the global distribution 
    of data when making top-level partitioning decisions.
    
'''