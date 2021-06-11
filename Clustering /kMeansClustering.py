'''
K-means is an unsupervised learning algorithm.
The algorithm will categorize the items into k groups of similarity. 
To calculate that similarity, we will use the euclidean distance as measurement.

***********************************************************************************************
The algorithm works as follows:
    1. First we initialize k points(in actual they are means), called means, randomly.
    2. We categorize each item to its closest mean and we update the mean’s coordinates, which are the averages of the items categorized in that mean so far.
    3. We repeat the process for a given number of iterations and at the end, we have our clusters.
    
************************************************************************************************
The other popularly used similarity measures are:-
    1. Cosine distance: It determines the cosine of the angle between the point vectors of the two points in the n dimensional space
        d = \frac{X.Y}{||X||*||Y||}\

    2. Manhattan distance: It computes the sum of the absolute differences between the co-ordinates of the two data points.
        d = \sum_{n} X{_{i}}-Y{_{i}}

    3. Minkowski distance: It is also known as the generalised distance metric. It can be used for both ordinal and quantitative variables
        d = (\sum _{n}|X_{i}-Y_{i}|^{\frac{1}{p}})^{p}

************************************************************************************************
Important Keypoints:
* The results produced by running the algorithm multiple times might differ. 
* K Means is found to work well when the shape of the clusters is hyper spherical (like circle in 2D, sphere in 3D).
* K Means clustering requires prior knowledge of K i.e. no. of clusters you want to divide your data into. 
* K Means clustering can handle big data well because of time complexity of O(n)

The Elbow Method is one of the most popular methods to determine this optimal value of k.


Visualizing the data alone cannot always give the right answer. Hence we demonstrate the following steps.
We now define the following:
1. Distortion: It is calculated as the average of the squared distances from the cluster centers of the respective clusters. Typically, the Euclidean distance metric is used.
2. Inertia: It is the sum of squared distances of samples to their closest cluster center.
'''