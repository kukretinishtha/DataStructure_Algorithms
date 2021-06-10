'''
K-Nearest Neighbors is the classification algorithms in Machine Learning.
It is a superivised learning algorithm.

Let m be the number of training data samples. Let p be an unknown point.
    1. Store the training samples in an array of data points arr[]. This means each element of this array represents a tuple (x, y).
    2. for i=0 to m:
            Calculate Euclidean distance d(arr[i], p).
    3. Make set S of K smallest distances obtained. Each of these distances corresponds to an already classified data point.
    4. Return the majority label among S.
    
It can be used to solve both classification and regression problems.
The KNN algorithm assumes that similar things exist in close proximity.
'''