"""
Fixed Point in an array is an index i such that arr[i] is equal to i. Given an array of n distinct integers (positive and negative) sorted in ascending order, write a function that returns a Fixed Point in the array, if there is any Fixed Point present in the array, else returns -1.
Input
arr[] = {-10, -5, 0, 3, 7}
arr_size = 5
Output
3
Input
 arr[] = {-10, -5, 3, 4, 7, 9}
arr_size = 6
Output
-1
"""

def fixedpoint(arr, n , ele):
    midpoint = 0
    if n%2 == 0:
        midpoint = n/2
    else:
        midpoint = (n+1/2)
        
    if ele == arr[int(midpoint)]:
        print(ele)
        
    elif ele < arr[int(midpoint)]:
        arr = [0,int(midpoint)]
        n = len(arr)
        return fixedpoint(arr, n, ele)
    else:
        arr = [int(midpoint),n]
        n = len(arr)
        return fixedpoint(arr,n ,ele)

arr = [-10, -5, 3, 4, 7, 9]
ele = 3
n = len(arr)
fixedpoint(arr, n , ele)