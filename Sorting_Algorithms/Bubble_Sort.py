# Implementation of Bubble Sort in Python
"""
Returns a sorted array.
"""
def bubble_sort(arr):
    """
    ================================================================
    Input: 
    ================================================================
    arr : array containing a unsorted list of elements
    ================================================================
    Output: returns a sorted array
    =================================================================
    Time Complexity : O(n^2)
    """
    length = len(arr)
    for i in range(length):
        for j in range(length-i-1):
            # Compares with the next element in the array
            if arr[j] > arr[j + 1]:
                # Swap two numbers
                arr[j], arr[j+1] = arr[j+1], arr[j]   

    return arr   

# Driver Code
X = [12, 3, 8, 35]  
sorted_array = bubble_sort(X)
print(f'The sorted array of {X} is {sorted_array}')