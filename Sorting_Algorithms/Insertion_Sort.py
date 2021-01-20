# Implementation of Insertion Sort in Python
"""
Returns a sorted array.
"""
def insertion_sort(arr):
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
    for i in range(1, len(arr)):
        ele = arr[i]
        j = i-1
        while j>=0 and arr[j] > ele:
            # Swap 
            arr[j+1] = arr[j]
            j -= 1
        # Swap
        arr[j+1] = ele
    return arr

# Driver Code
X = [12, 3, 8, 35]  
sorted_array = insertion_sort(X)
print(f'The sorted array is {sorted_array}')