# Implementation of Binary Search in Python
"""
If element is present the return the index from the sorted array.
Otherwise return -1
"""
def binary_search(arr, element):
    """
    ================================================================
    Input: 
    ================================================================
    arr : array containing a list of elements
    element : element to be serachable in arr
    ================================================================
    Output: returns the index at which element is present in arr
            and if element is not present then return -1
    =================================================================
    Time Complexity : O(log n)     
    """
    # Initializing the values
    left = 0
    right = len(arr)-1
    mid =0

    # Condition
    while left < right:
        # calculating the value of mid
        mid = (left + right)//2

        # Check if element is present in mid
        if arr[mid] < element:
            left = mid + 1

        # If element is greater than mid value ignore the left side of array
        elif arr[mid] > element:
            right = mid - 1

        # If element is smaller than mid value ignore the left side of array
        else:
            return mid
    
    # elemnt is not present then return -1
    return -1


# Driver Code
x = [4, 10, 16, 19]  # array
ele = 6     # element

result = binary_search(x, ele) # binary search
if result != -1:    # result
    print(f'Element {ele} is present in {x} and is present at index {result}')
else:  
    print(f'Element {ele} is not present in {x}')