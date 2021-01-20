# Implementation of Linear Search in Python
"""
If element is present the return the index.
Otherwise return -1
"""
def linear_search(arr, element):
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
    Time Complexity : O(n)     
    """
    for i in range(0, len(arr)):
        if arr[i] == element:
            return i
    return -1


# Driver Code

x = [4, 5, 8, 10, 6] # array
ele = 8 # element
result = linear_search(x,ele) # return result

if result != -1:    
    print(f'Element {ele} is present in {x} and is present at index {result}')
else:  
    print(f'Element {ele} is not present in {x}')