# Implementation of Jump Search in Python
"""
If element is present the return the index.
Otherwise return -1
"""
import math

def jump_search(arr, element):
    """
    ================================================================
    Input: 
    ================================================================
    arr : array containing a list of elements
    element : element to be searchable in arr
    ================================================================
    Output: returns the index at which element is present in arr
            and if element is not present then return -1
    =================================================================
    Time Complexity : O(n)     
    """

    # Finding the length of arr
    length = len(arr)

    # Finding the step size
    step_size = math.sqrt(length)

    #Initializing the initial_step
    initial_step = 0

    while arr[int(min(step_size, length))] < element:
        initial_step = step_size
        step_size += math.sqrt(length)
        if initial_step >= length:
            return -1

    # Linear search for element in the step size defined block
    while arr[int(initial_step)] < element:
        initial_step += 1

        if initial_step == min(initial_step, length):
            return -1

    # Element is found then return the index
    if arr[int(step_size)] == element:
        return initial_step

    return -1

# Driver Code

x = [4, 5, 8, 10] # array
ele = 4 # element
result = jump_search(x,ele) # return result

if result != -1:    
    print(f'Element {ele} is present in {x} and is present at index {result}')
else:  
    print(f'Element {ele} is not present in {x}')