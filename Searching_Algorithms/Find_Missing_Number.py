def find_Missing_Number(arr, n):
    if n > 1:
        res = ((n+1)*(n+2)/2) - sum(arr)
        return res
    else:
        return -1

# Driver Code
arr = [1,2,3,5]   # Input Array
ele = len(arr)   # Number of elements

result = find_Missing_Number(arr, ele) # return result

if result != -1:    
    print(f'The missing element is {result}')
else:  
    print(f'Error! Array length is insuffient')