def pivotedBinarySearch(arr, n, key):
    """
    ================================================================
    Input: 
    ================================================================
    arr : array containing a list of elements
    n : number of elements in array
    key : element to be serachable in arr
    ================================================================
    Output: returns the index at which element is present in arr
            and if element is not present then return -1
    =================================================================
    Time Complexity : O(n)     
    """
    pivot = findPivot(arr, 0, n-1)
    if pivot ==-1:
        return binarySearch (arr, 0, n-1, key)

def findPivot(arr, low, high):
    if high < low:
        return -1
    if high ==low:
        return low

    mid = int((low+high)/2)

    if mid < high and arr[mid]> arr[mid+1]:
        return mid
    if mid > low and arr[mid] < arr[mid-1]:
        return (mid-1)
    if arr[low] >= arr[mid]:
        return findPivot(arr, low, mid-1)
    return findPivot(arr,mid+1, high)


def binarySearch(arr, low, high, key):
    if high < low:
        return -1
    mid = int((low+high)/2)
    if key == arr[mid]:
        return mid
    if key > arr[mid]:
        return binarySearch(arr, (mid +1), high, key)
    return binarySearch(arr, (mid+1), high, key)

# Driver Code
arr = [3, 4, 5, 6, 7, 8, 1, 2, 3]  # array
n = len(arr)  # number of elements
key = 3 # find the element
result = pivotedBinarySearch(arr, n , key) # Result

print(f'Index of the element is {result}')