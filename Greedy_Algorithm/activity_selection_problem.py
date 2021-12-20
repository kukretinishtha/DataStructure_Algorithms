# The problem has been taken from geeks for geeks

def MaxActivities(arr, n):
    selected = []
    Activity.sort(key=lambda x : x[1])
    i = 0
    selected.append(arr[i])
    for j in range(1, n):
        if arr[j][0] >= arr[i][1]:
            selected.append(arr[j])
            i = j
    return selected

# Driver code
Activity = [[5, 9], [1, 2], [3, 4], [0, 6],[5, 7], [8, 9]]
n = len(Activity)
selected = MaxActivities(Activity, n)
print("Following activities are selected :")
print(selected)

