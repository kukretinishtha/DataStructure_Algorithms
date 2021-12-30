def minStartValue(nums):
        ans=0
        sum=0
        for ele in nums:
            if sum+ele<1:
                diff=(-1*(sum+ele))+1
                ans+=diff
                sum=1
            else:
                sum+=ele
        if ans==0:return 1
        return ans

# Driver Code
arr = [-4, 3, 2, 0, 1]
print(f'Minimum start value is {minStartValue(arr)}')