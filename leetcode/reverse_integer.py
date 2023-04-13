"""
Given a signed 32-bit integer x, return x with its digits reversed. If reversing x causes the value to go outside the signed 32-bit integer range [-231, 231 - 1], then return 0.
Assume the environment does not allow you to store 64-bit integers (signed or unsigned).

Example 1:
Input: x = 123
Output: 321

Example 2:
Input: x = -123
Output: -321

Example 3:
Input: x = 120
Output: 21
"""

class Solution:
    def reverse(self, x: int) -> int:
        if x<0:
            temp = str(-1*x)[::-1]
        else:
            temp = str(x)[::-1]
        if pow(-2,31)<=int(temp)<pow(2,31):
            if x<0:
                x = -1*x
                return int(str('-')+temp)
            else:
                return int(temp)
        else:
            return 0