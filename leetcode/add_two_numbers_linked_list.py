"""
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.
You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Example 1:
Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]
Explanation: 342 + 465 = 807.

Example 2:
Input: l1 = [0], l2 = [0]
Output: [0]

Example 3:
Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
Output: [8,9,9,9,0,0,0,1]
 

Constraints:
The number of nodes in each linked list is in the range [1, 100].
0 <= Node.val <= 9
It is guaranteed that the list represents a number that does not have leading zeros.
"""

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        # ******** FIRST APPROACH **************
        # l1_num, l2_num = '', ''
        # while l1:
        #     l1_num = l1_num+str(l1.val)
        #     l1 = l1.next
        # while l2:
        #     l2_num = l2_num+str(l2.val)
        #     l2 = l2.next
        # l1_num = eval(l1_num[::-1])
        # l2_num = eval(l2_num[::-1])
        # res = int(l1_num) + int(l2_num)
        # res = list(str(res)[::-1])
        # ********** FIRST APPROACH ENDED *******
        # ********** Second Approach ************
        # res = []
        # carry = 0
        # while(l1 or l2):
        #     if l1 and l2:
        #         temp = (l1.val+l2.val+carry)%10
        #         carry = int((l1.val+l2.val+carry)/10)
        #         res.append(temp)
        #         l1 = l1.next
        #         l2 = l2.next
        #     elif l1 and not l2:
        #         temp = (l1.val+carry)%10
        #         carry = int((l1.val+carry)/10)
        #         l1 = l1.next
        #         res.append(temp)
        #     elif not l1 and l2:
        #         temp = (l2.val+carry)%10
        #         carry = int((l2.val+carry)/10)
        #         l2 = l2.next
        #         res.append(temp)
        # if carry != 0:
        #     res.append(carry)
        # ****** Second approach ended *****************
        # start_node = None
        # prev_node = None
        # curr_node = None
        # for idx, value in enumerate(res):
        #     if idx == 0 and idx == len(res)-1:
        #         return ListNode(int(value))
        #     elif idx == 0 and idx < len(res)-1:
        #         start_node = ListNode(int(value))
        #         prev_node = start_node
        #     elif idx == len(res)-1:
        #         curr_node = ListNode(int(value))
        #         prev_node.next = curr_node
        #         return start_node
        #     else:
        #         curr_node = ListNode(int(value))
        #         prev_node.next = curr_node
        #         prev_node = curr_node
        # return start_node

        if not l1 or not l2:
            return ListNode()

        carry = 0
        res = 0
        start_node = None
        prev_node = None
        curr_node = None
        while (l1 or l2 or carry):
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0
            res = (val1+val2+carry)%10
            carry = int((val1+val2+carry)/10)
            curr_node = ListNode(res)
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
            curr_node = ListNode(int(res))
            if prev_node == None and start_node == None:
                start_node =  curr_node
                prev_node = start_node
            elif prev_node != None and start_node != None:
                prev_node.next = curr_node
                prev_node = curr_node
        if carry>0:
            prev_node.next = ListNode(int(carry))
        return start_node