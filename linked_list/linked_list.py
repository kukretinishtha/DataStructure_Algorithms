
class Node:
	def __init__(self, data):
		self.data = data
		self.next = None


class LinkedList:
	def __init__(self):
		self.head = None
	
    def append(self, new_data):
		new_node = Node(new_data)
		if self.head is None:
			self.head = new_node
			return

		last = self.head
		while (last.next):
			last = last.next
		last.next = new_node
		
	def push(self, new_data):
		new_node = Node(new_data)
		new_node.next = self.head
		self.head = new_node


	def insertAfter(self, prev_node, new_data):
		if prev_node is None:
			print("The given previous node must inLinkedList.")
			return

		#	 Put in the data
		new_node = Node(new_data)
		new_node.next = prev_node.next
		prev_node.next = new_node




	def printList(self):
		temp = self.head
		while (temp):
			print(temp.data)
			temp = temp.next

if __name__=='__main__':
	
	llist = LinkedList()
	print(llist.printList())
	llist.append(6)
	llist.append(8)
	llist.append(9)
	print(llist.printList())
	# llist.push(7)
	# print(llist.printList())
	# llist.push(1)
	# print(llist.printList())
	# llist.append(4)
	# print(llist.printList())
	# llist.insertAfter(llist.head.next, 8)
	# print(llist.printList())
	print('Created linked list is:',llist)

