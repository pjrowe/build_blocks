"""Created on Mon Dec 21 16:33:40 2020.

@author: Trader
"""

# %% PROBLEM SOLVING BADGE
"""
Problem Solving (Data Structures) -
hackerrank_data_structures.py

 n  Category                     Done
--  --------                  -------
 6  Arrays                      6 /  6
15  Linked Lists               10 / 15 - mostly easy
17  Trees                       2 / 17 - one third easy
 3  Balanced Trees                /  3 - medium and hard
 9  Stacks                      2 /  9 - few easy
 5  Queues                        /  5 - more difficult
 4  Heap                          /  4 - 2 easy, 2 hard
 3  Multiple Choice             3 /  3
    --62                      -------
                                22 / 62 ; finish at least 30 of these

 4  Disjoint Set                 /  4 hard
 2  Trie                         /  2 hard
53  Advanced                     / 53 none are easy, 4 medium, rest hard
-----59                       ---------
                               15 /121
"""

# %% 6  Arrays                      4 / 6  only one difficult
# most too easy to record


# %% Arrays - Array Manipulation (Hard)
import os


def arrayManipulation(n, queries):
    arr = [0] * n
    for i, row in enumerate(queries):
        a, b, k = row[0], row[1], row[2]
        arr[a - 1] += k
        # a and b are 1 indexed values of arr,
        # so we subtract 1 from a to get actual index of arr
        if b < n:
        # if b == n, no need to decrement
            arr[b] -= k

    peak = 0
    level = 0
    for el in arr:
        level = level + el
        peak = max(peak, level)
    return peak


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    nm = input().split()
    n = int(nm[0])
    m = int(nm[1])
    queries = []

    for _ in range(m):
        queries.append(list(map(int, input().rstrip().split())))

    result = arrayManipulation(n, queries)
    fptr.write(str(result) + '\n')
    fptr.close()

# %% Arrays - Dynamic Array
# easy


def dynamicArray(n, queries):
    arr = [[] for i in range(n)]
    last_value = 0
    result = []

    for i, query in enumerate(queries):
        qtype, x, y = query
        if qtype == 1:
            idx = (x ^ last_value) % n
            arr[idx].append(y)
        elif qtype == 2:
            idx = (x ^ last_value) % n
            last_value = arr[idx][y % len(arr[idx])]
            result.append(last_value)

    return result


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    n = int(first_multiple_input[0])
    q = int(first_multiple_input[1])
    queries = []

    for _ in range(q):
        queries.append(list(map(int, input().rstrip().split())))

    result = dynamicArray(n, queries)
    fptr.write('\n'.join(map(str, result)))
    fptr.write('\n')
    fptr.close()

# %% 15  Linked Lists - Reverse a LL

def reverse(head):
    node = head
    prior = node.next
    node.next = None
    while prior:
        node = prior
        prior = node.next
        node.next =
    prior.next = node
    return prior


# %% 15  Linked Lists - Print in Reverse
# works

def reversePrint(head):
    output = []
    node = head
    while node:
        output.append(node.data)
        node = node.next
    while len(output) > 0:
        out = output.pop()
        print(out)


# %%  Linked Lists - Print the Elements of a Linked List

def printLinkedList(head):
    if head is not None:
        print(head.data)
        printLinkedList(head.next)


# %%  Linked Lists - Insert node at head of LL
# SinglyLinkedListNode:
#     int data
#     SinglyLinkedListNode next
# works

def insertNodeAtHead(llist, data):
    if llist is None:
        return SinglyLinkedListNode(data)
    else:
        a = SinglyLinkedListNode(data)
        a.next = llist
        return a

# %%  Linked Lists -Insert a Node at the Tail of a Linked List


def insertNodeAtTail(head, data):
    if head is None:
        return SinglyLinkedListNode(data)
    pnt = head
    while pnt.next:
        pnt = pnt.next
    pnt.next = SinglyLinkedListNode(data)
    return head


# %%  Linked Lists -Delete a Node
# works

def deleteNode(head, position):
    count = 0
    node = head
    while count < position:
        prior = node
        node = prior.next
        count += 1
    if count == 0 and head.next is None:
        head.data = None
        return head
    elif count == 0 and head.next is not None:
        head.data = None
        return head.next
    else:
        node.data = None
        prior.next = node.next
        return head

# %%  Linked Lists -removeDuplicates

def removeDuplicates(head):
    if head:
        node = head
        while node.next:
            if node.data == node.next.data:
                node.next = node.next.next
            else:
                node = node.next
    return head

# %% 17  Trees                       / 17

# %% 3  Balanced Trees               /  3
# %% 9  Stacks                       /  9

# %% 5  Queues                       / 5
# %% 4  Heap - QHEAP1
# after using heap, only 2 cases fail vs about half of cases before
# the deletion of arbitrary element in type 2 query makes a dictionary
# A better data structure
import heapq


q = int(input())
data = []
for i in range(q):
    query = list(map(int, input().split()))
    if query[0] == 1 and len(data) == 0:
        data.add(query[1])
    elif query[0] == 1 and len(data) > 0 and query[1] < data[0]:
        data = [query[1]] + data
    elif query[0] == 1 and len(data) > 0 and query[1] > data[0]:
        data.append(query[1])
    elif query[0] == 2:
        data = set(data)
        data.remove(query[1])
        heapq.heapify(data)
    elif query[0] == 3:
        print(data[0])


# %% 4  Heap - QHEAP1, try 2
# https://www.tutorialspoint.com/python_data_structure/python_heaps.htm

# =============================================================================
# A heap is created by using python’s inbuilt library named heapq.
# =============================================================================
# This library has the relevant functions to carry out various operations
#  on heap data structure. Below is a list of these functions.
# heapify - This function converts a regular list to a heap. In the
# resulting heap the smallest element gets pushed to the index position
# 0. But rest of the data elements are not necessarily sorted.
# heappush – This function adds an element to the heap without altering
#  the current heap.
# heappop - This function returns the smallest data element from the
# heap.
# heapreplace – This function replaces the smallest data element with a
#  new value supplied in the function.
# =============================================================================
import heapq


H = [21, 1, 45, 78, 3, 5]
# Use heapify to rearrange the elements
heapq.heapify(H)
print(H)

# %% 3  Heap - poorly written problem since a simple solution uses
# a set, not a heap; the arbitary element deletion in query 2 is the
# fatal part that makes a heap nonoptimal

q = int(input())
mymin = int(1e9)  # initialize as max of potential data
data = set([mymin])
for i in range(q):
    query = list(map(int, input().split()))
    if query[0] == 1:
        data.add(query[1])
        mymin = min(mymin, query[1])
    elif query[0] == 2 and query[1] == mymin:
        data.remove(query[1])
        mymin = min(data)
    elif query[0] == 2 and query[1] != mymin:
        data.remove(query[1])
    elif query[0] == 3:
        print(mymin)


# %% 3  Multiple Choice              / 3
# If we have a tree of n nodes, how many edges will it have?
# a. 1
# b. n*(n-1)
# c. n*(n-1)/2
# d. n-1  *** correct

# 2
# Which of the following data structures can handle updates and
# queries in log(n) time on an array?

# Linked list
# Stack (LIFO, a list is an example)
# Segment Tree  # ** correct
# Queue (FIFO)

# *** Segment Tree correct, see that all others are 0(1)
# insert/delete best case, O(n) worst case for insert/delete

# 3.Of the following data structures, which has a Last in First
# Out ordering (LIFO) ?
# QUEUE
# vector
# stack ** correct
# array list

# Answer is stack
# %% 4  Disjoint Set - componentsInGraph (medium)
# this one is giving me trouble -- this is new material so need to learn the
# basics of these new data structures
# works for 2 of 39 cases
# this runs through too many loops - should be more efficient way

from collections import defaultdict
import pprint
pp = pprint.PrettyPrinter(indent=4)


def componentsInGraph(gb):
    lookup = defaultdict(set)
    n = len(gb)  # max # of sets possible
    maxset = 0
    minset = 2 * n
    for i in range(n):
        # this will create all sets
        lookup[gb[i][0]].add(gb[i][0])
        lookup[gb[i][0]].add(gb[i][1])
    for i in range(n - 1):
        candelete = 0
        for j in range(i + 1, n):
            # if there is overlap, add missing members to each
            # node's list of connections
            if len(lookup[gb[i][0]].intersection(lookup[gb[j][0]])) > 0:
                lookup[gb[j][0]].update(lookup[gb[i][0]])
                candelete = 1
        print(i, lookup)
        if candelete == 1:
            print('we can delete ', gb[i][0], candelete)
            del lookup[gb[i][0]]
            print('\n---')
        # can get rid of node in dict if other node
        # was updated to include it
    print(lookup, 'should be zerod out?')
    for k, v in enumerate(lookup):
        maxset = max(maxset, len(lookup[v]))
        minset = min(minset, len(lookup[v]))
    pp.pprint(lookup)
    print(minset, maxset)
    return [minset, maxset]


# %% 4  Disjoint Set - componentsInGraph (medium)
# works for more cases, but still times out a lot
from collections import defaultdict
import pprint


pp = pprint.PrettyPrinter(indent=4)


def componentsInGraph(gb):
    lookup = defaultdict(set)
    n = len(gb)  # max # of sets possible
    maxset = 0
    minset = 2 * n  # this is upper limit of size of any set
    for i in range(n):
        # this will create all sets
        lookup[gb[i][0]].add(gb[i][0])
        lookup[gb[i][0]].add(gb[i][1])
    for i in range(1, n):
        candelete = 0
        for j in range(i + 1, n + 1):
            # don't do intersection operation unless the lookup exists
            # if there is overlap, add missing members to each
            # node's list of connections
            if len(lookup[i]) == 0:
                candelete = 1
            elif len(lookup[j]) == 0:
                pass
            elif len(lookup[j]) > 0 and len(lookup[i]) > 0:
                if len(lookup[i].intersection(lookup[j])) > 0:
                    lookup[j].update(lookup[i])
                    candelete = 1
        print(i, lookup)
        if candelete == 1:
            print('we can delete ', i, candelete)
            del lookup[i]
            print('\n---')
        if len(lookup[n]) == 0:
            del lookup[n]

        # can get rid of node in dict if other node
        # was updated to include it
    print(lookup, 'should be zerod out?')
    for k, v in enumerate(lookup):
        if len(lookup[v]) == 0:
            print(lookup[v])
            del lookup[v]
        maxset = max(maxset, len(lookup[v]))
        minset = min(minset, len(lookup[v]))
    pp.pprint(lookup)
    print(minset, maxset)
    return [minset, maxset]


# %% 2  Trie                         / 2
# %% 53  Advanced

# %% THE END