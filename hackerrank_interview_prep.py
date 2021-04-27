# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 16:33:40 2020

@author: Trader
"""

# %% INTERVIEW PREP
"""
- Tips and Guidelines (just hint videos)

69 Hackerrank Challenges broken into 15 subjects
- 4 Warmups                      4 / 4 x
- 5 Arrays                       5 / 5 x
- 5 Dictionaries and Hashmaps    5 / 5 x
- 5 Sorting                      4 / 5
- 5 String manipulation          5 / 5 x - 23

- 5 Greedy algorithms            4 / 5
- 7 Search                       4 / 7
- 4 Dynamic Programming          3 / 4
- 6 Stacks and Queues            5 / 6 - 15

- 5 Trees                        3 / 5 - 2     2 easy
- 5 Graphs                       0 / 5 - 5     0 easy
- 5 Linked Lists                 5 / 5 x
- 4 Recursion and Backtracking   4 / 4 x
- 4 Miscellaneous                3 / 4 - 1
===
- 69                            54 / 69 solved so far (78%)

"""

# %% INTERVIEW PREP - Warmups - Sock Merchant
from collections import Counter

n = int(input())
a = input().split()
y = Counter(a)

pairs = 0
for key, value in y.items():
    pairs += value // 2

print(pairs)


def sockMerchant(n, ar):
    y = Counter(ar)
    pairs = 0
    for key, value in y.items():
        pairs += value // 2
    return pairs


# %% INTERVIEW PREP - Warmups - Jumping on the Clouds
"""
test.

There is a new mobile game that starts with consecutively numbered
clouds. Some of the clouds are thunderheads and others are cumulus.
The player can jump on any cumulus cloud having a number that is equal
to the number of the
current cloud plus 1 or 2.
The player must avoid the thunderheads. Determine the minimum number of
jumps it will take to jump from the starting postion to the last cloud.
It is always possible to win the game.

For each game, you will get an array of clouds numbered
if they are safe or if they must be avoided.
"""


def jumpingOnClouds(c):
    jumps = 0
    position = 0
    n = len(c)
    for i in range(n):
        print(i, position)
        if position == n - 1:  # we've already reached end
            print('position ==')
            pass
        elif c[i] == 1:
            print('this is a 1, which we cannot rest on')
            pass
        elif i < position:  # we already moved beyond this cloud
            print('i < position')
            pass
        elif (((i + 2) <= (n - 1)) and (c[i + 2] == 0)):
            # we always take opportunity to jump two spaces forward if
            # it lands on a 0 instead of advancing just 1 index
            jumps += 1
            position = i + 2
        elif c[i + 1] == 0:
            # we only try this test case when we've
            # reached the second to last index, and because
            # game always end with a win, we could just pass
            # True as the condition instead of c[i+1]==0
            jumps += 1
            position = i + 1

    return jumps


if __name__ == '__main__':
    n = int(input())
    c = list(map(int, input().rstrip().split()))
    result = jumpingOnClouds(c)
    print(result)

# %% INTERVIEW PREP - Warmups - Repeated strings
""" takes string s and length n that an infinite stream of
characters generated from s

find how many times letere 'a' appears.

below function does the math on the main string * # times
it has to repeat,plus the count from remainder

this is preferable to searching through the full stream,
which could overflow memory
"""


def repeatedString(s, n):
    n_strings = n // len(s)
    partial = s[0: n % len(s)]
    count1 = 0
    count2 = 0
    for i in s:
        if i == 'a':
            count1 += 1
    for i in partial:
        if i == 'a':
            count2 += 1
    full_count = count1 * n_strings + count2
    print('Within %d characters of string "%s"' % (n, s))
    print('We find %d occurences of letter "a"' % (full_count))
    return full_count


s = input()
n = int(input())
print(repeatedString(s, n))
"""
Test inputs
-----------
a
100000

a;lksfjlkj
100

aba
10
"""
# %% INTERVIEW PREP - Arrays: Left Rotation
"""
write a function that takes spaced integers and n number of positions to left
shift the elements of that array
A: take the input into a list, then print the new string as
print(s[n:] + s{0:n])
shift left by 4
s = '01234'
print(s[4:] + s[0:4])

Out[7]: '40123'
"""

# %% INTERVIEW PREP - Arrays - 2D Array - DS
# !/bin/python3

import os


def hourglassSum(arr):
    themax = -1000000
    for row in range(4):
        for col in range(4):
            top = arr[row][col] + arr[row][col + 1] + arr[row][col + 2]
            middle = arr[row + 1][col + 1]
            bottom = arr[row + 2][col] + arr[row + 2][col + 1] + \
                arr[row + 2][col + 2]

            hour_sum = top + middle + bottom
            themax = max(hour_sum, themax)

    return themax


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    arr = []
    for _ in range(6):
        arr.append(list(map(int, input().rstrip().split())))

    result = hourglassSum(arr)
    print(result)
    fptr.write(str(result) + '\n')
    fptr.close()


"""
1 1 1 0 0 0
0 1 0 0 0 0
1 1 1 0 0 0
0 0 2 4 4 0
0 0 0 2 0 0
0 0 1 2 4 0

Answer: 19
"""
# =============================================================================
# %% INTERVIEW PREP - Arrays - Array Manipulation
# =============================================================================
"""
This exercise is rated 'Hard'
The arrayManipulation_brute() solution below works but exceeds
the time limit for 7 test cases.  The Peak detector solution is smarter
and passes all test cases.

For a large array (limited to 10e7 length), it takes too long to
update each cell for every query. That is potentially 10e7x2x10e5
write operations for the entire exercise.
The peak detector has a maximum of 4x10e5 write operations, since
the step up by k is written to the beginning index and the step down
by k is written to the end index, for a max of 2x10e5 queries.
Then one iteration through the 10e7-length array is done, summing the
steps up and down to find a max elevation (peak)
"""


def arrayManipulation_brute(n, queries):
    """
    For large n, this approach just takes too long.

    Parameters
    ----------
    n : number of items in initial array of 0s
        ex:
            n=3
            arr=[0, 0, 0]

    queries : q x 3 array where we add queries[q][2] to all elements
    inclusive from arr[queries[q][0]] to arr[queries[q][1]]

    max # queries will be 20,000

    Note: queries refers to arr with 1-indexing
    Returns
    -------
    max of arr after all operations in queries are performed.
    """
    arr = [0] * n

    for i, row in enumerate(queries):
        a, b, k = row[0], row[1], row[2]
        for j in range(a - 1, b):
            arr[j] = arr[j] + k
    print(f'array size {arr.__sizeof__()/1000000}')
    return max(arr)


# %% INTERVIEW PREP -  Arrays - Array Manipulation (continued)
# This is a peak detector, integrating the increases over the array
# This works by placing the increase in level at the starting point
# and decrementing the same k at the end ofthe interval;
# Then we sum the increases in levels, comparing to max found so far
# we iterate trhough queries 1x = max 200,000 x 3 = 600K elements
# then once through arr for max of 10e7 accesses

def arrayManipulation_peakdetector(n, queries):
    # list default is int64, unlimited integer
    # default numpy array is 32 bit integer
    arr = [0] * n
    max_k = 0

    for i, row in enumerate(queries):
        a, b, k = row[0], row[1], row[2]
        max_k = max(max_k, k)
        arr[a - 1] += k
        if b < n:
            # the array is one indexed, so if b==n, the interval included
            # the last element of arr; hence, no need to decrement because
            # there are no other elements in array fto consider
            arr[b] -= k
    maxval = 0
    cumsum = 0
    for i, el in enumerate(arr):
        cumsum += el
        maxval = max(maxval, cumsum)
    print(f'{maxval: <15,d}: Max value')
    arr_size = arr.__sizeof__() / 1000000
    print(f'{arr_size: <15.2f}: Array size(MB)')
    return maxval, arr_size


# %% INTERVIEW PREP -  Arrays - Array Manipulation (continued)
def arrayManipulation_peakdetector_numpyarr(n, queries):
    arr = np.array([0] * n_array, dtype='int64')
    # This is 80MB in size for test case 7
    # a list data type also is 80MB
    # We need 'int64' because dtype='int32' causes an overflow
    # for test case 7, and the maxval returned is 2,147,473,274
    # instead of the correct answer 2_497_169_732
    # The max for 32 bits is 2**31 =￼ ￼￼2_147_483_648

    max_k = 0

    for i, row in enumerate(queries):
        a, b, k = row[0], row[1], row[2]
        max_k = max(max_k, k)
        arr[a - 1] += k
        if b < n:
            # the array is one indexed, so if b==n, the interval
            # included the last element of arr;
            # this is why we decrement only if b<n as
            # there are no other elements in array to consider if b==n
            arr[b] -= k
    maxval = 0
    cumsum = 0
    for i, el in enumerate(arr):
        cumsum += el
        maxval = max(maxval, cumsum)

    print(f'{maxval: <15,d}: Max value')
    arr_size = arr.__sizeof__() / 1000000
    print(f'{arr_size: <15.2f}: Array size(MB)')
    return maxval, arr_size


# %% INTERVIEW PREP -  Arrays - Array Manipulation (continued)
def arrayManipulation_shortpeak(n, queries):
    """Explain more time and memory efficient approach (compression).

    We can reduce the time to calculate the answer by using a smaller
    array.  ie., the problem defines the max n as 10e7
    while the max # queries is 2x10e5. So, assuming all start and
    endpoints of intervals, a and b, are unique, we could have
    200,000 unique start points (2x10e5) and 200,000 endpoints.
    The array of 400,000 is a lot smaller than 10e7. We have basically
    compresssed the number of empty points that we need to iterate
    through to calcuate max level.
    The Timer function shows that we reduce time to complete from 5+
    seconds to 0.5 seconds.
    The preprocessing required is as follows:
    1. iterate through queries, collecting all starting and endpoints
        of intervals
    2. sort these indices and eliminate duplicates;
        the length of the resulting list is max 400,000 long
    3. initialize a new short_arr of length defined about (max 400K)
    4. create a dictionary that maps the index of the full n-long
        theoretical array to the index of the smaller array
    5. go through queries, incrementing the cell of short_arr that
        is mapped to n-long indexes a_s by k
    6. decrement short_arr by k in indexes corresponding to b_s
    7. iterate thru short_arr, finding the max elevation

    NOTE - this needlessly creates other objects. we do not need
    shorter array we can just preprocesses/sort the queries to calc a
    running total
    of the increments /decrements (see def arrayManip_noarr below)
    """
    a_s = []
    b_s = []
    k_s = []

    for i, row in enumerate(queries):
        a_s.append(row[0])
        b_s.append(row[1])
        k_s.append(row[2])

    # breakpoint()
    x = a_s + b_s
    all_indices = list(set(x))
    all_indices.sort()
    short_arr = [0] * len(all_indices)

    # mapping index of n-long array to index of shorter array
    index_lookup = {}
    for j, el in enumerate(all_indices):
        index_lookup[el] = j

    # breakpoint()
    for m in range(len(a_s)):
        short_arr[index_lookup[a_s[m]]] += k_s[m]
        short_arr[index_lookup[b_s[m]]] -= k_s[m]

    maxval = 0
    cumsum = 0
    for i, el in enumerate(short_arr):
        cumsum += el
        maxval = max(maxval, cumsum)

    print(f'{maxval: <15,d}: Max value')
    arr_size = short_arr.__sizeof__() / 1000000
    total = ((a_s.__sizeof__() / 1000000)
             + b_s.__sizeof__() / 1000000
             + k_s.__sizeof__() / 1000000
             + queries.__sizeof__() / 1000000
             + index_lookup.__sizeof__() / 1000000
             + short_arr.__sizeof__() / 1000000)
    print(f'{total: <15.2f}: All objects size(MB)')
    print(f'{arr_size: <15.2f}: Array size(MB)')
    return maxval, arr_size


# %% INTERVIEW PREP - Arrays - Array Manipulation (continued)
"""Can we optimize time and memory by using a generator?

there are two arrays:
    1. n-length array, max length 10e7
    2. queries array, max size 2x10e5 with 3 columns

    we do not need #1. We can simply have an array 200,000 elements
    long that stores all the increments and decrements for peak detection

    Is it possible to avoid creation/storage of this 200,000 x 8bytes
    or 1.6MB array. Hardly seems worth the trouble, considering it only
    takes 0.5 seconds to solve while using 1.6MB

    We can avoid creating an empty short_arr of max length 200,000
    by preprocessing ordering the queries so that we calculate the max
    as we go through the increment/decrement operations, avoiding having a
    short_arr in the firstplace.  This isn't really a generator though

    we should add memory of all items used by function to be pure
"""


def arrayManip_noarr(n, queries):
    q_s = []

    for i, row in enumerate(queries):
        # q = [(a_0, k_0), (b_0, -k_0)...]
        # we will sort q and then have an iterable of queries that is
        # max 400,000 long with 2x8 = 16 bytes per tuble
        q_s.append((row[0], row[2]))
        q_s.append((row[1], -row[2]))

    qs = sorted(q_s, key=lambda x: x[0])

    # breakpoint()
    maxval = 0
    cumsum = 0
    for i, el in enumerate(qs):
        cumsum += el[1]
        maxval = max(maxval, cumsum)

    print(f'{maxval: <15,d}: Max value')
    q_size = q_s.__sizeof__() / 1000000
    q_ord_size = qs.__sizeof__() / 1000000
    print(f'{q_size: <15.2f}: Unordered q_s array size(MB)\n'
          f'{q_ord_size: <15.2f}: Ordered qs array size(MB)')

    return maxval, q_ord_size


# %% INTERVIEW PREP -  Arrays - Array Manipulation (continued)
# Load Test case 7 data

import sys
from numpy import loadtxt
from timeit import Timer
import numpy as np

n_array = 10_000_000
m_queries = 100_000
max_output = 2_497_169_732  # THIS is the expected answer
arr_list = [0] * n_array
arr_numpy = np.array([0] * n_array, dtype='int64')
input_file = r'.\hackerrank_test_cases\interview_prep'
input_file = input_file + r'\array_manipulation_input7.txt'
data_in = loadtxt(input_file, delimiter=' ', dtype='int64')

print(f'{"arr_list": >10.10s} {"arr_numpy": >11.10s} {"data_in": >9.10s}')
print(f'{arr_list.__sizeof__()/1_000_000: >10.2f}'
      f'{arr_numpy.__sizeof__()/1_000_000: >12.2f}'
      f'{data_in.flatten().__sizeof__()/1_000_000: >10.2f} sizeof(MB)')
print(f'{sys.getsizeof(arr_list)/1_000_000: >10.2f}'
      f'{sys.getsizeof(arr_numpy)/1_000_000: >12.2f}'
      f'{sys.getsizeof(data_in.flatten())/1_000_000: >10.2f} getsizeof(MB)')
print(f'{len(arr_list)/1_000: >10,.0f}'
      f'{len(arr_numpy)/1_000: >12,.0f}'
      f'{len(data_in.flatten())/1_000: >10,.0f} # elements(000)')

# %% INTERVIEW PREP -  Arrays - Array Manipulation (continued)
# brute force times out for big arrays

test_input = data_in[0:10]
t_brute = Timer(lambda: arrayManipulation_brute(n_array, test_input))

# numpy doesn't help in brute force approach; too much memory and too big
# array to do a cell by cell increase; just 10 queries take longer
# than all 100K queries using the peakdetector approach
print(f'{round(t_brute.timeit(number=1), 1)} seconds: Time to complete '
      f'for {len(test_input)} queries using brute force')
# 9.9 seconds: Time to complete for 10 queries using brute force
# %% Arrays - Array Manipulation (continued)
# Timing on three peak detector functions on test case 7 data
t_peak = Timer(lambda: arrayManipulation_peakdetector(n_array, data_in))
t_peak_nparr = Timer(lambda: arrayManipulation_peakdetector_numpyarr(n_array,
                                                                     data_in))
t_short_arr = Timer(lambda: arrayManipulation_shortpeak(n_array, data_in))
t_no_arr = Timer(lambda: arrayManip_noarr(n_array, data_in))

maxval, arr_size1 = arrayManipulation_peakdetector(n_array, data_in)
maxval, arr_size2 = arrayManipulation_peakdetector_numpyarr(n_array, data_in)
maxval, arr_size3 = arrayManipulation_shortpeak(n_array, data_in)
maxval, arr_size4 = arrayManip_noarr(n_array, data_in)

time_peak = round(t_peak.timeit(number=1), 1)
time_peak_nparr = round(t_peak_nparr.timeit(number=1), 1)
time_peak_short_arr = round(t_short_arr.timeit(number=1), 1)
time_peak_noarr = round(t_no_arr.timeit(number=1), 1)

# %% INTERVIEW PREP -  Arrays - Array Manipulation (continued)

"""
Results

 Time(secs)     # queries  memory(MB)   function
       6.50     100,000        80.0     peak
       5.70     100,000        80.0     peak nparr
       0.60     100,000         1.6     peak short arr
       0.40     100,000         1.8     peak no arr

"""

print(f'{"Time(secs)": >11s} {"# queries": >13s}'
      f'{"memory(MB)": >12s} {" ": >1s} function')

print(f'{time_peak: >11.2f}'
      f'{len(data_in): >12,d}'
      f'{arr_size1: >12.1f}'
      f'{" ": >5s}'
      f'{"peak ": <15s}')

print(f'{time_peak_nparr: >11.2f}'
      f'{len(data_in): >12,d}'
      f'{arr_size2: >12.1f}'
      f'{" ": >5s}'
      f'{"peak nparr": <15s}')

print(f'{time_peak_short_arr: >11.2f}'
      f'{len(data_in): >12,d}'
      f'{arr_size3: >12.1f}'
      f'{" ": >5s}'
      f'{"peak short arr": <15s}')

print(f'{time_peak_noarr: >11.2f}'
      f'{len(data_in): >12,d}'
      f'{arr_size4: >12.1f}'
      f'{" ": >5s}'
      f'{"peak no arr": <15s}')


# %% INTERVIEW PREP - Arrays - New Year Chaos


def minimumBribes(q):
    # initialize the number of moves
    moves = 0
    #
    # decrease Q by 1 to make index-matching more intuitive
    # so that our values go from 0 to N-1, just like our
    # indices.  (Not necessary but makes it easier to
    # understand.)
    Q = [P - 1 for P in q]
    #
    # Loop through each person (P) in the queue (Q)
    for i, P in enumerate(Q):
        # i is the current position of P, while P is the
        # original position of P.
        #
        # First check if any P is more than two ahead of
        # its original position
        if P - i > 2:
            print("Too chaotic")
            return
        #
        # From here on out, we don't care if P has moved
        # forwards, it is better to count how many times
        # P has RECEIVED a bribe, by looking at who is
        # ahead of P.  P's original position is the value
        # of P.
        # Anyone who bribed P cannot get to higher than
        # one position in front if P's original position,
        # so we need to look from one position in front
        # of P's original position to one in front of P's
        # current position, and see how many of those
        # positions in Q contain a number large than P.
        # In other words we will look from P-1 to i-1,
        # which in Python is range(P-1,i-1+1), or simply
        # range(P-1,i).  To make sure we don't try an
        # index less than zero, replace P-1 with
        # max(P-1,0)
        for j in range(max(P - 1, 0), i):
            if Q[j] > P:
                moves += 1
    print(moves)


if __name__ == '__main__':
    t = int(input())

    for t_itr in range(t):
        n = int(input())
        q = list(map(int, input().rstrip().split()))
        minimumBribes(q)


# %% INTERVIEW PREP - Arrays - Minimum Swaps 2


def minimumSwaps(arr):
    min_swaps = 0
    correct_seq = [1 + i for i in range(len(arr))]
    for i in range(len(arr) - 1):
        val1 = arr[i]
        if val1 != correct_seq[i]:
            min_swaps += 1
            for j in range(i + 1, len(arr)):
                val2 = arr[j]
                if val2 == correct_seq[i]:
                    arr[i] = val2
                    arr[j] = val1
                    break
            print(arr, correct_seq)
    return min_swaps


print(minimumSwaps([2, 3, 4, 1, 8, 7, 6, 5]))

# %% INTERVIEW PREP - Arrays - Minimum Swaps 2

# but better to create a dictionary so we do not have to do a search each
# time to look for correct position to swap


def minimumSwaps(arr):
    ref_arr = [i + 1 for i in range(len(arr))]
#     ref_arr = sorted(arr)
    index_dict = {v: i for i, v in enumerate(arr)}
    swaps = 0
    for i, v in enumerate(arr):
        correct_value = ref_arr[i]
        if v != correct_value:
            to_swap_ix = index_dict[correct_value]
            print('\n')
            print(arr)
            print(to_swap_ix, ':', arr[to_swap_ix], i, ':', arr[i])
            arr[to_swap_ix], arr[i] = arr[i], arr[to_swap_ix]
            print(arr)
            print(to_swap_ix, ':', arr[to_swap_ix], i, ':', arr[i])
            print(index_dict)
            index_dict[v] = to_swap_ix
            index_dict[correct_value] = i
            print(index_dict)
            swaps += 1

            print(f'{"-" * 7} Swap {swaps:d} Complete {"-" * 7}')
    return swaps


print(minimumSwaps([2, 3, 4, 1, 8, 7, 6, 5]))

# %% INTERVIEW PREP - - 5 Dictionaries and Hashmaps 1/5
# this is a silly problem since it tests as little as a single character
# would need to form n-characters of various length if wanted to see if there
# was an intersection of substrings longer than a single character

s1 = 'abc'
s2 = 'dogs'

x = 'abcd'
y = 'axyzc'


def twoStrings(s1, s2):
    set1 = set(s1)
    set2 = set(s2)
    # we could pass a string instead of set2 into
    # .intersection() method

    # cannot test if =={} because it will not yield a True even if there is
    # no intersection, so need to test the length
    if len(set1.intersection(set2)) == 0:
        return 'NO'
    else:
        return 'YES'


print(twoStrings(s1, s2))  # NO
print(twoStrings(x, y))  # YES


# %% INTERVIEW PREP - Dictionaries and Hashmaps: Frequency Queries
"""
CODE BELOW WORKS

Test Case 10 and 11 were timing out until I made two changes:
    # only reduced count in deletion if count was >0
    (avoid creating extra entries into lookup table)
    # on 3 query, used set(integers.value()) as check


query is a tuple, with query[0] being the type of query,
and query[1] being the number to add/delete or the frequency
being queried
(0, 3) => add 0 to integers
(1, 2) => delete one instance of 2 from integers
(3, 20) => add 1 to outputs if there are any integers with 20
            instances/frequency in integers, otherwise, add 0
            to outputs

Another way to measure time:

    import time

start_time = time.time()
end_time = time.time()
duration = end_time - start_time

# As of 2020 using python 3.7 >=  Counter is faster that default dict
# https://stackoverflow.com/questions/27801945/surprising-results-with-python-timeit-counter-vs-defaultdict-vs-dict

"""
from collections import Counter


def freqQuery(queries):
    outputs = []
    integers = {}
    for i, query in enumerate(queries):
        if query[0] == 1:
            # add occurrence of integer to integers
            integers[query[1]] = integers.get(query[1], 0) + 1
        elif query[0] == 2:
            # delete occurrence of integer from integers
            # only delete if it's actually already in lookup table
            if integers.get(query[1], 0) > 0:
                integers[query[1]] = max(0, integers.get(query[1], 0) - 1)
        elif query[0] == 3:
            # find how many integers have the given frequency
            if query[1] in set(integers.values()):
                outputs.append(1)
            else:
                outputs.append(0)
#    print(f'integers: {integers}, type: {type(integers)}')
#    print(f'integers values: {integers.values()},'
#          f'type: {type(integers.values())}')
    print(f'{len(integers)} entries in integers lookup table')
    print(f'{len(set(integers.values()))} Unique frequencies')
    if len(set(integers.values())) < 10:
        print(f'{set(integers.values())} Unique frequencies')
    return outputs


queries = [(3, 4), (2, 1003), (1, 16), (3, 1)]
cnt = Counter(freqQuery(queries))
print(f'Queries results: Hits {cnt[1]}  Misses {cnt[0]}')

# %%

from numpy import loadtxt
from timeit import Timer

"""
# Here we see why cases 10 and 11 are failing; every single 3
# query needs to search through entire set because there are no hits
# there are a lot more unique integers in lookup
# Keeping track of frquencies at each update helps because lookup
# of a specific number is O(1) vs. the membership search in
# integer.values() which is O(n)

From .\build_blocks\freq_queries_analysis.xls
Test Case	    9	    10	    11
# outputs	33194	33317	33386
# hits	    5703	    8	    0
# adds	    33,455	33361	33300
# deletes	33,351	33322	33314
# query	    33,194	33317	33386
check	    100,000	100,000	100,000

Results for Case 9
----------------------
200 entries in integers lookup table
56 Unique frequencies
0.3 seconds: Time to complete

Results for Case 10
----------------------
1000 entries in integers lookup table
33 Unique frequencies
0.7 seconds: Time to complete

Results for Case 11
----------------------
33299 entries in integers lookup table
3 Unique frequencies
{0, 1, 2} Unique frequencies
8.6 seconds: Time to complete

"""

for n in [9, 10, 11]:
    input_file = 'freq_query_input' + str(n) + '.txt'
    output_file = 'freq_query_OUT' + str(n) + '.txt'
    data_in = loadtxt(input_file, delimiter=' ', dtype='int')
    data_out = loadtxt(output_file, delimiter=' ', dtype='int')
    print(f'Results for Case {n}\n {"-" * 10}')
    t = Timer(lambda: freqQuery(data_in))
    print(f'{round(t.timeit(number=1), 1)} seconds: Time to complete\n')


# %% INTERVIEW PREP - Dictionaries and Hashmaps: Frequency Queries


def freqQuery(queries):

    freq = Counter()
    # keys are integers added or decremented, values are frequency

    cnt = Counter()
    # keys are the frequency, values are the # integers with frequency
    arr = []
    for q in queries:
        if q[0] == 1:
            # there is now one less integer with frequency freq[q[1]]
            cnt[freq[q[1]]] -= 1

            # the frequency of q[1] must be increased
            freq[q[1]] += 1

            # the counter for that new freq must be increased as well
            cnt[freq[q[1]]] += 1

        elif q[0] == 2:
            # no need to do anything if the frequency of q[i] is 0
            if freq[q[1]] > 0:
                # the counter of integers having that freq is decreased
                cnt[freq[q[1]]] -= 1

                # then we update the frequency of that integer
                freq[q[1]] -= 1

                # then we increase count of integers with that new frequency
                cnt[freq[q[1]]] += 1
        else:
            # only output 1 if the # of integers with frequency q[1] is > 0
            if cnt[q[1]] > 0:
                arr.append(1)  # 1 signifies 'hit'
            else:
                arr.append(0)  # 0 signifies 'miss'
    return arr


for n in [9, 10, 11]:
    input_file = 'freq_query_input' + str(n) + '.txt'
    output_file = 'freq_query_OUT' + str(n) + '.txt'
    data_in = loadtxt(input_file, delimiter=' ', dtype='int')
    data_out = loadtxt(output_file, delimiter=' ', dtype='int')
    print(f'Results for Case {n}\n {"-" * 10}')
    Timer(lambda: freqQuery(data_in))
    print(f'{round(t.timeit(number=1), 1)} seconds: Time to complete\n')

"""
Below we see the results of keeping a frequency counter and
cnt counter both up to date on each query:  O(1) time for
each of the tests cases that were problematic for when we had to
create a set of the frequencies on each type 3 query

Results for Case 9
----------------------
0.2 seconds: Time to complete

Results for Case 10
----------------------
0.2 seconds: Time to complete

Results for Case 11
----------------------
0.2 seconds: Time to complete
"""

# %%
import pandas as pd
from timeit import Timer

data = pd.read_csv('input test5.txt', sep=' ')
out = pd.read_csv('output test5.txt')
q = data['query'].values.tolist()
num = data['num'].values.tolist()

queries = []
for i in range(250):
    queries.append((q[i], num[i]))

t = Timer(lambda: freqQuery(queries))
print('Time:', t.timeit(number=1_000) / 1000)


# %% INTERVIEW PREP - Dictionaries and Hashmaps : Count Triplets
import math

# this fails on 3 of 13 test cases, 2 were wrong, 1 was timing
# A better solution uses Counter, which preserves order
# of the original series


def countTriplets(arr, r):
    arr.sort()
    trips = 0
    mydict = {}
    for i, v in enumerate(arr):
        mydict[v] = mydict.get(v, []) + [i]

    if r == 1:
        for k, v in enumerate(mydict):
            n = len(mydict.get(v, []))
            k_denom = math.factorial(3) * (math.factorial(n - 3))
            trips = trips + int(math.factorial(n) / k_denom)

    else:
        for k, v in enumerate(mydict):
            trips = trips + len(mydict.get(v, [])) \
                * len(mydict.get(v * r, [])) * len(mydict.get(v * r * r, []))
            # print(k, v, trips)
            # print(mydict)
            # print('--')
    return trips


countTriplets(list(map(int, '1 3 9 9 27 81'.split())), 3)

print(countTriplets(list(map(int, '1 1 1 1 1 10 10 10 10 10'.split())), 1))

print(countTriplets(list(map(int, '1 1 1 1 1 10 10 10 10 10'.split())), 1))

math.factorial(100000) / math.factorial(3) / math.factorial(100000 - 3)
# %% I
count = 0
for i in range(1, 100000):
    count = count + i * (100000 - 1 - i)
print(count)

# this is the correct answer for a series with r=1, and 100,000 equal
# entries
# example of a series of 10 numbers, all the same, r=1

# 5 5 5 5 5 5 5 5 5 5
# 1: center point of triplet is i=1 or 2nd 5; there is 1 5 to left, 8
# to right
#     which equates to 1*8 triplets that can have 2nd 5 as middle
#     which is why we count the range 1,#_in_array
# 2nd 5 is middle:   1*8  = 8 additional
# 3rd 5 is middle:   + 2 5's to left * 7 5's to right = 14 additional
# 9th 5 is middle:   + 8 5's to left * 1 5 to right = +8
count = 0
for i in range(1, 10):
    count = count + i * (10 - 1 - i)
print(count)
120 == 8 + 2 * 7 + 3 * 6 + 4 * 5 + 5 * 4 + 18 + 14 + 8

# %% INTERVIEW PREP - Dictionaries and Hashmaps : Count Triplets
from collections import Counter

# THIS WORKS


def countTriplets(arr, r):
    a = Counter(arr)
    b = Counter()
    n_triples = 0
    for element in arr:
        j = element // r    # j = left most element of a triple
        k = element * r     # k is the right element of triple
        a[element] -= 1
        # we consider each element of arr once as being in
        # middle of a triple, so we reduce its freq in a so that we can
        # calculate the # of n_triples to add if next if loop below

        breakpoint()

        # if j is not an element in Counter b, then 'if b[j] evaluates
        # as False
        # not element%r means we only add to n_triples when element
        # is divisible by r
        print(f'Updated arr_counter:{a}  Counter for left element {b}')
        if b[j] and a[k] and not element % r:
            print(f'triples exist {j}, {element}, {k}')
            new = b[j] * a[k]
            n_triples += new
            print(f'adding {new} triples')
            breakpoint()
        b[element] += 1
        print('Updated n_triples:', n_triples)
        print(b)
        print('----')
        breakpoint()
    return n_triples


print(countTriplets(list(map(int, '1 1 1 1 1 10 10 10 10 10'.split())),
                    1))

# %% INTERVIEW PREP - Dictionaries and Hashmaps : Ransom Note
# m, n = list(map(int, input().split()))  # length magazine, length note
# magazine = input().split()  # magazines strings
# note = input().split()  # note strings

# there are NOT sufficient occurrences of exact word matches in magazine
# to write note
magazine = 'give me one grand dnarg today night'
note = 'give me grand grand'

# there are sufficient occurrences of exact word matches in mag2 to
# write note2
mag2 = 'give me one grand one one two'
note2 = 'give me one grand one'

from collections import Counter


def checkMagazine(magazine, note):
    if (Counter(note2.split()) - Counter(mag2.split())) == {}:
        print('Yes')
    else:
        print('No')


checkMagazine(magazine, note)


# %% INTERVIEW PREP - Dictionaries and Hashmaps : Sherlock and Anagrams
# given string s with max length 100, what is number of anagram pairs?
#
# we can use combinations of various lengths up to one less than len(s)
# and add them up

# 3 test cases timed out
from itertools import combinations
from collections import Counter


def sherlockAndAnagrams(s):
    count = 0
    for length in range(1, len(s)):
        subs = [s[i:i + length] for i in range(len(s) - length + 1)]
    #    print(subs)
        combos = list(combinations(subs, 2))
    #    print(combos)
        for comb in combos:
            if Counter(comb[0]) == Counter(comb[1]):
                count += 1
    print(count)
    return count


s = 'abba'
sherlockAndAnagrams(s)
s = 'kkkk'
sherlockAndAnagrams(s)
s = 'ifailuhkqq'
sherlockAndAnagrams(s)


# %% for pypy3
# 3 test cases fail from timeout under pypy3 as well using this code

n_cases = int(input())


for i in range(n_cases):
    s = input()
    # s = 'ifailuhkqq'
    count = 0
    for length in range(1, len(s)):
        subs = [s[i:i + length] for i in range(len(s) - length + 1)]
        print(f'substrings {subs}')
#        combos = list(combinations(subs, 2))
        combos2 = [x for x in combinations(subs, 2)
                   if Counter(x[0]) == Counter(x[1])]
#        print(combos)
        count += len(combos2)
# =============================================================================
#         for comb in combinations(subs, 2):
#             if Counter(comb[0]) == Counter(comb[1]):
#                 count += 1
# =============================================================================
    print(count)

# Where are cycles being spent? in the Counter test?  I imagine a
# majority of the combos should be rejected


# %%
# solution that works on all test cases
# this avoid creating all combinations and trying to check if anagram by
# using frozenset and counter to hash the substrings that are anagrams
from collections import Counter

s = 'abba'


def sherlockAndAnagrams(string):
    buckets = {}
    for i in range(len(string)):
        for j in range(1, len(string) - i + 1):
            key = frozenset(Counter(string[i:i + j]).items())
            # O(N) time key extract
            # frozenset allows Counter.items() to be hashable/immutable,
            # and therefore used as key to a dictionary
            # frozenset(Counter()) only hashes the keys in Counter,
            # not the values/counts as well; we need the counts to
            # define the anagram
            # but frozenset of two substrings that are anagrams
            # will be equivalent
            buckets[key] = buckets.get(key, 0) + 1
            breakpoint()
    count = 0
    for key in buckets:
        count += buckets[key] * (buckets[key] - 1) // 2
        # to count the number of combinations when the combos are
        # pairs, we take len(iterable) * len(iterable)-1/2
        # abc => 3*2/2 = 3 combos = ab, bc, ac
        # buckets will have keys that are substrings that only occur
        # 1x, so 1x0/2=0 addnothing to count
    # print(buckets)
    # print(count)
    return count


sherlockAndAnagrams(s)

"""
# buckets :
{   frozenset({('a', 1)}): 2,
    frozenset({('b', 1), ('a', 1)}): 2,
    frozenset({('b', 2), ('a', 1)}): 2,
    frozenset({('b', 2), ('a', 2)}): 1,
     # only appears once, so doesn't add to count since it is the full
     # string
    frozenset({('b', 1)}): 2,
    frozenset({('b', 2)}): 1}

import pprint
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(buckets)
"""
# %% INTERVIEW PREP - Sorting: Bubble Sort


def countSwaps(a):
    swaps = 0
    b = a.copy()

    keepsearching = True

    while keepsearching:
        this_round_swaps = 0
        for i in range(len(a)):
            if i == len(a) - 1 and this_round_swaps == 0:
                # print(i)
                keepsearching = False
            elif i == len(a) - 1 and this_round_swaps > 0:
                pass
            # print('keep searching...')
            elif b[i + 1] < b[i]:
                bigger = b[i]
                lower = b[i + 1]
                b[i] = lower
                b[i + 1] = bigger
                swaps += 1
                this_round_swaps += 1
    print('Array is sorted in %d swaps.' % swaps)
    print('First Element: %d' % b[0])
    print('Last Element: %d' % b[-1])
#     print('list', a)
#     print('sorted list', b)
#     print(this_round_swaps)

# %% INTERVIEW PREP - Sorting: Fraudulent Activity Notifications
# times out due to not allowing use of statistics package; they want a
# low level calcuation of median function for more time efficient calc
# when getting big data

# frequency table / dictionary solution works well, but it does so since
# we know prices only go up to 201, so we can arrive at the median from
# the bottom up by doing searching and then updating along the way by
# adding new prices of each new day and eliminating oldprices from the
# median window

# Test Case 5 has 200_000 elements in expenditure, and median window
# size of 40_001, so do a new median calculation using 40_001 elements
# is too time consuming; better to keep a running frequency count as
# explained above


"""def activityNotifications(expenditure, d):
    notifications = 0
    for i in range(len(expenditure)):
        if (d <= i):
            median_expense = median(expenditure[i-d:i])
            if ((median_expense * 2) <= expenditure[i]):
                notifications += 1
    return notifications
"""


def activityNotifications(expenditure, d):
    """Return how many times client notified of suspected fraud.

    Suspicious expenditures exceed 2x median of prior d days.

    expenditure: list of expenditures
    d: number of trailing days to check if current expenditure exceeds

    notify: total number of times client notified of a suspicious amount
    """
    freq = {}
    notify = 0

    def find(idx):
        total_count = 0
        for i in range(201):
            # we know that expenditures must be 0-200 inclusive, so we
            # run through all possibilities
            if i in freq:
                total_count = total_count + freq[i]
            if total_count >= idx:
                # when total_count = or exceeds the idx, i is the
                # expenditure
                return i

    for i, exp in enumerate(expenditure[:-1]):
        # don't need to calc median of last d expenditures, so we
        # only need to iterate up to second to last element
        if exp in freq:
            freq[exp] += 1
        else:
            freq[exp] = 1
        # print(f"i: {i},val: {expenditure[i]}, freq: {freq}")
        if i >= d - 1:
            # only calc median once we have d elements in window, or
            # i >= d-1
            if d % 2 == 0:
                median = (find(d // 2) + find(d // 2 + 1)) / 2
            else:
                median = find(d / 2)
                # ex: d=3, find(1.5) will return i as median when
                # total count = 2, which is middle element of 3 element
                # array; it works because we count up from 0 toward max
                # expenditure of 200
            # print("median: ",median)
            if expenditure[i + 1] >= (median * 2):
                notify += 1
            #     print("notify: ", notify)
            # remove the previous element from dictionary
            freq[expenditure[i - d + 1]] -= 1
    return notify


if __name__ == '__main__':
    #    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    nd = input().split()
    n = int(nd[0])
    d = int(nd[1])
    expenditure = list(map(int, input().rstrip().split()))
    result = activityNotifications(expenditure, d)
    print(result)

#        fptr.write(str(result) + '\n')
#        fptr.close()

# %% INTERVIEW PREP - Sorting: Comparator
# The players are first sorted descending by score, then ascending by name.
# ascending name means A before Z
from functools import cmp_to_key


class Player:
    def __init__(self, name, score):
        self.name = name
        self.score = score

    def comparator(a, b):
        if a.score > b.score:
            return -1
        elif a.score < b.score:
            return 1
        elif a.name > b.name:
            # 'alphabetical order means 'zzz' is > 'aaa'' so we put a 1 to
            # enforce it....numerically larger #s, however, need -1 to pull
            # larger # earlier in sorted list

            return 1
        return -1


n = int(input())
data = []
for i in range(n):
    name, score = input().split()
    score = int(score)
    player = Player(name, score)
    data.append(player)

data = sorted(data, key=cmp_to_key(Player.comparator))
for i in data:
    print(i.name, i.score)

# %% INTERVIEW PREP - Sorting: Mark and Toys
import os


def maximumToys(prices, k):
    prices.sort()
    allsum = 0
    maxtoys = 0
    for i, price in enumerate(prices):
        # add or 'buy' each new toy if still have $
        if allsum + price <= k:
            allsum = allsum + price
            maxtoys = i + 1
    return maxtoys


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    nk = input().split()
    n = int(nk[0])
    k = int(nk[1])
    prices = list(map(int, input().rstrip().split()))
    result = maximumToys(prices, k)
    fptr.write(str(result) + '\n')
    fptr.close()

# %% INTERVIEW PREP - String Manipulation: Making Anagrams
"""Strings: Making Anagrams.
solved 10/31/2020

Alice is taking a cryptography class and finding anagrams to be very
useful.
We consider two strings to be anagrams of each other if the first
string's
letters can be rearranged to form the second string. In other words, both
strings must contain the same exact letters in the same exact frequency
For example, bacdc and dcbac are anagrams, but bacdc and dcbad are not.

Alice decides on an encryption scheme involving two large strings where
encryption is dependent on the minimum number of character deletions
required to make the two strings anagrams. Can you help her find this
number?

Given two strings, and , that may or may not be of the same length, determine
the minimum number of character deletions required to make a and b anagrams.
Any characters can be deleted from either of the strings.

For example, if and , we can delete from string and from string so that both
remaining strings are and which are anagrams.

Function Description
Complete the makeAnagram function in the editor below. It must return an
integer representing the minimum total characters that must be deleted to make
the strings anagrams.

makeAnagram has the following parameter(s):
    a: a string
    b: a string
Input Format
The first line contains a single string,
The second line contains a single string,

Constraints
The strings and
    consist of lowercase English alphabetic letters ascii[a-z].

Output Format
Print a single integer denoting the number of characters you must delete to
make the two strings anagrams of each other.

Sample Input
cde
abc

Sample Output
4

Explanation
We delete the following characters from our two strings to turn them
into anagrams of each other:
    Remove d and e from cde to get c.
    Remove a and b from abc to get c.
We must delete characters to make both strings anagrams, so we print
on a new line.
"""


def makeAnagram(string_a, string_b):
    """Print integer of # of letters you must delete to make anagrams.

    # first process both strings into count of each letter of alphabet
    # numbersa is 26 long list of numbers, each # representing count of
    # a-z in string a
    # ord('a') = 97, ord('z') = 122
    """
    numbersa = [0] * 26
    numbersb = [0] * 26
    for letter in string_a:
        numbersa[ord(letter) - 97] += 1
    for letter in string_b:
        numbersb[ord(letter) - 97] += 1

    count = 0

    for i in range(26):
        count += abs(numbersa[i] - numbersb[i])
    return count


if __name__ == '__main__':
    # fptr = open(os.environ['OUTPUT_PATH'], 'w')
    a = input()
    b = input()
    res = makeAnagram(a, b)
    print(str(res) + '\n')
    # fptr.write(str(res) + '\n')cd
    # fptr.close()

# %% INTERVIEW PREP - String Manipulation: Alternating Characters
"""Strings: Alternating Characters.
solved 10/31/2020

You are given a string containing characters A and B only. Your task is
to change it into a string such that there are no matching adjacent
characters.
To do this, you are allowed to delete zero or more characters in the
string.

Your task is to find the minimum number of required deletions.

For example, given the string s=AABAAB, remove an A at positions 0 and
3 to make ABAB in deletions.

Function Description
Complete the alternatingCharacters function in the editor below.
It must return an integer representing the minimum number of deletions
to make the alternating string.

alternatingCharacters has the following parameter(s):
    s: a string
Input Format
The first line contains an integer q, the number of queries.
The next Q lines each contain a string s.

Constraints
Each string will consist only of characters A and B

Output Format
For each query, print the minimum number of deletions required on a new
line.

Sample Input
5
AAAA
BBBBB
ABABABAB
BABABA
AAABBB

Sample Output
3
4
0
0
4

Explanation
The characters marked red are the ones that can be deleted so that the
string doesn't have matching consecutive characters.
import math
import random
import re
import sys
"""
import os


# Complete the alternatingCharacters function below.
def alternatingCharacters(s):
    # MY CODE BELOW
    prior = s[0]
    deletions = 0
    for letter in s[1:]:
        if letter == prior:
            deletions += 1
        else:
            prior = letter
    return deletions


if __name__ == '__main__':
    fptr = open('output2.txt', 'w')
    q = int(input())
    for q_itr in range(q):
        s = input()
        result = alternatingCharacters(s)
        fptr.write(str(result) + '\n')
    fptr.close()


# %% INTERVIEW PREP - String Manipulation: Sherlock and the Valid String

# Complete the isValid function below.
def isValid(input_str):
    letters = {}
    for let in input_str:
        if let in letters:
            letters[let] += 1
        else:
            letters[let] = 1
    inv_map = {}
    for k, v in letters.items():
        inv_map[v] = inv_map.get(v, []) + [k]
    print(inv_map)
    values_ = list(inv_map.values())
    keys_ = list(inv_map.keys())

    if len(keys_) == 1:
        return 'YES'
    elif len(keys_) == 2:
        if ((keys_[0] == 1) and len(values_[0]) == 1)\
           or (keys_[1] == 1) and len(values_[1]) == 1:
            return 'YES'
        elif keys_[1] - keys_[0] == 1 and (len(values_[1]) == 1):
            return 'YES'
        elif keys_[0] - keys_[1] == 1 and (len(values_[0]) == 1):
            return 'YES'
        else:
            return 'NO'
    else:
        return 'NO'


if __name__ == '__main__':
    #    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    s = input()
    result = isValid(s)
    print(result)
    #    fptr.write(result + '\n')
    #    fptr.close()


# %% INTERVIEW PREP - String Manipulation: Special String Again
# THESE TRIES DID NOT WORK; I needed to refer to the discussions section
"""
# this works on some cases but has runtime errors
x = input()

def substrCount(n, s):
    y = len(s)

    count = 1
    max_count = 0

    for i in range(1, y):
        if s[i] == s[i-1]:
            count += 1
            max_count = max(max_count, count)
        else:
            count = 1
    max_window = 2*max_count + 1

    result = 0
    for window_size in range(1, len(s)+1):
        print('ws=', window_size)
        for j in range(len(s) - window_size+1):
            print('\nj', j, 'ws', window_size)
            print(s[j:j+window_size])
            print('len', len(set(s[j:j+window_size]))==2)
            print('ws odd?', window_size % 2 == 1)
            print('symmetric', s[j:j+window_size//2]==s[j+window_size//2 + 1:j+window_size])
            if len(set(s[j:j+window_size]))==1:
                print(s[j:j+window_size])
                print(window_size, j, j + window_size)
                result += 1
                print('res', result)
            elif (len(set(s[j:j+window_size]))==2) and (window_size % 2 == 1) and (s[j:j+window_size//2]==s[j+window_size//2+1:j+window_size]):
                print(s[j:j+window_size])
                print(window_size, j, j + window_size)
                result += 1
                print('res', result)


    return result

print(substrCount(7, x))
# %% INTERVIEW PREP - String Manipulation: Special String Again

# remove prints; also overshoots time

def substrCount(n, s):
    y = len(s)

    count = 1
    max_count = 0

    for i in range(1, y):
        if s[i] == s[i-1]:
            count += 1
            max_count = max(max_count, count)
        else:
            count = 1
    max_window = 2*max_count + 1
    result = 0
    for window_size in range(1, len(s)+1):
        for j in range(len(s) - window_size+1):
            if len(set(s[j:j+window_size]))==1:
                result += 1
            elif (len(set(s[j:j+window_size]))==2) and (window_size % 2 == 1) and (s[j:j+window_size//2]==s[j+window_size//2+1:j+window_size]):
                result += 1

    return result
# %% INTERVIEW PREP - String Manipulation: Special String Again

# remove prints; also overshoots time

from collections import Counter

x = input()

def substrCount(n, s):
    y = len(s)

    count = 1
    max_count = 1

    for i in range(1, y):
        if s[i] == s[i-1]:
            count += 1
            max_count = max(max_count, count)
            print(count, max_count)
        else:
            count = 1
            print(count, max_count)
    max_window = 2*max_count + 1
    print('maxw', max_window)
    result = 0

    for window_size in range(1, max_window+2):
        for j in range(len(s) - window_size + 1):
            if s[j:j+window_size] == s[j]*window_size:
                print(window_size, 'ws')
                result += 1
            elif (window_size % 2 == 1) and (s[j:j+window_size//2]==s[j]*int((window_size//2))) and (s[j+window_size//2+1]==s[j]*int((window_size//2))):
                print(window_size, 'ws')
                result += 1
    print(result)
    return result

substrCount(len(x), x)
"""


# %% INTERVIEW PREP - String Manipulation: Special String Again
# solution found online uses formula k*(k+1)/2 to find
# # valid strings that are all same from substring of all same
# character
# aaa = 3*4/2 = 6 {a,a,a,a,aa, aa, aaa}
# I didn't know of this formula
# provided solution for finding symmetric valid strings was
# weird to me

"""
THIS WORKS
"""
from itertools import groupby

x = "aaaabbbabccdd"
y = "bcbcbaacacbccacbbbcbbbaaacccaaabbcaacbbbcbaaabbcbcbbabbbccbccacb"
y = y + "bababcaccbbbabccccccbacacbbcbbabcbccacbaaccccbcaaaabccccaba"


def substrCount(n, x):
    case1 = 0
    groups = []
    for k, g in groupby(x):
        groups.append(((k, len(list(g)))))
    for g in groups:
        case1 += g[1] * (g[1] + 1) / 2

# Case2 explanation:
# Counting valid groups which have a single middle character different from
# surrounding characters that are all the same: eg, aba, bab, bbabb, aabaa, ..
#
# We only need to scan the range of the string where there is room for a
# single middle character:  the earliest this can happen is when i=1
# (second character in string)
# and the furthest this can happen is one before end of string, or
# len(groups) - 1; then we just add the length of the smaller group on either
# side of the middle character
# Example 1:
#  string = 'aaaabbacccccdccchxiallljii...'
# groups     a,4 b,2 a,1 c,5 d,1 c,3 ...
#  case1=    10 + 3 + 1 + 15 + 1 + 6 + ....
#  case2=         a,1  c,3 d,1 c,3
#                  ^       ^
#  the single d character with same character 'c' on either side identifies as
#  valid characters... Note, the single 'a' has a b and c on either side, so
# is ignored.  But since the group of 'c's after 'd,1' has only 3, we count
# those 3 in case2: specifically, there is 'cdc', 'ccdcc', 'cccdccc'
# 'ccccdccch' is not valid, on the other hand, because of the 'h'
    case2 = 0
    for i in range(1, len(groups) - 1):
        if groups[i][1] == 1 and groups[i - 1][0] == groups[i + 1][0]:
            case2 += min(groups[i - 1][1], groups[i + 1][1])
    return int(case1 + case2)


print(substrCount(len(x), x))

z = input()
print(substrCount(len(z), z))


# %% INTERVIEW PREP - String Manipulation: Common Child
# try groupby
# A string is said to be a child of a another string if it can be formed
# by deleting 0 or more characters from the other string. Given two
# strings of equal length, what's the longest string that can be
# constructed such that it is a child of both?

# Length of strings is 5000 or less so running through string a few times
# should not be problem with time

# my approach won't work; this is a common algo problem for which there
# are canned approach come back to this later
from itertools import groupby


a = input()
b = input()
x = 'abcdea'
y = 'bdaff'


def commonChild(s1, s2):
    gb1 = groupby(s1)
    gb2 = groupby(s2)
    s1_group = []
    s2_group = []
    s1_letters = []
    s2_letters = []
    for k, g in gb1:
        s1_group.append(((k, len(list(g)))))
        s1_letters.append(k)
    for k, g in gb2:
        s2_group.append(((k, len(list(g)))))
        s2_letters.append(k)

    set1 = set(s1_letters)
    set2 = set(s2_letters)
    common_letters = set1.intersection(set2)
    for tup in s1_letters:
        if tup[0] not in common_letters:
            s1_letters.remove(tup)
    for tup in s2_letters:
        if tup[0] not in common_letters:
            s2_letters.remove(tup)
    # now s1_letters and s2_letters have common letters


commonChild(a, b)

# %% INTERVIEW PREP - Greedy Algorithms: Minimum Absolute Difference in an
# Array

# this solution works for small arrays in test cases
# but not for 10,000 or 100,000 long arrays
# hence the need for a greedy algorithm to avoid the need
# to create a 100,000x99,999 long list of combinations

from itertools import combinations


def minimumAbsoluteDifference(arr):
    # create combinations tuples
    # calc abs diff for each tuple
    # print min diff
    combos = list(combinations(arr, 2))

    min_diff = min([abs(combo[0] - combo[1]) for combo in combos])
    return min_diff


# %% INTERVIEW PREP - Greedy Algorithms: Minimum Absolute Difference in an
# Array
# sort first
# test first pair
# go through array, testing consecutive diff and comparing to first diff


def minimumAbsoluteDifference(arr):
    arr.sort()
    min_diff = abs(arr[0] - arr[1])
    for i in range(1, len(arr) - 1):
        min_diff = min(abs(arr[i] - arr[i + 1]), min_diff)
    return min_diff

# %% INTERVIEW PREP - Greedy Algorithms: Luck Balance

    # best to sort the important prelims (T=1) by increasing
    # increasing importance (L=1, 2, etc)
    # then just sum up first k of those luck as negatives
    # to add to rest of lucks
    # eg:
    #  n = 6, k = 3
    # contest [luck, 0 or 1]  0 = not important (can lose), 1 important
    #  contests = [[5, 1], [2, 1], [1, 1], [8, 1], [10, 0], [5, 0]]


def luckBalance(k, contests):
    lookup = {'important': [], 'notimportant': []}

    for i in range(len(contests)):
        if contests[i][1] == 1:
            lookup['important'].append(contests[i][0])
        else:
            lookup['notimportant'].append(contests[i][0])

    num_imp = len(lookup['important'])

    # correcting for edge case in which k allows us to lose
    # all important prelim contests
    if k >= num_imp:
        k = num_imp

    lookup['important'].sort()
    print('num_imp:', num_imp, 'k:', k)
    maxluck = -sum(lookup['important'][0:num_imp - k]) \
        + sum(lookup['important'][num_imp - k:]) \
        + sum(lookup['notimportant'])
    return int(maxluck)


"""debugging helpers

    print(-sum(lookup['important'][0:num_imp-k]))
    print(sum(lookup['important'][num_imp-k:]))
    print(sum(lookup['notimportant']))
    print(lookup)
"""

k = 3
contests = [[5, 1], [2, 1], [1, 1], [8, 1], [10, 0], [5, 0]]
print(luckBalance(k, contests))


# %% INTERVIEW PREP - Greedy Algorithms: Max Min
def maxMin(k, arr):
    arr.sort()
    unfairness = arr[k - 1] - arr[0]
    for i in range(k, len(arr)):
        unfairness = min(arr[i] - arr[i - k + 1], unfairness)
    return unfairness

# %% INTERVIEW PREP - Greedy Algorithms: Greedy Florist


def getMinimumCost(k, c):
    c.sort(reverse=True)
    n = len(c)  # how many flowers
    # k = # friends
    cost = 0
    for i in range(n):
        cost += (i // k + 1) * c[i]
    return cost, c


# to minimize cost of group of customers buying all of flowers,
# do a reverse sort so they don't pay any increase on the k mostexpensive
# flowers

# 0 1 2 3 4 5 6 7 = index
# 9 8 7 5 5 3 2 1 = original prices in revrese roder
# 9 8 7 10 10 6 6 3 = prices paid by group of 3
# sum of prior line is 59

c = [9, 8, 7, 5, 5, 3, 2, 1]
k = 3
print(getMinimumCost(k, c))

# %% ##########################################################################
# INTERVIEW PREP - Search: Hash Tables: Ice Cream Parlor
###############################################################################
# Search is best done using bounds and triangulating
# We can go through the prices array twice, first to create the dictionary
# and second to check if complement to current element is in dictionary
#
# Order 2n of time required to calculate


def whatFlavors(cost_of_flavors, money):
    cost_lookup = {}
    for i_of_flavor, cost in enumerate(cost_of_flavors):
        cost_lookup[cost] = cost_lookup.get(cost, []) + [i_of_flavor + 1]
        # cost_lookup keys are the prices, the values are the id# of
        # the flavor, which is the 1-based index of cost input
        # ex: if cost = [1, 5], then cost_lookup is{5: 2, 1: 1} where
        # flavor 2 has a cost of 5 and flavor 1 has a cost of 1
    # print(cost_lookup)
    for i_of_flavor, cost in enumerate(cost_of_flavors):
        # if cost and its complement are both in dict, choose them
        # if money == 2*cost[i], then cost[i] must appear twice in
        # cost_lookup
        # EX: money = 10, and cost= [1, 5, 6, 5] then
        # cost_lookup[5] = [2, 4], so flavors 2 and 4 each have cost 5
        # There is only solution
        cost_of_flavor2 = money - cost
        # print(cost_of_flavor2)
        if cost_of_flavor2 not in cost_lookup:
            pass
        elif cost_of_flavor2 in cost_lookup\
            and cost_of_flavor2 == cost\
                and len(cost_lookup[cost_of_flavor2]) == 1:
            pass
            # we pass BECAUSE we only have one flavor with the cost,
            # i.e., cost_lookup[cost] =[cost], but we need two flavors
            # to add up to the money
        else:
            # only remaining cases are when cost_of_flavor2 is in
            # cost_lookup
            # and when there is another flavor whose cost adds up to money,
            # because problem states that all test cases have a
            # unique solution
            result = [cost_lookup[cost][0], cost_lookup[cost_of_flavor2][-1]]
            # print('result', result)
            result.sort()
            print(' '.join(list(map(str, result))))
            return result


cost = [2, 2, 4, 3, 8]
flavors = whatFlavors(cost, 10)

"""condensed"""


def whatFlavors(cost_of_flavors, money):
    cost_lookup = {}
    for i_of_flavor, cost in enumerate(cost_of_flavors):
        cost_lookup[cost] = cost_lookup.get(cost, []) + [i_of_flavor + 1]
    for i_of_flavor, cost in enumerate(cost_of_flavors):
        cost_of_flavor2 = money - cost
        if cost_of_flavor2 not in cost_lookup:
            pass
        elif cost_of_flavor2 in cost_lookup\
            and cost_of_flavor2 == cost\
                and len(cost_lookup[cost_of_flavor2]) == 1:
            pass
        else:
            result = [cost_lookup[cost][0], cost_lookup[cost_of_flavor2][-1]]
            result.sort()
            print(' '.join(list(map(str, result))))
            return result


# %% INTERVIEW PREP - Search: Minimum Time Required - 1 of 5 inccorrect

# This works but times out for several cases
# try using a dicitonary instead of adding up production each
# time
def minTime(machines, goal):
    mindays = 0
    production = 0

    while production < goal:
        mindays += 1
#        print('days', mindays)
        production = 0
        for m in machines:
            production += mindays // m
#        print('production=', production)
#        print(machines)
#        print('---')
    return mindays


print(minTime([4, 5, 6], 12))

# %% INTERVIEW PREP - Search:Minimum Time Required - 2 of 5 inccorrect
# 7 timeouts


def minTime(machines, goal):
    production = 0
    mindays = 0

    while production < goal:
        mindays += 1
        production = sum([mindays // m for m in machines])
        print('days', mindays)
        print('production=', production)
        print(machines)
        print('---')
    return mindays
# same timout issues


print(minTime([2, 3], 5))

# %% INTERVIEW PREP - Search:Minimum Time Required  - 3 of 5 inccorrect
from collections import Counter
# 4 timeouts : 1000, 10,000, 50,000 and 100,000 machines with
# a big total production needed


def minTime(machines, goal):
    production = 0
    mindays = 0
    mach = Counter(machines)

    while production < goal:
        mindays += 1
        production = sum([(mindays // item[1]) * mach[item[1]]
                          for item in enumerate(mach)])
    return mindays


# same timout issues

print(minTime([2, 3], 5))
print(minTime([4, 5, 6], 12))
# %% INTERVIEW PREP - Search: Minimum Time Required  - 4 of 5 inccorrect
# 5, 7, 8 and 9 test cases still didn't pass
import math
from collections import Counter


def minTime(machines, goal):
    max_days = goal * min(machines)
    # we don't need to look at machines that take longer
    # than max_days to produce 1, because fractional
    # product does not count and we can produce our goal in max_days with
    # just 1 machine with the shortest production time

    machines2 = list(filter(lambda x: x <= max_days, machines))
    machines2.sort()

    production = 0
    mindays = 0
    mach = Counter(machines2)

    while production < goal:
        mindays += 1
        production = sum([(mindays // item[1]) * mach[item[1]]
                          for item in enumerate(mach)])
    return mindays

# daily_prod = sum([1/mach for mach in machines2])
# min_increment = min(machines2, min([machines[i+1]-machines[i] i  for ]))
# same timout issues


print(minTime([2, 3], 5))
print(minTime([4, 5, 6], 12))

# %% INTERVIEW PREP - Search: Minimum Time Required  - 5 of 5 inccorrect
# 6 testcasess timed out because I used brute force approach of calculating
# production instead of using search approach for min and max, until min, max
# converge to a single number
import math
from collections import Counter


def minTime(machines, goal):
    max_days = goal * min(machines)
    min_days = goal * min(machines) // len(machines)
    print('mintime', min_days)
    # we don't need to look at machines that take longer
    # than max_days to produce 1, because fractional
    # product does not count and we can produce our goal in max_days with
    # just 1 machine with the shortest production time

    machines2 = list(filter(lambda x: x <= max_days, machines))
    machines2.sort()

    total_production = 0
    mindays = min_days - 1
    mach = Counter(machines2)

    while total_production < goal:
        mindays += 1
        daily_production = 0
        for i in range(1, mindays + 1):
            if mindays % i == 0:
                daily_production += mach[i]
                print(i, 'daily', 'mindays', mindays,
                      daily_production, total_production)
        total_production += daily_production
    return mindays

# daily_prod = sum([1/mach for mach in machines2])
# min_increment = min(machines2, min([machines[i+1]-machines[i] i  for ]))
# same timout issues


print(minTime([2, 3], 5))
print(minTime([4, 5, 6], 12))

# %% INTERVIEW PREP - Search: Minimum Time Required - CORRECT APPROACH
# Correct strategy is to divide and conquer ie., find min bound and max
# bound
# and iteratively move closer to the value that satisfies the solution
# This approach avoids needing to calculate the sum of production for a
# large number of machines. This is the general approach for search -
# we trinagulate rather than do brute force.

import math
from collections import Counter


def minTime(machines, goal):
    """Calculate minimum days required to produce goal items.

    machines= array of integerswith #days toproduce one item per machine
    goal = # items we want to produce
    """
    max_days = goal * min(machines) / 1
    # fastest machine has smallest # days to produce one item
    # Rather than doing brute force, we just assume we only have the
    # fastest machine working; this is upper bound
    # total days = units goal * days needed per 1 unit

    # Minimum bound assumes all machines are as fast as fastest
    # total days = units wanted * days needed per unit per 1 machine
    # total days for all machines = units * fastest machine/# machines
    min_days = math.ceil(goal * min(machines) / len(machines))

    machines2 = list(filter(lambda x: x <= max_days, machines))
    # this filters out any machines that take longer than upper bound
    # estimate of days to produce single unit; they will not help us
    # ex: if machines = [1,2,3, 100] and goal is 5 units, then our
    # max_days upper bound estimate is 5*1= 5 days; no need to use the
    # machines thatproduces one unit every 100 days
    # machines2 in this case = [1,2,3]
    mach = Counter(machines2)
    days = list(mach.keys())
    days.sort()
    guess = min_days
    # first guess try theoretical minimum in

    # if min_days = 40, max_days = 41, ceiling of mean is ceil(81/2)=41
    # if min_days = 41, max_days = 42, ceiling of mean is ceil(83/2)=42
    # if there is an integer in between, then there is another guess to try

    while guess != max_days and max_days > min_days + 1:
        # when max_days estimate is one day more than min_days, we are
        # done
        mach_production = [mach[n_days] * (guess // n_days) for n_days in days]
        # only those machines whole units produced for our current guess
        # will count toward production
        production = sum(mach_production)
        print('\nMachine production', mach_production)
        print('guess:', guess, ', mindays:', min_days, ', maxdays:', max_days,
              ', Total Production:', production)
        if production < goal:
            old_guess = guess
            guess = math.ceil(sum([guess, max_days]) / 2)
            # next guess will be halfway between current guess and
            # current upper bound

            min_days = old_guess
            print('Production < Goal')
            print('old_guess-->', old_guess, 'new_min', min_days,
                  'Calculated new guess-->', guess)
        elif production > goal:
            old_guess = guess
            guess = math.ceil(sum([min_days, guess]) / 2)
            max_days = old_guess
            print('Production > Goal')
            print('old_guess-->', old_guess, 'new_max', max_days,
                  'Calculated new guess-->', guess)
        else:
            # this is edge case where maybe a fewer number of days
            # doesn't decrease unit production, so we need to try
            # another lower guess
            old_guess = guess
            guess = math.ceil(sum([min_days, guess]) / 2)
            max_days = old_guess
            print('Production == Goal')
            print('old_guess-->', old_guess, 'new_max', max_days,
                  'Calculated new guess-->', guess)

    print('---done---')
    return guess


print(minTime([1, 3, 10, 100], 60))

# %% # %% INTERVIEW PREP - Search: Minimum Time Required - CORRECT
# CLEANER version with fewer comments or debugging printouts


def minTime(machines, goal):
    # a tigher upper bound is
    # only use fastest machine
    max_days = goal * min(machines) / 1
    # assume all machines are as fast as fastest; need to round up
    min_days = math.ceil(goal * min(machines) / len(machines))

    machines2 = list(filter(lambda x: x <= max_days, machines))
    mach = Counter(machines2)
    days = list(mach.keys())
    days.sort()
    guess = min_days  # first guess try minimum in theory

    while guess != max_days and max_days > min_days + 1:
        mach_prod = [mach[n_days] * (guess // n_days) for n_days in days]
        production = sum(mach_prod)
        if production < goal:
            min_days = guess
            guess = math.ceil(sum([min_days, max_days]) / 2)
        elif production > goal:
            max_days = guess
            guess = math.ceil(sum([min_days, max_days]) / 2)
        else:
            max_days = guess
            guess = math.ceil(sum([min_days, max_days]) / 2)
    return guess


print(minTime([1, 3, 10, 100], 60))

# test cases
# print(minTime([4, 5, 6], 12))
# print(minTime([63, 2, 26, 59, 16, 55, 99, 21, 98, 65], 56))

# %% # %% INTERVIEW PREP - Search: Pairs
# that was easy
# only 2x through array, 2n
# finds how many 'pairs' there are in arr where pair of elements has
# difference of 5
from collections import Counter


def pairs(interval, arr):
    count = 0
    pairs = []
    int_counter = Counter(arr)
    for integer in arr:
        if int_counter[integer + interval]:
            count += 1
            pairs.append((integer, integer + interval))
    print(pairs)
    return count


interval = 5
arr = [0, 5, 10, 11, 16]
print(f'There are {pairs(interval, arr)} pairs'
      f'in {arr} with difference of {k}')


# %% # %% INTERVIEW PREP - Search: Triples
# 5 test cases fail

from collections import Counter


def triplets(a, b, c):
    # there is a runtime error for several large test cases
    a = list(sorted(set(a)))
    b = list(sorted(set(b)))
    c = list(sorted(set(c)))
    result = 0

    def make_dict(a, b):
        a_dict = {}
        lowers_a = 0
        j = 0
        print('\n', a, b)
        print('dict\n-------')
        for i in range(len(b)):
            while b[i] >= a[j] and j < len(a) - 1:
                lowers_a = j + 1
                j += 1
                print(j)

            a_dict[b[i]] = lowers_a

            if b[i] >= a[j] and j == len(a) - 1:
                lowers_a = j + 1

            a_dict[b[i]] = lowers_a
            print('b_i', i, 'on b', b[i], 'dict', a_dict)
        return a_dict

    a_dict = make_dict(a, b)
    c_dict = make_dict(c, b)

    for b_el in b:
        result += a_dict[b_el] * c_dict[b_el]
    return result


print(triplets([1, 3, 5], [2, 3, 3], [1, 2, 3]))
# %%
print(triplets([0, 1, 2, 5], [1, 2, 3, 5, 6], [0, 5]))
print(triplets([0, 1, 2], [0, 1, 2], [0, 1, 2]))

print(triplets([1, 3, 5, 7], [5, 7, 9], [7, 9, 11, 13]))

# %% # %% INTERVIEW PREP - Search: Triples


def triplets(a, b, c):
    """Find how many distinct triplet combos (a_el,b_el,c_el) are possible.

    a,b,c : lists
    a_el<=b_el and b_el >= c_el in all triplets

    This is almost brute force; not sure how this is a search problem
    The solution below is simple but even simpler is filter solution
    """
    a = list((set(a)))
    a.sort()
    b = list((set(b)))
    b.sort()
    c = list((set(c)))
    c.sort()

    ai = 0
    bi = 0
    ci = 0
    ans = 0

    while bi < len(b):
        # this is the useful part ofthe code: nested while loops
        while ai < len(a) and a[ai] <= b[bi]:
            ai += 1

        while ci < len(c) and c[ci] <= b[bi]:
            ci += 1

        ans += ai * ci
        bi += 1

    return ans


# %% # %% INTERVIEW PREP - Search: Triples
from itertools import product


def triplets(a, b, c):
    # This works for some test cases but there are 6 that have runtime
    # issues
    # The length of inputs on these cases is 10,000+ each for a, b, c
    # so the product operation
    # below creates a very large list that is too time consuming
    # to filter; the nested while loops above work better because they
    # do not need to store the entire list of combinations. Only a
    # running count of combinations is needed as the while loops
    # check that a_len <= b_el and c_el <= b_el
    a = list((set(a)))
    a.sort()
    b = list((set(b)))
    b.sort()
    c = list((set(c)))
    c.sort()

    triplets = list(product(a, b, c))
    trips = list(filter(lambda x: x[0] <= x[1] and x[2] <= x[1], triplets))
    # print(trips)
    return len(trips)


a = (1, 2, 3)
b = (2, 4, 6)
c = (-1, 0)
# this should yield 4 + 3*2 + 3*2 = 16 triplets
print(f'The number of special triplets formed from a={a} b={b} and c={c} is'
      f' {triplets(a,b,c)}.')
# 16

# %% INTERVIEW PREP - Dynamic Programming: Max Sum Array
# the array is 100,000 long at max, so we may have time problem
# [1, 2, 3, 4, 5]-> has 4 master sets [1,3,5], [1,4] [2,4] [2,5]
# [3, 5] is just a subset of indexed, so there is not any others
# However, if there was a six, then [3, 6] would be another
# if also 7, 8, then [3, 6, 8] would be the set
# and then [4,7] would be another
# from these we can eliminate numbers if that would increase max


# %% INTERVIEW PREP - Dynamic Programming: Max Sum Array
# a number of incorrect test cases

def maxSubsetSum(arr):
    arr2 = [0] + arr + [0]
    for i in range(len(arr2)):
        arr2[i] = max(arr2[i], 0)

    max_even = 0
    max_odd = 0
    max_even_off = 0
    max_odd_off = 0

    i = 1
    while i <= len(arr):
        if i % 2 == 0:  # even indices
            max_even += max(0, arr2[i])
            max_even_off = max(max_even_off, max(arr2[i], 0)
                               - max(arr2[i - 1], 0)
                               - max(arr2[i + 1], 0))

        elif i % 2 == 1:  # odd indices
            max_odd += max(0, arr2[i])
            max_odd_off = (max(max_odd_off, max(arr2[i], 0)
                               - max(arr2[i - 1], 0)
                               - max(arr2[i + 1], 0)))

        print('i', i, 'max_even:', max_even, 'max_odd', max_odd,
              'max_even_off', max_even_off, 'max_odd_off', max_odd_off)
        i += 1

    result = max(max_even, max_odd, max_even_off + max_odd,
                 max_odd_off + max_even)

    print(result)
    print(arr2)
    return result


# %% INTERVIEW PREP - Dynamic Programming: Max Sum Array - Solution


def maxSubsetSum(arr):
    maxes = {}
    maxes[0] = max(0, arr[0])
    if len(arr) > 1:
        maxes[1] = max(arr[0], arr[1])
    i = 2
    while i < len(arr):
        maxes[i] = max(maxes[i - 2], maxes[i - 1], arr[i],
                       arr[i] + maxes[i - 2])
        breakpoint()
        i += 1

    result = maxes[len(arr) - 1]
    return result


arr = [3, 7, 4, 6, 5]
arr = [2, 1, 5, 8, 4]
arr = [3, 5, -7, 8, -10]
print(maxSubsetSum(arr))

# %% INTERVIEW PREP - Dynamic Programming: Max Sum Array - Solution


def maxSubsetSum(arr):
    dp = {}  # key : max index of subarray, value = sum
    dp[0], dp[1] = arr[0], max(arr[0], arr[1])
    for i, num in enumerate(arr[2:], start=2):
        dp[i] = max(dp[i - 1], dp[i - 2] + num, dp[i - 2], num)
    return dp[len(arr) - 1]


arr = [3, 7, 4, 6, 5]
arr = [2, 1, 5, 8, 4]
arr = [3, 5, -7, 8, -10]
print(maxSubsetSum(arr))


# %%

def abbreviation(a, b):
    #    a='aaBcd'
    #    b = 'ABC'
    A = [[None for j in range(len(b))] for i in range(len(a))]
    # construct base cases
    # for our base case it's only going to be true if it's all lower case
    # and one of them is equal to B[0]
    # or there's only been one upper case letter and it's equal to b[j]
    # upper_encountered means that we encountered that upper case letter

#    A is a grid which we will use to find if a can satisfy b
#  we need to search all of a, and all of b, and end result is in
# cell A[len(a)-1][len(b)-1]
    #   example: b is 'ABC', a = 'aaBcd'
    #   j =     0      1     2
# a[i]b[j] pairs as follows
# i = 0      aA     aB     aC
# i = 1      aA     aB     aC
# i = 2      BA     BB     BC
# i = 3      cA     cB     cC
# i = 4      dA     dB     dC

# we can see that first two rows will show True only for first element aA

    j = 0
    if a[0].upper() == b[0]:
        A[0][0] = True
    upper_encountered = a[0].isupper()
    for i in range(1, len(a)):
        print(i, ':', a[i], b[j])
        if a[i].isupper() and upper_encountered:
            A[i][j] = False
            # because we already had an upper case in 'a' at i-1, we cannot
            # satisfy the first b character with this second capital

        elif a[i].isupper() and not upper_encountered and a[i] == b[j]:
            A[i][j] = True
            upper_encountered = True
        elif a[i].isupper() and not upper_encountered and a[i] != b[j]:
            A[i][j] = False
            upper_encountered = True
        elif a[i].islower() and a[i].upper() == b[j] and not upper_encountered:
            A[i][j] = True
            print('found a')

        # a[i].islower()
        else:
            A[i][j] = A[i - 1][j]

        print(A[i][j])
    print(A)
    print('after first round')
    # since a[i] is only length 1 anything in A[0][1:] will be False
    i = 0
    for j in range(1, len(b)):
        A[i][j] = False
    # now find the solution
    print(A)
    for i in range(1, len(a)):
        print('-------')
        print(i, ':i', a[i], i, ':j', b[j])
        for j in range(1, len(b)):
            print(i, ':i', a[i], j, ':j', b[j])
            if a[i].upper() == b[j] and a[i].islower():
                A[i][j] = A[i - 1][j - 1] or A[i - 1][j]
            elif a[i].upper() == b[j] and a[i].isupper():
                A[i][j] = A[i - 1][j - 1]
            elif a[i].upper() != b[j] and a[i].islower():
                A[i][j] = A[i - 1][j]
            else:
                A[i][j] = False
    print(A)
    if A[len(a) - 1][len(b) - 1]:
        return "YES"
    return "NO"


print(abbreviation('AaBcd', 'ABC'))

# %% timeit is suited for code snippets; probably not long functions
if __name__ == '__main__':
    import timeit
    timeit.timeit("abbreviation('AaBcd','ABC')",
                  setup='from __main__ import abbreviation')

# %% INTERVIEW PREP - Stacks and Queueus - Queues: A Tale of Two Stacks
# times out on a number of cases because the put operation requires a
# rewriting of the whole stack, shifting over every element, which takes
# O(n_i) time for every put operation, for the length n_i of the queue
# at the time of the put operation
import random


class MyQueue(object):
    def __init__(self):
        self.queue = []

    def peek(self):
        # print(self.queue[-1])
        return self.queue[-1]

    def pop(self):
        return self.queue.pop()

    def put(self, value):
        self.queue = [value] + self.queue


queue = MyQueue()
n = 1000  # int(input())
for i in range(t):
    # values = map(int, input().split())
    # values = list(values)
    values = [random.randint(1, 3) for _ in range(n)]
    if values[i] == 1:
        queue.put(random.randint(0, 10))
    elif values[i] == 2:
        queue.pop()
    else:
        print(queue.peek())


"""
Sample Input

10
1 42
2
1 14
3
1 28
3
1 60
1 78
2
2

Sample Output

14
14
"""

# %% INTERVIEW PREP - Stacks and Queueus - Min Max Riddle
# brute force times out on three cases with long arrays
# 6 test cases work fine

# DIscussion:
# =============================================================================
# A few hints for people who are banging their heads on the wall trying
# to pass all test cases:
# 1) O(N) solution is possible using stacks; avoid DP for this problem
# 2) Think about how to identify the largest window a number is the
# minimum for (e.g. for the sequence 11 2 3 14 5 2 11 12 we would make a
# map of number -> window_size as
# max_window = {11: 2, 2: 8, 3: 3, 14: 1, 5: 2, 12: 1}) -
# this can be done using stacks in O(n)
# 3) Invert the max_window hashmap breaking ties by taking the maximum
# value to store a mapping of windowsize -> maximum_value
# (continuing with example above inverted_windows = {1: 14, 8:2, 3:3, 2:11}
# 4) starting from w=len(arr) iterate down to a window size of 1,
# looking up the corresponding values in inverted_windows and fill
# missing values with the previous largest window value
# (continuing with the example result = [2, 2, 2, 2, 2, 3, 11, 14] )
# 5) Return the result in reverse order (return [14, 11, 3, 2, 2, 2, 2, 2])
# =============================================================================


def riddle(arr):
    n = len(arr)
    max_mins = [max(arr)]
    for i in range(1, n):
        mins = []
        for j in range(n - i):
            mins.append(min(arr[j:j + i + 1]))
        max_mins.append(max(mins))
    return max_mins


# %% INTERVIEW PREP - Stacks and Queueus - Min Max Riddle
# answer - wow, so strange a solution
# =============================================================================


def riddle(arr):
    n = len(arr)
    max_mins = [None] * n
    stack = []  # will store (num, index)
    for i in range(n):
        print('stack', stack)
        print('max_mins', max_mins)
        # remember to "push back"
        _m = i
        while len(stack) > 0 and stack[-1][0] > arr[i]:
            _v, _i = stack.pop(-1)
            w = i - _i
            for _w in range(w):  # note that it's zero indexed and shifted down
                if max_mins[_w] is None:
                    max_mins[_w] = _v
                else:
                    max_mins[_w] = max(max_mins[_w], _v)
            _m = _i  # get the smallest index at which we could start
        stack.append((arr[i], _m))

    # these were the minima for all this time
    while len(stack) > 0:
        print('stack', stack)
        print('max_mins', max_mins)
        _v, _i = stack.pop(-1)
        w = n - _i
        for _w in range(w):
            if max_mins[_w] is None:
                max_mins[_w] = _v
            else:
                max_mins[_w] = max(max_mins[_w], _v)
    return max_mins


arr = [1, 2, 3, 5, 1, 13, 3]
riddle(arr)
# %% INTERVIEW PREP - Stacks and Queueus - Queues: A Tale of Two Stacks
# This works
"""
you need 2 stacks to make a queue, one covers the push operation and
one covers the pops and tops. The pushing stack is what the queue
looks like from the back of the line the pop/top stack is looking at
the queue from the front of the line. if your queue was 3,4,5 the
pusher sees 3,4,5 and the poper sees 5,4,3. The trick is the queue
is virtual and there only needs to be 1 copy of each number across
the two stacks so if the pop/top stack has 5,4,3 then the push stack
would be empty. New items go on the stack that is the back of the line
while they come off the stack that is the front of the line
"""
import random
import time


class MyQueue(object):
    def __init__(self):
        self.inward = []
        self.outward = []

    def peek(self):
        if not self.outward:
            # perform following when outward is empty
            # reverse inward so that FIRST pushed item is now -1
            # item in list
            self.outward = list(reversed(self.inward))

            # and empty the inward list
            self.inward = []
        return self.outward[-1]

    def pop(self):
        head = self.peek()
        del self.outward[-1]
        return head

    def put(self, value):
        self.inward.append(value)


queue = MyQueue()
n = 1000
put_duration = peek_duration = pop_duration = 0

for i in range(n):
    values = [random.randint(1, 6) for _ in range(n)]
    values = list(map(lambda x: bool(x == 1 or x == 4 or x == 5 or x == 6)
                      + 2 * bool(x == 2) + 3 * bool(x == 3), values))

    if values[i] == 1:
        start = time.time()
        queue.put(random.randint(0, 10))
        end = time.time()
        put_duration += end - start
    elif values[i] == 2:
        start = time.time()
        queue.pop()
        end = time.time()
        pop_duration += end - start
    else:
        start = time.time()
        print(queue.peek())
        end = time.time()
        peek_duration += end - start

n_put = sum([1 for x in values if x == 1])
n_pop = sum([1 for x in values if x == 2])
n_peek = sum([1 for x in values if x == 3])

print(f'put_duration \t{put_duration:>4.2f} n_put {n_put}'
      f' avg {put_duration/n_put:>7.6f}\n'
      f'peek_duration \t{peek_duration:>4.2f} n_pop {n_peek}'
      f' avg {pop_duration/n_peek:>7.6f}\n'
      f'pop_duration \t{pop_duration:>4.2f} n_peek {n_pop}'
      f' avg {peek_duration/n_pop:>7.6f}')

# pop and peek are the time intensive tasks, as the dequeue is the
# expensive task when using a stack to implement a queue
# %% INTERVIEW PREP - Stacks and Queueus - Queues: A Tale of Two Stacks
# This also works with the double ended queue 'deque' data structure
# the above solution uses two lists to do the same thing

from collections import deque

mydeque = deque()
t = int(input())
for line in range(t):
    values = map(int, input().split())
    values = list(values)
    if values[0] == 1:
        # push operation equivalent to enqueue, which we arbitrarily
        # choose to put on right side
        mydeque.append(values[1])
    elif values[0] == 2:
        # popleft operation is equivalent to dequeue, which we must do
        # on left side since push is on right side
        mydeque.popleft()
    elif values[0] == 3:
        # we want to see value of next pop, without dequeueing
        print(mydeque[0])

# %% INTERVIEW PREP - Stacks and Queueus - Balanced Brackets
# this works; we should refactor so we have fewer conditions, if possible


def isBalanced(s):
    """Verify s is a balanced string of brackets.

    s is a string of brackets like '([{}])
    Return 'YES' if s is balanced
    'NO' if no
    My idea is to turn into list, pop values off end of it, checking for if
    it is opening or closing bracket. First opening bracket should match
    with a closing bracket on 2nd queue
    queue 1  test   queue2
    ([{}])   -        -
    ([{}]    )        )       is closing, so just append to q2
    ([{}     ]        )]      is closing, so just append to q2
    ([{      }        )]}     is closing, so just append to q2
    ([       {        )]}     is opening, so must match pop of q2? YES
    (        [        )]      YES
    -        (        )       YES
    return 'NO' if opening doesn't match
    """
    left = list(s)
    right = []

    result = 'YES'
#    breakpoint()
    while left or right:  # if not empty, enter loop
        if (not left) and right:
            # left is empty but right is not empty, we have a problem
            result = 'NO'
            break

        else:
            x = left.pop()
            if x in set('})]'):
                right.append(x)
            elif x == '{' and not right:
                # cannot have hanging opening brackets; right.pop()
                # when right is empty generates error
                result = 'NO'
                break

            elif x == '{' and right[-1] == '}':
                right.pop()
            elif x == '(' and not right:
                result = 'NO'
                break

            elif x == '(' and right[-1] == ')':
                right.pop()
            elif x == '[' and not right:
                result = 'NO'
                break
            elif x == '[' and (right[-1] == ']'):
                right.pop()
            else:
                result = 'NO'
                break
    print(result)
    return result


# isBalanced('{[(])}')
# isBalanced('}]') #'[}}(}][))]')

inp = ['}][}}(}][))]',
       '[](){()}',
       '()',
       '({}([][]))[]()',
       '{)[](}]}]}))}(())(',
       '([[)']

for i in range(6):
    isBalanced(inp[i])


# %% INTERVIEW PREP - Stacks and Queueus - Balanced Brackets
# refactored a little; works
# not sure there is benefit to trimming it down fruther in terms of
# # of levels for visual


def isBalanced(s):
    left = list(s)
    right = []

    result = 'YES'
    while left or right:  # if not empty, enter loop
        if (not left) and right:
            # left is empty but right is not empty, we have a problem
            result = 'NO'
            break
        # hanging opening bracket is not good
        elif left[-1] in set('{[(') and not right:
            result = 'NO'
            break
        else:
            x = left.pop()
            if x in set('})]'):
                right.append(x)
            elif ((x == '{' and right[-1] == '}')
                  or (x == '(' and right[-1] == ')')
                  or (x == '[' and right[-1] == ']')):
                right.pop()
            else:
                result = 'NO'
                break
    print(result)
    return result


# %% INTERVIEW PREP - Stacks and Queueus - Largest Rectangle
# this works fine, no problems with timing
# not sure how a stack or queue could have helped this problem; lists
# allow quick random access with the while loops controlling flow

# all buildings are of equal width 1, so the area of the building is
# simply 1xh = h, so we just add h of the adjacent buildings to current
# building as long as those buildings' are of equal or higher height

def largestRectangle(heights):
    i = 0
    maxarea = 0
    while i < len(heights):
        # start with building of width
        area = heights[i]
        j = 1
        # breakpoint()
        # adding buildings further along the array to our current h,
        # if they are as tall or taller than our current building
        while j + i < len(heights) and heights[i + j] >= heights[i]:
            area = area + heights[i]
            j += 1
        k = 1
        # adding buildings to the left of our current building, if they
        # are taller or equal height
        while i - k >= 0 and heights[i - k] >= heights[i]:
            area = area + heights[i]
            k += 1
        maxarea = max(area, maxarea)
        i += 1
    print(maxarea)
    return maxarea


h = [1, 2, 3, 4, 5]
h = [11, 11, 10, 10, 10]
largestRectangle(h)

# %% INTERVIEW PREP - Stacks and Queueus - Castle on the Grid
"""
This works

This is a more elegant form of what I was trying to do above
The visited before ensures there are no backtrackings and each possible
move reaches to all the grid and each step of the while loop; much better
that way that trying to trace paths the way I was doing
"""


def minimumMoves(grid, startX, startY, goalX, goalY):
    moves = ((1, 0), (-1, 0), (0, 1), (0, -1))
    visit = {(startX, startY): 0}
    rows = cols = len(grid)
    n_blocked = sum([1 for x in ''.join(grid) if x == 'X'])
    blocked = set()

    # print out grid ##########################################################
    print('Grid:'
          '\n------------------------------')
    print(f'1st row   {"".join([str((0,i)) for i in range(rows)])}')
    print('1st col   \t\t')
    for i in range(len(grid)):
        print(f'{i, 0} \t\t{"      ".join(grid[i])}')
    print('\n------------------------------')
    print(f'Start: ({startX, startY}) Target ({goalX, goalY})')
    print('--------------------------------')

    q = [[startX, startY, 0]]

    while len(q) > 0:
        path_base, q = q[0], q[1:]
        # base cell; we explore end of all moves from this before
        # we move starting cell one further down line
        row, col, val = path_base
        print(f'-----------\nStart at {row, col} moves so far {val}')
        for move in moves:
            nrow, ncol = row, col
            print(f'\tmove direction {move}')
            while True:
                # moves one step in same direction until X found or
                # destination reached or edge of grid reached, at
                # which point we break and choose the next
                # move/direction
                # each square gets val+1
                nrow, ncol = nrow + move[0], ncol + move[1]

                if (
                    nrow >= 0 and ncol >= 0 and nrow < rows
                    and ncol < cols and grid[nrow][ncol] == 'X'
                ):
                    # add to blocked cells set if we find them
                    blocked.add((nrow, ncol))
                    print('\treached X, cannot proceed in this direction')
                    break

                elif (
                        nrow >= 0 and ncol >= 0 and nrow < rows
                        and ncol < cols and grid[nrow][ncol] == '.'
                ):
                    # arriving at open cell, we test if it is
                    # destination or add it and n_moves to visit
                    # dictionary
                    if (nrow, ncol) == (goalX, goalY):
                        visit[(nrow, ncol)] = val + 1
                        print(f'\tArrived at {(goalX, goalY)}')
                        print(f'\tVisited {len(visit)} cells of {rows**2}'
                              f' and {n_blocked} blocked')
                        print(f'\tVisits {visit}')
                        print(f'\tBlocked found {blocked}')
                        print(f'\tq {q}\n')
                        print(f'\tMin # turns: {val + 1}')
                        print(f'\tpath {path_base}')
                        return val + 1
                    elif (nrow, ncol) not in visit:
                        visit[(nrow, ncol)] = val + 1
                        # q is list, so this is append operation
                        q += [[nrow, ncol, val + 1]]

                        print(f'\tAdding {nrow, ncol} to Visits: '
                              f'cell: n_visits {visit}')
                        print(f'\tq {q}\n')
                    else:
                        print(f'\t{nrow,ncol} Already visited')

                else:
                    print(f'\t{nrow, ncol} outside grid, try another move\n')
                    break
        print(f'No more moves to try for starting cell {row,col}')
    return -1


grid = ['.X.', '...', '...']
minimumMoves(grid, 0, 0, 0, 2)



# %% INTERVIEW PREP - Graphs:

# %% INTERVIEW PREP - Trees: Tree: Huffman Decoding
# Medium

# %% INTERVIEW PREP - Trees: Balanced Forest
# Hard

# %% INTERVIEW PREP - Trees: Is This a Binary Search Tree?

"""
For the purposes of this challenge, we define a binary search tree to
be a binary tree with the following properties:

1. The  value of every node in a node's left subtree is less than the
    data value of that node.
2. The  value of every node in a node's right subtree is greater than the
    data value of that node.
3. The value of every node is distinct.
We can recursively determine test if each node violates any of the 3
criteria and combine all the results in a boolean expression that
yields true of there is not one violation
"""
# failed 9 of 15 test cases
# I think an example diagram incorrectly allowed a child value in right
# tree to be below the parent; so the code below is wrong because it
# only tested vs. the origional headvalue vs. the value of each parent down
# the heirarchy


def checkBST(root):
    all_values = set([root.data])

    def check_node(values, node, comp, headvalue):
        if not node:
            return True
        elif (node.data in values
              or eval(str(node.data) + comp + str(headvalue))):
            return False
        else:
            values.add(node.data)
            return (check_node(values, node.left, comp, headvalue)
                    and check_node(values, node.right, comp, headvalue))

    if root is None:
        return 'Yes'
    else:
        condition = (check_node(all_values, root.left, '>=', root.data)
                     and check_node(all_values, root.right, '<=', root.data))

    if condition:
        return 'Yes'
    else:
        return 'No'


# %% solution

def checkBST(root):
    return check(root, -float('inf'), float('inf'))


def check(root, min_, max_):
    # no need for a set passed down the tree because each split in
    # tree divides into lesser and greater and lower and upper limits
    # and there is no equality allowed
    return (root is None or (root.data < max_
                             and root.data > min_
                             and check(root.left, min_, root.data)
                             and check(root.right, root.data, max_))
            )


# %% INTERVIEW PREP - Trees: Height of a Binary Tree

# Enter your code here. Read input from STDIN. Print output to STDOUT
"""
class Node:
      def __init__(self,info):
          self.info = info
          self.left = None
          self.right = None

       // this is a node of the tree , contains info as data, left , right
"""


def height(root):
    if not root:
        return -1
    return max(height(root.left), height(root.right)) + 1


tree = BinarySearchTree()
t = int(input())

arr = list(map(int, input().split()))

for i in range(t):
    tree.create(arr[i])

print(height(tree.root))


# %% INTERVIEW PREP - Trees: Binary Search Tree : Lowest Common Ancestor

class Node:
    def __init__(self, info):
        self.info = info
        self.left = None
        self.right = None
        self.level = None

    def __str__(self):
        """HOLDER."""
        return str(self.info)


class BinarySearchTree:
    def __init__(self):
        self.root = None

    def create(self, val):
        if self.root is None:
            self.root = Node(val)
        else:
            current = self.root

            while True:
                if val < current.info:
                    if current.left:
                        current = current.left
                    else:
                        current.left = Node(val)
                        break
                elif val > current.info:
                    if current.right:
                        current = current.right
                    else:
                        current.right = Node(val)
                        break
                else:
                    break


def lca(root, a, b):
    node = root
    while node:
        if max(a, b) < node.info:
            node = node.left
        elif min(a, b) > node.info:
            node = node.right
        else:
            break
    return node


tree = BinarySearchTree()
t = 6  # int(input())

arr = [4, 2, 3, 1, 7, 6]
v = [1, 7]

# v = list(map(int, input().split()))
# list(map(int, input().split()))

for i in range(t):
    tree.create(arr[i])

ans = lca(tree.root, v[0], v[1])
print(ans.info)
# %% INTERVIEW PREP - Linked Lists: Insert a node at a specific
# position in a linked list

#
# For your reference:
#
# SinglyLinkedListNode:
#     int data
#     SinglyLinkedListNode next


def insertNodeAtPosition(head, data, position):
    # this works
    newnode = SinglyLinkedListNode(data)
    Headvalue = head

    if position == 0:
        newhead = newnode
        newhead.next = head
        return newhead
    i = 0
    while i < position:
        # traverse list
        previous = Headvalue
        Headvalue = Headvalue.next
        i += 1
    previous.next = newnode
    newnode.next = Headvalue
    return head


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    llist_count = int(input())
    llist = SinglyLinkedList()

    for _ in range(llist_count):
        llist_item = int(input())
        llist.insert_node(llist_item)

    data = int(input())
    position = int(input())
    llist_head = insertNodeAtPosition(llist.head, data, position)

    print_singly_linked_list(llist_head, ' ', fptr)
    fptr.write('\n')
    fptr.close()


# %% Reference
# https://pythonguides.com/linked-lists-in-python/#:~:text=What%20are%20linked%20lists%20in%20python?%201%20A,link%20to%20the%20next%20node.%20More%20items...


class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None


class Linkedlist:
    def __init__(self):
        self.head = None
        self.last_node = None

    def Atbegining(self, data_n):
        NewNode = Node(data_n)
        NewNode.next = self.head
        self.head = NewNode

    def append(self, data):
        if self.last_node is None:  # nothing in list
            self.head = Node(data)
            self.last_node = self.head
        else:
            self.last_node.next = Node(data)
            self.last_node = self.last_node.next

    def display(self):
        curr = self.head
        while curr is not None:
            print(curr.data, end=' ')
            curr = curr.next

    def AtEnd(self, newdata):
        NewNode = Node(newdata)
        if self.headvalue is None:
            self.headvalue = NewNode
            return
        last = self.headvalue
        while(last.next):
            last = last.next
        last.next = NewNode

    def insertNode_at(self, data_n, position):
        newnode = Node(data_n)
        Headvalue = self.head
        i = 0

        if position == 0:
            newhead = newnode
            newhead.next = self.head
            return

        while i < position:
            # traverse list
            previous = Headvalue
            Headvalue = Headvalue.next
            i += 1
        previous.next = newnode
        newnode.next = Headvalue
        return

    def RemoveNode(self, Removekey):
        Headvalue = self.head
        if (Headvalue is not None):
            if (Headvalue.data == Removekey):
                self.head = Headvalue.next
                Headvalue = None
                return
        while (Headvalue is not None):
            if Headvalue.data == Removekey:
                break
            previous = Headvalue
            Headvalue = Headvalue.next
        if (Headvalue is None):  #
            return
        previous.next = Headvalue.next
        Headvalue = None

    def list_print(self):
        printvalue = self.head
        while (printvalue):
            print(printvalue.data),
            printvalue = printvalue.next


my_list = Linkedlist()
my_list.Atbegining("jan")
my_list.Atbegining("feb")
my_list.Atbegining("march")
my_list.RemoveNode("jan")
my_list.list_print()


# %% INTERVIEW PREP - Linked Lists: Find Merge Point of Two Lists
"""
The first line contains an integer , the number of test cases.

Each of the test cases is in the following format:
The first line contains an integer, , the node number where the merge
    will occur.
The next line contains an integer,  that is the number of nodes in the
    first list.
Each of the following  lines contains a value for a node. The next
line contains an integer,  that is the number of nodes in the second
    list.
Each of the following  lines contains a value for a node.
1 # t, how many test cases
1 # index, the node number where merge occurs
3 # number of nodes in first list
1 # data value of 1st node in 1st list
2 # data value of 2nd node in 1st list
3 # data value of 3rd node in 1st list
1 # number of nodes in second list
1 # data value of node in 2nd list
"""


def findMergeNode(head1, head2):
    # traverse data
    list1_node = head1
    list2_node = head2
    while list1_node != list2_node:
        list1_node = list1_node.next
        if list1_node is None:
            list1_node = head1
            list2_node = list2_node.next
    merge_node = list1_node
    return merge_node.data


# %% INTERVIEW PREP - Linked Lists: Inserting a Node Into a Sorted
# Doubly Linked List
# n the number of nodes in list is 1 or greater, so no need to test for
# empty list

def sortedInsert(head, data):
    newnode = DoublyLinkedListNode(data)
    Headvalue = head

    if head.data >= data:
        # insert node before current head
        newnode.next = head
        head.prev = newnode
        return newnode

    while Headvalue.data < data:
        # traverse list until Headvalue.data is >=, then insert in
        # front of Head
        if Headvalue.next is None:
            Headvalue.next = newnode
            newnode.prev = Headvalue
            return head
        Headvalue = Headvalue.next

    # when while loop breaks, newnode can be inserted before
    # need to rewire connections on either side of new node,
    # so 2x2 = 4 reconnections
    newnode.prev = Headvalue.prev
    Headvalue.prev.next = newnode
    Headvalue.prev = newnode
    newnode.next = Headvalue
    return head

# %% INTERVIEW PREP - Linked Lists: Reverse a doubly linked lis
# could be null / empty list
"""
Do in place reassignment
"""


def reverse(head):
    if head.data is None or head.next is None:
        return head
        # empty list means head has a value but data is None because
        # there is no node inserted; no need to do any reversing
        # List with only one node also requires no change
    Headvalue = head
    while Headvalue.next is not None:
        # need to reverse the 2 connections at each intersection
        Headvalue.next, Headvalue.prev = Headvalue.prev, Headvalue.next
        Headvalue = Headvalue.prev
        # .prev is now pointing to what was next node in original list
    Headvalue.next, Headvalue.prev = Headvalue.prev, Headvalue.next
    # must perform one last reassignment on the last node
    return Headvalue


# %% INTERVIEW PREP - Linked Lists: Linked Lists: Detect a Cycle
# 25 points
"""
A Node is defined as:

    class Node(object):
        def __init__(self, data = None, next_node = None):
            self.data = data
            self.next = next_node
"""


def has_cycle(head):
    if head is None:
        return 0
    else:
        pointers = set()
        Headvalue = head
        while Headvalue.next is not None:
            pointers.add(Headvalue)
            if Headvalue.next in pointers:
                return 1
            Headvalue = Headvalue.next
    return 0

# %% INTERVIEW PREP - Recursion and Backgracking: Recursion: Fibonacci
# passed
from functools import lru_cache


# @lru_cache   # caching doesn't work on hackerrank - generates error
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

# %% INTERVIEW PREP - Recursion: Davis' Staircase

# can take 1, 2, or 3 steps to complete n-high staircase
from functools import lru_cache


@lru_cache
def stepPerms(n):
    if n == 1:
        return 1
    if n == 2:
        return 2
    if n == 3:
        return 4
    else:
        return stepPerms(n - 1) + stepPerms(n - 2) + stepPerms(n - 3)


# %%
cache = {}


def stepPerms(n):
    if n == 1:
        return 1
    if n == 2:
        return 2
    if n == 3:
        return 4
    if n not in cache:
        cache[n] = stepPerms(n - 1) + stepPerms(n - 2) + stepPerms(n - 3)

    return cache[n]


print(stepPerms(6))


# using dynamic programming
def stepPerms(n):
    dp = [0] * (n + 1)
    dp[1], dp[2], dp[3] = 1, 2, 4
    for i in range(4, n + 1):
        dp[i] = sum(dp[i - 3:i])
    return dp[n]


print(stepPerms(5))

# %% INTERVIEW PREP - Recursion: Crossword Puzzle - First try
"""
From discussion:
"This is a very poor puzzle, for the simple reason that there does not
seem to be an elegant solution for it. Just brute-force it.
So the solution is very easy, but the implementation is very tedious
and boring."

"For what it's worth, I think that the actual difficulty of the problem
is medium level, but there were several different tasks that you had
 to do rather than the single task that most other questions ask of you.

It's medium in that you can be inefficient in both time and space and
 use no algorithms or data structures other than arrays/vectors
 (although I created a custom class to make things easier) and come
 up with the solution.

So still medium, but definitely more than 30 points ... took me like
200 lines of code to solve."

Editorial (hackerrank solution) is also empty.
"""
# Test cases

words = 'LONDON;DELHI;ICELAND;ANKARA'
crossword = ['+-++++++++',
             '+-++++++++',
             '+-++++++++',
             '+-----++++',
             '+-+++-++++',
             '+-+++-++++',
             '+++++-++++',
             '++------++',
             '+++++-++++',
             '+++++-++++']
words = 'ICELAND;MEXICO;PANAMA;ALMATY'
crossword = ['++++++-+++',
             '++------++',
             '++++++-+++',
             '++++++-+++',
             '+++------+',
             '++++++-+-+',
             '++++++-+-+',
             '++++++++-+',
             '++++++++-+',
             '++++++++-+']

words = 'SYDNEY;TURKEY;DETROIT;EGYPT;PARIS'
crossword = ['+-++++++++',
             '+-++++++++',
             '+-------++',
             '+-++++++++',
             '+-----++++',
             '+-+++-++++',
             '+++-----++',
             '+++++-++++',
             '+++++-++++',
             '+++++-++++']

# %%
import re
from itertools import permutations


words = 'ANDAMAN;MANIPUR;ICELAND;ALLEPY;YANGON;PUNE'
crossword = ['+-++++++++',
             '+-------++',
             '+-++-+++++',
             '+-------++',
             '+-++-++++-',
             '+-++-++++-',
             '+-++------',
             '+++++++++-',
             '++++++++++',
             '++++++++++']


def twistgrid(crossword):
    """Transpose position of column and row elements.

    (This will allow re.finditer to find column blank spaces as
     columns become strings in the transposed grid.)
    Returns list of strings; each string is a column, with index 0
    representing the row of original crossword
    """
    n = len(crossword)
    twisted = []
    for col in range(n):
        tw_row = [crossword[i][col] for i in range(n)]
        tw_row = ''.join(tw_row)
        twisted.append(tw_row)
    # print(twisted)
    return twisted


def crosswordPuzzle(crossword, words):
    """Fill in empty crossword grid with provided words."""
    words = words.split(';')
    n_rows = len(crossword)
    lines = []
    columns = []
    row = 0
    twisted = twistgrid(crossword)
    while row < n_rows:
        lines.append(list(re.finditer(r'(-){2,}', crossword[row])))
        columns.append(list(re.finditer(r'(-){2,}', twisted[row])))
        row += 1
    # lines and columns are a list of lists; both are n_rows long.
    # If there are any matches in the column or row, they will appear
    # as a match object in one of the sublists
    # A match requires 2 or more blank spaces consecutively +--+
    # A single blank is not a word / match
    # print('lines', lines)
    # print(columns)
    row_words = []
    col_words = []
    blank_lengths = []
    for irow, matches in enumerate(lines):
        row_words.extend([((irow, x.span()[0]),
                           (irow, x.span()[1] - 1)) for x in matches])
        # row_words is a list of 2-tuples, which have the coord of
        # start and end of blank space for a word
        # ex: ((3, 0), (3, 4)) represents a space at (row 3, col 0),
        # which extends 5 blank space to terminate on (row 3, col 4)
        # .extend method is used because we could have more than one
        # match object per line and append method only takes one
        # argument
        blank_lengths.extend([x.span()[1] - x.span()[0] for x in matches])
        # blank_lengths is a list that counts how long each of the row
        # words is, going top to bottom in grid.
        # a match object's span()[1] coordinate is one greater than the
        # last index of the match, so subtracting gives the length

    for icol, matches in enumerate(columns):
        col_words.extend([((x.span()[0], icol),
                           (x.span()[1] - 1, icol)) for x in matches])
        blank_lengths.extend([x.span()[1] - x.span()[0] for x in matches])

    intersections = {'row_words': [], 'col_words': []}
    n_intersections = 0
    # we want coordinates of where the intersections occur of the row
    # words with column words to verify a matching letter later on
    # intersections is a dictionary of two lists and each list contains
    # 2-tuples:
    #    (p, q) p = index of row_words or col_words
    #           q = index of row_words[p] that corresponds to the letter
    #               in the word that needs to be verified

    for i, rword in enumerate(row_words):
        for j, cword in enumerate(col_words):
            if rword[0][0] >= cword[0][0] and rword[0][0] <= cword[1][0] and\
               cword[0][1] >= rword[0][1] and cword[0][1] <= rword[1][1]:
                # rword[0][0] is row of rword, which needs to fall
                # within
                # the range of the column word in order to intersect
                # cword[0][1] is column row of cword, which needs to
                # fall within of the row word in order to intersect
                intersections['row_words'].append((i,
                                                   cword[0][1] - rword[0][1]))
                intersections['col_words'].append((j,
                                                   rword[0][0] - cword[0][0]))
                n_intersections += 1

    guesses = list(permutations(words))
    # there might be better way to choose ways to check potential solutions
    # but max # words in test cases was 6, so 6x5x4x3x2 = 720 guesses
    right_length = []
    for i, guess in enumerate(guesses):
        if all((len(guess[k]) == blank_lengths[k] for k in range(len(guess)))):
            # Only pass guesses where every single word fits in the blank.
            # Rows are filled first, then columns
            right_length.append(guess)

    for k, guess in enumerate(right_length):
        row_intersections = []
        col_intersections = []
        i = 0
        while col_intersections == row_intersections and i < n_intersections:
            word = guess[intersections['row_words'][i][0]]
            letter_ind = intersections['row_words'][i][1]
            row_intersections.append(word[letter_ind])

            # now find letters of columns that are intersections
            word = guess[intersections['col_words'][i][0] + len(row_words)]
            # need offset because the first words in guess fall into rows
            letter_ind = intersections['col_words'][i][1]
            col_intersections.append(word[letter_ind])
            i += 1
        if col_intersections == row_intersections:
            # print(f'Intersections match for guess {guess}')
            break  # don't keep changing guess even after you found fit
    print(f'Found correct guess on search {k+1} out of {len(guesses)} choices')

    # Now let's insert solution into grid, replacing the blank spaces '-'
    # with the words
    # To mutate elements of crossword, we need to mutate it
    # so we need to turn it into list.
    # We also need to transpose it to fill in the columns, then transpose
    # it back to normal orientation

    out = [list(row) for row in crossword]
    for i, word in enumerate(row_words):
        out[word[0][0]][word[0][1]:word[1][1] + 1] = list(guess[i])

    out = [''.join(row) for row in out]
    out = twistgrid(out)

    out = [list(row) for row in out]
    for i, word in enumerate(col_words):
        out[word[0][1]][word[0][0]:word[1][0] + 1] = list(guess[i + len(row_words)])
        # the indexing is different from row above because we need
        # to update the range of rows that column word occupies
        # need offset because row words come first in guesses list
    out = [''.join(row) for row in out]
    out = twistgrid(out)
    out = [''.join(row) for row in out]
    print('\n'.join(out))

    return out


import time
start = time.time()
for i in range(1000):
    answers = crosswordPuzzle(crossword, words)
end = time.time()
print(end - start)
# 6.75 seconds for 1000 runs

# %% INTERVIEW PREP - Recursion and Backtracking: Crossword Puzzle
# This is my original brute force solution, but with no comments


def twistgrid(crossword):
    n = len(crossword)
    twisted = []
    for col in range(n):
        tw_row = [crossword[i][col] for i in range(n)]
        tw_row = ''.join(tw_row)
        twisted.append(tw_row)
    # print(twisted)
    return twisted


def crosswordPuzzle(crossword, words):
    """Fill in empty crossword grid with provided words."""
    words = words.split(';')
    n_rows = len(crossword)
    lines = []
    columns = []
    row = 0
    twisted = twistgrid(crossword)
    while row < n_rows:
        lines.append(list(re.finditer(r'(-){2,}', crossword[row])))
        columns.append(list(re.finditer(r'(-){2,}', twisted[row])))
        row += 1
    row_words = []
    col_words = []
    blank_lengths = []
    for irow, matches in enumerate(lines):
        row_words.extend([((irow, x.span()[0]),
                           (irow, x.span()[1] - 1)) for x in matches])
        blank_lengths.extend([x.span()[1] - x.span()[0] for x in matches])

    for icol, matches in enumerate(columns):
        col_words.extend([((x.span()[0], icol),
                           (x.span()[1] - 1, icol)) for x in matches])
        blank_lengths.extend([x.span()[1] - x.span()[0] for x in matches])

    intersections = {'row_words': [], 'col_words': []}
    n_intersections = 0

    for i, rword in enumerate(row_words):
        for j, cword in enumerate(col_words):
            if rword[0][0] >= cword[0][0] and rword[0][0] <= cword[1][0] and\
               cword[0][1] >= rword[0][1] and cword[0][1] <= rword[1][1]:
                intersections['row_words'].append((i,
                                                   cword[0][1] - rword[0][1]))
                intersections['col_words'].append((j,
                                                   rword[0][0] - cword[0][0]))
                n_intersections += 1

    guesses = list(permutations(words))
    right_length = []
    for i, guess in enumerate(guesses):
        if all((len(guess[k]) == blank_lengths[k] for k in range(len(guess)))):
            right_length.append(guess)

    for k, guess in enumerate(right_length):
        row_intersections = []
        col_intersections = []
        i = 0
        while col_intersections == row_intersections and i < n_intersections:
            word = guess[intersections['row_words'][i][0]]
            letter_ind = intersections['row_words'][i][1]
            row_intersections.append(word[letter_ind])

            # now find letters of columns that are intersections
            word = guess[intersections['col_words'][i][0] + len(row_words)]
            # need offset because the first words in guess fall into rows
            letter_ind = intersections['col_words'][i][1]
            col_intersections.append(word[letter_ind])
            i += 1
        if col_intersections == row_intersections:
            print(f'Intersections match for guess {guess}')
            break  # don't keep changing guess even after you found fit
    print(f'Found correct guess on search {k+1} out of {len(guesses)} choices')

    out = [list(row) for row in crossword]
    for i, word in enumerate(row_words):
        out[word[0][0]][word[0][1]:word[1][1] + 1] = list(guess[i])

    out = [''.join(row) for row in out]
    out = twistgrid(out)

    out = [list(row) for row in out]
    for i, word in enumerate(col_words):
        out[word[0][1]][word[0][0]:word[1][0] + 1] = list(guess[i + len(row_words)])
    out = [''.join(row) for row in out]
    out = twistgrid(out)
    out = [''.join(row) for row in out]
    print('\n'.join(out))

    return out


# %% INTERVIEW PREP - Recursion and Backtracking: Crossword Puzzle
# Alternative solution using recursion as opposed to brute force that I
# used

import sys


def print_board(board):
    for row in board:
        print(''.join(row))


def get_poss_locs(board, word):
    # get possible locations
    poss_locs = []
    length = len(word)
    # horizontal possible location
    for i in range(10):
        for j in range(10 - length + 1):
            # j is starting index, can only get within (length-1) from
            # edge of board
            good = True
            for k in range(len(word)):
                if board[i][j + k] not in ['-', word[k]]:
                    good = False
                    break
                if not (
                        ((j == 0) or board[i][j - 1] == '+')
                        and ((j + length) == 10 or board[i][j + length] == '+')
                ):
                    # This ensures that only exact fit passes as 'good';
                    # condition above will pass words in blank spaces
                    # that are longer, which is not ideal.
                    good = False
                    break

            if good:
                poss_locs.append((i, j, 0))  # 0 is axis indicator

    # vertical possible location,where i is limited by length of word
    for i in range(10 - length + 1):
        for j in range(10):
            good = True
            for k in range(len(word)):
                if board[i + k][j] not in ['-', word[k]]:
                    good = False
                    break
                if not (
                        ((i == 0) or board[i - 1][j] == '+')
                        and ((i + length) == 10 or board[i + length][j] == '+')
                ):
                    # this ensures that only exact fit passes as 'good'
                    good = False
                    break
            if good:
                poss_locs.append((i, j, 1))  # 0 is axis indicator

    return poss_locs


def revert(board, word, loc):
    # revert move
    i, j, axis = loc
    if axis == 0:  # axis 0 is horizontal
        for k in range(len(word)):
            board[i][j + k] = '-'
    else:  # axis 1 is vertical
        for k in range(len(word)):
            board[i + k][j] = '-'


def move(board, word, loc):
    # write the word on board at specified loc
    i, j, axis = loc
    if axis == 0:
        for k in range(len(word)):
            board[i][j + k] = word[k]
    else:
        for k in range(len(word)):
            board[i + k][j] = word[k]


def solve(board, words):
    global solved
    global count
    count = 1
    if len(words) == 0:
        if not solved:
            print_board(board)
            solved = True
            print('\nSolved')
            return [''.join(row) for row in board]

    board = [list(line) for line in board]
    word = words.pop()
    pos_locs = [loc for loc in get_poss_locs(board, word)]
    print(f'{word} {pos_locs}')
    # breakpoint()
    for loc in pos_locs:
        move(board, word, loc)
        solve(board, words)
        if solved is True:
            # when we find a solution, no need to continue
            return [''.join(row) for row in board]

        revert(board, word, loc)
        # if we start reverting, it means pos_locs became empty
        # before we emptied words, and so we do not even enter the for
        # loop. The word is appended and we try next word, etc.
    words.append(word)
    count += 1
    print(f'appended {word}')

import time

start = time.time()
count = 0

for i in range(1000):
    words = 'ANDAMAN;MANIPUR;ICELAND;ALLEPY;YANGON;PUNE'
    solved = False
    words = words.split(';')
    board = ['+-++++++++',
             '+-------++',
             '+-++-+++++',
             '+-------++',
             '+-++-++++-',
             '+-++-++++-',
             '+-++------',
             '+++++++++-',
             '++++++++++',
             '++++++++++']
    solve(board, words)
end = time.time()
print(end - start)  # running 1000x is 2.62 seconds,much faster than mine


# %% ==========================================================================
# INTERVIEW PREP - Recursion and Backtracking - Recursive Digit Sum
# =============================================================================
# this passed
# it is important use multiplication of the sum instead
# of multiplying the string, because a very large string
# multiplied by a big number can overload memory; the end result using multiply
# is the same

def superDigit(n, k):
    if k == 1 and len(n) == 1:
        return n
    else:
        p_list = list(n)
        return superDigit(str(k * sum(list(map(int, p_list)))), 1)


print(superDigit('148', 3))

# %%===========================================================================
# INTERVIEW PREP - Miscellaneous -Time Complexity: Primality
# =============================================================================
import math


def primality(n):
    if n == 1:
        return 'Not prime'
    elif n == 2:
        return 'Prime'

    for i in range(2, math.floor(n**0.5) + 1):
        if n % i == 0 and n != 2:
            return 'Not prime'
    return 'Prime'


print(primality(4))


x = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256,
     289, 324, 361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841, 907]

for el in x:
    print(primality(el))


# %%===========================================================================
# INTERVIEW PREP - Miscellaneous - Friend Circle Queries
# =============================================================================
# this works
# This was classified as medium, but that is correct only if one knows
# about the theory behind this problem.  Otherwise, it's a mess trying
# to attachk with brute force of dictionaries/lists, as my examples below
# show.  My attempts work for some but not all test cases, as the union
# operation is not comprehensive.  Thus,
"""
    This problem requires a special approach
    Union Find Data structure
    https://www.hackerearth.com/practice/notes/disjoint-set-union-union-find/
    https://code.activestate.com/recipes/215912-union-find-data-structure/#:~:text=Union%20Find%20data%20structure%20(Python%20recipe)%20The%20union,able%20to%20combine%20sets%20quickly.%20Python,%20136%20lines
    https://www.semanticscholar.org/paper/The-Union-Find-Problem-Is-Linear-Zhang/4991f813ba6d316a7a77e9d4b61900616c5b86cd#:~:text=The%20union-find%20problem,%20also%20known%20as%20the%20disjoint,that%20its%20worst%20case%20time%20complexity%20is%20linear.
    log*n is log(log(......log(n)))
    (recursive log that many times till the number becomes 1
"""

"""
The problem statement is finding friend circles, but it's more
intuitive if we call it families, where a single person family is
his own parent.  And the multi-person families have one parent, which
facilitates storage in the library of the number of members in the
family.


"""


def init_parent_lookup(parent_lookup, x, y):
    """Key of dictionary is the member of family, value is parent.

    In case of single person family, parent is the individual, or
    key = value

    x , y are family members who may or may not already have an assigned
    parent; if not, they are their own paretnt
       which becomes the single parent is decided by maxCircle function
    """
    if x not in parent_lookup:
        parent_lookup[x] = x
        # single person family has himself as parent
        # maxCircle function decides
    if y not in parent_lookup:
        parent_lookup[y] = y


def init_member_counter(cc, x, y):
    """Family member counter.

    key is parent, value is # of people in family, including parent
    """
    if x not in cc:
        cc[x] = 1
    if y not in cc:
        cc[y] = 1


def get_parent(parent_lookup, x):
    while parent_lookup[x] != x:
        x = parent_lookup[x]
    return x


def maxCircle(queries):
    """Turn queries into families.

    Returns size of largest family after each query

    Variables
    ---------
    queries: list of lists, each q being [#, #] where a and b are two
                                          family members

    parent_lookup: key is family member, value is parent

    counter = key is parent, value is # members in family, including parent

    largest_family_count: type int, largest family size

    size_largest_family_so_far: type list of length len(queries)
                contains the value of largest_family_count after every
                query
    """
    parent_lookup = {}
    counter = {}
    largest_family_count = 0
    size_largest_family_so_far = []
    for q in queries:
        init_parent_lookup(parent_lookup, q[0], q[1])
        init_member_counter(counter, q[0], q[1])
#        breakpoint()
        parent1 = get_parent(parent_lookup, q[0])
        parent2 = get_parent(parent_lookup, q[1])
        if parent2 != parent1:
            if counter[parent1] > counter[parent2]:
                parent_lookup[parent2] = parent1
                counter[parent1] = counter[parent1] + counter[parent2]
            else:
                parent_lookup[parent1] = parent2
                counter[parent2] = counter[parent1] + counter[parent2]
            largest_family_count = max(largest_family_count,
                                       max(counter[parent1], counter[parent2]))
        size_largest_family_so_far.append(largest_family_count)
    print(size_largest_family_so_far)
    return size_largest_family_so_far


x = input().split('\n')
"""
1 2
3 4
1 3
5 7
5 6
7 4
"""
y = [list(map(int, x[i].split())) for i in range(6)]
maxCircle(y)
# %%===========================================================================
# INTERVIEW PREP - Miscellaneous - Friend Circle Queries
# =============================================================================
# Fails on 12 of 15 test cases - i.e., it only works on small number
# of queries, by accident


def maxCircle(queries):
    """Output list of max # in a friend circle after each handshake.

    Queries is a list of 2 dimensional lists
    [[a,b], [c,d], ..]
    There are up to 10_000 queries
    a, b are integers that represent people
    a, b, etc. <=10e9
    """
    circles = {}
    max_circle = 0
    max_each_query = []
    for i, friends in enumerate(queries):
        if friends[0] in circles.keys() and friends[1] in circles.keys():
            circles[friends[0]].update(circles[friends[1]])
            circles[friends[1]].update(circles[friends[0]])

        elif friends[0] in circles.keys() and friends[1] not in circles.keys():
            circles[friends[0]].add(friends[1])
            circles[friends[1]] = circles[friends[0]]
        elif friends[0] not in circles.keys() and friends[1] in circles.keys():
            circles[friends[1]].add(friends[0])
            circles[friends[0]] = circles[friends[1]]
        elif (friends[0] not in circles.keys()
              and friends[1] not in circles.keys()):
            circles[friends[0]] = set(friends)
            circles[friends[1]] = set(friends)
        max_circle = max(len(circles[friends[1]]), len(circles[friends[0]]),
                         max_circle)
        max_each_query.append(max_circle)
        print(circles)
    return max_each_query


maxCircle(y)


# %%===========================================================================
# INTERVIEW PREP - Miscellaneous - Friend Circle Queries
# =============================================================================
# This works for 7 of 15 Test cases, but fails for 8 on Runtime Errors
# See theory behind the Union-Find problem and the solution above

# This approach won't work for very long queries with few overlaps
# because we would have to check every item of
# the circles every time in worst case, which would approach O(n^2) complex
# where n is the number of queries

def maxCircle(queries):
    circles = []
    max_circle = 0
    max_each_query = []

    for i, friends in enumerate(queries):
        k_circle = -1
        m_circle = -1
        for k, circle in enumerate(circles):
            if friends[0] in circle:
                k_circle = k
                break
        for m, circle in enumerate(circles):
            if friends[1] in circle:
                m_circle = m
                break
        # 4 cases to deal with
        if k_circle == -1 and m_circle == -1:  # neither in a circle
            circles.append(set(friends))
            max_circle = max(max_circle, 2)
        elif k_circle == -1 and m_circle != -1:  # only friends[1] in a circle
            circles[m_circle].add(friends[0])
            max_circle = max(max_circle, len(circles[m_circle]))
        elif k_circle != -1 and m_circle == -1:  # only friends[0] is in
            circles[k_circle].add(friends[1])
            max_circle = max(max_circle, len(circles[k_circle]))
        elif k_circle != -1 and m_circle != -1 and k_circle != m_circle:
            # both in a circle
            circles[k_circle].update(circles[m_circle])  # join the 2
            max_circle = max(max_circle, len(circles[k_circle]))
            del circles[m_circle]  # and delete one of them

        max_each_query.append(max_circle)
        print(circles)
    return max_each_query


# %%===========================================================================
# INTERVIEW PREP - Miscellaneous - Maximum Xor
# =============================================================================
# 13/16 time out; 3 are correct
# there must be trick with XOR that speeds up so we do not need to go
# through
"""
one case that didn't pass had 74672 queries and arr of size 14941,
both arr and queries containing large numbers, which equates to
75K x 15K xor operations (1.125 billion) at worst

we want the two largest query and val such that they don't have same
upper bits, otherwise they'll negate

we should probably try to create a lookup of values to test from array,
where they are sorted by their top bit
upper limit of array value is 1e9, which is 30 bits binary
so

lookup = {30: [arr[a:-1]], 29:[arr[b:a], etc.]

          this way we can test just the subset of array with query
          that doesn't have the same top bit, because they would
          cancel.
this seems like a lot of work, probably a more efficient trick
"""


def maxXor(arr, queries):
    i = 0
    result = []
    while i < len(queries):
        maxval = -math.inf
        query = queries[i]
        for j, val in enumerate(arr):
            maxval = max(maxval, query ^ val)
        result.append(maxval)
        i += 1
    return result


# %%

from collections import defaultdict
subarrays = defaultdict(lambda: [])

# maybe we only need to keep index ranges of largest numbers
indexes = []

powers = [0, 0, 1, 3, 20, 21, 21, 25, 25, 26, 26]
base = [0, 5, 6, 3, 44, 25, 1, 65, 35, 155, 226]
arr = [2**powers[i] + base[i] for i in range(len(powers))]
arr.sort()
power = 19
j = 0
indexes = []
while j < len(arr):
    print('\nthreshold', 2**power)
    print(j, arr[j], powers[j], 'POWER', int(math.log2(arr[j])))
    # breakpoint()
    if int(math.log2(arr[j])) > power:
        print('help')
        power = int(math.log2(arr[j]))
        indexes.append((j, power))
        print(j, power, 'added')
    j += 1


# %%
for i in range(31):
    if i % 5 == 0:
        print('\n')
    print(f'2^{i:2d}: {2**i: >31,d}')
    print(f'2^{i:2d}: {2**i: >31b}')

print(f'max value involved {10**9: 12,d}')
"""

2^ 0:                                1
2^ 0:                                1
2^ 1:                                2
2^ 1:                               10
2^ 2:                                4
2^ 2:                              100
2^ 3:                                8
2^ 3:                             1000
2^ 4:                               16
2^ 4:                            10000


2^ 5:                               32
2^ 5:                           100000
2^ 6:                               64
2^ 6:                          1000000
2^ 7:                              128
2^ 7:                         10000000
2^ 8:                              256
2^ 8:                        100000000
2^ 9:                              512
2^ 9:                       1000000000


2^10:                            1,024
2^10:                      10000000000
2^11:                            2,048
2^11:                     100000000000
2^12:                            4,096
2^12:                    1000000000000
2^13:                            8,192
2^13:                   10000000000000
2^14:                           16,384
2^14:                  100000000000000


2^15:                           32,768
2^15:                 1000000000000000
2^16:                           65,536
2^16:                10000000000000000
2^17:                          131,072
2^17:               100000000000000000
2^18:                          262,144
2^18:              1000000000000000000
2^19:                          524,288
2^19:             10000000000000000000


2^20:                        1,048,576
2^20:            100000000000000000000
2^21:                        2,097,152
2^21:           1000000000000000000000
2^22:                        4,194,304
2^22:          10000000000000000000000
2^23:                        8,388,608
2^23:         100000000000000000000000
2^24:                       16,777,216
2^24:        1000000000000000000000000


2^25:                       33,554,432
2^25:       10000000000000000000000000
2^26:                       67,108,864
2^26:      100000000000000000000000000
2^27:                      134,217,728
2^27:     1000000000000000000000000000
2^28:                      268,435,456
2^28:    10000000000000000000000000000
2^29:                      536,870,912
2^29:   100000000000000000000000000000


2^30:                    1,073,741,824
2^30:  1000000000000000000000000000000
2^31:                    2,147,483,648
2^31: 10000000000000000000000000000000
max value involved  1,000,000,000
"""
# %%
# INTERVIEW PREP - Miscellaneous - Maximum Xor
# the full explanation of these types of problems can be found at
# https://www.geeksforgeeks.org/maximum-xor-of-two-numbers-in-an-array/
# the use of a mask for shielding certain bits and a few bit operators
# are new to me
# the brute force approach I tried is O(n^2) complexity, but the optimum
# solution is O(N log M), where M is the max number in the array

# Function to return the maximum xor


def max_xor(arr, n):
    maxx = 0
    mask = 0

    se = set()

    for j in range(30, -1, -1):
        # set the i'th bit in mask
        # like 100000, 110000, 111000..
        mask |= (1 << j)
        # left shift 1 i times
        # 1<<2 means 4 or 0b100
        # |= is the bitwise OR operation, meaning
        # mask = mask | (0b00)
        # I was unaware of this shortcut before this problem
        newMaxx = maxx | (1 << j)
        # each iteration makes the precision finer for the search of
        # what the real maximum xor is
        print(maxx, bin(newMaxx))
        prev = len(se)

        for i in range(n):

            # Just keep the prefix till
            # i'th bit neglecting all
            # the bit's after i'th bit
            se.add(arr[i] & mask)  # adds element to
            # print(len(se))
            if len(se) > prev:
                print(f'we added {len(se)-prev} elements to set'
                      f' using mask {bin(mask)}:{se}')
                prev = len(se)
            else:
                print('nothing added, i=', i)
        print('---')
        for prefix in se:

            # find two pair in set
            # such that a^b = newMaxx
            # which is the highest
            # possible bit can be obtained
            print(f'prefix {prefix} newMaxx {newMaxx} se{se}')
            if (newMaxx ^ prefix) in se:
                # property of xor:
                # if this is true: A^B = C
                # then it's also true that : B^C = A
                # A^C = B
                # basically, the XOR connects the three numbers
                #
                #
                print(f'found {newMaxx^prefix} in {se} newMaxx {newMaxx} ')
                maxx = newMaxx
                pair = (prefix, newMaxx ^ prefix)
                # we set the maxx to the newMaxx, which
                # is the
                break

        # clear the set for next
        # iteration
        se.clear()
    print(pair)
    print(bin(pair[0]), bin(pair[1]))
    return maxx


# Driver Code
arr = [25, 10, 2, 8, 5, 3]
n = len(arr)
print(max_xor(arr, n))
# correct answer is 28

# %% operators
"""

Operator	Example		Equivalent to
--------    --------    -------------
*=			x *=  5		x = x *  5
/=			x /=  5		x = x /  5
%=			x %=  5		x = x %  5
//=			x //= 5		x = x // 5
**=			x **= 5		x = x ** 5

&=			x &=  5		x = x &  5
|=			x |=  5		x = x |  5
^=			x ^=  5		x = x ^  5
>>=			x >>= 5		x = x >> 5
<<=			x <<= 5		x = x << 5

"""
# %% End
"""
arr = [0, 1, 2, 2]
queries = [3, 7, 2]
"""
arr = [1, 3, 5, 7]
queries = [17, 6]


def maxXor(arr, queries):
    maxes = []
    arr_set = set(arr)
    arr = list(arr_set)
    arr.sort(reverse=True)
    n = len(arr) + 1
    for query in queries:
        maxx = 0
        mask = 0
        arr.append(query)
        # print(arr)
        for k in range(31, -1, -1):
            se = set()
            mask |= (1 << k)
            newMaxx = maxx | (1 << k)
            for i in range(n):
                se.add(arr[i] & mask)  # adds prefix to
            # if k < 3:
            #      breakpoint()
            # print(bin(mask), 'newmax', newMaxx, '=', bin(newMaxx), maxx, se, k)
            for prefix in se:
                if (newMaxx ^ prefix) in se:
                    # print(newMaxx ^ prefix)
                    maxx = newMaxx
                    break

            se.clear()
        maxes.append(maxx)
        arr.pop()
    # print(pair)
    # print(bin(pair[0]), bin(pair[1]))
    print(maxes)
    return maxes


# %%
import numpy as np
queries_len = 26867
arr_len = 71132
input_file = r'.\hackerrank_test_cases\interview_prep'
arr_data = input_file + r'\maxxor.txt'
query_data = input_file + r'\maxxor_queries.txt'
output_data = input_file + r'\maxxor_output.txt'
arr = np.loadtxt(arr_data, delimiter=' ', dtype='int32')
queries = np.loadtxt(query_data, delimiter='\n', dtype='int32')
output = np.loadtxt(output_data, delimiter='\n', dtype='int32')
maxes = maxXor(arr, queries[0:5])

maxXor(arr.tolist(), list(queries[0:5].tolist()))

# maxXor(arr.tolist(), list(queries[0:5].tolist()))


# %%
class Rectangle():
    def __init__(self, width, length):
        self.width = width
        self.length = length

    def area(self):
        return self.length * self.width


class Square(Rectangle):
    # all squares are rectangles, but the behavior here is nonintuitive
    # because we can later change square's width to something different
    # and then it won't be a square anymore, because length doesn't
    # automatically get changed as well
    def __init__(self, sidelength):
        return super().__init__(sidelength, sidelength)
