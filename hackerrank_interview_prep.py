# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 16:33:40 2020

@author: Trader
"""

# %% INTERVIEW PREP
"""
- Tips and Guidelines (just hint videos)

69 Hackerrank Challenges broken into 15 subjects
- 4 Warmups                      4 / 4  x
- 5 Arrays                       4 / 5
- 5 Dictionaries and Hashmaps    4 / 5
- 5 Sorting                      4 / 5
- 5 String manipulation          4 / 5

- 5 Greedy algorithms            4 / 5
- 7 Search                       4 / 7  # 3 more tomorrow
- 4 Dynamic Programming          1 / 4
- 6 Stacks and Queues             -

- 5 Graphs                        -
- 5 Trees                         -
- 5 Linked Lists                  -
- 4 Recursion and Backtracking    -
- 4 Miscellaneous                1 / 4
===
- 69

30 / 69 solved so far

5 per day = 25 this week
41 by end of week
"""

# %% INTERVIEW PREP - Warmups - Sock Merchant
from collections import Counter

n = int(input())
a = input().split()
y = Counter(a)

pairs = 0
for key, value in y.items():
    pairs += value//2

print(pairs)


def sockMerchant(n, ar):
    y = Counter(ar)
    pairs = 0
    for key, value in y.items():
        pairs += value//2
    return pairs


# %% INTERVIEW PREP - Warmups - - Jumping on the Clouds
"""
test.

There is a new mobile game that starts with consecutively numbered clouds.
Some of the clouds are thunderheads and others are cumulus. The player can
jump on any cumulus cloud having a number that is equal to the number of the
current cloud plus 1 or 2.
The player must avoid the thunderheads. Determine the minimum number of
jumps it will take to jump from the starting postion to the last cloud. It
is always possible to win the game.

For each game, you will get an array of clouds numbered
if they are safe or if they must be avoided.
"""


def jumpingOnClouds(c):
    jumps = 0
    position = 0
    n = len(c)
    for i in range(n):
        print(i, position)
        if position == n-1:  # we've already reached end
            print('position ==')
            pass
        elif c[i] == 1:
            print('this is a 1, which we cannot rest on')
            pass
        elif i < position:  # we already moved beyond this cloud
            print('i < position')
            pass
        elif (((i+2) <= (n-1)) and (c[i+2] == 0)):
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
    n_strings = n//len(s)
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

# %% INTERVIEW PREP - Arrays - New Year Chaos


def minimumBribes(q):
    # initialize the number of moves
    moves = 0
    #
    # decrease Q by 1 to make index-matching more intuitive
    # so that our values go from 0 to N-1, just like our
    # indices.  (Not necessary but makes it easier to
    # understand.)
    Q = [P-1 for P in q]
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
        for j in range(max(P-1, 0), i):
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
    correct_seq = [1+i for i in range(len(arr))]
    for i in range(len(arr)-1):
        val1 = arr[i]
        if val1 != correct_seq[i]:
            min_swaps += 1
            for j in range(i+1, len(arr)):
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
    ref_arr = [i+1 for i in range(len(arr))]
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

            print('-------- Swap %d Complete --------' % swaps)
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
# this works for all but 2 test cases due to timeout
# both had 100000 queries
# the query 3 is the problem; looking up takes too long
# need to do a counter, perhaps

def freqQuery(queries):
    outputs = []
    integers = {}
    for i in range(len(queries)):
        if queries[i][0] == 1:
            # add occurrence of integer to integers
            integers[queries[i][1]] = integers.get(queries[i][1], 0) + 1
        elif queries[i][0] == 2:
            # delete occurrence of integer to integers
            integers[queries[i][1]] = max(0, integers.get(queries[i][1], 0)-1)
        elif queries[i][0] == 3:
            # find how many intengers have the given frequency
            if queries[i][1] in integers.values():
                outputs.append(1)
            else:
                outputs.append(0)
    print(integers)
    return outputs


queries = [(3, 4), (2, 1003), (1, 16), (3, 1)]
print(freqQuery(queries))

# %% INTERVIEW PREP - Dictionaries and Hashmaps: Frequency Queries
from collections import Counter
# only one wrong test case answer for this code; 100,000 queries
# but this outputs identical results to the one given by the testcase
# not sure where I went wrong


def freqQuery(queries):
    outputs = []
    integers = {}
    freqs = Counter()
    for i in range(len(queries)):
        if queries[i][0] == 1:
            # add occurrence of integer to integers
            newfreq = integers.get(queries[i][1], 0) + 1
            integers[queries[i][1]] = newfreq
            freqs[newfreq] += 1
            if newfreq > 1:
                freqs[newfreq - 1] -= 1
            # no need to update freqs[0] or freqs[negative#]

        elif queries[i][0] == 2:
            # delete occurrence of integer to integers
            update_freq = max(0, integers.get(queries[i][1], 0)-1)
            integers[queries[i][1]] = update_freq
            if update_freq > 0:
                freqs[update_freq] += 1
                freqs[update_freq+1] -= 1
        elif queries[i][0] == 3:
            # find how many intengers have the given frequency
            if freqs[queries[i][1]] > 0:
                outputs.append(1)
            else:
                outputs.append(0)
    print(integers)
    return outputs

import pandas as pd

data = pd.read_csv('input10.txt', sep='\t')
out_given = pd.read_csv('output10.txt')
q = data['q'].values.tolist()
num = data['num'].values.tolist()

queries = []
for i in range(100000):
    queries.append((q[i], num[i]))

out_calc = freqQuery(queries)
datalen = len(out_calc)
print(datalen, 'datalength')
comp = pd.DataFrame(out_given['out'][0:datalen])
comp['out_calc'] = pd.Series(out_calc)

comp['comp'] = comp['out'] == comp['out_calc']
comp.tail(50)

sum(comp['comp'])
print(datalen)
# %% INTERVIEW PREP - Dictionaries and Hashmaps: Frequency Queries
# this works - was provided


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

            # the counter have that new freq must be increased as well
            cnt[freq[q[1]]] += 1

        elif q[0] == 2:
            # no need to do anything if the frequency of integer q[i] is 0
            if freq[q[1]] > 0:
                # the counter of integers having that freq is decreased
                cnt[freq[q[1]]] -= 1

                # then we update the frequency of that integer
                freq[q[1]] -= 1

                # then we increase count of integers with that new frequency
                cnt[freq[q[1]]] += 1
        else:
            # only output 1 if the # of integers with frequency q[1] is >0
            if cnt[q[1]] > 0:
                arr.append(1)
            else:
                arr.append(0)
    return arr


# %% INTERVIEW PREP - Dictionaries and Hashmaps: Frequency Queries
import pandas as pd

data = pd.read_csv('input test5.txt', sep=' ')
out = pd.read_csv('output test5.txt')
q = data['query'].values.tolist()
num = data['num'].values.tolist()

queries = []
for i in range(250):
    queries.append((q[i], num[i]))


def freqQuery(queries):
    outputs = []
    freq = {}
    integers = {}
    for i in range(len(queries)):
        q_type = queries[i][0]
        val_or_freq = queries[i][1]
        if q_type == 1:
            # add occurrence of integer to integers
            print('q_type', q_type)
            print('val or freq', val_or_freq)
            print('ints', integers)
            newfreq = integers.get(val_or_freq, 0) + 1
            integers[val_or_freq] = newfreq
            freq[newfreq-1] = freq[newfreq-1] -1
            freq[newfreq] = freq.get(newfreq, 0) + 1
            print('ints', integers)
            print('---')
        elif q_type == 2:
            print('q_type', q_type)
            print(val_or_freq)
            print(integers)
            try:
                newfreq = max(0, integers[val_or_freq]-1)
                integers[val_or_freq] = newfreq
                freq[newfreq] += 1
                freq[newfreq+1] -= 1
            except Exception:  # if exception, it's a useless query
                pass
        elif q_type == 3:
            # find how many intengers have the given frequency

            try:
                if freq[queries[i][1]]:  # append 1 if there is a True for this freq
                    outputs.append(1)
            except Exception:  # if exception, it's a useless query
                outputs.append(0)
    return outputs, integers, freq

print(freqQuery(queries))
outputs, integers, freq = freqQuery(queries)
# %% INTERVIEW PREP - Dictionaries and Hashmaps: Frequency Queries
# Here we see that when we have a queries list of length 10,000, the three
# output times from this code are as follows
"""
query_1 total: 0.03
query_2 total: 0.0
query_3 total: 0.13
query_3 =  4.0 x more than query_1 (although some runs were up to 150x)
"""
import time

query_1 = 0
query_2 = 0
query_3 = 0


outputs = []
freq = {}
integers = {}
for i in range(len(queries)):
    if queries[i][0] == 1:
        start_time = time.time()
        # add occurrence of integer to integers
        newval = integers.get(queries[i][1], 0) + 1
        integers[queries[i][1]] = newval
        freq[newval] = True
        end_time = time.time()
        duration = end_time - start_time
        query_1 += duration
    elif queries[i][0] == 2:
        start_time = time.time()
        # delete occurrence of integer to integers
        # don't add zero if not already in dictionary
        try:
            integers[queries[i][1]] -= 1
            pass
        except Exception:
            pass
        end_time = time.time()
        duration = end_time - start_time
        query_2 += duration
    elif queries[i][0] == 3:
        # find how many intengers have the given frequency
        start_time = time.time()

        try:
            if freq[queries[i][1]]:
                # append 1 if there is a True for this freq
                outputs.append(1)
        except Exception:  # if exception, it's a useless query
            outputs.append(0)
        end_time = time.time()
        duration = end_time - start_time
        query_3 += duration

print('query_1 total:', round(query_1, 2))
print('query_2 total:', round(query_2, 2))
print('query_3 total:', round(query_3, 2), '\nquery_3 = ',
      round(query_3 / query_1, 0), 'x more than query_1')

# %% INTERVIEW PREP - Dictionaries and Hashmaps :Sherlock and Anagrams
# what's the point of this one?
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
            k_denom = math.factorial(3)*(math.factorial(n-3))
            trips = trips + int(math.factorial(n)/k_denom)

    else:
        for k, v in enumerate(mydict):
            trips = trips + len(mydict.get(v, [])) \
                * len(mydict.get(v*r, [])) * len(mydict.get(v*r*r, []))
            # print(k, v, trips)
            # print(mydict)
            # print('--')
    return trips


countTriplets(list(map(int, '1 3 9 9 27 81'.split())), 3)

print(countTriplets(list(map(int, '1 1 1 1 1 10 10 10 10 10'.split())), 1))

print(countTriplets(list(map(int, '1 1 1 1 1 10 10 10 10 10'.split())), 1))

math.factorial(100000)/math.factorial(3)/math.factorial(100000-3)
# %% I
count = 0
for i in range(1, 100000):
    count = count + i * (100000 - 1 - i)
print(count)

# this is the correct answer for a series with r=1, and 100,000 equal entries
# example of a series of 10 numbers, all the same, r=1

# 5 5 5 5 5 5 5 5 5 5
# 1: center point of triplet is i=1 or 2nd 5; there is 1 5 to left, 8 to right
#     which equates to 1*8 triplets that can have 2nd 5 as middle
#     which is why we count the range 1,#_in_array
# 2nd 5 is middle:   1*8  = 8 additional
# 3rd 5 is middle:   + 2 5's to left * 7 5's to right = 14 additional
# 9th 5 is middle:   + 8 5's to left * 1 5 to right = +8
count = 0
for i in range(1, 10):
    count = count + i * (10-1-i)
print(count)
120 == 8 + 2*7 + 3*6 + 4*5 + 5*4 + 18 + 14 + 8

# %% INTERVIEW PREP - Dictionaries and Hashmaps :Sherlock and Anagrams
from collections import Counter

#
# THIS WORKS
#

def countTriplets(arr, r):
    a = Counter(arr)
    b = Counter()
    n_triples = 0
    for element in arr:
        j = element // r    # j = left most element of a triple
        k = element * r     # k is the right element of triple
        a[element] -= 1     # we consider each element of arr once as being in
        # middle of a triple,so we reduce its freq in a

        # if j is not an element in Counter b, then 'if b[j] evaluates as False
        # not element%r means we only add to n_triples when element
        # is divisible by r
        print(arr)
        print(a, b)
        if b[j] and a[k] and not element % r:
            n_triples += b[j] * a[k]
            print('triple exists')
            print('1/r to left of element:', b[j])
            print('*r to right of element:', a[k])
        b[element] += 1
        print('new triples:', n_triples)
        print(b)
        print('----')
    return n_triples


print(countTriplets(list(map(int, '1 1 1 1 1 10 10 10 10 10'.split())), 1))

# %% INTERVIEW PREP - Dictionaries and Hashmaps : Ransom Note
# m, n = list(map(int, input().split()))  # length magazine, length note
# magazine = input().split()  # magazines strings
# note = input().split()  # note strings

# there are NOT sufficient occurrences of exact word matches in magazine to
# write note
magazine = 'give me one grand dnarg today night'
note = 'give me grand grand'

# there are sufficient occurrences of exact word matches in mag2 to write
# note2
mag2 = 'give me one grand one one two'
note2 = 'give me one grand one'

from collections import Counter


def checkMagazine(magazine, note):
    if (Counter(note2.split()) - Counter(mag2.split())) == {}:
        print('Yes')
    else:
        print('No')


checkMagazine(magazine, note)

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
            elif b[i+1] < b[i]:
                bigger = b[i]
                lower = b[i+1]
                b[i] = lower
                b[i+1] = bigger
                swaps += 1
                this_round_swaps += 1
    print('Array is sorted in %d swaps.' % swaps)
    print('First Element: %d' % b[0])
    print('Last Element: %d' % b[-1])
#     print('list', a)
#     print('sorted list', b)
#     print(this_round_swaps)


# %% INTERVIEW PREP - Sorting: Fraudulent Activity Notifications
# times out due to not allowing use of statistics package; they want a lowlevel
# calcuation of median function for more timeefficient calc when getting big
# data

# frequency table / dictionary solution works well, but it does so since
# we know prices only go up to 201, so we can arrive at the median from the
# bottom up by doing searching and then updating along the way by adding new
# prices of each new day and eliminating oldprices from the meidan window


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
    freq = {}
    notify = 0


    def find(idx):
        total_count = 0
        for i in range(201):
            if i in freq:
                total_count = total_count + freq[i]
            if total_count >= idx:
                return i
    for i in range(len(expenditure) - 1):
        if expenditure[i] in freq:
            freq[expenditure[i]] += 1
        else:
            freq[expenditure[i]] = 1
        # print(f"i: {i},val: {expenditure[i]}, freq: {freq}")
        if i >= d - 1:
            if d % 2 == 0:
                median = (find(d // 2) + find(d // 2 + 1)) / 2
            else:
                median = find(d/2)
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

# Complete the maximumToys function below.
def maximumToys(prices, k):
    prices.sort()
    allsum = 0
    maxtoys = 0
    for i in range(len(prices)):  # add or 'buy' each new toy if still have $
        if allsum + prices[i] <= k:
            allsum = allsum + prices[i]
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

Alice is taking a cryptography class and finding anagrams to be very useful.
We consider two strings to be anagrams of each other if the first string's
letters can be rearranged to form the second string. In other words, both
strings must contain the same exact letters in the same exact frequency
For example, bacdc and dcbac are anagrams, but bacdc and dcbad are not.

Alice decides on an encryption scheme involving two large strings where
encryption is dependent on the minimum number of character deletions
required to make the two strings anagrams. Can you help her find this number?

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
We delete the following characters from our two strings to turn them into
anagrams of each other:
    Remove d and e from cde to get c.
    Remove a and b from abc to get c.
We must delete characters to make both strings anagrams, so we print on a
new line.
"""

# Complete the makeAnagram function below.


def makeAnagram(a, b):
    # first process both strings into count of each letter of alphabet
    # numbersa is 26 long list of numbers, each # representing count of a-z
    # in string a
    numbersa = [0]*26
    numbersb = [0]*26
    for i in a:
        numbersa[ord(i) - 96 - 1] += 1
    for i in b:
        numbersb[ord(i)-96-1] += 1

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

You are given a string containing characters A and B only. Your task is to
change it into a string such that there are no matching adjacent characters.
To do this, you are allowed to delete zero or more characters in the string.

Your task is to find the minimum number of required deletions.

For example, given the string s=AABAAB, remove an A at positions 0 and 3 to
make ABAB in deletions.

Function Description
Complete the alternatingCharacters function in the editor below.
It must return an integer representing the minimum number of deletions to
make the alternating string.

alternatingCharacters has the following parameter(s):
    s: a string
Input Format
The first line contains an integer q, the number of queries.
The next Q lines each contain a string s.

Constraints
Each string will consist only of characters A and B

Output Format
For each query, print the minimum number of deletions required on a new line.

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
        if ((keys_[0] == 1) and len(values_[0]) == 1) or (keys_[1] == 1) and len(values_[1]) == 1:
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
THIS WORKS!!!!!!!!!!!!!
yeaaaaaaaaaaaaaaaaaaaaaaaaaaaaahhhhhhhhhhhhhhhhhhhhhh
"""
from itertools import groupby

x = "aaaabbbabccdd"
y = "bcbcbaacacbccacbbbcbbbaaacccaaabbcaacbbbcbaaabbcbcbbabbbccbccacbbababcaccbbbabccccccbacacbbcbbabcbccacbaaccccbcaaaabccccaba"""


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
    for i in range(1, len(groups)-1):
        if groups[i][1] == 1 and groups[i - 1][0] == groups[i + 1][0]:
            case2 += min(groups[i - 1][1], groups[i + 1][1])
    return int(case1 + case2)


print(substrCount(len(x), x))

z = input()
print(substrCount(len(z), z))


# %% INTERVIEW PREP - String Manipulation: Common Child
# try groupby
# A string is said to be a child of a another string if it can be formed by
# deleting 0 or more characters from the other string. Given two strings of
# equal length, what's the longest string that can be constructed such that
# it is a child of both?

# Length of strings is 5000 or less so running through string a few times
# should not be problem with time

# my approach won't work; this is a common algo problem for which there are
# canned approach
# come back to this later

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

    min_diff = min([abs(combo[0]-combo[1]) for combo in combos])
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
        min_diff = min(abs(arr[i] - arr[i+1]), min_diff)
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
    unfairness = arr[k-1] - arr[0]
    for i in range(k, len(arr)):
        unfairness = min(arr[i] - arr[i-k+1], unfairness)
    return unfairness

# %% INTERVIEW PREP - Greedy Algorithms: Greedy Florist


def getMinimumCost(k, c):
    c.sort(reverse=True)
    n = len(c)  # how many flowers
    # k = # friends
    cost = 0
    for i in range(n):
        cost += (i//k + 1) * c[i]
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

# %% INTERVIEW PREP - Search: Hash Tables: Ice Cream Parlor
# search is best done using bounds and triangulating
# we can go through the prices array twice, first to create the dictionary
# and second to check if complement to current element is in dictionary
# Order 2n of time required to calculate


def whatFlavors(cost, money):
    costs = {}
    for i in range(len(cost)):
        costs[cost[i]] = costs.get(cost[i], []) + [(i+1)]
#    print(costs)
    for i in range(len(cost)-1):

        # if cost and its complement are both in dict, choose them
        # if money == 2*cost[i], then cost[i] must appear twice in dictionary
        # value, ie, length must be 2
        # EX: money = 10, cost=5, then costs[5] = [5,5], or costs must
        # have entry {5:[5,5], ...}
        # if 5:[5], then the two friends cannot spend all the money, because
        # only one ice cream costs $5
        i_of_complement = costs.get(money - cost[i], [])
        print(costs)
        if len(i_of_complement) == 1 and cost[i] != money - cost[i]:
            result = [i + 1, costs[money - cost[i]][0]]
            result.sort()
            result = str(result[0]) + ' ' + str(result[1])
            print(result)
            return
        elif len(i_of_complement) == 2 and cost[i] == money - cost[i]:
            # two occurrences of same price adding up to money
            result = [i + 1, costs[money - cost[i]][1]]
            result.sort()
            result = str(result[0]) + ' ' + str(result[1])
            print(result)
            return
        else:
            pass  # only one unique solution, so no other possible


cost = [2, 2, 4, 3]
costs = whatFlavors(cost, 4)


# %% INTERVIEW PREP - Search: Minimum Time Required - 1 of 5 inccorrect

# this works but times out for several cases
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
            production += mindays//m
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
        production = sum([mindays//m for m in machines])
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
        production = sum([(mindays//item[1])*mach[item[1]] for item in enumerate(mach)])
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
        production = sum([(mindays // item[1]) * mach[item[1]] \
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
    mindays = min_days-1
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
# correctstrategy is to divide and conquer ie., find min bound and max bound
# and iteratively move closer to the value that satisfies the solution
# This approach avoids needing to calculate the sum of production for a
# large number of machines. This is the general approach for search - we
# trinagulate rather than do brute force.

# THIS WORKS!
import math
from collections import Counter


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
    guess = min_days
    # first guess try theoretical minimum in

    # if min_days = 40, max_days = 41, ceiling of mean is ceil(81/2)=41
    # if min_days = 41, max_days = 42, ceiling of mean is ceil(83/2)=42
    # if there is an integer in between, then there is another guess to try

    while guess != max_days and max_days > min_days+1:
        mach_production = [mach[n_days] * (guess // n_days) for n_days in days]
        production = sum(mach_production)
        print('\nMachine production', mach_production)
        print('guess:', guess, ', mindays:', min_days, ', maxdays:', max_days,
              ', Total Production:', production)
        if production < goal:
            old_guess = guess
            guess = math.ceil(sum([guess, max_days])/2)
            min_days = old_guess
            print('Production < Goal')
            print('old_guess-->', old_guess, 'new_min', min_days,
                  'Calculated new guess-->', guess)
        elif production > goal:
            old_guess = guess
            guess = math.ceil(sum([min_days, guess])/2)
            max_days = old_guess
            print('Production > Goal')
            print('old_guess-->', old_guess, 'new_max', max_days,
                  'Calculated new guess-->', guess)
        else:
            old_guess = guess
            guess = math.ceil(sum([min_days, guess])/2)
            max_days = old_guess
            print('Production == Goal')
            print('old_guess-->', old_guess, 'new_max', max_days,
                  'Calculated new guess-->', guess)

    print('---done---')
    return guess


print(minTime([1, 3, 10, 100], 60))

# %% # %% INTERVIEW PREP - Search: Minimum Time Required - CORRECT APPROACH
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

    while guess != max_days and max_days > min_days+1:
        mach_prod = [mach[n_days] * (guess // n_days) for n_days in days]
        production = sum(mach_prod)
        if production < goal:
            min_days = guess
            guess = math.ceil(sum([min_days, max_days])/2)
        elif production > goal:
            max_days = guess
            guess = math.ceil(sum([min_days, max_days])/2)
        else:
            max_days = guess
            guess = math.ceil(sum([min_days, max_days])/2)
    return guess


print(minTime([1, 3, 10, 100], 60))

# test cases
# print(minTime([4, 5, 6], 12))
# print(minTime([63, 2, 26, 59, 16, 55, 99, 21, 98, 65], 56))

# %% # %% INTERVIEW PREP - Search: Pairs
# that was easy
# only 2x through array, 2n

from collections import Counter


def pairs(k, arr):
    count = 0
    int_counter = Counter(arr)
    for integer in arr:
        if int_counter[integer + k]:
            count += 1
    return count


print(pairs(5, [0, 5, 10, 11, 16]))

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


# %% INTERVIEW PREP - Dynamic Programming: Max Sum Array
    # the array is 100,000 long at max, so we may have time problem
    # [1, 2, 3, 4, 5]-> has 4 master sets [1,3,5], [1,4] [2,4] [2,5]
    # [3, 5] is just a subset of indexed, so there is not any others
    # However, if there was a six, then [3, 6] would be another
    # if also 7, 8, then [3, 6, 8] would be the set
    # and then [4,7] would be another
    # from these we can eliminate numbers if that would increase max


# %% INTERVIEW PREP - Dynamic Programming: Max Sum Array
# a number of incorrect test cases. why do I always try the wrongapproach?

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
            max_even_off = max(max_even_off, max(arr2[i], 0) -
                               max(arr2[i-1], 0)
                               - max(arr2[i + 1], 0))

        elif i % 2 == 1:  # odd indices
            max_odd += max(0, arr2[i])
            max_odd_off = max(max_odd_off, max(arr2[i], 0) -
                              max(arr2[i-1], 0) -
                              max(arr2[i + 1], 0))

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
    if len(arr)>1: maxes[1] = max(arr[0], arr[1])
    i = 2
    while i < len(arr):
        maxes[i] = max(maxes[i-2], maxes[i-1], arr[i], arr[i] + maxes[i-2])
        i += 1

    result = maxes[len(arr)-1]
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
        print(i,':', a[i], b[j])
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
            A[i][j] = A[i-1][j]

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
        print(i,':i', a[i], i,':j', b[j])
        for j in range(1, len(b)):
            print(i,':i', a[i], j,':j', b[j])
            if a[i].upper() == b[j] and a[i].islower():
                A[i][j] = A[i-1][j-1] or A[i-1][j]
            elif a[i].upper() == b[j] and a[i].isupper():
                A[i][j] = A[i-1][j-1]
            elif a[i].upper() != b[j] and a[i].islower():
                A[i][j] = A[i-1][j]
            else:
                A[i][j] = False
    print(A)
    if A[len(a)-1][len(b)-1]:
        return "YES"
    return "NO"


print(abbreviation('AaBcd','ABC'))
# %% INTERVIEW PREP - Stacks and Queueus

# %% INTERVIEW PREP - Graphs:

# %% INTERVIEW PREP - Trees:

# %% INTERVIEW PREP - Linked Lists:

# %% INTERVIEW PREP - Recursion and Backgracking:

# %% INTERVIEW PREP - Miscellaneous































# %% End
