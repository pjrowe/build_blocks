"""Created on Mon Dec 21 16:33:40 2020.

@author: Trader
"""

# %% PROBLEM SOLVING BADGE
"""
Problem Solving (Algorithms)

6 Star - Gold  2200
5 Star - Gold   850

Subdomains
Algorithms
n   Category                  Done
--  --------                  -------
10  Warmup                    10 / 10 x
66  Implementation            49 / 66
48  Strings                   22 / 48
15  Sorting                   13 / 15
26  Search                    .  /
    --94/127                  -------

64  Graph Theory              . / 64
24  Greedy                    1 / 24
99  Dynamic Programming       . / 99
11  Constructive Algos        1 / 11
27  Bit Manipulation          1 / 27
    --3/225                   -------

11  Recursion                 . / 11
33  Game Theory               . / 33
 4  NP Complete               . /  4
 5  Debugging                 2 /  5
--  -2/ 53                    -------
//  99/405 so far

405 Problem Solving - Algos (hackerrank_problem_solving.py)
121 Problem Solving - Data Structures(see hackerrank_data_structures.py)
---
526?? why does Problem Solving have 563 challenges then?

as of 1/07/2021:  861/2200 points
as of 1/19/2021: 1701/2200 points
as of 1/24/2021:  105/ 563 challenges solved


"""
# =============================================================================
# %% Practice - Algorithms - Warmup - Birthday Cake Candles
# function that counts how many of the tallest candles are on cake
# =============================================================================
from collections import Counter

candles = [1, 1, 2, 4, 9, 5, 9, 7, 9]


def birthdayCakeCandles(candles):
    """birthdayCakeCandles."""
    count = Counter(candles)
    max_ = max(count)
    return count[max_]


# max of a counter gives the value that is greatest; then we lookup how many
# of them appear in teh Counter or candles list input

print(birthdayCakeCandles(candles))

# in this case, therea are 3 candles of 9 tall, so the answer is 3

# %% Practice - Algorithms - Warmup -

# most were quite easy, so no need to record them here


# %% Practice Algorithms: Implementation - Number Line Jumps

# we test to see if two lines will cross at an integer point of jumps
# the kangaroos starting position and length of jump represents a line
# x1 + n*v1 = position along y axis vs. n integer # jumps (x-axis)
# the lines will cross so long as x1>x2 and v1<v2 or vice versa, but the
# question is if the n value is an integer
# x1 + n*v1 = x2 + n*v2
# rearrange: n = (x1-x2)/(v1-v2); so long as this value has no remainder,
# n will be an integer; hence the % operator
def kangaroo(x1, v1, x2, v2):
    """kangaroo."""
    # lines cross at integer # jumps
    if x1 < x2 and v1 > v2 and (x2 - x1) % (v1 - v2) == 0:
        return 'YES'
    elif x1 > x2 and v1 < v2 and (x1 - x2) % (v2 - v1) == 0:
        return 'YES'
    else:
        return 'NO'

# %% Practice Algorithms: Implementation - Sequence Equation


def permutationEquation(p):
    """Calculate sequence."""
    xlookup = {}
    for i in range(len(p)):
        xlookup[p[i]] = i + 1
    result = [xlookup[xlookup[i]] for i in range(1, len(p) + 1)]
    print(result)
    return result


# %% Practice Algorithms: Implementation - Forming a Magic Square
# nxn matrix with values [1,n^2] is magic if sum of rows and diags and cols
# is always same constant...
# - will have one instance of each int in range
# - sum will be (n^2+1)*n/2
#     if n = 3, 10 * 3 / 2 = 15 will be magic constant

class Magic(object):

    pre = [[[8, 1, 6], [3, 5, 7], [4, 9, 2]],
           [[6, 1, 8], [7, 5, 3], [2, 9, 4]],
           [[4, 9, 2], [3, 5, 7], [8, 1, 6]],
           [[2, 9, 4], [7, 5, 3], [6, 1, 8]],
           [[8, 3, 4], [1, 5, 9], [6, 7, 2]],
           [[4, 3, 8], [9, 5, 1], [2, 7, 6]],
           [[6, 7, 2], [1, 5, 9], [8, 3, 4]],
           [[2, 7, 6], [9, 5, 1], [4, 3, 8]]
           ]

    def evaluate(self, s):
        totals = []
        for p in self.pre:
            total = 0
            for p_row, s_row in zip(p, s):
                for i, j in zip(p_row, s_row):
                    if not i == j:
                        total += max([i, j]) - min([i, j])
            totals.append(total)
        return min(totals)


def evaluate(s):
    """Calculate distance of given s matrix from all of the magic squares.

    Then calc minimum
    """
    # 5 is the only integer that can be placed in middle because all others
    # would result in excess ofthe magic number 15 in one ofthediagonals, cols,
    # or rows
    pre = [[[8, 1, 6], [3, 5, 7], [4, 9, 2]],
           [[6, 1, 8], [7, 5, 3], [2, 9, 4]],
           [[4, 9, 2], [3, 5, 7], [8, 1, 6]],
           [[2, 9, 4], [7, 5, 3], [6, 1, 8]],
           [[8, 3, 4], [1, 5, 9], [6, 7, 2]],
           [[4, 3, 8], [9, 5, 1], [2, 7, 6]],
           [[6, 7, 2], [1, 5, 9], [8, 3, 4]],
           [[2, 7, 6], [9, 5, 1], [4, 3, 8]]
           ]

    totals = []
    for p in pre:
        total = 0
        for p_row, s_row in zip(p, s):
            for i, j in zip(p_row, s_row):
                if not i == j:
                    total += abs(i - j)  # or += max([i, j]) - min([i, j])
        totals.append(total)
    return min(totals)


s = [[5, 3, 4], [1, 5, 8], [6, 4, 2]]
print(evaluate(s))

# %% Practice Algorithms: Implementation - Between Two Sets
# perhaps klunky, but oh well


def getTotalX(a, b):
    """getTotalX."""
    # find integers across sets where elements of a are its factors and
    # elements of b are evenly divided by such integers
    possible = [x for x in range(1, 101)]
    ints = set(possible)

    # a and b are two arrays
    print('Starting out with full set of solutions\n', ints)

    # eliminate integers that are not divisible by factors(elements of 'a')
    for divisor in a:
        for el2 in possible:
            if el2 % divisor != 0:
                # print('removing el2', el2, '/', divisor)
                try:
                    ints.remove(el2)
                except Exception:
                    pass
    print('\nRemaining possibilities after eliminating for factors in a',
          ints)

    # eliminate remaining integers that do not divide evenly into elements
    # of 'b'
    newit = list(ints)  # sets are not iterable
    for factor2 in newit:
        for num in b:
            if num % factor2 != 0:
                try:
                    ints.remove(factor2)
                except Exception:
                    pass
    print('\nIntegers that solve:', ints)

    return len(list(ints))


getTotalX([2, 3, 6], [12, 60])
print('\nSolution, i.e., # of integers:', getTotalX([1], [72, 48]))

# %% Practice Algorithms - Implementation - Breaking the Records


def breakingRecords(scores):
    """breakingRecords."""
    least_score = 0
    most_score = 0
    record_high = scores[0]
    record_low = scores[0]
    for i in range(1, len(scores)):
        if scores[i] > record_high:
            most_score += 1
            record_high = scores[i]
        elif scores[i] < record_low:
            least_score += 1
            record_low = scores[i]

    return [most_score, least_score]


x = [3, 4, 21, 36, 10, 28, 35, 5, 24, 42]
print(breakingRecords(x))


# %% Practice Algorithms - Implementation - Sub-array Division


def birthday(s, d, m):
    """birthday."""
    ways = 0
    for i in range(m, len(s) + 1):
        print(i - m, i, s[i - m:i])
        if sum(s[i - m:i]) == d:
            ways += 1
    return ways


print(birthday([2, 2, 1, 3, 2], 4, 2))


# %% Practice Algorithms - Implementation - Divisible Sum Pairs

from itertools import combinations


def divisibleSumPairs(n, k, ar):
    """divisibleSumPairs."""
    combos = list(combinations(ar, 2))
    result = 0
    for pair in combos:
        if sum(pair) % k == 0:
            result += 1
    return result


ar = [1, 3, 2, 6, 1, 2]
print(divisibleSumPairs(5, 3, ar))

# %% Practice Algorithms - Implementation -  migratoryBirds


def migratoryBirds(arr):
    """migratoryBirds."""
    max_freq = 0
    birds = {}

    for bird in arr:
        birds[bird] = birds.get(bird, 0) + 1
        if birds[bird] > max_freq:
            max_freq = birds[bird]
            most_freq_birds = [bird]
        elif birds[bird] == max_freq:
            most_freq_birds.append(bird)

    most_freq_birds.sort()
    print(birds)
    print(most_freq_birds)
    return most_freq_birds[0]


ar = [1, 3, 2, 6, 6, 6, 6, 1, 2, 2]
print(migratoryBirds(ar))


# %% Practice Algorithms - Implementation - Day of the Programmer


def dayOfProgrammer(year):
    """dayOfProgrammer."""
    Jan_to_August_days_less_Feb = 31 + 31 + 30 + 31 + 30 + 31 + 31

    if year == 1918:  # transition year from Julian to Gregorian in Russia
        feb_days = 15
        x = Jan_to_August_days_less_Feb + feb_days
        day = 256 - x
        return str(day) + '.09.' + str(year)
    elif year >= 1700 and year < 1918:  # Julian calendar
        if year % 4 == 0:
            # then a leap year
            feb_days = 29
            x = Jan_to_August_days_less_Feb + feb_days
            day = 256 - x
            return str(day) + '.09.' + str(year)
        else:  # not a leap year
            feb_days = 28
            x = Jan_to_August_days_less_Feb + feb_days
            day = 256 - x
            return str(day) + '.09.' + str(year)

    elif year >= 1919 and year <= 2700:  # Gregorian calendar
        if year % 400 == 0 or (year % 4 == 0 and year % 100 != 0):
            # then a leap year
            print('a leap year after 1919')
            # j F    m   APR    MAY ...August
            feb_days = 29
            x = Jan_to_August_days_less_Feb + feb_days
            day = 256 - x
            return str(day) + '.09.' + str(year)
        else:  # not a leap year
            feb_days = 29
            x = Jan_to_August_days_less_Feb + feb_days
            day = 256 - x
            return str(day) + '.09.' + str(year)


print(dayOfProgrammer(1800))
print(dayOfProgrammer(1917))
print(dayOfProgrammer(1984))

# %% Practice Algorithms - Implementation - Bill Division


def bonAppetit(bill, k, b):
    """bonAppetit."""
    fairsplit = (sum(bill) - bill[k]) / 2
    if fairsplit == b:
        print('Bon Appetit')
    else:
        print(int(b - fairsplit))


# %% # %% Practice Algorithms - Implementation - Drawing Book
import math


def pageCount(n, p):
    """pageCount."""
    if n == p:
        return 0
    elif n != p and n % 2 == 0:
        return min(math.ceil((n - p) / 2), math.ceil((p - 1) / 2))
    elif n != p and n % 2 == 1:   # odd # pages
        return min((n - p) // 2, (p - 0) // 2)

# %%  Practice Algorithms - Implementation - Electronics Shop


def getMoneySpent(keyboards, drives, b):
    """getMoneySpent."""
    maxspent = -1
    pairs = [(keyboard, drive) for keyboard in keyboards for drive in drives]
    for pair in pairs:
        if sum(pair) <= b and sum(pair) > maxspent:
            maxspent = sum(pair)
    return maxspent

# %%  Practice Algorithms - Implementation - Cat and Mouse


def catAndMouse(x, y, z):
    """catAndMouse."""
    if (abs(x - z)) < (abs(y - z)):
        return 'Cat A'
    elif (abs(x - z)) > (abs(y - z)):
        return 'Cat B'
    else:
        return 'Mouse C'


# %%  Practice Algorithms - Implementation - Picking Numbers
# longest subarray with difference of at most 1


def pickingNumbers(arr):
    """pickingNumbers."""
    arr.sort()
    max_length = 1
    length = 1
    start = arr[0]
    print(arr)

    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1] and arr[i] == start:
            # same # as start
            length += 1
            max_length = max(max_length, length)
        elif arr[i] == arr[i - 1] + 1 and arr[i] == start + 1:
            # ending value of subarr
            length += 1
            max_length = max(max_length, length)
        elif arr[i] == arr[i - 1] and arr[i] == start + 1:
            # continuing onending value
            length += 1
            max_length = max(max_length, length)
        else:
            # here is where we start measuring a new subarray
            start = arr[i]
            length = 1

    return max_length


# %%  Practice Algorithms - Implementation - Designer PDF Viewer
import string


def designerPdfViewer(h, word):
    """designerPdfViewer."""
    lookup = dict(zip(string.ascii_letters, h))
    maxheight = 0
    for letter in word:
        maxheight = max(maxheight, lookup[letter])

    return 1 * len(word) * maxheight

# %%  Practice Algorithms - Implementation -  Utopian Tree


def utopianTree(n):
    """utopianTree."""
    height = 1
    if n == 0:
        return height
    cycle = 1
    while cycle <= n:
        if cycle % 2 == 1:
            height = height * 2
        elif cycle % 2 == 0:
            height += 1

        cycle += 1
    return height


# %%  Practice Algorithms - Implementation -  Beautiful Numbers


def revnum(k):
    """revnum."""
    skip = 0
    rev = str(k)[-1::-1]
    for i in range(len(rev)):
        if rev[i] != '0':
            break
        else:
            skip += 1
    return int(rev[skip:])

# %%  Practice Algorithms - Implementation - Save the Prisoner!
# key to this problem is knowing that the edge case of evenly divided candies
# results in a zero, which means we need to add the 'OR n' in the return


def saveThePrisoner(n, m, s):
    """saveThePrisoner."""
    return (m + s - 1) % n or n

    # when divides evenly, need to land on same
    # seat as n
    # first operand will pass instead of second when both are nonzero
    # second will pass when first is0
    # 3 or 4 ==> 3
    # 0 or 4 ==> 4


print(saveThePrisoner(352926151, 380324688, 94730870))
print(saveThePrisoner(94431605, 679262176, 5284458))
print(saveThePrisoner(999999999, 999999998, 2))


# %%  Practice Algorithms - Implementation - Circular Rotation


def circularArrayRotation(a, k, queries):
    """circularArrayRotation."""
    newlist = []
    if k % len(a) == 0:
        newlist = a
    else:
        newlist = a[-k % len(a):]
        newlist.extend(a[0: -k % len(a)])
    result = []
    for query in queries:
        result.append(newlist[query])
    return result


x = circularArrayRotation([1, 2, 3], 2, [0, 1, 2])
print(x)

arr = ['a', 'b', 'c']
k = 2
queries = [0, 1, 2]

print(circularArrayRotation(arr, k, queries))


# %%  Practice Algorithms - Implementation - Find Digits
# lesson here is to make an integer an iterable with str() and then back to
# an integer again with int()

def findDigits(n):
    """findDigits."""
    n_divisors = 0
    for i in range(len(str(n))):
        if int(str(n)[i]) == 0:
            pass
        elif n % int(str(n)[i]) == 0:
            n_divisors += 1

    return n_divisors


# %%  Practice Algorithms - Implementation - Jumping on the Clouds: Revisited
# could probably calculate a specific answer using Least CommonDenominator, but
# since number of clouds is limited to 25, we can just use a loop and
# iterate through to find when we land back at zero

# test boundary cases


def jumpingOnClouds(c, k):
    """jumpingOnClouds."""
    i = (0 + k) % len(c)
    e = 100
    n_jumps = 0

    while i != 0:
        n_jumps += 1
        if c[i] == 1:
            e = e - 1 - 2
        else:
            e -= 1
        i += k
        i = i % len(c)

    if c[0] == 1:
        e = e - 1 - 2
    else:
        e -= 1
    return e


print(jumpingOnClouds([0, 0, 1, 0, 0, 1, 1, 0], 2))

# %%  Practice Algorithms - Implementation - Append and Delete


def appendAndDelete(s, t, k):
    """appendAndDelete."""
    sameletters = 0

    i = 0
    while i < min(len(s), len(t)) - 1 and s[i] == t[i]:
        # a while loop is the way to go, since we don't want to count same
        # letters after the first character from left to rigth that isn't
        # equal...while loop breaks when first unequal character appears
        sameletters += 1
        i += 1

    special_value = len(s) + len(t) - 2 * sameletters

    print('\n', s, t, 'k=', k)
    print('sameletters', sameletters)
    print('special_value', special_value)
    print('len s', len(s))
    print('len t', len(t))
    print('both', len(s) + len(t))
    print('-' * 3)

    if k >= len(s) + len(t):
        print('Yes')
        return 'Yes'
    elif k >= special_value and k < len(s) + len(t) and\
            (k - special_value) % 2 == 0:
        print('Yes')
        return 'Yes'

    else:
        print('No')
        return 'No'


s = 'word'
t = 'caps'
appendAndDelete(s, t, 6)
appendAndDelete(s, t, 4)
appendAndDelete(s, t, 8)
appendAndDelete(s, t, 10)


s = 'capitalsg'
t = 'caps'
appendAndDelete(s, t, 6)
appendAndDelete(s, t, 7)
appendAndDelete(s, t, 8)
appendAndDelete(s, t, 9)
appendAndDelete(s, t, 10)
appendAndDelete(s, t, 11)
appendAndDelete(s, t, 12)
appendAndDelete(s, t, 13)

# %%


def squares(a, b):
    """Squares."""
    count = 0
    for i in range(a, b + 1):
        count += ((i**0.5 - int(i**0.5)) == 0)
    print(count)
    return count


print(squares(3, 9))
print(squares(17, 24))
# faster version


def squares(a, b):
    """Squares."""
    extra = 0
    if int(a**0.5) == a**0.5:
        extra = 1
    # this tests if we include lower integer in count
    count = int(b**0.5) - int((a**0.5)) + extra
    return count


print(squares(3, 9))
print(squares(17, 24))

# %% Library Fine


def libraryFine(d1, m1, y1, d2, m2, y2):
    """LibraryFine."""
    # y2 is due date
    fine = 0
    if y1 > y2:
        fine = 10000
    elif y2 == y1 and m1 > m2:
        fine = 500 * (m1 - m2)
    elif y2 == y1 and m2 == m1 and d1 > d2:
        fine = 15 * (d1 - d2)
    return fine

# %% Cut the sticks


def cutTheSticks(arr):
    """CutTheSticks."""
    arr.sort()
    result = []
    while len(arr) > 0:
        print(len(arr))
        result.append(len(arr))
        mincut = arr[0]
        i = 0
        while i < len(arr) and arr[i] == mincut:
            i += 1
        arr = arr[i:]
        for j in range(len(arr)):
            arr[j] = arr[j] - mincut
    return result


cutTheSticks([5, 4, 4, 2, 2, 8])

# %% Equalize the Array

from collections import Counter


def equalizeArray(arr):
    """Array."""
    freqs = Counter(arr)
    max_freq = max(freqs.values())
    print(len(arr) - max_freq)
    return len(arr) - max_freq


equalizeArray([3, 3, 2, 1, 3])

# %% ACM ICPC Team
from itertools import combinations
# timed out


def acmTeam(topic):
    """AcmTeam."""
    students = [i for i in range(len(topic))]
    teams = combinations(students, 2)

    max_known = -1
    teams_knowing = 0
    for team in teams:
        x = topic[team[0]]
        y = topic[team[1]]
        x_l = list(map(int, list(x)))
        y_l = list(map(int, list(y)))
        team_topics = [a[0] | a[1] for a in list(zip(x_l, y_l))]
        known = sum(team_topics)
        if known > max_known:
            teams_knowing = 1
            max_known = known
        elif known == max_known:
            teams_knowing += 1
    print([max_known, teams_knowing])
    return [max_known, teams_knowing]


# this also timed out
acmTeam(['10101', '11100', '11010', '00101'])
topic = ['10101', '11100', '11010', '00101']


def acmTeam(topic):
    """AcmTeam."""
    students = ['0b' + i for i in topic]
    teams = list(combinations(students, 2))

    max_known = -1
    teams_knowing = 0
    team_topics = [int(a[0], 2) | int(a[1], 2) for a in teams]
    results = [sum(list(map(int, list(bin(x)[2:])))) for x in team_topics]
    for result in results:
        if result > max_known:
            teams_knowing = 1
            max_known = result
        elif result == max_known:
            teams_knowing += 1
    print([max_known, teams_knowing])
    return [max_known, teams_knowing]

# %%
# this had runtime issue with long test cases...try generators
# instead of lists


def acmTeam(topic):
    """AcmTeam."""
    teams = list(combinations(topic, 2))
    max_known = -1
    teams_knowing = 0
    for team in teams:
        x = team[0]
        y = team[1]
        result = 0
        for i in range(len(x)):
            print(i, result, x, y)
            if int(x[i]) or int(y[i]):
                result += 1
        print('Team', team, 'result', result, 'maxknown', max_known,
              'teams knowing', teams_knowing)
        if result > max_known:
            teams_knowing = 1
            max_known = result
        elif result == max_known:
            teams_knowing += 1
        print('Team', team, 'result', result, 'maxknown', max_known,
              'teams knowing', teams_knowing)

    print([max_known, teams_knowing])
    return [max_known, teams_knowing]


acmTeam(['10101', '11100', '11010', '00101'])
acmTeam(['11101', '10101', '11001', '10111', '10000', '01110'])

# %%
from itertools import combinations
import pandas as pd


def acmTeam(topic):
    """AcmTeam."""
    teams = combinations(topic, 2)
    # to use generator, the only code I changed was prior line; didn't use
    # the list() function; but still didn't work on 5 test cases due to
    # time limitexceeded

    max_known = -1
    teams_knowing = 0
    for team in teams:
        x = team[0]
        y = team[1]
        result = 0
        for i in range(len(x)):
            # print(i, result, x, y)
            if int(x[i]) or int(y[i]):
                result += 1
                # print('Team', team, 'result', result, 'maxknown', max_known,
                # 'teams knowing', teams_knowing)
        if result > max_known:
            teams_knowing = 1
            max_known = result
        elif result == max_known:
            teams_knowing += 1
#        print('Team', team, 'result', result, 'maxknown', max_known,
#              'teams knowing', teams_knowing)

#    print([max_known, teams_knowing])
    return [max_known, teams_knowing]


acmTeam(['10101', '11100', '11010', '00101'])
# acmTeam(['11101', '10101', '11001', '10111', '10000', '01110'])
# %%
# this code worked on my machine when I loaded in the test case...yes, it took
# a while, but it worked with a list
import pandas as pd

topic = pd.read_csv('students3.txt')
data = topic['students'].values.tolist()


def acmTeam(topic):
    """AcmTeam."""
    teams = combinations(topic, 2)
    # to use generator, the only code I changed was prior line; didn't use
    # the list() function; but still didn't work on 5 test cases due to
    # time limitexceeded

    max_known = -1
    teams_knowing = 0
    for team in teams:
        x = team[0]
        y = team[1]
        result = 0
        for i in range(len(x)):
            # print(i, result, x, y)
            if int(x[i]) or int(y[i]):
                result += 1
        # print('Team', team, 'result', result, 'maxknown', max_known,
        #  'teams knowing', teams_knowing)
        if result > max_known:
            teams_knowing = 1
            max_known = result
        elif result == max_known:
            teams_knowing += 1
#        print('Team', team, 'result', result, 'maxknown', max_known,
#              'teams knowing', teams_knowing)

    print([max_known, teams_knowing])
    return [max_known, teams_knowing]


acmTeam(data)

# Test case 3 [467, 1] is output
# Test case 4 [467, 1] is output as well
# TEST CASE 5 [416, 2] worked fine with my code as well
# %%
from itertools import combinations

# suggested solution works on all cases
# so simple; my prior attempts were basically this, but this makes the students
# a list of integers from the beginning, avoiding needing to make all
# combinations of students into integers/lists


def acmTeam(topic):
    """AcmTeam."""
    students = [list(map(int, list(x))) for x in topic]
    sums_elements = [[x for x in list(zip(*i))] for i in
                     combinations(students, 2)]
    sums = [sum([x[0] or x[1] for x in list(zip(*i))]) for i in
            combinations(students, 2)]
    print(students)
    print(sums_elements)
    print(sums)
    print(max(sums), sums.count(max(sums)))
    return [max(sums), sums.count(max(sums))]


# test case 3 data read in from text file
topic = pd.read_csv('students3.txt')
data = topic['students'].values.tolist()
data = ['10101', '11100', '11010', '00101']
acmTeam(data)

# %% Python solution, which is more elegant way than my bitwise solution


def acmTeam(data):
    """AcmTeam."""
    n = len(data)
    maxi = 0
    cnt = 0
    for i in range(n):   # we can create all combinations with 2 loops
        for j in range(i + 1, n):
            set_bit = bin(int(data[i], 2) | (int(data[j], 2))).count("1")
            # bin() turns it into a string again with prefix  '0b',
            # count() counts the '1's
            if set_bit > maxi:
                maxi = set_bit
                cnt = 1
            elif set_bit == maxi:
                cnt += 1
    print([maxi, cnt])
    return [maxi, cnt]


data = ['10101', '11100', '11010', '00101']
acmTeam(data)

# %% Practice Algorithms: Implementation - The Grid Search

# this worksfor allbutone test case which  is huge and multiple
# tests


def gridSearch(G, P):
    pr = len(P)
    pc = len(P[0])
    gr = len(G)
    gc = len(G[0])
    result = 'NO'
    for i in range(gr - pr + 1):
        print('checking row', i)
        for j in range(gc - pc + 1):
            # if we find the pattern, no need to continue searching other
            # columns j; thus we check result and return 'yes' to break out
            if result == 'YES':
                return result
            print('checking column', j)
            if G[i][j:j + pc] == P[0][0:pc]:
                print('match')
                result = 'YES'
                # iterate through to see if rest of pattern matches
                for row in range(1, pr):
                    if G[i + row][j:j + pc] != P[row][0:pc]:
                        print('mismatch')
                        result = 'NO'

    return result


G = ['123412', '561212', '123634', '781288']
P = ['12', '34']

gridSearch(G, P)
P = ['1234', '4321', '5678', '8765']

# %% Practice Algorithms: Implementation - The Grid Search
# this is leaner; fewer debugging code, which passes test


def gridSearch(G, P):
    pr = len(P)
    pc = len(P[0])
    gr = len(G)
    gc = len(G[0])
    lines = 0
    for i in range(gr - pr + 1):
        for j in range(gc - pc + 1):
            if G[i][j:j + pc] == P[0]:
                # iterate through to see if rest of pattern matches
                for row in range(1, pr):
                    if G[i + row][j:j + pc] == P[row]:
                        lines += 1
                        if lines == pr - 1:
                            return 'YES'
                    else:
                        lines = 0
    return 'NO'


G = ['123412', '341212', '123634', '341288', '341288']
P = ['12', '34', '56']
print(gridSearch(G, P))
# %% Practice Algorithms: Implementation - The Grid Search
# this works


def gridSearch(G, P):
    lineChecks = 0
    for i in range(len(G[0]) - len(P[0]) + 1):
        for j in range(len(G) - len(P) + 1):
            if G[j][i:i + len(P[0])] == P[0]:
                for x in range(1, len(P)):
                    if G[j + x][i:i + len(P[0])] == P[x]:
                        lineChecks += 1
                        if lineChecks == len(P) - 1:
                            return "YES"
                    else:
                        lineChecks = 0
    return "NO"

# %% Practice Algorithms: Implementation - Taum and B'day


# quite simple


def taumBday(n_b, n_w, cost_b, cost_w, z):
    """Min cost for presents."""
    # 3 cases for how much it would cost to buy black or white present
    case1 = n_w * cost_w + n_b * cost_b
    case2 = n_w * (cost_b + z) + n_b * cost_b
    case3 = n_w * cost_w + n_b * (cost_w + z)
    return min(case1, case2, case3)


# %% Practice Algorithms: Implementation - Modified Kaprekar Numbers

def kaprekarNumbers(p, q):
    """Numbers."""
    results = []
    for n in range(p, q + 1):
        square = n**2
        str_sq = str(square)
        d = len(str(n))
        right = str_sq[-d:]
        left = str_sq[0:-d]
        if left == '':
            left = '0'
        if int(right) + int(left) == n:
            results.append(n)

    if len(results) == 0:
        print('INVALID RANGE')

    else:
        print(' '.join(map(str, results)))


kaprekarNumbers(1, 1000)

# %%
# this code timed out for 2 test case arrays that were 10,000 in length


def beautifulTriplets(d, arr):
    """BeautifulTriplets."""
    result = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[j] - arr[i] == d:
                for k in range(i + 2, len(arr)):    # go search for third index
                    if arr[k] - arr[j] == d:
                        result.append((i, j, k))
    print(len(result))

    return len(result)


beautifulTriplets(1, [2, 2, 3, 4, 5])

# %% easier to use a Counter and only one pass through array
from collections import Counter


def beautifulTriplets(d, arr):
    """BeautifulTriplets."""
    result = 0
    x = Counter(arr)
    for i in range(1, len(arr) - 1):
        result += x[arr[i] - d] * x[arr[i] + d]
    print(result)
    return result


beautifulTriplets(1, [2, 2, 3, 4, 5])

# %% Practice Algorithms: Implementation - Minimum Distance


def minimumDistances(arr):
    """MinimumDistances."""
    distance = -1
    pairs = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] == arr[j]:
                pairs.append((i, j))
    if len(pairs) != 0:
        distance = min([abs(x[0] - x[1]) for x in pairs])

    return distance

# %% Practice Algorithms: Implementation - Halloween Sale


def howManyGames(p, d, m, budget):
    """Return the number of games you can buy."""
    cost = p
    i = 0
    add = 0
    while budget >= cost:
        budget = budget - cost
        add += cost
        print(add, cost)

        i += 1

        if (p - i * d) <= m:
            cost = m
        else:
            cost = p - i * d
    print(i)
    return i


howManyGames(20, 3, 6, 80)

# %% Practice Algorithms: Implementation -


def chocolateFeast(money, cost, m):
    """ChocolateFeast."""
    n_bars = money // cost
    wrappers = n_bars
    while wrappers >= m:
        n_bars += 1
        wrappers = wrappers - m + 1
    print(n_bars)
    return n_bars


# %% Practice Algorithms: Implementation - serviceLane

def serviceLane(widths_array, cases):
    """ServiceLane."""
    min_widths = []
    for case in cases:
        entry_i = case[0]
        exit_i = case[1] + 1
        min_widths.append(min(widths_array[entry_i:exit_i]))
    print(min_widths)
    return min_widths


# %% Practice Algorithms: Implementation - Lisa's Workbook
import math


def workbook(n, k, arr):
    """Workbook."""
    last_page = 0
    magic_numbers = 0
    for chapter in range(1, n + 1):
        n_probs = arr[chapter - 1]
        n_pages = math.ceil(n_probs / k)
        print('\nnprobs', n_probs, 'npages', n_pages)
        prob_range = [1, min(k, n_probs)]
        curr_page = last_page + 1
        last_page = curr_page + n_pages - 1

        while curr_page < (last_page + 1):
            print('p', curr_page, prob_range)
            if curr_page >= prob_range[0] and curr_page <= prob_range[1]:
                magic_numbers += 1
                print('found magic problem')
            prob_range[0] = prob_range[0] + k
            prob_range[1] = min(prob_range[1] + k, n_probs)
            curr_page += 1
    print(magic_numbers)
    return magic_numbers


workbook(5, 3, [4, 2, 6, 1, 10])

# %% Practice Algorithms: Implementation - Viral Advertising


def viralAdvertising(n):
    """viralAdvertising."""
    cum_liked = 0
    liked = 0
    shared = 5
    for day in range(1, n + 1):
        liked = shared // 2
        cum_liked = liked + cum_liked
        print(day, shared, liked, cum_liked)
        shared = liked * 3

    return cum_liked


viralAdvertising(5)


# %% Practice Algorithms: Implementation -


def flatlandSpaceStations(n, c):
    """FlatlandSpaceStations."""
    c.sort()
    maxdist = 0
    maxdist = max(c[0], n - 1 - c[-1])
    print(maxdist)

    for i in range(1, len(c)):
        dist = math.ceil((c[i] - c[i - 1] - 1) / 2)
        maxdist = max(maxdist, dist)

    print(maxdist)
    return maxdist


flatlandSpaceStations(90, [85, 44, 25, 67, 20, 83, 50, 88, 2, 32, 16])

# %% Practice Algorithms: Implementation -


def fairRations(B):
    """fairRations."""
    if sum(B) % 2 == 1:
        return 'NO'
    else:
        i = 1
        loaves = 0
        while i < len(B):
            if B[i] % 2 == 0 and B[i - 1] % 2 == 0:
                i += 1

            # even, odd -> pass
            elif B[i] % 2 == 1 and B[i - 1] % 2 == 0:
                i += 1

            elif B[i] % 2 == 1 and B[i - 1] % 2 == 1:
                loaves += 2
                i += 2
            elif B[i] % 2 == 0 and B[i - 1] % 2 == 1:
                B[i] += 1
                loaves += 2
                i += 1
    print(loaves)
    return loaves


fairRations([2, 3, 4, 5, 6])


# %% Practice Algorithms: Implementation - cavityMap
import copy


def cavityMap(grid):
    """cavityMap."""
    n_grid = []
    n = len(grid)
    for i in range(n):
        # each row's numbers need to be turned into a
        # list of strings, and each one turned to int
        numbers = list(map(int, list(grid[i])))
        n_grid.append(numbers)

    second_grid = copy.deepcopy(n_grid)
    # cannot keep it in strings, because strings cannot be changed
    #    print(n_grid)
    #   print(second_grid)

    for row in range(1, n - 1):
        for col in range(1, n - 1):
            cavity = n_grid[row][col]
            upper = n_grid[row - 1][col]
            lower = n_grid[row + 1][col]
            left = n_grid[row][col - 1]
            right = n_grid[row][col + 1]

            is_cavity = (cavity > upper) and (cavity > lower) and \
                (cavity > left) and (cavity > right)
            if is_cavity:
                second_grid[row][col] = 'X'

    for i in range(len(grid)):
        second_grid[i] = ''.join(list(map(str, second_grid[i])))

    print(n_grid)
    print(second_grid)
    return second_grid


cavityMap(['989', '191', '111'])

# %% Practice Algorithms: Implementation -


def stones(n, a, b):
    """Stones."""
    last_numbers = []
    for i in range(n):
        last_numbers.append(a * (n - 1 - i) + b * i)
    result = list(set(last_numbers))
    result.sort()
    print(result)
    out = ' '.join(map(str, result))
    print(out)
    return result


stones(2, 2, 3)


# %% Practice Algorithms: Implementation -

import string
from collections import Counter


def happyLadybugs(b):
    """HappyLadybugs."""
    c = Counter(b)
    letters = list(set(b).difference('_'))
    if '_' in b:
        # test there is at least two of each letter
        for i in range(len(b)):
            if b[i] in letters and c[b[i]] < 2:
                print('NO')
                return 'NO'
        print('YES')
        return 'YES'

        # if no _, then we need to see that they are already next to
        # one another; need to check if
    else:
        if len(b) == 1 and b[0] in letters:
            return 'NO'
        elif len(b) == 2 and b[0] != b[1]:
            return 'NO'

        elif b[0] != b[1] or b[-1] != b[-2]:
            return 'NO'
        for i in range(1, len(b) - 1):
            if not (b[i] == b[i - 1] or b[i] == b[i + 1]):
                print('NO')
                return 'NO'
        print('YES')
        return 'YES'

# %% Practice Algorithms: Implementation -


def strangeCounter(t):
    """StrangeCounter."""
    power = 0
    upper = 3 * 2**power

    while t > upper:
        power += 1
        upper = upper + 3 * 2**power
    value = upper - t + 1
    print(value)
    return value


# %% Practice Algorithms: Implementation


def superReducedString(s):
    """SuperReducedString."""
    reduced = list(s)
    i = 0
    length = len(s)
    while i < length and reduced != []:

        if i == length - 1:
            i += 1
        elif reduced[i] == reduced[i + 1]:
            reduced.pop(i)
            reduced.pop(i)
            length = len(reduced)
            i = 0
        else:
            i += 1

    if reduced == []:
        print('Empty String')
        return 'Empty String'
    else:
        print(reduced)
        return ''.join(reduced)


superReducedString('abbcad')


# %% Practice Algorithms: Data Structures/Strings -
import re


def minimumNumber(n, password):
    """Return the minimum number of characters to make password strong."""
    lower = 0
    upper = 0
    num = 0
    special = 0

    for let in password:
        if re.search(r'[a-z]', let):
            lower = 1
        elif re.search(r'[A-Z]', let):
            upper = 1
        elif re.search(r'[0-9]', let):
            num = 1

    # - needs to be in front or end of [] so as not to be interpreted as escape
        elif re.search(r'[-!@#$%^&*()+ ]', let):
            special = 1
    categories = lower + upper + num + special

    if n >= 6 and categories == 4:
        return 0
    elif n >= 6 and categories < 4:
        print('here', categories)
        return (4 - categories)
    elif n < 6 and categories == 4:
        return (6 - n)
    elif (n < 6) and (categories < 4):
        if (n + 4 - categories) >= 6:
            return 4 - categories
        elif (n + 4 - categories) < 6:
            return 6 - n


print(minimumNumber(3, 'Ab1'))
print(minimumNumber(7, 'AUzs-nV'))


# %% Practice Algorithms: Data Structures/Strings -
# What is max length of alternating character string ofjust twocharacters, if
# we eliminate all other characters?

from itertools import combinations


def alternate(s):
    """Alternate."""
    def is_valid(st):
        result = True
        for i in range(1, len(st)):
            if st[i] == st[i - 1]:
                result = False
        return result

    s_list = list(s)
    max_len = 0
    letters = list(set(s))

    for combo in combinations(letters, 2):
        letters_to_keep = []
        for let in s_list:
            if let in combo:
                letters_to_keep.append(let)
        if is_valid(letters_to_keep):
            max_len = max(max_len, len(letters_to_keep))
    print(max_len)
    return max_len


alternate('abcdebc')


# %% Practice Algorithms: Data Structures/Strings - HackerRank in a String!

def hackerrankInString(s):
    """HackerrankInString."""
    pointer = 0

    word = 'hackerrank'
    i = 0
    while pointer < len(word) and i < len(s):
        print(pointer, 'p', 'i', i)
        if s[i] == word[pointer] and pointer <= len(word) - 1:
            pointer += 1
        i += 1
    if (pointer == len(word)):
        return 'YES'
    else:
        return 'NO'


# %% Practice Algorithms: Data Structures / Strings - Caesar Cipher
import string


def caesarCipher(s, k):
    """CaesarCipher."""
    encoded = ''
    for letter in s:
        if letter in string.ascii_letters:
            i = 0
            while string.ascii_letters[i] != letter:
                i += 1
            if letter.isupper():
                encoded_letter = string.ascii_letters[(i + k) % 26].upper()
            else:
                encoded_letter = string.ascii_letters[(i + k) % 26]
            encoded = encoded + encoded_letter
        else:
            encoded = encoded + letter
    return encoded


# %% Practice Algorithms: Data Structures / Strings - Mars Explorer
# Complete the marsExploration function below.
def marsExploration(s):
    """MarsExploration."""
    count = 0
    for i in range(len(s)):
        if i % 3 == 0 or (i + 1) % 3 == 0:
            if s[i] != 'S':
                count += 1
        else:
            if s[i] != 'O':
                count += 1
    return count


# %% Practice Algorithms: Data Structures / Strings - Pangram

import string

s = 'We promptly judged antique ivory buckles for the next prize'
s = 'We promptly judged antique ivory buckles for the prize'
# problem with this solution is that


def pangrams(s):
    """Pangrams."""
    characters = set(s.lower())
    superset = set(string.ascii_letters[0:26] + ' ')
    # A.difference(B) only returns what is in A and not in B, not what is in B
    # and not A
    return ('pangram' if len(set(superset).difference(characters)) == 0
            else 'not pangram')


print(pangrams(s))


def pangrams(s):
    """Pangrams."""
    # this works if we can be sure input is only A-Z and a-z and ' '
    return ('pangram' if len(set(s.lower())) == 27 else 'not pangram')


# returns integer ASCII code associated with the letter 'a', or 97
ord('a')
# returns integer ASCII code associated with the letter 'z', or 97+25=122
ord('z')
ord('A')  # 65
ord('Z')  # 90
ord('0')  # 48
ord('1')  # 49
ord('9')  # 57
ord('#')  # 35

letters = string.ascii_letters[0:26]
for i in range(10):
    print(ord(str(i)))
# %% Practice Algorithms: Data Structures / Strings - Weighted Uniform Strings
# Query 3 was too time consuming; should use dictionary/lookup as we do in
# try 2


with open("weightedstring_case5.txt", "r") as myfile:
    string_queries = myfile.readlines()

# Edit the size of these inputs before timing
string_ = string_queries[0][0:10000]  # -3]
queries = string_queries[0:10000]
# When string and queries are 0:10000
# Query 1 0.003000497817993164
# Query 2 0.009000301361083984
# Query 3 2.4431400299072266   ==> so we see here there is a bottleneck

# it would be good toanalyze string with a counter like function:
# to produce list of tuples for 'abbcccc' as[(a,1), (b,2), (c,4)]
print('Length of string  =', len(string_))  # = 61542
print("Number of queries =", len(queries))  # = 61542

import string
import time
from collections import Counter


def weightedUniformStrings(s, queries):
    """WeightedUniformStrings."""
    query_1 = 0
    query_2 = 0
    query_3 = 0

    start_time = time.time()
    letters = dict(zip(string.ascii_letters[0:26], range(1, 27)))
    s_values = [letters[letter] for letter in s]
    print(letters, s_values)

    end_time = time.time()
    duration = end_time - start_time
    query_1 += duration

    print(s_values, '<==str_to_numbers')
    cum_values = [s_values[0]]
    # query 2
    start_time = time.time()

    for i in range(1, len(s_values)):
        if s_values[i] == s_values[i - 1]:
            cum_values.append(cum_values[i - 1] + s_values[i])
        else:
            cum_values.append(s_values[i])
    print(cum_values, '<==cum_values', 'len of array', len(cum_values))
    results = []

    end_time = time.time()
    duration = end_time - start_time
    query_2 += duration

    print(Counter(cum_values))
# query 3
#   start_time = time.time()
#   for query in queries:
#      if query in cum_values:
#         results.append('Yes')
#     else:
#          results.append('No')
#    end_time = time.time()
#    duration = end_time - start_time
#    query_3 += duration

    print(results)
    print('Query 1', query_1)
    print('Query 2', query_2)
    print('Query 3', query_3)

    return results


weightedUniformStrings(string_, queries)

# %% Practice Algorithms: Data Structures/Strings-Weighted Uniform Strings 2
# Attempt to simplify query 2 and query3 by making a counter of string
# THIS WORKED!!
# example:
# weightedUniformStrings('abccddde', [1, 3, 12, 5, 9, 10])
# 'abccddde'
# 1, 2, 3, 2*3=6, 4, 2*4=8, 3*4=12, 5 = this is the decoded sequence
# Value_lookup: {1: True, 2: True, 3: True, 6: True, 4: True, 8: True,
# 12: True, 5: True}
# queries = [1, 3, 12, 5, 9, 10]
# Results of queries : ['Yes', 'Yes', 'Yes', 'Yes', 'No', 'No']

with open("weightedstring_case5.txt", "r") as myfile:
    string_queries = myfile.readlines()

# Edit the size of these inputs before timing
string_ = string_queries[0][0:10]
queries = string_queries[0:10]
# When string and queries are 0:10000
# Query 1 0.003000497817993164
# Query 2 0.009000301361083984
# Query 3 2.4431400299072266   ==> so we see here there is a bottleneck

# it would be good toanalyze string with a counter like function:
# to produce list of tuples for 'abbcccc' as[(a,1), (b,2), (c,4)]
print('Length of string  =', len(string_))  # = 61542
print("Number of queries =", len(queries))  # = 61542
import string
import time


def weightedUniformStrings(s, queries):
    """WeightedUniformStrings."""
    # Create dictionary for lowercase letters and 1:26
    letters = dict(zip(string.ascii_letters[0:26], range(1, 27)))

    # Create dictionary showing if a certain integer exists for the decoded
    # string
    value_lookup = {}
    count = 1
    value_lookup[count * letters[s[0]]] = True
    # This loop adds True to value_lookup for numbers
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
            value_lookup[count * letters[s[i]]] = True
        else:
            count = 1
            value_lookup[count * letters[s[i]]] = True
    print('Value_lookup:', value_lookup)
    results = []

    # Now lookup to see if the integer exists in the decoded string
    for query in queries:
        if value_lookup.get(query, False):
            results.append('Yes')
        else:
            results.append('No')

    print('Results:', results)
    # print(results)
    return results


# weightedUniformStrings(string_, queries)

weightedUniformStrings('abccddde', [1, 3, 12, 5, 9, 10])

# %% Separate the Numbers
# Works for all but Test Case #20
# This finally works


def separateNumbers(s):
    """SeparateNumbers."""
    n = len(s)
    length = 1
    l_init = length
    result = 'YES '
    start = 0
    end = start + length
    current_end = end
    current = int(s[start:end])
    next_int = current + 1
    print('\n', s, 'Length:', len(s))
    if len(s) == 1:
        result = 'NO'

    else:
        while current_end != n and (end + len(str(next_int))) <= n:
            print('-' * 6, '\nCurrent:', current, 'from', start, ':', end)
            print('Next   :', next_int, 'from', start + length, ':',
                  end + len(str(next_int)))
            if s[start + len(str(current)):
                 end + len(str(next_int))] == str(next_int):
                start = start + len(str(current))
                end = end + len(str(next_int))
                current_end = end
                current = next_int
                next_int = next_int + 1
                result = 'YES '

            else:
                print('Next is not valid')
                length += 1
                l_init = length
                start = 0
                end = start + length
                current = int(s[start:end])
                next_int = current + 1

    # need to make sure the last successful run ends on
    # last character of string; otherwise, could have a few remainder numbers
    # on end that are never verified to be consecutive
    print('curr_end:', current_end, 'end:', end)
    if result == 'YES ' and current_end == n:  #
        print(result + s[0:l_init])
        return result + s[0:l_init]
    else:
        print('NO')


separateNumbers('99910001001')
separateNumbers('7891011')
separateNumbers('9899100')
separateNumbers('999100010001')
separateNumbers('101103')
separateNumbers('1')
separateNumbers('13')
separateNumbers('010203')


# %% test case #20 for separating numbers

# this is only one returning YES
separateNumbers('429496729542949672964294967297')

# rest should be 'NO'
separateNumbers('429496729542949672964294967296')
separateNumbers('429496729542949672964294967287')
separateNumbers('429496729542949672964294967197')

# these two WERE saying YES because of remainder characters; had to fix by
# conditioning result on current_end==end
separateNumbers('42949672954294967296429496729')
separateNumbers('4294967295429496729642949672')

separateNumbers('429496729500000000000000000001')
separateNumbers('42949672950123456789')
separateNumbers('4294967295000010020030000456789')
separateNumbers('4294967295000102003004005')

"""
Testcase # 20 output
YES 4294967295
NO
NO
NO
NO
NO
NO
NO
NO
NO
"""
# %% Practice Algorithms: Data Structures / Strings-
# Funny Numbers


def funnyString(s):
    """funnyString."""
    x = [abs(ord(s[i]) - ord(s[i - 1])) for i in range(1, len(s))]
    y = [abs(ord(s[len(s) - 1 - i]) - ord(s[len(s) - 1 - (i - 1)]))
         for i in range(1, len(s))]

    if x == y:
        return 'Funny'
    else:
        return 'Not Funny'


print(funnyString('abc'))
print(funnyString('abd'))


# %% Practice Algorithms: Data Structures/Strings- String Construction
# this was easy - don't need to worry about long substrings, because even
# one letter in p can be repeated/appended for no cost

def stringConstruction(s):
    """Minimize cost of re-constructing string 's'.

    Cost 1 for appending from s to p=''
    Cost 0 for appending any substring of p to p
    """
    p = ''
    i = 0
    mincost = 0
    while p != s:
        if s[i] in p:
            p = p + s[i]
            # no cost since it is substring of p
        else:
            p = p + s[i]
            mincost += 1
        i += 1
    return mincost


print(stringConstruction('abab'))

# %% Practice Algorithms: Data Structures/Strings - Game of Thrones I
from collections import Counter


def gameOfThrones(s):
    """Determine if s can be arranged into palindrome.

    If s length is odd, then only one letter can have frequency of just 1
    All others must have even frequency

    If s length is even, every letter must have even frequency
    """
    y = Counter()
    x = Counter(s)

    for a in x:
        if x[a] % 2 == 1:
            y['odd'] += 1

    if len(s) % 2 == 1 and y['odd'] > 1:
        return 'NO'
    elif len(s) % 2 == 1 and y['odd'] == 1:
        return 'YES'
    elif len(s) % 2 == 0 and y['odd'] > 0:
        return 'NO'
    elif len(s) % 2 == 0 and y['odd'] == 0:
        return 'YES'


# %% Practice Algorithms: Data Structures/Strings- Anagram
# need the Counters to be equivalent, so that same freq of letters in both
# words
from collections import Counter


def makingAnagrams(s1, s2):
    """Return min number of deletions for two strings to be anagrams.

    Uses two counters and reduces frequency every time a letter is found
    that has higher freq in one string vs. the second string

    Needs two passes, one through first string, second through second.
    """
    min_del = 0

    x = Counter(s1)
    y = Counter(s2)

    for key in x:
        if x[key] > 0 and x[key] > y[key]:
            min_del += x[key] - y[key]
            x[key] -= x[key] - y[key]
        elif x[key] == y[key]:
            pass
        elif x[key] > 0 and x[key] < y[key]:
            min_del += y[key] - x[key]
            y[key] -= y[key] - x[key]

    for key in y:
        if y[key] > 0 and y[key] > x[key]:
            min_del += y[key] - x[key]
            y[key] -= y[key] - x[key]
        elif x[key] == y[key]:
            pass
        elif y[key] > 0 and y[key] < x[key]:
            min_del += x[key] - y[key]
            x[key] -= x[key] - y[key]

    return min_del


print(makingAnagrams('abc', 'cde'))

# %% Practice Algorithms: Data Structures/Strings- Making Anagrams


def anagram(s):
    """Return min number of changes to make word1 and word2 anagrams."""
    minchanges = 0

    if len(s) % 2 == 1:
        return -1
    else:
        word1 = s[0:len(s) // 2]
        word2 = s[len(s) // 2:]
        print(word1, word2)
        c1 = Counter(word1)
        for letter in word2:
            if c1[letter] > 0:
                c1[letter] -= 1
            elif c1[letter] == 0:
                minchanges += 1
            else:
                minchanges += 1

        return minchanges


# %% Practice Algorithms: Data Structures/Strings- Gemstones


def gemstones(arr):
    """gemstones."""
    possible_gems = set(arr[0])
    for rock in arr[1:]:
        possible_gems = possible_gems.intersection(set(rock))

    n_gemstones = len(list(possible_gems))
    return n_gemstones


arr = ['abcdde', 'baccd', 'eeabg']
print(gemstones(arr))
# %% Practice Algorithms: Data Structures/Strings- Beautiful binary String


def beautifulBinaryString(b):
    """Minimum Number of Element changes to turn string into 'beautiful.

    beautiful = no substring '010'
    We go straight through string, changes the last 0 to a 1 in any substring
    encountered.
    """
    b_list = list(b)
    min_changes = 0
    for i in range(1, len(b_list) - 1):
        if b_list[i - 1] == '0' and b_list[i] == '1' and b_list[i + 1] == '0':
            b_list[i + 1] = '1'
            min_changes += 1
    return min_changes


# %% Practice Algorithms: Data Structures/Strings - Palindrome Index
#  This is the simplest solution


def find_mismatching_pair(s):
    """find_mismatching_pair.
    j will be <=i if all letters are palindromic through the middle index"""
    i = 0
    j = len(s) - 1
    while i < j and s[i] == s[j]:
        i += 1
        j -= 1
    return i, j


def is_palindrome(s):
    """is_palindrome."""
    i, j = find_mismatching_pair(s)
    return True if j <= i else False


def palindromeIndex(s):
    """correct."""
    i, j = find_mismatching_pair(s)
    return -1 if j <= i else i if is_palindrome(s[i + 1:j + 1]) else j


print(palindromeIndex('mmbiefhflbeckaecprwfgmqlydfroxrblulpasumubqhhbvlqpixvvxipqlvbhqbumusaplulbrxorfdylqmgfwrpceakceblfhfeibmm'))

# %% Practice Algorithms: Data Structures/Strings-


def theLoveLetterMystery(s):
    """Turn each string into palindrome in min number of operations.

    Can only reduce value of letter, and each reduction counts as 1 operation
    """
    mincount = 0
    for i in range(len(s) // 2):
        mincount += abs(ord(s[i]) - ord(s[-1 - i]))

    return mincount

# %% Practice Algorithms: Data Structures/Strings-


# %% Practice Algorithms - Sorting
"""Outline.
-------
Insertion Sort challenges
- Insertion Sort 1 - Inserting       x
- Insertion Sort 2 - Sorting         x
- Correctness and loop invariant     x
- Running Time of Algorithms         x

Quicksort challenges
- Quicksort 1 - Partition            X
- Quicksort 2 - Sorting              X
- Quicksort In-place (advanced)      -
- Running time of Quicksort          -

Counting sort challenges
- Counting Sort 1 - Counting         -
- Counting Sort 2 - Simple sort
- Counting Sort 3 - Preparing
- Full Counting Sort (advanced)      -

Insertion sort algo - O(N^2). for 2N elements, 4N^2
Quicksort (comparison sort) - n log n is worst case run time, because that
many comparisons need to be done

"""


# %% Practice Algorithms - Sorting - Big Sorting

from collections import defaultdict


def bigSorting(unsorted):
    """Sort list of strings / numbers of various lengths.

    Using default dictionary allows us to put each string into list of equal
    length strings. Avoid need to turn everything into an integer before
    sorting.
    """
    lookup = defaultdict(lambda: [])
    print(lookup)
    for num_string in unsorted:
        lookup[len(num_string)].append(num_string)

    results = []
    lengths = list(lookup.keys())
    lengths.sort()
    for length in lengths:
        x = lookup[length]
        x.sort()
        results = results + x
    print(results)
    return results


x = bigSorting(['31415926535897932384626433832795', '1',
                '31415926535897932384626433832500', '3', '10', '3', '5'])


# %% Practice Algorithms - Sorting - Insertion Sort - Part 1

def insertionSort1(n, arr):
    """Sort list of strings / numbers of various lengths."""
    unsorted_val = arr[n - 1]
    i = 1

    while unsorted_val < arr[n - i - 1] and (i < n):
        arr[n - i] = arr[n - i - 1]
        print(' '.join(map(str, arr)))
        i += 1
    arr[n - i] = unsorted_val
    print(' '.join(map(str, arr)))

# %% Practice Algorithms - Sorting - Insertion Sort - Part 2


def insertionSort2(n, arr):
    """Adapt code from insertionSort1() to use on subarray of length i + 1."""
    i = 1
    while i < n:
        unsorted_val = arr[i]

        m = i + 1
        j = 1
        while unsorted_val < arr[m - j - 1] and (j < m):
            arr[m - j] = arr[m - j - 1]
            j += 1
        arr[m - j] = unsorted_val
        print(' '.join(map(str, arr)))

        i += 1


insertionSort2(6, [1, 4, 3, 5, 6, 2])

# %% Practice Algorithms - Sorting - - Correctness and loop invariant


def insertion_sort(arr):
    """Insertion_sort.

    Find the error"""

    n = len(arr)
    i = 1
    while i < n:
        unsorted_val = arr[i]

        m = i + 1
        j = 1
        while unsorted_val < arr[m - j - 1] and (j < m):
            arr[m - j] = arr[m - j - 1]
            j += 1
        arr[m - j] = unsorted_val
        i += 1


arr = [1, 4, 3, 5, 6, 2]
insertion_sort(arr)
print(" ".join(map(str, arr)))


# %% Practice Algorithms - Sorting - quickSort

def quickSort(arr):
    """quickSort."""
    equal = [arr[0]]
    left = []
    right = []
    for i in range(1, len(arr)):
        if arr[i] == arr[0]:
            equal.append(arr[i])
        elif arr[i] < arr[0]:
            left.append(arr[i])
        elif arr[i] > arr[0]:
            right.append(arr[i])

    return left + equal + right

# %% Practice Algorithms - Sorting - Running Time of Algorithms


def runningTime(arr):
    """runningTime."""
    n = len(arr)
    i = 1
    times = 0

    while i < n:
        unsorted_val = arr[i]

        m = i + 1
        j = 1
        while unsorted_val < arr[m - j - 1] and (j < m):
            arr[m - j] = arr[m - j - 1]
            j += 1
            times += 1
        arr[m - j] = unsorted_val
        i += 1
    # print(arr)
    return times


arr = [2, 1, 3, 1, 2]
print(runningTime(arr))


# %% Practice Algorithms - Sorting - countingSort 1


def countingSort(arr):
    """countingSort.

    Returns the frequency of arr's elements, which are 0<=arr[i]<100
    """
    freq = [0] * 100
    for el in arr:
        freq[el] += 1
    return freq


# %% Practice Algorithms - Sorting - Counting Sort 2


def countingSort2(arr):
    """Return arr sorted.

    by going through frequency array and building sorted
    from ground up
    """
    freq = [0] * 100
    for el in arr:
        freq[el] += 1

    sorted_arr = []
    for i in range(len(freq)):
        sorted_arr = sorted_arr + [i] * freq[i]
    return sorted_arr


# %% Practice Algorithms - Sorting - The Full Counting Sort


def fullcountSort(arr):
    """Return associated strings in proper order."""
    sorted = [[] for i in range(100)]
    for i in range(int(len(arr) / 2)):
        sorted[int(arr[i][0])].append('-')
    for i in range(int(len(arr) / 2), len(arr)):
        sorted[int(arr[i][0])].append(arr[i][1])

    output = ''
    for item in sorted:
        if item == []:
            pass
        else:
            output = output + ' '.join(item) + ' '
    print(output)


# %% Practice Algorithms - Sorting - Closest Numbers


def closestNumbers(arr):
    """Sort so we can find smallest difference between pairs."""
    arr.sort()
    mindiff = abs(arr[1] - arr[0])

    for i in range(1, len(arr)):
        mindiff = min(mindiff, abs(arr[i] - arr[i - 1]))

    pairs = []
    for i in range(1, len(arr)):
        if arr[i] - arr[i - 1] == mindiff:
            pairs.append(arr[i - 1])
            pairs.append(arr[i])
    return pairs


# %% Find the Median - very simple - sort and then calculate middle index value
# for odd number


def findMedian(arr):
    arr.sort()
    mid = len(arr) // 2
    return arr[mid]


# %% Practice Algorithms - Search


def missingNumbers(arr, brr):
    bcounter = Counter(brr)
    acounter = Counter(arr)
    missing = []
    for el in list(acounter):
        bcounter[el] -= acounter[el]
    for el in list(bcounter):
        if bcounter[el] > 0:
            missing.append(el)
    missing.sort()
    return missing

# %% Practice Algorithms - Search - Sherlock and Array


def balancedSums(arr):
    total = sum(arr)
    cumsumleft = 0
    i = 0
    while i < len(arr):
        if cumsumleft == (total - arr[i] - cumsumleft):
            return 'YES'
        else:
            cumsumleft += arr[i]
        i += 1
    return 'NO'


# %% Practice Algorithms - Search - Ice Cream Parlor

from collections import defaultdict


def icecreamParlor(m, arr):
    """Return index + 1 of two flavors that use all budget m from arr."""
    lookup = defaultdict(lambda: [])
    for i in range(len(arr)):
        lookup[arr[i]] = lookup[arr[i]] + [i + 1]

    for i in range(len(arr)):
        if lookup[m - arr[i]] != [] and len(lookup[m - arr[i]]) == 2:
            # We know there will be a unique solution, i.e., two indices whose
            # arr[i] add up to m.  This if clause finds the case where
            # m/2 = arr[i] and arr[j], both i and j stored in lookup[arr[i]]
            # which is equivalent to lookup[m-arr[i]]; if the length were 1,
            # then it would not be a solution
            print([lookup[m - arr[i]][0], lookup[m - arr[i]][1]])
            return [lookup[m - arr[i]][0], lookup[m - arr[i]][1]]
        elif lookup[m - arr[i]] != [] and len(lookup[m - arr[i]]) == 1 and \
                arr[i] != m - arr[i]:
            print([i + 1, lookup[m - arr[i]][0]])
            return [lookup[arr[i]][0], lookup[m - arr[i]][0]]


icecreamParlor(8, [1, 2, 4, 4, 5])
icecreamParlor(100, [5, 75, 25])


# %% Practice Algorithms - Greedy - Grid Challenge


def gridChallenge(grid):
    for i in range(len(grid)):
        row_list = list(grid[i])
        row_list.sort()
        grid[i] = ''.join(row_list)
    ascending = 'YES'
    col = 0
    while col < len(grid[0]):
        row = 1
        while row < len(grid):
            if grid[row][col] < grid[row - 1][col]:
                ascending = 'NO'
                return ascending
            row += 1
        col += 1
    return ascending


# %% Practice Algorithms - Greedy - Candies

def givecandies(arr):
    """Give students seated in arr by their scores at least one candy.

    such that the total candies is minimized and that student with higher
    score will get more than student next to them

    Algo is to go through front to back and then back to front, increasing
    candy by one when 'next' student has higher score
    """
    n = len(arr)
    candies = [1] * n
    print(arr, 'Student scores')
    print(candies, 'Initital/minimum candies')
    for i in range(1, n):
        if arr[i] > arr[i - 1]:
            candies[i] = 1 + candies[i - 1]
    for i in range(1, n):
        if (
            arr[n - i - 1] > arr[n - i]
            and candies[n - i - 1] <= candies[n - i]
        ):
            candies[n - i - 1] = candies[n - i] + 1
    print(candies, 'Minimum candies rewarding for higher score')
    print(sum(candies), 'Total candies')
    return sum(candies)


givecandies([2, 1, 3, 4, 5, 6, 2, 1])
givecandies([4, 6, 4, 5, 6, 2])
givecandies([3, 1, 3, 10, 12, 2])

# %% Practice Algorithms - Greedy - Sherlock and The Beas
# n
# 1, 2, 4, 7 cannot be formed
# any mulitple of 3 will be maxed by having all 5s
# all other n are covered by remaining three


def decentNumber(n):
    """Examples.

    case 1: 5, 10, 20, 25, 35, 40, 45, 50 55, etc.
    case 2: 3+5=8, 6+5=11, 9+5=14, 12+5=17,etc.
    case 3: 3+10=13, 6+10=16, 9+10=19
        10 case 1 = all 3s
        11 case 2 = 555 555 33333
        12 %3==0  = 12x'5'
        13 case 3 = 555 33333 33333
        14 case 2 = 555 555 555 33333
        15 %3==0
        16 case 3 = 555 555 33333 33333
        17 case 2
        18 %3==0
        19 case 3
        20 case 1
        etc.
    """
    if n < 3 or n == 4 or n == 7:
        print(-1)
    elif n % 3 == 0:
        print(int('5' * n))
    elif (n % 15) != 0 and (n % 15) % 5 == 0:  # covers 5, 10, 20, 25, etc.
        print(int(('5' * (n - (n % 15))) + '3' * (n % 15)))
    elif n % 3 != 0 and n % 5 != 0 and ((n - 5) % 3) == 0:
        print(int('5' * (n - 5) + '3' * (5)))
    elif n % 3 != 0 and n % 5 != 0 and ((n - 10) % 3) == 0:
        print(int('5' * (n - 10) + '3' * (10)))


# %% Practice Algorithms - Greedy -Largest Permutation
import copy

# probably want to create lookup table to store index of each unique array
# value, which we will update each time a swap is performed
# also need a sorted list so that we know which values to lookup each time
# the lookup tables will help us avoid searching through array each time we
# want next largest value to swap into next position in permutation


def largestPermutation(k, arr):
    ref = copy.deepcopy(arr)
    lookup = {}
    for i in range(len(arr)):
        lookup[arr[i]] = i
    ref.sort(reverse=True)
    print('Original', arr)
    print('Lookup', lookup)
    print('Goal', ref)

    if k > len(arr):
        return ref
    else:
        swaps = 0
        i = 0
        while swaps < k and i < len(arr):
            if arr[i] != ref[i]:
                new1 = lookup[ref[i]]
                new2 = lookup[arr[i]]
                lookup[arr[i]] = new1
                lookup[ref[i]] = new2
                arr[new1] = arr[i]
                arr[i] = ref[i]
                swaps += 1
            i += 1
    print(arr)
    return arr


k = 1
arr = [4, 2, 3, 5, 1]
largestPermutation(k, arr)


# %% Practice Algorithms - Greedy - Jim and the Orders
def jimOrders(orders):
    lookup = {}
    delivered = []
    delivery = []
    for i in range(len(orders)):
        d_time = orders[i][0] + orders[i][1]
        lookup[d_time] = lookup.get(d_time, []) + [i + 1]
        delivery.append(d_time)
    delivery.sort()
    delivered = [lookup[time].pop(0) for time in delivery]
    print(delivered)
    return delivered


# %%
# can be done shorter because each element of arr should be element of first
# N numbers; my approach is omre general, consideringelements could be
# skipping around and not consecutive integers


def largestPermutation(k, arr):
    a = dict(enumerate(arr))
    b = {v: k for k, v in a.items()}
    length = len(arr)
    for i in range(length):
        if k and a[i] != length - i:
            x = a[i]
            y = b[length - i]
            a[i] = length - i
            a[y] = x
            b[x] = y
            k -= 1
        yield a[i]

# %% Practice Algorithms - Greedy - Priyanka and Toys


def toys(w):
    w.sort()
    containers = 1
    minval = w[0]
    for el in w:
        if el > minval + 4:
            containers += 1
            minval = el
    return containers
# %% Practice Algorithms - Greedy - Permuting Two Arrays


def twoArrays(k, A, B):
    A.sort(reverse=True)
    B.sort()
    for i in range(len(A)):
        if B[i] + A[i] < k:
            print('NO')
            return 'NO'
    print('YES')
    return 'YES'


twoArrays(10, [7, 1, 1], [8, 9, 3])

# %% Practice Algorithms - Greedy - Beautiful Pairs
# if there are more than 2 occurrences more in B than in A of a certain item,
# it will be impossible to be disjoint

# find number of matching elements in A/B, then if freq of one element occurs
# more in B than A, we can increase the number of beautiful pairs by one, at
# most ; if sum of pairs is < len(B) and there are no freq differences of
# common elements, then we can increase # pairs by one by changing one element
# of B into same as A
from collections import Counter


def beautifulPairs(A, B):
    pairs = 0
    a_ctr = Counter(A)
    b_ctr = Counter(B)
    a_elements = list(a_ctr)
    for el in a_elements:
        pairs = pairs + min(a_ctr[el], b_ctr[el])
    if pairs < len(A):
        pairs += 1
    else:
        pairs -= 1
        # we must make one change, so it already identical sets in A and B,
        # only change will reduce the number of pairs by one; pretty stupid

    print(pairs)
    return pairs


A = [3, 5, 7, 11, 5, 8]
B = [5, 7, 11, 10, 5, 8]

beautifulPairs(A, B)
# %%PracticeAlgorithmsGreedyMaximum Perimeter Triangle
# How is this not just testing the combinations


def maximumPerimeterTriangle(sticks):
    sticks.sort()
    triangles = []
    for i in range(len(sticks) - 2):
        for j in range(i + 1, len(sticks) - 1):
            for k in range(j + 1, len(sticks)):
                if sticks[i] + sticks[j] > sticks[k]:
                    triangles.append((sticks[i], sticks[j], sticks[k]))

    if triangles == []:
        print(triangles)
        return [-1]
    else:
        print(triangles[-1])
        return triangles[-1]


maximumPerimeterTriangle([1, 1, 1, 3, 3])

# %% Practice Algorithms - Bit Manipulation - maximizing Xor


def maximizingXor(left, right):
    maxval = left ^ left
    for i in range(left, right + 1):
        for j in range(left, right + 1):
            maxval = max(maxval, i ^ j)
    return maxval

# %% Practice Algorithms - Bit Manipulation SumXor
# 4 test cases dont work becuase n can be very large; timeout


def sumXor(n):
    which = []
    values = 0
    for i in range(n + 1):
        if n + i == i ^ n:
            which.append(i)
            values += 1
    print(which, values)
    return values


sumXor(100)

# %% Practice Algorithms - Bit Manipulation SumXor
# solution using trick or XOR
"""
the justification is that the xor simulates binary addition without the
carry over to the next digit. For the zero digits of n you can either add a
1 or 0 without getting a carry which implies xor = + whereas if a digit in n
 is 1 then the matching digit in x is forced to be 0 on order to avoid carry.
 For each 0 in n in the matching digit in x can either being a 1 or 0 with a
 total combination count of 2^(num of zero).

 for i in range(6):
    print(i^5, i+5)

5 5
4 6
7 7
6 8
1 9
0 10
"""


def sumXor(n):
    mylist = list(str(bin(n))[2:])
    nzeros = len(mylist) - sum(list(map(int, mylist)))
    if n == 0:
        # this case makes no sense; how can 0 ^ 0 is 0; this was test case 1
        print(1)
        return 1
    else:
        result = 2**nzeros
        print(result)
        return result


sumXor(5)
sumXor(0)


# %% Practice Algorithms - Game Theory - Tower Breakers
"""
Two players are playing a game of Tower Breakers! Player 1 always moves first,
and both players always play optimally.The rules of the game are as follows:

Initially there are n towers.
Each tower is of height m.
The players move in alternating turns.
In each turn, a player can choose a tower of height x and reduce its height
to y, where 1 <= y < x and y evenly divides x.
If the current player is unable to make a move, they lose the game.
Given the values of n and m , determine which player will win.
If the first player wins, return 1. Otherwise, return 2

y must be a factor of x at every step, with lowest value of y being 1,so
each tower can be lowered to 1 and then it offers no more moves

Thus, a tower of6 can be lowered to 3 or 2 or 1, because 3%6 = 0 or 2%6 or 0
so the set of possible moves on this tower is 3. if there is only one tower
with remaining moves, Player 1 could cut all way down to 1, leaving no more
moves.
So we need to count the number of towers with remaining moves, since # or
remaining moves can always be reduced to 1 given the opportunity.
Can calculate rmainging moves on each tower using this method.
"""

def towerBreakers(n, m):

    def get_all_factors(n):
        factors = []
        for i in range(1, n + 1):
            if n % i == 0:
                factors.append(i)

        return factors

    z = get_all_factors(m)
    nfactors = len(z)

    towers = [nfactors] * n
    print('towers:', towers, n, 'towers, of height', m)

    # coordinates are (n_towers, max_factors)

    def coords(towers):
        greater_than_one = 0
        equal_one = 0
        for el in towers:
            if el == 1:
                equal_one += 1
            elif el > 1:
                greater_than_one += 1
        x0 = greater_than_one
        y0 = equal_one
        return x0, y0

    def nextstep(x0, y0):
        thenext = set()
        # we need to change one tower; reducing it to 1, or 0 should
        # cover all cases, as 0 is even and 1 is odd
        if x0 > 0 and y0 > 0:
            thenext.add((max(x0 - 1, 0), y0 + 1))  # drop to 1 left in tower
            thenext.add((max(x0 - 1, 0), y0))      # drop to 0 in tower
            thenext.add((x0, max(y0 - 1, 0)))          # drop to 0 in tower

        elif x0 > 0 and y0 == 0:
            thenext.add((max(x0 - 1, 0), y0 + 1))  # drop to 1 left in tower
            thenext.add((max(x0 - 1, 0), y0))      # drop to 0 in tower
        elif x0 == 0 and y0 > 0:
            thenext.add((x0, max(y0 - 1, 0)))          # drop to 0 in tower
        print('current', x0, y0, 'the next', thenext)
        return thenext

#    nextstep(0,1)
#    nextstep(1,1)
#    nextstep(1,0)
#    nextstep(2,0)

    x0, y0 = coords(towers)
    print('starting coords of towers (', x0, ',', y0, ')')
    if x0 == 0:  # then towers is all ones, and we just find out if odd /even
        if y0 % 2 == 1:
            print(1, 'wins: odd number of towers with 1\n')
            return 1
        else:
            print(2, 'wins: even number of towers with 1\n')
            return 2
    else:

        winning_start = set([(0, 1), (1, 0), (0, 3), (1, 1)])
        losing_start = set([(0, 2), (2, 0)])
        i = 0
        while i < x0 + 1:
            j = 0
            while j < 10:
                thenext = nextstep(i, j)
                if (i, j) in winning_start:
                    print('in table building', 1)
                    pass
                elif (i, j) in losing_start:
                    print('in table building', 2)
                elif len(losing_start.intersection(thenext)) > 0:
                    # if player 1 has any option that is a losing start,he wins
                    print('(', i, ',', j, ')', ': Adding to winning start',
                          'because next', losing_start.intersection(thenext),
                          'is losing\n')
                    winning_start.add((i, j))

                elif len(winning_start.intersection(thenext)) == len(thenext):
                    # if player 1 gives player 2 only winning options, p1 loses
                    print('(', i, ',', j, ')', ': Adding to losing start',
                          thenext, ': because next is all winning\n')
                    losing_start.add((i, j))

                j += 1
            i += 1

    print('winning start', winning_start)
    print('losing start', losing_start)

    if (x0, y0) in winning_start:
        print('starting coords are winning for 1\n')
        return 1
    elif (x0, y0) in losing_start:
        print('starting coords are winning for 2\n')
        return 2
    else:
        print('need to do more searching')


# %%
towerBreakers(1, 1)  # n = # towers, m = height of each initially

towerBreakers(2, 1)

towerBreakers(1, 2)

towerBreakers(2, 2)

towerBreakers(3, 3)


towerBreakers(100, 1)

import pprint
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(winning_start)
pp.pprint(losing_start)

# %%
from collections import Counter


def towerBreakers(n, m):

    def get_all_factors(n):
        factors = []
        for i in range(1, n + 1):
            if n % i == 0:
                factors.append(i)

        return factors

    z = get_all_factors(m)
    nfactors = len(z)

    towers = [nfactors] * n
    print('towers:', towers, n, 'towers, of height', m)

    ctower = Counter(towers)
    print(ctower, sum(ctower.values()))
    print(Counter([0, 0]) == Counter([0, 1]))

    def nextstep(ctower):
        thenext = []
        keys = list(ctower)
        for key in keys:
            if key > 1:
                ctr_copy = copy.deepcopy(ctower)
                ctr_copy[key] = ctr_copy[key] - 1
                ctr_copy[1] = ctr_copy[1] + 1
                ctr_copy2 = copy.deepcopy(ctower)
                ctr_copy2[key] = ctr_copy2[key] - 1
                thenext.extend([ctr_copy, ctr_copy2])
        for key in to_remove:
            del counter_dict[key]

        print(thenext)
        thenext = set()
        # we need to change one tower; reducing it to 1, or 0 should
        # cover all cases, as 0 is even and 1 is odd
        if x0 > 0 and y0 > 0:
            thenext.add((max(x0 - 1, 0), y0 + 1))  # drop to 1 left in tower
            thenext.add((max(x0 - 1, 0), y0))      # drop to 0 in tower
            thenext.add((x0, max(y0 - 1, 0)))          # drop to 0 in tower

        elif x0 > 0 and y0 == 0:
            thenext.add((max(x0 - 1, 0), y0 + 1))  # drop to 1 left in tower
            thenext.add((max(x0 - 1, 0), y0))      # drop to 0 in tower
        elif x0 == 0 and y0 > 0:
            thenext.add((x0, max(y0 - 1, 0)))          # drop to 0 in tower
        print('current', x0, y0, 'the next', thenext)
        return thenext

    # if list(ctower.keys()) == [1]


towerBreakers(2, 1)

towers = [3, 3, 3]

# %% Practice Algorithms - Chess game


def chessboardGame(x, y):
    move = 0
    origin = [(x, y)]
    winning = set([(1, 1), (2, 1), (1, 2), (2, 2)])

    def isvalid(coord):
        return True if coord[0] >= 1 and coord[1] >= 1 else False

    def iswinning(coords):
        val = len(set(coords).intersection(winning))
        return (True if val > 0 else False)

    def nextsquares(origin):
        next = []
        for coords in origin:
            x = coords[0]
            y = coords[1]
            temp = [(x - 2, y + 1), (x - 2, y - 1),
                    (x + 1, y - 2), (x - 1, y - 2)]
            for coord in temp:
                if isvalid(coord) and coord not in next:
                    next.append(coord)
        return next

    next = nextsquares(origin)
    print(origin, '->', next)
    if iswinning(origin):
        print('origin prevents move')
        return 'Second'
    elif iswinning(next):
        print('First move can win')
        return 'First'
    while not iswinning(next):
        move += 1
        print('Player has a legal, nonwinning move', move, next)
        next = nextsquares(next)

    move += 1

    print('\n', next, 'Total moves', move)

    if move % 2 != 0:
        print('First')
        return 'First'
    else:
        print('Second')
        return 'Second'


chessboardGame(5, 3)


# %% Practice Algorithms - Game Theory -  Final chessgame

def chessboardGame(x, y):
    """Knight-like game.

    Need to determine winner based on starting position of 15x15 board
    """
    xin = x
    yin = y

    # These squares have no possible move, therefore, are losing;
    # we chose these squares by sight; while loop below expands these sets
    # until we encompass whole board
    # it was not clear to me in the beginning that every square has a unique
    # determinant ending under optimal play
    losing_start = set([(1, 1), (2, 1), (1, 2), (2, 2)])

    # These squares can jump to losing_start in one move, so are winning
    winning_start = set([(1, 3), (1, 4), (2, 3), (2, 4),
                         (3, 1), (3, 2), (3, 3), (3, 4),
                         (4, 1), (4, 2), (4, 3)])

    def nextset(x, y):
        def isvalid(coord):
            return True if coord[0] >= 1 and coord[1] >= 1 \
                and coord[0] <= 15 and coord[1] <= 15 else False

        nextsquares = [(x - 2, y + 1), (x - 2, y - 1), (x + 1, y - 2),
                       (x - 1, y - 2)]
        nextsquares = set([*filter(isvalid, nextsquares)])
        # print(nextsquares)
        return nextsquares

    # run a few times through whole board;
    # it takes 5 times to find a definitive win path for all 225 squares
    # 161 squares are winning for first player
    # 64 squares are losing starting for first player
    test_set = [(i, j) for i in range(1, 16) for j in range(1, 16)]
    times = 1
    while (len(winning_start) + len(losing_start)) < 225:
        for coords in test_set:
            x_ = coords[0]
            y_ = coords[1]
            thenextset = nextset(x_, y_)
            # print('testing', x_, y_, thenextset)

            if (x_, y_) in losing_start:
                # print('No Path, Second wins')
                pass
            elif (x_, y_) in winning_start:
                # print('One jump to terminal square, First wins')
                pass
            elif (len(winning_start.intersection(thenextset))
                  == len(thenextset)):
                # if next set ONLY includes winning_starts, First loses because
                # he has no choice but give win to opponent
                # need to add x,y to losing_start
                losing_start.add((x_, y_))
                # print('we lose, Second wins')
            elif len(losing_start.intersection(thenextset)) > 0:
                # if next set includes ANY losing_start, we win by choosing it
                # need to add x,y to winning_start
                winning_start.add((x_, y_))
                # print('First wins')
            else:
                # print('do not know')
                pass

        print('Run', times, len(winning_start) + len(losing_start))
        times += 1

    print(len(winning_start))
    print(len(losing_start))

    # prints schematic of Winor Loss of each of 15x15 squares

    print(' '.join(map(str, [i for i in range(1, 16)])))
    for i in range(15):
        row = ''
        for j in range(15):
            if test_set[i * 15 + j] in winning_start:
                row = row + 'W '
            else:
                row = row + 'L '
        print(row + str(i))

    if (xin, yin) in winning_start:
        print('First wins with', xin, yin)
        return 'First'
    else:
        print('Second wins with', xin, yin)
        return 'Second'


chessboardGame(15, 15)

# %% Practice Algorithms - Game Theory - Game of Stones


def gameOfStones(n):
    """Find who wins when given number of starting stones.

    Player1 and Player 2 move alternately, removing 2, 3, or 5 stones.
    Loser cannot make a move.  With optimal play, who wins?

    Three options after each play:
        lose?  nstones< 2
        win?  2, 3, 4, 5, 6 stones would give Player 1 the win because they
            could remove enough stones to leave only 1 stone.
        run through range of options, moving gradually higher, to see if
        player would win or lose with each higher n starting stones

        Could probably do a recursive solution as well.
    """
    winning_start = set([2, 3, 4, 5])  # 7 we don't know who would win yet
    losing_start = set([0, 1])

    def n_remain(n_):
        n_next = set([n_ - x for x in [2, 3, 5] if n_ - x >= 0])
        return n_next

    i = 0
    while i < n + 1:
        x = n_remain(i)
        # let's build lookup table to see what each value of up to n yields as
        # far as winner under optimal play
        if i in winning_start:
            # print('Player 1 wins')
            pass
        elif i in losing_start:
            # print('Player 2 wins')
            pass
        elif len(losing_start.intersection(x)) > 0:
            # if player 1 has any option that is a losing start, he win
            print(i, 'Adding to winning start', x)
            winning_start.add(i)
        elif len(winning_start.intersection(x)) == len(x):
            # if player 1 gives player 2 only winning options, p1 loses
            print(i, 'Adding to losing start', x)
            losing_start.add(i)
        else:
            print('this should never happen')
        i += 1

    if n in winning_start:
        print('First')
        return 'First'
    else:
        print('Second')
        return 'Second'


for i in range(1, 8):
    gameOfStones(i)

gameOfStones(10)

# %% Practice Algorithms - Debugging


def findZigZagSequence(a, n):
    a.sort()
    mid = int((n + 1) / 2) - 1  # 1st change
    a[mid], a[n - 1] = a[n - 1], a[mid]

    st = mid + 1
    ed = n - 2  # 2nd change
    while(st <= ed):
        a[st], a[ed] = a[ed], a[st]
        st = st + 1
        ed = ed - 1  # 3rd change

    for i in range(n):
        if i == n - 1:
            print(a[i])
        else:
            print(a[i], end=' ')
    return


arr = [1, 2, 3, 4, 5, 6, 7]
findZigZagSequence(arr, len(arr))

# %%  Hackerrank Python basic Certification - Classes problem, 1 of 2
# All tests passed
# Finished in about 20minutes
# Had 35 min to spare at end
import os


class Multiset:
    """Init method.

    Py 3 has no ()s in class definition, but in Python 2 one could extend
    from another object using Multiset(<object>)
    """

    def __init__(self):
        """Init method."""
        self.lookup = {}

    def add(self, val):
        """Add one occurrence of val from the multiset, if any."""
        self.lookup[val] = self.lookup.get(val, 0) + 1

    def remove(self, val):
        """Remove one occurrence of val from the multiset, if any."""
        if self.lookup.get(val, 0) > 0:
            self.lookup[val] = self.lookup.get(val, 0) - 1

    def __contains__(self, val):
        """Return True when val is in the multiset, else returns False."""
        if self.lookup.get(val, 0) > 0:
            return True
        else:
            return False

    def __len__(self):
        """Return the number of elements in the multiset."""
        return sum(list(self.lookup.values()))


if __name__ == '__main__':

    def performOperations(operations):
        """performOperations."""
        m = Multiset()
        result = []
        for op_str in operations:
            elems = op_str.split()
            if elems[0] == 'size':
                result.append(len(m))
            else:
                op, val = elems[0], int(elems[1])
                if op == 'query':
                    result.append(val in m)
                elif op == 'add':
                    m.add(val)
                elif op == 'remove':
                    m.remove(val)
        return result

    q = int(input())
    operations = []
    for _ in range(q):
        operations.append(input())

    result = performOperations(operations)

    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    fptr.write('\n'.join(map(str, result)))
    fptr.write('\n')
    fptr.close()


"""Test cases
12
query 1
add 1
query 1
remove 1
query 1
add 2
add 2
size
query 2
remove 2
query 2
size

Expected Output
False
True
False
2
True
True
1
"""


# %%  Hackerrank Python basic Certification - Python String Representation
# All tests passed
# Finished in 25 minutes
import os


class Car:
    """Init car object."""

    def __init__(self, max_speed, speed_unit):
        """Init method."""
        self.max_speed = max_speed
        self.speed_unit = speed_unit

    def __str__(self):
        """For printing method output."""
        return "Car with the maximum speed of " + str(self.max_speed) + \
            " " + self.speed_unit


class Boat:
    """Init car object."""

    def __init__(self, max_speed):
        """Init method."""
        self.max_speed = max_speed

    def __str__(self):
        """For printing method output."""
        return "Boat with the maximum speed of " + str(self.max_speed) \
            + " knots"


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    q = int(input())
    queries = []
    for _ in range(q):
        args = input().split()
        vehicle_type, params = args[0], args[1:]
        if vehicle_type == "car":
            max_speed, speed_unit = int(params[0]), params[1]
            vehicle = Car(max_speed, speed_unit)
        elif vehicle_type == "boat":
            max_speed = int(params[0])
            vehicle = Boat(max_speed)
        else:
            raise ValueError("invalid vehicle type")
        fptr.write("%s\n" % vehicle)
    fptr.close()

"""
2
car 151 km/h
boat 77

Car with the maximum speed of 151 km/h
Boat with the maximum speed of 77 knots

"""

# %%# %%  Hackerrank Problem Solving Basic
# !/bin/python3
import os


def filledOrders(order, k):
    """Complete the 'filledOrders' function below.

    The function is expected to return an INTEGER.
    The function accepts following parameters:
        1. INTEGER_ARRAY order
        2. INTEGER k
    """
    max_filled = 0
    cum_ordered = 0
    order.sort()
    for i in range(len(order)):
        cum_ordered = cum_ordered + order[i]
        if cum_ordered <= k:
            max_filled = i + 1
    return max_filled


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    order_count = int(input().strip())
    order = []

    for _ in range(order_count):
        order_item = int(input().strip())
        order.append(order_item)

    k = int(input().strip())

    result = filledOrders(order, k)

    fptr.write(str(result) + '\n')

    fptr.close()
# %%# %%  Hackerrank Problem Solving Basic

# !/bin/python3
# import math
import os


def getMinCost(crew_id, job_id):
    """Complete the 'getMinCost' function below.

    The function is expected to return a LONG_INTEGER.
    The function accepts following parameters:
        1. INTEGER_ARRAY crew_id
        2. INTEGER_ARRAY job_id
    """
    crew_id.sort()
    job_id.sort()    # Write your code here
    min_dist = 0
    for i in range(len(crew_id)):
        min_dist = min_dist + abs(crew_id[i] - job_id[i])

    return min_dist


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    crew_id_count = int(input().strip())

    crew_id = []

    for _ in range(crew_id_count):
        crew_id_item = int(input().strip())
        crew_id.append(crew_id_item)

    job_id_count = int(input().strip())

    job_id = []

    for _ in range(job_id_count):
        job_id_item = int(input().strip())
        job_id.append(job_id_item)

    result = getMinCost(crew_id, job_id)

    fptr.write(str(result) + '\n')

    fptr.close()

# %%# %%  Hackerrank Problem Solving Intermediate
# this is bad  approach
freq = [0, 1, 3]


def taskOfPairing(freq):
    """Docstring."""
    wts = []
    for i in range(len(freq)):
        wts.append((i + 1) * freq[i])
    print(wts)
    weights = ''.join(wts)
    print(weights)
    i = 1
    pairs = 0
    while i < len(weights):
        if int(weights[i]) <= int(weights[i - 1]) + 1:
            pairs += 1
            i = i + 2
    print(pairs)
    return pairs


freq = [0, 1, 0, 1, 0, 1, 0]
taskOfPairing(freq)
# %%
# not sure why it didn't work...this was fairly efficient I think


def taskOfPairing(freq):
    """Docstring."""
    pairs = freq[0] // 2
    leftover = freq[0] % 2
    print(freq)
    for i in range(1, len(freq)):
        pairs = pairs + freq[i] // 2
        if leftover == 1 and freq[i] % 2 == 1:
            pairs += 1
            leftover = 0
            print('i', i, 'Pairs:', pairs, '0 lo', leftover)
        elif leftover == 0 and freq[i] % 2 == 1:
            leftover = 1
            print('i', i, 'Pairs:', pairs, '+1 lo', leftover)
        else:  # this case is freq[i]%2 == 0
            leftover = 0
            print('i', i, 'Pairs:', pairs, 'no leftovers', leftover)

    print(pairs)
    return pairs


freq = [0, 1, 3, 4, 5, 6, 7, 7, 7, 7]
freq = [0, 1, 0, 1, 0, 1, 3]


taskOfPairing(freq)
#  this has memory error

# %% this still had only 3 test cases passing


def taskOfPairing(freq):
    """Create lists - memory intensive therefore a bad approach."""
    n_pairs = list(map(lambda x: x // 2, freq))
    remainders = list(map(lambda x: x % 2, freq))
    print(freq)
    print(n_pairs)
    print(remainders)

    total = sum(n_pairs)
    print(total)
    i = 1
    while i < len(freq):
        print('i', i, remainders[i], remainders[i - 1])
        if remainders[i] + remainders[i - 1] == 2:
            print('total', total, 'i', i)
            total = total + 1
            i = i + 2
            print('newtotal', total, 'i', i)
        else:
            i += 1
    print(total)
    return total


freq = [0, 1, 3, 4, 5, 6, 7, 7, 7, 7]
taskOfPairing(freq)


# %% Hackerrank Certification - Intermediate Problem Solving -
# this is bitwise operation exercise
import math


def countPairs(arr):
    """Find pairs where bitwise &  x&y is a power of 2.

    i.e.,log2(x&B) is equal to round integer
    key issues I encountered:
        - using math.log2 (log2 is undefined)
        - anticipating boundary condition that log(0)= Inf, which cannot be
          turned into integer for comparison in the elif

    THIS IS THE TRICK
    Any power of 2, minus 1 would have binary representation where all
    bits are set except the bit that represents the next number
    (which is a power of 2).

    Using the bitwise AND with the number n and n-1, we can conclude
    that the number in question is a power of two if it
    returns 0. These are just some of the many use cases of bitwise
    operators.
    n   &  n-1
    ---    ---
    10 & 01 = 0
    100 & 011 = 0
    """
    pairs = 0
    for i in range(len(arr) - 1):
        for j in range(i + 1, len(arr)):
            if arr[i] & arr[j] == 0:  # log 0 = Infinity...avoid this problem
                pass
            elif int(math.log2(arr[i] & arr[j])) == math.log2(arr[i] & arr[j]):
                print('(', arr[i], '&', arr[j], ')', '=', arr[i] & arr[j],
                      '=2^', int(math.log2(arr[i] & arr[j])))
                pairs += 1
                # print('Pairs:', pairs, '\n')
    print(pairs)
    return pairs


countPairs([10, 2, 3, 4, 8])


# math domain error == CANNOT CALCULATE LOG 0, which could
# happen with bitwise operation,would generate infinity, which cannt be turned
# into integer

for i in range(100):
    print(i, i & (i - 1), 'a power of two' * (not i & (i - 1)))

# %%
# https://medium.com/better-programming/5-pairs-of-magic-methods-in-python-you-should-know-f98f0e5356d6


class Product:
    """Example."""

    def __init__(self, name, price):
        self.name = name
        self.price = price

    def __repr__(self):
        """Reproduce."""
        print(f"Product({self.name!r}, {self.price})")
        # we need the !r because {self.name} generates thing with no
        # quotes, which is not correct;
        # {self.price} genreates same as {self.price!r}
        print(f"Product({self.name!s}, {self.price})")

        return f"Product({self.name!r}, {self.price!r})"

    def __str__(self):
        """Print."""
        return f"Product: {self.name}, ${self.price:.2f}"


# =============================================================================
# Three conversion flags are currently supported: '!s' which calls
# str() on the value, '!r' which calls repr() and '!a' which calls
# ascii().
#
# Some examples:
#
# "Harold's a clever {0!s}"        # Calls str() on the argument first
# "Bring out the holy {name!r}"    # Calls repr() on the argument first
# "More {!a}"                      # Calls ascii() on the argument first
#
# =============================================================================
x = Product('thing', 150)
print(x)
print(repr(x))
