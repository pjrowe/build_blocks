# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 16:33:40 2020

@author: Trader
"""

# %% PROBLEM SOLVING BADGE
"""
Problem Solving (Algorithms and Data Structures)

4 Star - Silver 475 --> Currently I have 366 points for 17 solved
5 Star - Gold   850
6 Star - Gold  2200

Subdomains
n   Category                  Done
--  --------                  -------
10  Warmup                    10 / 10 x
66  Implementation            46 / 66
10  Strings                   13 / 45
15  Sorting                   2  / 15
26  Search                    .
    --127                     -------

64  Graph Theory              .
24  Greedy                    1 /
99  Dynamic Programming       .
11  Constructive Algos        1 /
27  Bit Manipulation          1 /
    --225                     -------

11  Recursion                 .
33  Game Theory               .
 4  NP Complete               .
 5  Debugging                 .
--  --------53                -------
//405 so far                  17 for

as of 1/7/2020: 861/2200 points
"""

# %% Practice - Algorithms - Warmup - Birthday Cake Candles
# function that counts how many of the tallest candles are on cake
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
    if x1 < x2 and v1 > v2 and (x2-x1) % (v1-v2) == 0:
        return 'YES'
    elif x1 > x2 and v1 < v2 and (x1-x2) % (v2-v1) == 0:
        return 'YES'
    else:
        return 'NO'

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
    for i in range(m, len(s)+1):
        print(i-m, i, s[i-m:i])
        if sum(s[i-m:i]) == d:
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
    fairsplit = (sum(bill)-bill[k])/2
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
        return min((n-p) // 2, (p-0) // 2)

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
    if (abs(x-z)) < (abs(y-z)):
        return 'Cat A'
    elif (abs(x-z)) > (abs(y-z)):
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
        if arr[i] == arr[i-1] and arr[i] == start:
            # same # as start
            length += 1
            max_length = max(max_length, length)
        elif arr[i] == arr[i-1] + 1 and arr[i] == start + 1:
            # ending value of subarr
            length += 1
            max_length = max(max_length, length)
        elif arr[i] == arr[i-1] and arr[i] == start + 1:
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

    return 1*len(word)*maxheight

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
    print('both', len(s)+len(t))
    print('---')

    if k >= len(s) + len(t):
        print('Yes')
        return 'Yes'
    elif k >= special_value and k < len(s) + len(t) and (k-special_value) %2 == 0:
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
    for i in range(a, b+1):
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
    sums_elements = [[x for x in list(zip(*i))] for i in combinations(students, 2)]
    sums = [sum([x[0] or x[1] for x in list(zip(*i))]) for i in combinations(students, 2)]
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
    for n in range(p, q+1):
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
        for j in range(i+1, len(arr)):
            if arr[i] == arr[j]:
                pairs.append((i, j))
    if len(pairs) != 0:
        distance = min([abs(x[0]-x[1]) for x in pairs])

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

        if (p - i*d) <= m:
            cost = m
        else:
            cost = p - i*d
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
        exit_i = case[1]+1
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
        print('\nnprobs',  n_probs, 'npages', n_pages)
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

# %% Practice Algorithms: Implementation -


def flatlandSpaceStations(n, c):
    """FlatlandSpaceStations."""
    c.sort()
    maxdist = 0
    maxdist = max(c[0], n - 1 - c[-1])
    print(maxdist)

    for i in range(1, len(c)):
        dist = math.ceil((c[i]-c[i-1]-1)/2)
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
            if B[i] % 2 == 0 and B[i-1] % 2 == 0:
                i += 1

            # even, odd -> pass
            elif B[i] % 2 == 1 and B[i-1] % 2 == 0:
                i += 1

            elif B[i] % 2 == 1 and B[i-1] % 2 == 1:
                loaves += 2
                i += 2
            elif B[i] % 2 == 0 and B[i-1] % 2 == 1:
                B[i] += 1
                loaves += 2
                i += 1
    print(loaves)
    return loaves


fairRations([2, 3, 4,  5, 6])


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

    for row in range(1, n-1):
        for col in range(1, n-1):
            cavity = n_grid[row][col]
            upper = n_grid[row-1][col]
            lower = n_grid[row+1][col]
            left = n_grid[row][col-1]
            right = n_grid[row][col+1]

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
        last_numbers.append(a*(n-1-i) + b*i)
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
            if not (b[i] == b[i-1] or b[i] == b[i+1]):
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
        upper = upper + 3*2**power
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
        elif reduced[i] == reduced[i+1]:
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
        return (6-n)
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
            if st[i] == st[i-1]:
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
        if s_values[i] == s_values[i-1]:
            cum_values.append(cum_values[i-1] + s_values[i])
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
        if s[i] == s[i-1]:
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
        while current_end != n and (end+len(str(next_int))) <= n:
            print('------\nCurrent:', current, 'from', start, ':', end)
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
# %% Practice Algorithms: Data Structures/Strings-
# Funny Numbers


def funnyString(s):
    """funnyString."""
    x = [abs(ord(s[i])-ord(s[i-1])) for i in range(1, len(s))]
    y = [abs(ord(s[len(s) - 1 - i])-ord(s[len(s) - 1 - (i-1)]))
         for i in range(1, len(s))]

    if x == y:
        return 'Funny'
    else:
        return 'Not Funny'


print(funnyString('abc'))
print(funnyString('abd'))


# %% Gemstones

def gemstones(arr):
    """gemstones."""
    possible_gems = set(arr[0])
    for rock in arr[1:]:
        possible_gems = possible_gems.intersection(set(rock))

    n_gemstones = len(list(possible_gems))
    return n_gemstones


arr = ['abcdde', 'baccd', 'eeabg']
print(gemstones(arr))
# %%
