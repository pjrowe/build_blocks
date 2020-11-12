"""Hackerrank quizzes

Reference and progress for hackerrank skills / problems

Four skills
------------------
1. Python 86/115
    - most of the examples below are for python challenges
    **end python
2. SQL 52/58
    - the challenges are quite simple, so I don't include them here
    **end sql
3. Regex 34/47
    - see below for cells with challenges, title includes Regex
4. Interview Prep 7/69

"""
# %% PYTHON - 115
"""
- 7 Introduction
- 6 Basic Data Types
- 14 Strings
- 13 Sets
- 7 Math
    47
- 7 Itertools
- 8 Collections
- 2 Date and Time
- 2 Errors and Exceptions
- 2 Classes
    21

- 6 Built-Ins
- 3 Python Functionals
-17 Regex and Parsing
- 2 XML
- 2 Closures and Decorators
    30

-15 Numpy
- 2 Debugging
    17
===
115
"""

# %%  PYTHON - BASIC DATA TYPES - Find the Runner-Up Score!
# First Try 1 : no need to do a sort
# Find the Runner-Up Score!

if __name__ == '__main__':
    n = int(input())
    array = []
    a_ = input().split(' ')

    for i in range(n):
        array.append(int(a_[i]))

    array.sort()
    # print(array)
    runnerup = array[-1]
    for i in range(2, n + 1):
        # note how we need to added n+1 to get to
        # n in reverse parsing
        if array[-i] < array[-1]:
            # print('check',array[-i])
            runnerup = array[-i]
            break

    print(runnerup)

# Try 2 : more straightforward is to use set operation
# https://www.hackerrank.com/challenges/find-second-maximum-number-in-a-list
# solved 10/31/2020

if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    arrset = set(arr)
    lessset = arrset - {max(arrset)}
    runnerup = max(lessset)
    print(runnerup)

# %% PYTHON - BASIC DATA TYPES - Finding the percentage - 06/19/2020

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
#    print(n)
    for _ in range(n):
        line = input().split()
        name, scores = line[0], line[1:]
        scores = [float(scores[i]) for i in range(3)]
        student_marks[name] = scores
    query_name = input()
    out = sum(student_marks[query_name]) / len(student_marks[query_name])
    print("%.2f" % round(out, 2))

# %% PYTHON - BASIC DATA TYPES - List Comprehensions
# solved 6/20/2020
# return 3d space in cube defined by origin and X,Y,Z where i + j + k != n

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    out1 = [[i, j, k] for i in range(x + 1) for j in range(y + 1) for k in
            range(z + 1) if sum([i, j, k]) != n]
    print(out1)


# %% PYTHON - BASIC DATA TYPES - Nested lists
# - print alphabetically sorted list of student names that
# have second highest scores
#  completed 6/20/2020

if __name__ == '__main__':
    low_score = 100
    second_low_score = 100
    name_score = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        insert_list = [name, score]
        if score < low_score:
            second_low_score = low_score
            low_score = score
        elif (score > low_score) & (score < second_low_score):
            second_low_score = score
        name_score.append(insert_list)
#        print(low_score, second_low_score)
    out = [name_score[i][0] for i in range(len(name_score)) if
           name_score[i][1] == second_low_score]
    out.sort()
    for _ in out:
        print(_)


# %% PYTHON - STRINGS - Merge the Tools!
# worked on all 16 test cases 6/23/2020


def merge_the_tools(string, k):
    # your code goes here
    n = int(len(string)/k)
    for i in range(n):
        ministr = string[i*k:i*k+k]
        mini = ministr[0]
        for i in range(1, len(ministr)):
            if ministr[i] in ministr[0:i]:
                pass
            else:
                mini = mini + ministr[i]
        print(mini)

# testing
# string = 'abccefggi'
# k=3
# merge_the_tools(string, k)


if __name__ == '__main__':
    string = input()
    k = int(input())
    merge_the_tools(string, k)


# %%  PYTHON - STRINGS - Capitalize

def solve(s):
    mylist = s.split(' ')  # this doesn't join consecutive whitespace
    for i in range(len(mylist)):
        mylist[i] = mylist[i].capitalize()
    return ' '.join(mylist)



# %% PYTHON - ITERTOOLS -

"""
itertools.permutations(iterable[, r])
This tool returns successive
length permutations of elements in an iterable.
If is not specified or is None, then defaults to the length of the
iterable, and all possible full length permutations are generated.

Permutations are printed in a lexicographic sorted order. So, if the
input iterable is sorted, the permutation tuples will be produced in a
sorted order.
"""
#mylist = 'HACK 2'.split()

from itertools import permutations

mylist = input().split()
strings = list(mylist[0])
k =  int(mylist[1])
strings.sort()
for i in permutations(strings, k):
    print(''.join(i))

# %% PYTHON - ITERTOOLS -itertools.combinations_with_replacement(iterable, r)

"""
itertools.combinations_with_replacement(iterable, r)
This tool returns

length subsequences of elements from the input iterable allowing individual
elements to be repeated more than once.

Combinations are emitted in lexicographic sorted order. So, if the input
iterable is sorted, the combination tuples will be produced in sorted order.
"""
from itertools import combinations_with_replacement

mylist = input().split()
strings = list(mylist[0])
k =  int(mylist[1])
strings.sort()
for i in combinations_with_replacement(strings, k):
    print(''.join(i))


# %% PYTHON - ITERTOOLS - Iterables and Iterators

from itertools import combinations

mylist = input().split()
strings = list(mylist[0])
k =  int(mylist[1])
strings.sort()
for j in range(1,k+1):
    for i in combinations(strings, j):
        print(''.join(i))

# %%

mylist = list(map(int, input().split()))
N = mylist[0]
M = mylist[1]

lists =[]
for i in range(N):
    lists.append(list(map(int, input().split())))

# %% PYTHON - ITERTOOLS - Iterables and Iterators
# rather than trying to figure out the theory and math, just implement
# the code to figure it out

N = int(input())
mylist = input().split()
k = int(input())

comb = list(combinations(mylist, k))
tot = len(comb)

count = 0
for i in comb:
    if 'a' in i: count+=1

print(count/tot)


# %% PYTHON - BASIC DATA TYPES - Lists
# take N on first line as the # lines you will get as input
# evaluate thelist commands given on the list

n = int(input())
mylist = []

for _ in range(n):
    s = input().split()
    cmd = s[0]
    args = s[1:]
    if cmd != "print":
        cmd += "(" + ",".join(args) + ")"
        eval("mylist." + cmd)
        # eval is the key command I was unaware of
    else:
        print(mylist)

# %% PYTHON - STRINGS -swap case

    n = int(input())
    integer_list = map(int, input().split())
    print(hash(tuple(integer_list)))

s.swapcase()  # changes case of string character by Character

# %% PYTHON - STRINGS - text alignment
thickness = int(input())  # This must be an odd number
c = 'H'

# Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1) + c + (c*i).ljust(thickness-1))

# Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2) +
          (c * thickness).center(thickness * 6))

# Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))

# Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)
          + (c*thickness).center(thickness*6))

# Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness) + c +
           (c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

# %% PYTHON - STRINGS - String validation
s = input()
x = list(s)
# need to break string into separate elements to test if
# any character satisfies condition
print(sum([y.isalnum() for y in x]) > 0)
print(sum([y.isalpha() for y in x]) > 0)
print(sum([y.isdigit() for y in x]) > 0)
print(sum([y.islower() for y in x]) > 0)
print(sum([y.isupper() for y in x]) > 0)
"""
test if alphanumeric characters
test if alphabetical characters
if digits
if any lowercase
if uppercase

s.isalnum() is True if all alphanumeric
s.isalpha() True if all alphabetic
s.isdigit() True if all numbers
s.islower() True if all lower
s.isupper() True if all upper
"""
# %% PYTHON - SETS - set operations Set Mutations
# Both the discard() and remove() functions take a single value as an
# argument and removes that value from the set. If that value is not present,
# discard() does nothing, but remove() will raise a KeyError exception.

"""
.union()
.difference()
.intersection()
.discard()
.remove()
.update()

.update() or |=
Update the set by adding elements from an iterable/another set.

.intersection_update() or &=
Update the set by keeping only the elements found in it and an iterable/
another set.

.difference_update() or -=
Update the set by removing elements found in an iterable/another set.

.symmetric_difference_update() or ^=
Update the set by only keeping the elements found in either set, but not in
both.

"""

n = int(input())
A = set(map(int, input().split()))
m = int(input())

sets = []
for i in range(m):
    command, _ = input().split()
    subset = set(map(int, input().split()))
    eval('A.' + command + '(subset)')
print(sum(list(A)))

# %% PYTHON - SETS - set add
# gets N, which is number of lines we need to read in
# .add() allows us to add elements to a set
# len(stamps) returns how many elements in set at end
x = int(input())
stamps = set()
for i in range(x):
    stamps.add(input())
print(len(stamps))

# %% PYTHON - SETS - symmetric difference
# returns elements outside intersection oftwo iterables, ie.,in one or other
# but not in both

s ^ set('Rank')  # ^ replaces .symmetric_difference but only operates on
                 # in set elements
                 # s is not mutated by ^
s.symmetric_difference({'Rank':1})

# %% PYTHON - MATH
import cmath

print(*cmath.polar(complex(input())), sep='\n')

# %% PYTHON - MATH -

import cmath
print(*cmath.polar(complex(input())), sep='\n')
x=cmath.polar(complex(input()))

# %% PYTHON - ITERTOOLS - max product
# max product, done with generator object, which I should learn how to use
# better

from itertools import product

K, M = map(int,input().split())
N = list((square(list(map(int, input().split()))[1:]) for _ in range(K)))
results = map((lambda x: sum(i) for i in x) % M, product(*N))
print(max(results))
# correct answer is 206, but there is some deprecated code here

# %%  PYTHON - ITERTOOLS - max product
from itertools import product

N, M = map(int, input().split())

sq_lists =[]
for i in range(N):
    x = list(map(int, input().split()))
    sq_lists.append([x**2 for x in x[1:]])

checklist = list(product(*sq_lists))

newmax = 0
imax = 0
for i in range(len(checklist)):
    if sum(checklist[i]) % M > newmax:
        newmax = sum(checklist[i]) % M
    imax = i

checkmax = checklist[imax]
print(newmax)  # correct answer is 206

# %% PYTHON - ITERTOOLS - itertools
# other to do puzzles of hackerrank
"""
itertools.product()

"""
from itertools import product

x = list(map(int, input().split()))
y = list(map(int, input().split()))
print(*product(x, y))  # unpack the product iterable


# %%
# generator object
gen = ((x,y) for x in [1,2] for y in [3,4])
for i in gen:
    print(i)

# %%
n = int(input())
print(all([int(input().split()) > 0 for _ in range(n)]))

# %% PYTHON - COLLECTIONS - Word Order
# completed all 7 test cases 6/24/2020

n = int(input())
words = []
lookup = {}

for i in range(n):
    word = input()
    if word in lookup:
        lookup[word] += 1
    else:
        words.append(word)
        lookup[word] = 1

out = str(lookup[words[0]])
for i in words[1:]:
    out = out + ' ' + str(lookup[i])

print(len(words))
print(out)

# %% # PYTHON - COLLECTIONS - Company Logo
# this works but uses higher order python (pandas)
# not sure how I would do with just low level

import math
import os
import random
import re
import sys
import pandas as pd


if __name__ == '__main__':
    s = input()
    letters = []
    lookup = {}

    for letter in s:
        if letter in lookup:
            lookup[letter] += 1
        else:
            letters.append(letter)
            lookup[letter] = 1

x = pd.DataFrame([lookup.keys(),lookup.values()])
x = x.transpose()
x = x.sort_values(by=[1, 0], ascending=[False, True]).reset_index(drop=True)

for i in range(3):
    print(x.iloc[i, 0], x.iloc[i, 1])

# %% # PYTHON - COLLECTIONS - Company Logo --- suggested solution
# but doesn't work because requires import of already written code

from collections import Counter
z = Counter()

x = 'hellfireehi'
y = 'akxlljookbbccc'
[print(*c) for c in Counter(sorted(x)).most_common(3)]

Counter(sorted(x)).items()

[print(*c) for c in Counter(sorted(input())).most_common(3)]

# %%  PYTHON - COLLECTIONS - Counter
# done 11/2/2020
# https://www.hackerrank.com/challenges/collections-counter/problem
from collections import Counter

n_shoes = int(input())
sizes = list(map(int, input().split()))
n_customers = int(input())
count_sizes = Counter(sizes)

sold = 0
shoe = 0

for i in range(n_customers):
    data = list(map(int, input().split()))
    shoe = data[0]
    if count_sizes[shoe] > 0:
        sold += data[1]
        count_sizes[shoe] -= 1
print(sold)

# %% PYTHON - COLLECTIONS - default dict
"""The defaultdict tool is a container in the collections class of Python.
It's similar to the usual dictionary (dict) container, but the only difference
 is that a defaultdict will have a default value if that key has not been set
 yet.
 If you didn't use a defaultdict you'd have to check to see if that key
 exists, and if it doesn't, set it to what you want.
"""

from collections import defaultdict

d = defaultdict(list)
d['python'].append("awesome")
d['something-else'].append("not relevant")
d['python'].append("language")
for i in d.items():
    print(i)

# %% PYTHON - COLLECTIONS - deque

from collections import deque

d = deque()
N = int(input())

for i in range(N):
    command = input().split()
    if command[0] == 'pop':
        d.pop()
    elif command[0] == 'popleft':
        d.popleft()
    else:
        eval('d.' + command[0] + '(' + command[1] + ')')

print(*d)


# %% PYTHON - COLLECTIONS - named tuple
"""
collections.namedtuple()

Basically, namedtuples are easy to create, lightweight object types.
They turn tuples into convenient containers for simple tasks.
With namedtuples, you don’t have to use integer indices for accessing members
of a tuple.

Example

Code 01

>>> from collections import namedtuple
>>> Point = namedtuple('Point', 'x, y')
>>> pt1 = Point(1, 2)
>>> pt2 = Point(3, 4)
>>> dot_product = ( pt1.x * pt2.x ) + ( pt1.y * pt2.y )
>>> print dot_product
11

Code 02

>>> from collections import namedtuple
>>> Car = namedtuple('Car','Price Mileage Colour Class')
>>> xyz = Car(Price=100000, Mileage=30, Colour='Cyan', Class='Y')
>>> print xyz
Car(Price=100000, Mileage=30, Colour='Cyan', Class='Y')
>>> print xyz.Class
Y
"""
from collections import namedtuple

N = int(input())
fields = input().split()

total = 0
for i in range(N):
    students = namedtuple('student',fields)
    field1, field2, field3,field4 = input().split()
    student = students(field1,field2,field3,field4)
    total += int(student.MARKS)
print('{:.2f}'.format(total/N))
# %% PYTHON - TIME DATE - calendar
import calendar

print(calendar.TextCalendar(firstweekday=6).formatyear(2015))
month, day, year = map(int, input().split())
d = calendar.weekday(year, month, day)
print(calendar.day_name[d].upper())

# %% PYTHON - COLLECTIONS - default dict

n, m = (int(x) for x in input().split())
A = defaultdict(list)
B = defaultdict(list)
for i in range(1, n + 1):
    A[input()].append(i)
for i in range(1, m + 1):
    B[input()].append(i)
for word in B.keys():
    if word in A:
        print(*(x for x in A[word]))
    else:
        print(-1)

# %% PYTHON - COLLECTIONS - default dict
from collections import defaultdict

A = defaultdict(list)
B = []

n, m = map(int, input().split())

for i in range(0, n):
    A[input()].append(i+1)

for i in range(0, m):
    B = B + [input()]

for i in B:
    if i in A:
        print(" ".join(map(str, A[i])))
    else:
        print(-1)

# %% PYTHON - COLLECTIONS - ordered dict

# An OrderedDict is a dictionary that remembers the order of the keys that
# were inserted first. If a new entry overwrites an existing entry, the
# original insertion position is left unchanged.
from collections import OrderedDict

ordered_dictionary = OrderedDict()
ordered_dictionary['a'] = 1
ordered_dictionary['c'] = 3
ordered_dictionary['b'] = 2
ordered_dictionary['c'] = 4

print(ordered_dictionary)
# OrderedDict([('a', 1), ('c', 4), ('b', 2)])

from collections import OrderedDict

shop_list = OrderedDict()
n = int(input())
for i in range(n):
    item, space, price = list(input().rpartition(' '))
    shop_list[item] = shop_list.get(item, 0) + int(price)
for item, price in shop_list.items():
    print(item, price)


# %%  PYTHON - ERRORS AND EXCEPTIONS - Exceptions
"""ONLY 2 EXERCISES IN ERRORS/EXCEPTIONS"""

x = int(input())
for i in range(x):
    try:
        a, b = map(int, input().split())
        print(a//b)
    except Exception as e:
        print("Error Code:", e)

        # except ZeroDivisionError as e:
        # except ValueError as e:

# %% PYTHON - ERRORS AND EXCEPTIONS - Incorrect Regex

# 89,928 rank before, 80200 after; wow, that's quite a jump
import re
for _ in range(int(input())):
    ans = True
    try:
        reg = re.compile(input())
    except re.error:
        ans = False
    print(ans)

# %% PYTHON - BUILT INS - any and all
# - are all integers in array positive?
# - are any of them palindromes, i.e., reflect around vertical
# axis in middle of number?

n = int(input())
ns = list(map(int, input().split()))
palin = any([str(x)[0:len(str(x))//2]==str(x)[-1:len(str(x))//2-1:-1] for x in ns])
print(bool(all([x > 0 for x in ns])*palin))

# %% PYTHON - FUNCTIONALS - map and lambda

def fib(n):
    if n == 0: mylist = []
    elif n == 1: mylist = [0]
    elif n == 2: mylist = [0, 1]
    elif n == 3:  mylist = [0, 1, 1]
    else:
        mylist = [0, 1, 1]
        for i in range(n-3):
            mylist.append(mylist[-2] + mylist[-1])
    return mylist

print(list(map(lambda x: x**3, fib(5))))

# %% PYTHON - REGEX AND PARSING - re.split
import re

x = '100,000,000.000'
print('\n'.join(re.split('[,.]', x)))


# %% PYTHON - REGEX AND PARSING - HTML Parser - Part 1

"""

from HTMLParser import HTMLParser

# create a subclass and override the handler methods
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print "Found a start tag  :", tag
    def handle_endtag(self, tag):
        print "Found an end tag   :", tag
    def handle_startendtag(self, tag, attrs):
        print "Found an empty tag :", tag

# instantiate the parser and fed it some HTML
parser.feed("<html><head><title>HTML Parser - I</title></head>"
            +"<body><h1>HackerRank</h1><br /></body></html>")
"""
from html.parser import HTMLParser
# HTMLParser ONLY EXISTS IN PYTHON 2 -- so needed to use html.parser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Start :", tag)
        for name, value in attrs:
            try:
                print("-> {0} > {1}".format(name, value))
            except Exception:
                print("-> {0} > None".format(name))

    def handle_endtag(self, tag):
        print("End   :", tag)
        # very important that we have 3 spaces after 'End' so that colons
        # line up for after Start, End, and Empty.

    def handle_startendtag(self, tag, attrs):
        print("Empty :", tag)
        for name, value in attrs:
            try:
                print("-> {0} > {1}".format(name, value))
            except Exception:
                print("-> {0} > None".format(name))

parser = MyHTMLParser()

N = int(input())
for i in range(N):
    html_text = input()
    parser.feed(html_text)

# %% PYTHON - REGEX AND PARSING - check if input is a float
import re

for _ in range(int(input())):
    print(bool(re.match(r'^[-+]?[0-9]*\.[0-9]+$', input())))

# %% PYTHON - REGEX AND PARSING - findall, finditer
import re

s = input()
pat = r'(?<=[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm])([aeiouAEIOU]{2,})(?=[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm])'
matches = list(map(lambda x: x.group(), re.finditer(pat, s)))
if matches==[]:
    print(-1)
else:
    for x in range(len(matches)):
        print(matches[x])


# %% PYTHON - REGEX AND PARSING - .group(). .groups(), .groupdict()
"""
group()
group() expression returns one or more subgroups of the match.
>>> import re
>>> m = re.match(r'(\w+)@(\w+)\.(\w+)','username@hackerrank.com')
>>> m.group(0)       # The entire match
'username@hackerrank.com'
>>> m.group(1)       # The first parenthesized subgroup.
'username'
>>> m.group(2)       # The second parenthesized subgroup.
'hackerrank'
>>> m.group(3)       # The third parenthesized subgroup.
'com'
>>> m.group(1,2,3)   # Multiple arguments give us a tuple.
('username', 'hackerrank', 'com')

groups() expression returns a tuple containing all the subgroups of the match.

>>> import re
>>> m = re.match(r'(\w+)@(\w+)\.(\w+)','username@hackerrank.com')
>>> m.groups()
('username', 'hackerrank', 'com')

groupdict()
A groupdict() expression returns a dictionary containing all the named
subgroups of the match, keyed by the subgroup name.

>>> m = re.match(r'(?P<user>\w+)@(?P<website>\w+)\.(?P<extension>\w+)',
                 'myname@hackerrank.com')
>>> m.groupdict()
{'website': 'hackerrank', 'user': 'myname', 'exten
"""

import re

s = input()
m = re.search(r'([A-Za-z0-9])\1+', s)  # two consec appearances
print(m.group(1) if m else -1)

# NOTE - re.match only works from beginning of string??

# %% PYTHON - REGEX AND PARSING - Validating Credit Card Numbers
"""
valid
4253625879615786
4424424424442444
5122-2368-7954-3214

invalid
42536258796157867       #17 digits in card number → Invalid
4424444424442444        #Consecutive digits are repeating 4 or more
                        times → Invalid
5122-2368-7954 - 3214   #Separators other than '-' are used → Invalid
"""
import re

for _ in range(int(input())):
    u = input()

    try:
        assert re.match(r'([456](\d){15}|[456](\d){3}-(\d){4}-(\d){4}-(\d){4})$', u)
        assert not re.search(r'(\d)\1\1\1', ''.join(u.split('-')))
        assert len(u) == 16 or len(u) == 19
    except Exception:
        print('Invalid')
    else:
        print('Valid')

# %% PYTHON - REGEX AND PARSING - Validating Roman Numerals
"""
CDXXI

Code below doesn't seem to work on machine but does on web

"""
import re

thousand = 'M{0, 3}'
hundred = '(C[DM]|D?C{0, 3})'  # D? means it is optional in that location
ten = '(X[CL]|L?X{0, 3})'
digit = '(I[VX]|V?I{0, 3})'


pattern = thousand + '$'  # + hundred + ten + digit + '$'
print(bool(re.match(r'((C[DM])|(DC{0, 3}))$', input())))
# print(bool(re.findall(r'M{0, 3}(C[MD]|D?C{0, 3})(X[CL]|L?X{0, 3})(I[VX]|V?I{0, 3})$', input())))

# %% PYTHON - REGEX AND PARSING - Substitution
#   Worked on all 9 cases 6/22/2020

# Fixed first fail - 3 of 9 test cases failed 6/22/2020
#    Out[302]: ['x&& &&& and && x or | ||\\|| x']
# comes out  ['x&& &&& and && x or | ||\\|| x'] because the space
# in between ' && && ' is matched with first && and not second...

import re


def my_filter(match):
    return ' and '


def my_filter2(match):
    return ' or '


n = int(input())

outp = []
for i in range(n):
    x = input()
    y = re.sub(r"\s[&]{2}\s", my_filter, x)
    while y != re.sub(r"\s[&]{2}\s", my_filter, y):
        y = re.sub(r"\s[&]{2}\s", my_filter, y)
    z = re.sub(r"\s[||]{2}\s", my_filter2, y)
    while z != re.sub(r"\s[||]{2}\s", my_filter2, z):
        z = re.sub(r"\s[||]{2}\s", my_filter2, z)
    outp.append(z)

for i in range(n):
    print(outp[i])

# test
# i want && and '&&& or ||  |||  &&
#
print(re.sub(r"\s[&]{2}\s", my_filter, "1 2 3 && 2d bbd"))

# %% PYTHON - REGEX AND PARSING - Validating and Parsing Email Addresses
"""
2
DEXTER <dexter@hotmail.com>
VIRUS <virus!@variable.:p>
"""
import email.utils
import re

for i in range(int(input())):
    t = email.utils.parseaddr(input())  # creates tuple with two items
    # it looks like the name is not checked, and the problem does in fact NOT
    # say anything about the name, only the user; so it was pointless to do
    # prior match on name in section above;  some names are in fact not user
    # \w includes _
    # name <user@email.com>
    # re.match automatically starts at beginning, so any characters at front
    # that are not alphabetic causes False
    if bool(re.match('[a-zA-Z](\w|-|\.)*@[a-zA-Z]*\.[a-zA-Z]{1,3}$', t[1])):
        print(email.utils.formataddr(t))
    # probably change to + on group after @ to be more accurate

# %% PYTHON - REGEX AND PARSING - Validating UID
"""
User ids must contain
at least 2 uppercase English alphabet characters.
at least 3 digits
only contain alphanumeric characters
No character should repeat.
There must be exactly 10 characters

2
B1CD102354
B1CDEF2354

Invalid
Valid
"""
import re

for _ in range(int(input())):
    # sorting works because they rules can be jumbled
    u = ''.join(sorted(input()))
    try:
        assert re.search(r'[A-Z]{2}', u)  # must have at least 2 caps
        assert re.search(r'\d\d\d', u)  # must have at least 3 digits
        assert not re.search(r'[^a-zA-Z0-9]', u)  # must not have non alphanum
        assert not re.search(r'(.)\1', u)  # \1 means repeat of prior match
        assert len(u) == 10
    except Exception:
        print('Invalid')
    else:
        print('Valid')


# %% PYTHON - REGEX AND PARSING - HEX color code capture
"""
#fff123
"""
import re

N = int(input())
pat = r'(?<!^)(#(?:[\da-fA-F]{3}){1,2})'
#  { #fff returns #fff, not (#fff, fff) because grouping with # is optional
# when theres a subgroup
#  { #fff123 returns #fff123, not (#fff123, 123)
# the {1,2} outside of () group means you could repeat that (), which is
# why the second three hex digits wouldn't need #

pat2 = r'(?<!^)(#([\da-fA-F]{3}){1,2})'  #  { #fff would return
#  { #fff returns (#fff, fff)
#  { #fff123 returns (#fff123, 123)

#  (?<!^) - exclude matches which are aligned on left margin
# (?:[\da-fA-F]{3} noncapturing is needed to avoid catching the
# second fff in fff)


for i in range(N):
    code = input()
    m = re.findall(pat, code)
    for x in m:
        print('(#(?:[', x)
    n = re.findall(pat2, code)
    for x in n:
        print('(#([', x)

# %% PYTHON - REGEX AND PARSING - Validating phone numbers
# A valid mobile number is a ten digit number starting with a 7,8, or 9.
"""
input
-----
2
9587456281
1252478965

Sample Output
YES
NO
"""
import re

for i in range(int(input())):
    if bool(re.match(r'[789]\d{9}$', input())):
        print('YES')
    else:
        print('NO')

# %% PYTHON - Debugging - Default arguments
class EvenStream(object):
    def __init__(self):
        self.current = 0

    def get_next(self):
        to_return = self.current
        self.current += 2
        return to_return

class OddStream(object):
    def __init__(self):
        self.current = 1

    def get_next(self):
        to_return = self.current
        self.current += 2
        return to_return

def print_from_stream(n, stream=EvenStream()):
    stream.__init__()   # this is the key line to fix error; need to init
    # the Stream object, because we were just passed it as arg or as dafault
    for _ in range(n):
        stream == print(stream.get_next())


queries = int(input())
for _ in range(queries):
    stream_name, n = input().split()
    n = int(n)
    if stream_name == "even":
        print_from_stream(n)
    else:
        print_from_stream(n, OddStream())

# **end python

# %% HACKERRANK SQL SKILLS - 58 TOTAL EXERCISES
"""
Topics covered in 58 questions on hackerrank, only some examples included
        as reference in this file
 #Qs Topic
---- ------------
- 20 Basic Select
-  5 Advanced Select
- 17 Aggregation
-  8 Basic Join
-  5 Advanced Join
-  3 Alternative Queries
====
 58
"""

# %% HACKERRANK SQL - Occupations
# works!
"""
set @r1=0, @r2=0, @r3=0, @r4=0;
select
    min(Doctor), min(Professor), min(Singer), min(Actor)
FROM (SELECT
    case
          when Occupation='Doctor' then (@r1:=@r1+1)
          when Occupation='Professor' then (@r2:=@r2+1)
          when Occupation='Singer' then (@r3:=@r3+1)
          when Occupation='Actor' then (@r4:=@r4+1) end as RowNumber,
    case when Occupation='Doctor' then Name end as Doctor,
    case when Occupation='Professor' then Name end as Professor,
    case when Occupation='Singer' then Name end as Singer,
    case when Occupation='Actor' then Name end as Actor
    FROM occupations order by Name) Temp
GROUP BY RowNumber;


# ---------------- ---------------- ----------------
# draw triangle 1

set @number = 21;
select
repeat('* ', @number := @number - 1)
from information_schema.tables;

# ---------------- ---------------- ----------------
# draw triangle 2
# note that update of row occurs after repeating operation
#i.e., @row =0 on row that first single * is printed
# so on 21st row, @row = 20, so no * is output
# ---------------- ---------------- ----------------
set @row := 0;
select repeat('* ', @row := @row + 1)
from information_schema.tables where @row < 20


"""
# %% Binary tree

# The reshape tool gives a new shape to an array without changing its
# data. It creates a new array and does not modify the original array itself.
import numpy

my_array = numpy.array([1,2,3,4,5,6])
print numpy.reshape(my_array,(3,2))

# %% Binary tree
"""
SELECT
    N,
    IF(P IS NULL, 'Root',
       IF((SELECT COUNT(*) FROM BST WHERE P=B.N) > 0, 'Inner',
          'Leaf'))
FROM BST AS B
ORDER BY N;

OR simpler, but less efficient for large tables

select
    N,
    case
        when P is NULL then 'Root'
        when N in (select P from BST) then 'Inner'
        else 'Leaf' end as Node
from BST
order by N;
"""
# **end sql

# %%  HACKERRANK - REGEX - 47 total
"""
Course Outline (not all problems in this outline are included in this file)
---------------------------------------------------------------------------

- 6 Introduction
- 3 Character Class
- 5 Repetitions
- 3 Grouping and Capturing
- 4 Backreferences - 2 problems not supported with Python
- 4 Assertions
- 22 Applications
===
47
"""

# %% REGEX - Introduction - start() and end()
# Return the indices of the start and end of the substring matched.
# by the group.
""" Notes
The dot (.) matches anything (except for a newline).
\d digit
\D non-digit
\s whitespace [ \r\n\t\f ].
\S  nonwhitespace
\w word (alphanumeric and numbers and _)
\W non word
^ start
$ end
[] only matches one character []
[^] matches any character NOT in[]
+ one ore more
* 0 or more
() captureing group
(?:) noncapturing
Here, b? is optional and matches nothing.
Thus, (b?) is successfully matched and capture nothing.
o is matched with o and \1 successfully matches the nothing captured by the group.

(b?)o\1 matches 'o' because the first group did match nothing, and so didthe \1
(b)?o\1 does not match 'o', because (b)? didn't even participate, so \1
    cannot match
c(?=o) matches 2nd o in chocolate - POSITive lookahead
c(?!o) matches 1st o in chocolate - negative loohahead
(?<=[a-z])[aeiou] matches e in he10 because h comes before - pos lookbehind
(?<![a-z])[aeiou] matches 0 in he10 because 1 comes before, not a letter
    -- neg lookbehind


"""
import re

m = re.search(r'\d+', '1234')
print(m.end())  # end() is always one character after last match
print(m.start())

# %% REGEX - Introduction - start() and end()
import re

s = input()
pat = input()
m = re.search(pat, s)
all = re.findall(pat, s)
if m is None:
    print((-1, -1))
else:
    print((m.start(), m.end()-1))
    print(all)

# %% REGEX - Introduction - search for start/end index of matches in string

import re

s = input()
k = input()
index = 0

if re.search(k, s):
    while index+len(k) < len(s):
        m = re.search(k, s[index:])  # begins search with new index

        if m is None:
            break
        else:
            print("({0}, {1})".format(index+m.start(), index+m.end()-1))
            index += m.start() + 1  # assign new index by +1
else:
    print((-1, -1))

#%%  REGEX - pattern match
import re
import sys

regex_pattern = r"...\....\....\...."	# Do not delete 'r'.
test_string = input()
match = re.match(regex_pattern, test_string) is not None
print(str(match).lower())

#%%  REGEX - pattern match

"""
pattern below didn't work because 2 of the test cases out of 7 had 7 character
length, unlike the specification below

with the following conditions:

must be of length: 6
First character: 1, 2 or 3
Second character: 1, 2 or 0
Third character: x, s or 0
Fourth character: 3, 0 , A or a
Fifth character: x, s or u
Sixth character: . or ,
"""
Regex_Pattern = r'[1-3][0-2][x,s,0][3,0,A,a][x,s,u][\.\,]'

print(bool(re.match(r'[abc][1-3]', 'a5')))  # False
print(bool(re.match(r'[abc][1-3]', 'a3')))  # True

#%%  REGEX - excluding

rp = r'^\D[^aeiou][^bcDF]\S[^AEIOU][^.,]$'  # not ANY of abcde
print(bool(re.match(rp, 'trains')))  # True

# \D is digit
# \S is whitespace
# ^ is start, $ is end

rp = r'^[a-z][1-9][^a-z][^A-Z][A-Z].*'  # .* pads anything at end
print(bool(re.search(rp, 'trains')))

Regex_Pattern = r'^[a-zA-z02468]{40}[13579\s]{5}$'
# {5} means prior char repeated up to 5 timesDo not delete 'r'.

# The {x,y} tool will match between and (both inclusive) repetitions
# of character/character class/group.

# + tool will match one or more repetitions of character/class/group
# * tool will match zero or more repetitions of character/ class/grou.
# $ boundary matcher matches occurrence of a character/character class/group
# at the end of a line.

# \b assert position at a word boundary.
# 3 positions for \b
# Before the first character in the string, if the first character is a word
# character.
# Between two characters in the string, where one is a word character and the
# other is not a word character.
# After the last character in the string, if the last character is a word
# character.

# () groups part of regex together
# (?:) can be part of match that is optional

# (Al|Bo) matches either Al or Bo

# (\d)\1: It can match 00, 11, 22, 33, 44, 55, 66, 77, 88
# or 99.
# This tool (\1 references the first capturing group) matches the same text as
# previously matched by the capturing group.

r'([a-z])([\w])([\s])\1\2\3'
# lowercase, word, whitespace, then repeat 1st (lowercase), then word,
# whitespace

# %% REGEX - Parse for digits
rp = r'^\d{2}(-?)(\d{2}\1){2}\d{2}$'

print(bool(re.search(rp, '12-34-56-78')))


# %% REGEX - Lookahead and behind
# The positive lookahead (?=) asserts regex_1 to be immediately followed by
# regex_2. The lookahead is excluded from the match. It does not return
#  matches of regex_2. The lookahead only asserts whether a match is
#  possible or not.
rp = r'o(?=oo)'
print(re.search(rp, 'xxoooxx'), bool(re.search(rp, 'xxoooxx')))

# (?<=regex_2)regex_1
rp = r'(?<=xx)o'
print(re.search(rp, 'xxo'), bool(re.search(rp, 'xxo')))

# The positive lookbehind (?<=) asserts regex_1 to be immediately
# preceded by regex_2. Lookbehind is excluded from the match
# (do not consume matches of regex_2), but only assert whether a match is
# possible or not.

# nEGATIVE LOOKbehind
rp = r'(?<!xx)o'
print(re.search(rp, 'xxo'), bool(re.search(rp, 'xxo')))

# %% REGEX -GROUPING AND CAPTURING - Alternative Matching
u = ''
v = 'Mr.Voshi'

pat2 = r'^(Mrs|Mr|Dr|Er)\.[a-zA-Z]+$'

print(re.match(pat2, v))
# re.match(pat1, u)

if '-' in num:
    num = num.split('-')
elif ' ' in num:
    num = num.split(' ')
# %% REGEX - APPLICATIONS - Detect HTML links
# one in 10 test cases failed
# comments said regex is not the industry standard way to do this task
# html = pd.read_csv('html.csv')  # used to debug failing case

import pandas as pd
import re

n = int(input())
pat = r'<a href="(.*?)".*?>([\w ,./]*)(?=</)'
#       this is one match,   this is second match for word, space, , or . or /
for i in range(n):
    text = input()
    # text = html.loc[i].values[0]
    alllinks = re.findall(pat, text)
    for link, title in alllinks:
        print(link.strip() + ',' + title.strip())


# %% Regex - Test HTML

text = '<p><a href="http://www.quackit.com/html/tutorial/html_links.cfm">Example Link</a></p>'
all = re.findall(pat, text)
print(all[0][0] + ',' + all[0][1])

# %% REGEX - APPLICATIONS - Find a Word
# text = 'foo bar (foo) bar foo-bar foo_bar foo\'bar bar-foo bar, foo.'

import re

text = 'colourfuture saturday.future face_future(anxiety obtain.future surroundings\'futurerefrigerator alone)futurecomparison wine-future,tight futureimpatient bodyfuture excite(future(grandfather'
N = int(input())

sentences = []
for i in range(N):
    sentences.append(input())

m = int(input())

words = []
for i in range(m):
    words.append(input())

for w in words:
    count = 0
    for s in sentences:
        # print(re.findall(rf'\b{w}\b', s))
        count += len(re.findall(rf'\b{w}\b', s))
    print(count)


# %% REGEX - APPLICATIONS - Utopian Identification Number
"""New ID.

A new identification number is given for every Citizen of the Country Utopia
 and it has the following format.

- The string must begin with between 0-3 (inclusive) lowercase letters.
- Immediately following the letters, there must be a sequence of digits
    (0-9). The length of this segment must be between 2 and 8, both inclusive.
- Immediately following the numbers, there must be atleast 3 uppercase
    letters.
2
abc012333ABCDEEEE
0123AB

Sample Output

VALID
INVALID

"""
import re

for _ in range(int(input())):
    u = input()

    try:
        assert re.match(r'([a-z]{0,3})(\d{2,8})([A-Z]{3,})$', u)
    except Exception:
        print('INVALID')
    else:
        print('VALID')
# **end regex

"""conditions:
must be of length: 6
First character: 1, 2 or 3
Second character: 1, 2 or 0
Third character: x, s or 0
Fourth character: 3, 0 , A or a
Fifth character: x, s or u
Sixth character: . or ,
"""
pat = r'^[1-3][0-2][xs0][30Aa][xsu][.,]$'

"""
: If you want to match (.) in the test string,
you need to escape the dot by using a slash \..
"""
r'^...\....\....\....$'


# %% REGEX - APPLICATIONS - Find a Sub-word

import re

n = int(input())
text = "\n".join(input() for _ in range(n))
t = int(input())
for _ in range(t):
    print(len(re.findall(r'\B(%s)\B' % input().strip(), text)))
# \B means not a boundary of a word

# %% REGEX - APPLICATIONS - Alien Username
# what is diff between () and []
# order matters for (), BUT NOT for []
# [] stands for one of the characters inside
"""
3
_0898989811abced_
_abce
_09090909abcD0

Sample Output
VALID
INVALID
INVALID
"""

import re

for _ in range(int(input())):
    u = input()
    pat1 = '^([_.])\d+([a-zA-Z]){0,}_?$'

    try:
        x = re.match(pat1, u)
        assert x
    except Exception:
        print('INVALID')
    else:
        print('VALID')

# %% REGEX - APPLICATIONS - Alien Username
# troubleshooting

import re

for _ in range(int(input())):
    u = 'x0898989811abced_'
# _abce
# _09090909abcD0'
    pat1 = '^([_.])\d+([a-zA-Z]){0,}_?$'
    # inside the [], . means ., no escape \ needed
    pat2 = '^([._])\d+([a-zA-Z]){0,}_?$'


    pat3 = '^(_.)\d+([a-zA-Z]){0,}_?$'
    # pat3 catches '_,0898989811abced_' because . is any
    # character, not just .; if we want to specify ., we
    # need \.

    pat4 = '^(._)\d+([a-zA-Z]){0,}_?$'
    pat5 = '^(_\.)\d+([a-zA-Z]){0,}_?$'
    # here the (._) means any character and then _

    x = re.match(pat1, u)
    y = re.match(pat2, u)
    z = re.match(pat3, u)
    a = re.match(pat4, u)
    b = re.match(pat5, u)
    print(x)
    print(y)
    print(z)
    print(a)
    print(b)

# %% REGEX - APPLICATIONS - hackerrank language
    # make sure no spaces between word and :
    languages = 'C:CPP:JAVA:PYTHON:PERL:PHP:RUBY:CSHARP:HASKELL:CLOJURE:BASH:SCALA:ERLANG:CLISP:LUA:BRAINFUCK:JAVASCRIPT:GO:D:OCAML:R:PASCAL:SBCL:DART:GROOVY:OBJECTIVEC'
    languages = languages.split(':')

n = int(input())
for i in range(n):
    code, language = input().split()
    if language in languages:
        print('VALID')
    else:
        print('INVALID')


# %% REGEX - APPLICATIONS - Split the phone numbers
import re

for i in range(int(input())):
    num = input()

    x = re.findall(r'(\d{1,3})[ -]', num)
    y = re.findall(r'(?<=[ -])(\d{4,10})$', num)
    answer = 'CountryCode=' + x[0] + ',LocalAreaCode='
    + x[1] + ',Number=' + y[0]
    print(answer)

# %% REGEX - APPLICATIONS - Split the phone numbers

    x = re.findall(r'(\d{1,3})[ -](\d{1,3})[ -](\d{4,10})$', num)
    answer = 'CountryCode=' + x[0][0] + ',LocalAreaCode='
    + x[0][1] + ',Number=' + x[0][2]

    print(answer)

# %% REGEX - APPLICATIONS - Detecting Valid Latitude and Longitude Pairs
"""Description.

Given a line of text which possibly contains the latitude and longitude of a
point, can you use regular expressions to identify the latitude and longitude
referred to (if any)?

Input Format
The first line contains an integer N, which is the number of tests to follow.
This is followed by N lines of text. Each line contains a pair of co-ordinates
which possibly indicate the latitude and longitude of a place.

Constraints
1 <= N <= 100
The latitude and longitude, if present will always appear in the form of (X, Y)
where X and Y are decimal numbers.
For a valid (latitude, longitude) pair:
-90<=X<=+90 and -180<=Y<=180.
They will not contain any symbols for degrees or radians or N/S/E/W. There
may or may not be a +/- sign preceding X or Y.
There will be a space between Y and the comma before it.
There will be no space between X and the preceding left-bracket, or between Y
and the following right-bracket.
There will be no unnecessary zeros (0) before X or Y.

Output Format
"Valid" where X and Y are the latitude and longitude which you found to be a
valid (latitude, Longitude) pair.
If the given pair of numbers are not a valid (latitude,longitude) pair,
output "Invalid".
"""
import re

for i in range(int(input())):
    coords = input()
    coords = '_' + coords[1:-1] + '_'
    nums = coords.split(', ')
    lat = nums[0][1:]
    lon = nums[1][0:-1]

    try:
        assert abs(float(lat)) <= 90.0
        assert abs(float(lon)) <= 180.0

        # no hanging decimal point
        assert len(re.findall(r'(\d[.][,]|\d[.]_)', coords)) == 0
        # no zero padding
        assert len(re.findall(r'_[+-]?0\d+', nums[0])) == 0

        # no zero padding
        assert len(re.findall(r'^0\d+', lon)) == 0

    except Exception:
        print('Invalid')
    else:
        print('Valid')

# %% REGEX - APPLICATIONS - HackerRank Tweets

import re

N = int(input())
count = 0
for i in range(N):
    text = input().lower()
    if re.match('.*hackerrank.*', text):
        count += 1
print(count)

# %% REGEX - APPLICATIONS - Find Hackerrank


import re

N = int(input())
for i in range(N):
    text = input()
    if re.match('^(hackerrank|hackerrank.*hackerrank)$', text):
        # need to control for if hackerrank occurs once only or more than 1
        # print('at both front and back', 0)
        print(0)
    elif re.match('^hackerrank.*[^k]$', text):
        # print('at beginning only', 1)
        print(1)
    elif re.match('^[^h].*hackerrank$', text):
        # print('at end only', 2)
        print(2)
    else:
        # print('neither', -1)
        print(-1)


# %% REGEX - APPLICATIONS - IP Address Validation

N = int(input())

for i in range(N):
    text = input()

    if '.'  in text:
        # test for ipv4
        try:
            address = text.split('.')
            assert len(address) == 4
            nums = list(map(int, address))
            assert sum([x >= 0 and x <= 255 for x in nums]) == 4
            # each number in range of 0 to 255
        except Exception:
            print('Neither')
        else:
            print('IPv4')
    elif ':' in text:
        try:
            address = text.split(':')
            assert len(address) == 8
            assert sum([int(x, 16) for x in address])
            # only sums if all elements are converted to hex

        except Exception:
            print('Neither')
        else:
            print('IPv6')
    else:
        print('Neither')


# %% INTERVIEW PREP
"""
69 CHALLENGES broken into 15 subjects
- 4 Warmups
- 5 Arrays
- 5 Dictionaries and Hashmaps
- Tips and Guidelines (just hint videos)
- 5 Sorting

- 5 String manipulation
- 5 Greedy algorithms
- 7 Search
- 4 DynamicProgramming
- 6 Stacks and Queues

- 5 Graphs
- 5 Trees
- 5 Linked Lists
- 4 Recursion and backtracking
- 4 Miscellaneous
===
-69

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

import math
import os
import random
import re
import sys


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
            print('one a 1')
            pass
        elif i < position:  # we already moved beyond this cloud
            print('i < position')
            pass
        elif (((i+2) <= (n-1)) and (c[i+2] == 0)):
            jumps += 1
            position = i+2
        elif c[i+1] == 0:
            # by the time we reach here, it means that we've
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
    print('Within %d characters of string %s' % (n, s))
    print('We find %d occurences of letter a' % (full_count))
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
the minimum number of character deletions required to make and anagrams.
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

import math
import os
import random
import re
import sys

# Complete the makeAnagram function below.
def makeAnagram(a, b):
    # first process both strings into count of each letter of alphabet
    # numbersa is 26 long list of numbers, each # representing count of a-z
    # in string a
    numbersa = [0]*26
    numbersb = [0]*26
    for i in a:
        numbersa[ord(i)-96-1]+=1
    for i in b:
        numbersb[ord(i)-96-1]+=1

    count = 0
    for i in range(26):
        count += abs(numbersa[i] - numbersb[i])
    return count


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    a = input()
    b = input()
    res = makeAnagram(a, b)
    fptr.write(str(res) + '\n')
    fptr.close()

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
"""
import math
import os
import random
import re
import sys


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
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    q = int(input())
    for q_itr in range(q):
        s = input()
        result = alternatingCharacters(s)
        fptr.write(str(result) + '\n')
    fptr.close()


# %% # %% ARTIFICIAL INTELLIGENCE The Best Aptitude Test
# https://www.hackerrank.com/challenges/the-best-aptitude-test/problem
# 1 <= T <= 10
# 4 <= N <= 100
# 0.0 <= k <= 10.0, where k is the GPA of every student.
# 0.0 <= s <= 100.0, where s is the score of every student in any of the 5
# aptitude tests.
# Enter your code here. Read input from STDIN. Print output to STDOUT


def get_gpa(n):
    # n GPAs for first year, one per student
    gpas_ = input().split(' ')
    gpas = []
    for i in range(n):
        gpas.append(float(gpas_[i])*10)
    return gpas


def get_scores(n):
    scores_ = []
    scores = []
    for i in range(5):
        scores_.append(input().split(' '))
        scores.append([])
        for j in range(n):
            scores[i].append(float(scores_[i][j]))
    return scores


def get_relscore(gpas, scores):
    correct = 0
    for i in range(1, len(gpas)):
        if scores[i] > scores[i - 1]:
            correct += 1
    return correct


def get_error(gpas, scores):
    error = 0
    for i in range(len(gpas)):
        error += (gpas[i]-scores[i])**2
    return error


if __name__ == '__main__':

    t = int(input())  # test cases

    for i in range(t):
        n = int(input())  # # of students, each case
        gpas = get_gpa(n)
    #    print(gpas)
        scores = get_scores(n)
    #    print(scores)
        best_test = 0
        correct = get_relscore(gpas, scores[0])
    # print(correct)

        for j in range(1, 5):
            next_correct = get_relscore(gpas, scores[j])
            # print(next_correct)
            if next_correct > correct:
                best_test = j
                correct = next_correct
        print(best_test + 1)

# %% MISCELLANEOUS PYTHON - PRINT formatting
# read 2 int from stdin, print sum, diff, product

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a + b)
    print(a - b)
    print(a * b)

"""
# ALTERNATIVES, showing formatting options
    print('{0}\n{1}\n{2}'.format((a + b), (a - b), (a * b)))
# or
    print('{:d}\n{:d}\n{:d}'.format(a + b, a - b, a*b))
# if we wanted to pad a few spaces before number
    print('{:3d}\n{:2d}\n{:d}'.format(a+b, a-b, a*b))

# %% int division, float division
print('{:d}\n{:f}'.format(a // b, a / b))
print('{0}\n{1}'.format(a // b, a / b))
"""

# %% MISCELLANEOUS PYTHON - unpack an iterable with *
#  print 12345...n where n in input; *turns range into output
print(*range(1, int(input())+1), sep='')

if __name__ == '__main__':
    n = int(input())
    print(*range(1, n+1), sep='')

"""
# Less succinct solution, the * causes unpacking.
# You can do it in any iterable, not just range

    output = ''
    for i in range(n):
        output = output + str(i + 1)

    print(output)


# 2nd example of unpacking an iterable
x='1234'
print(*x, sep='-')
> 1-2-3-4
"""

# %% MISCELLANEOUS HACKERRANK
# Day 5: Computing the Correlation
# no special library support
# Pearson product-moment correlation coefficient
#                       sum(x_i * y_i) - n*x_avg * y_avg
# r_xy = num/denom =   ----------------------------------
#                       (n-1)*s_x*s_y
#
#
# num        sum(x_i * y_i) - n*x_avg * y_avg
# ----  =   ----------------------------------
# denom      (n-1)*s_x*s_y
#
# -------------- -------------- --------------

import math


def std(x, avg, n):
    ans = 0
    for i in x:
        ans += (avg - i) ** 2

    return math.sqrt(ans / (n - 1))


def cof(u, v, avg_u, avg_v, std_u, std_v, n):
    ans = 0
    for i in range(n):
        ans += u[i] * v[i]

    ans -= (n * avg_u * avg_v)
    return ans / ((n - 1) * std_u * std_v)


n = int(input())
m = []
p = []
c = []
sum_m = 0
sum_c = 0
sum_p = 0

for _ in range(n):
    m_, p_, c_ = map(int, input().split('\t'))
    m.append(m_)
    p.append(p_)
    c.append(c_)
    sum_m += m_
    sum_p += p_
    sum_c += c_

avg_m = sum_m / n
avg_c = sum_c / n
avg_p = sum_p / n

std_m = std(m, avg_m, n)
std_p = std(p, avg_p, n)
std_c = std(c, avg_c, n)

print(round(cof(m, p, avg_m, avg_p, std_m, std_p, n), 2))
print(round(cof(c, p, avg_c, avg_p, std_c, std_p, n), 2))
print(round(cof(m, c, avg_m, avg_c, std_m, std_c, n), 2))



# %% Overly dense list comprehension for hackerrank problem

nstud, ntests = list(map(int, input().split()))
scores = [list(map(float, input().split())) for _ in range(ntests)]
avgs = list(map(lambda x: sum(x)/ntests, list(zip(*scores))))

for i in range(nstud):
    print(avgs[i])
