"""Hackerrank quizzes.

Reference and progress for hackerrank skills / problems

3 skills covered in this file
------------------------------------------------------------
1. Python              102/115   (I earned badge on Hackerrank)
2. SQL                  54/58    (I earned badge on Hackerrank)
    - the challenges are quite simple, so I only more difficult ones
      included here for reference
    **end sql
3. Regex                 45/47    (I earned badge on Hackerrank)
4. Interview Prep          /69
    -- see separate file hackerrank_interview_prep.py

"""
# =============================================================================
# %% PYTHON PRACTICE - 115 QUESTIONS/DRILLS from hackerrank.com
# =============================================================================
"""
#    Topics                          # left to do
-  7 Introduction                   x
-  6 Basic Data Types               x
- 14 Strings                        x
- 13 Sets                           x (operations)
-  7 Math                           x
---  47
-  7 Itertools                      x (permutations, groupby, product,
                                       combinations,
                                       combinations_with_replacement...)
-  8 Collections                    x (Counter, DefaultDict, namedtuple,
                                       OrderedDict, deque, etc.)
-  2 Date and Time                  x (datetime, timedelta,)
-  2 Errors and Exceptions          x
-  2 Classes                        1
---  21

-  6 Built-Ins                      x (any, all, zip, input, sort,
                                       etc.see realpython)
-  3 Python Functionals             x (map, lambda, reduce, filter)
- 17 Regex and Parsing              1
-  2 XML                            x
-  2 Closures and Decorators        x
---  30

- 15 Numpy                          x
-  2 Debugging                      x
---  17
=======
113/115

classes, regex, nummpy
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
        # n+1 to get to n in reverse parsing
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

    for _ in range(n):
        line = input().split()
        name, scores = line[0], line[1:]
        scores = [float(scores[i]) for i in range(3)]
        student_marks[name] = scores
    query_name = input()
    out = sum(student_marks[query_name]) / len(student_marks[query_name])
    print("%.2f" % round(out, 2))

    width = 10
    precision = 4
#    print(f'{out:{width}.{precision}}')
    print(f'{out:10d}')  # {width}.{precision} specified in that order


# %% PYTHON - BASIC DATA TYPES - List Comprehensions
# solved 6/20/2020
# return 3d space in cube defined by origin and X,Y,Z where i + j + k != n
# permutations problem, excluding

def combos(x, y, z, n):
    return [[i, j, k] for i in range(x + 1) for j in range(y + 1) for k in
            range(z + 1) if sum([i, j, k]) != n]


if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    output = combos(x, y, z, n)
    print(f'Output has {len(output)} combinations: {output}')


# %% PYTHON - BASIC DATA TYPES - Nested lists
# Print sorted list of students that have 2nd highest scores
# completed 6/20/2020

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
    names = [name_score[i][0] for i in range(len(name_score)) if
             name_score[i][1] == second_low_score]
    out.sort()
    for name in names:
        print(name)


# %% PYTHON - BASIC DATA TYPES - Nested lists
# Print sorted list of students that have 2nd highest scores
# better way is to use set


if __name__ == '__main__':
    name_score = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        insert_list = [name, score]
        scores.add(score)
        name_score.append(insert_list)
    second_low_score = min(scores - {min(scores)})
    names = [name_score[i][0] for i in range(len(name_score)) if
             name_score[i][1] == second_low_score]
    names.sort()
    for name in names:
        print(name)


# %% PYTHON - STRINGS - Alphabet Rangoli
from string import ascii_lowercase


def print_rangoli(n):
    rows = ['x'] * (n * 2 - 1)
    # initialize the list as of 2*n -1 strings of 'x'

    for i in range(n):
        pattern = ascii_lowercase[n-i-1:n]
        fullpat = pattern[-1:0:-1] + pattern
        rowpat = '-' * (n*2-2-2*i) + '-'.join(fullpat) + '-'*(n*2-2-2*i)
        rows[i] = rowpat
        rows[2*n-2-i] = rowpat
    print(*rows, sep='\n')


if __name__ == '__main__':
    n = int(input())
    print_rangoli(n)


"""
# when n=4, rows is as follows:

------d------
----d-c-d----
--d-c-b-c-d--
d-c-b-a-b-c-d
--d-c-b-c-d--
----d-c-d----
------d------
"""

# %% PYTHON - STRINGS - String Formatting
"""Format String.

Given an integer, n, print the following values for each integer from
1 to n:

Decimal Octal  Hexadecimal (capitalized)   Binary
Decimal Octal  Hexadecimal (capitalized)   Binary
Decimal Octal  Hexadecimal (capitalized)   Binary

The four values must be printed on a single line in the order specified
above for each i from 1 to n. Each value should be space-padded to match
the width of the binary value of n.
"""


def print_formatted1(number):
    """First try."""
    for i in range(1, number + 1):
        print(i, '\t', oct(i), '\t', hex(i), '\t', bin(i))


def print_formatted2(n):
    """Use .format()"""
    width = len("{0:b}".format(n))

    for i in range(1, n + 1):
        print("{0:{width}d} {0:{width}o} {0:{width}X} \
              {0:{width}b}".format(i, width=width))


def print_formatted3(n):
    """Use fstring."""
    width = len(f"{n:b}")

    for i in range(1, n + 1):
        print("{0:{width}d} {0:{width}o} {0:{width}X} \
              {0:{width}b}".format(i, width=width))


print_formatted1(10)
print_formatted2(10)
print_formatted3(10)
# %% PYTHON - STRINGS - Minion Game


def minion_game(string):
    """Determine who is winner and their score."""
    str_vowels = set(string).intersection(set('AEIOU'))
    str_consonants = set(string).difference(str_vowels)

    cpoints = 0
    vpoints = 0
    # breakpoint()

    for i in range(len(string)):
        if string[i] in str_consonants:
            # number of words to add is len(strin)-i because
            # words of length 1 to len(string) are included, but we are
            # at position i so we subtract i
            cpoints = cpoints + len(string) - i
        elif string[i] in str_vowels:
            vpoints = vpoints + len(string) - i

    if cpoints > vpoints:
        # more unique consonants => Stuart wins
        print('Stuart', cpoints)
    elif cpoints < vpoints:
        print('Kevin', vpoints)
    elif cpoints == vpoints:
        print('Draw')


minion_game('BANANA')


# %% PYTHON - STRINGS - Designer Door Mat
# 5 < n < 101
# 15 < m < 303

n, m = list(map(int, input().split()))
rows = ['x'] * n

for i in range(n//2):
    pattern = '-' * (m//2-1-3*i) + '.|.' * (2*i+1) + '-' * (m//2-1-3*i)
    rows[i] = pattern
    rows[n - 1 - i] = pattern
rows[n // 2] = '-' * (m // 2 - 3) + 'WELCOME' + '-' * (m // 2 - 3)
print(*rows, sep='\n')

# output for n,m = 6, 16
"""
-------.|.-------
----.|..|..|.----
-.|..|..|..|..|.-
-----WELCOME-----
----.|..|..|.----
-------.|.-------
"""

# %% PYTHON - STRINGS - Merge the Tools!
# worked on all 16 test cases 6/23/2020


def merge_the_tools(string_in, k):
    """Print out the substrings after removing duplicate characters.

    string: length is a multiple of k
    k = length of substring to analyze

    1. break string into len(string)/k substrings
    2. remove duplicate characters from substring
    3. print out each processed substring

    This code passes all test cases, but we could probably optimize
    membership test of characters by turning ministr into set instead
    of iterating through the string with the 'in' operator
    """
    n_substrings = int(len(string_in)/k)  # guaranteed to be an integer
    out_substrings = []

    for i in range(n_substrings):
        ministr = string_in[i*k:i*k+k]
        mini = ministr[0]
        for i in range(1, len(ministr)):
            if ministr[i] in ministr[0:i]:
                pass
            else:
                mini = mini + ministr[i]
        # print(mini)
        out_substrings.append(mini)

    return out_substrings


def merge_the_tools2(string_in, k):
    """Print out the substrings after removing duplicate characters.

    This version of function tests if each character of substring
    has already occurred ministr and has been added to passed substring
    """
    n_substrings = int(len(string_in)/k)  # guaranteed to be an integer
    out_substrings = []

    for i in range(n_substrings):
        ministr = string_in[i*k:i*k+k]
        mini = ministr[0]
        for i in range(1, len(ministr)):
            if ministr[i] in mini:
                pass
            else:
                mini = mini + ministr[i]
        # print(mini)
        out_substrings.append(mini)

    return out_substrings


#if __name__ == '__main__':
#    string_in, k = input(), int(input())
#    merge_the_tools(string_in, k)


# testing
# string = 'abccefggi'
# k=3
# merge_the_tools(string, k)

from timeit import Timer

for n in [10, 11, 15]:
    in_prefix = r'.\hackerrank_test_cases\python\merge_tools_input_'
    out_prefix = r'.\hackerrank_test_cases\python\merge_tools_OUTPUT_'
    input_file = in_prefix + str(n) + '.txt'
    output_file = out_prefix + str(n) + '.txt'
    data_in = loadtxt(input_file, delimiter=' ', dtype='str')
    data_in, k = data_in[0], data_in[1]
    k = int(k)
    # data_out = loadtxt(output_file, delimiter=' ', dtype='str')
    # generated_output = merge_the_tools(data_in, k)
    # print(f'Test Case {n}: '
    #       f'{"PASS" if list(data_out)==generated_output else "FAIL"}')
    t = Timer(lambda: merge_the_tools(data_in, k))
    t2 = Timer(lambda: merge_the_tools2(data_in, k))
    print(f'2000x Test Case {n}, using list: '
          f'{round(t.timeit(number=2000), 2)} seconds')
    print(f'2000x Test Case {n}, using set : '
          f'{round(t2.timeit(number=2000), 2)} seconds\n')

"""
OUTPUT
Note Test Case 15 doesn't improve because the string mini is only 1 char long'
2000x Test Case 10, using list: 6.09 seconds
2000x Test Case 10, using set : 1.89 seconds

2000x Test Case 11, using list: 4.22 seconds
2000x Test Case 11, using set : 1.69 seconds

2000x Test Case 15, using list: 10.28 seconds
2000x Test Case 15, using set : 10.29 seconds
"""

# %%  PYTHON - STRINGS - Capitalize


def solve(s):
    """Could use built in which iterates through list to capitalize."""
    mylist = s.split(' ')
    print(mylist)
    for i, letter in enumerate(mylist):
        mylist[i] = letter.capitalize()
    return ' '.join(mylist)


s ='this is a sentence i want to make a title 23 43f f88 ;ad'
print(solve(s))   # capitalizes each word
print(s.title())  # capitalizes each word, ignoring non-letter prefix
print(s.upper())  # capitalizes every letter of every word
"""
This Is A Sentence I Want To Make A Title 23 43f F88 ;ad
This Is A Sentence I Want To Make A Title 23 43F F88 ;Ad
THIS IS A SENTENCE I WANT TO MAKE A TITLE 23 43F F88 ;AD
"""
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

# %% PYTHON - STRINGS - Text wrap

import textwrap


def wrap(string, max_width):
    """wrap."""
    return '\n'.join(textwrap.wrap(string, max_width))


if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)

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
# %% PYTHON - SETS - Reference List of Commands
# set cannot contain another set

numbers = set()
print(numbers)
print(type({}))

numbers = set([1, 2])
print(numbers)  # Output: {1, 2}
string_set = set("hello")
print(string_set)  # Output: {'o', 'e', 'l', 'h'}

numbers = {1, 2, 1, 2, 3, 5}
print(numbers)  # Output: {1, 2, 3, 5}

random_set = {'a', 'a', 1, 2, 1}
print(random_set)  # Output: {1, 2, 'a'}

################################################
# Sets can contain only hashable elements.
# Sets cannot contain other sets.
my_set = {1, 2}
set_to_add = {3, 4}
my_set.add(set_to_add)  # Raises "TypeError: unhashable type: 'set'"
# To create set of sets, use a frozenset
my_set.add(frozenset(set_to_add))
print(my_set)  # Output: {1, 2, frozenset({3, 4})}

# Length of a set.
numbers = {1, 2, 3}
print(len(numbers))  # Output: 3

print(1 in {1, 2, 3})  # Output: True
print(5 in {1, 2, 3})  # Output: False
print(1 not in {1, 2, 3})  # Output: False

# Access elements in a set
numbers = {1, 2, 3}
for number in numbers:
    print(number)

# Union: Elements in either set_1, set_2 or both.
set_1 = {1, 2, 3, 4}
set_2 = {3, 4, 5, 6}
print(set_1.union(set_2))  # Output: {1, 2, 3, 4, 5, 6}
print(set_1 | set_2)  # Output: {1, 2, 3, 4, 5, 6}

# Intersection: Elements in both set_1 and set_2
set_1 = {1, 2, 3, 4}
set_2 = {3, 4, 5, 6}
print(set_1.intersection(set_2))  # Output: {3, 4}
print(set_1 & set_2)   # Output: {3, 4}

# Set difference: Elements in set_1 not in set_2
set_1 = {1, 2, 3, 4}
set_2 = {3, 4, 5, 6}
print(set_1.difference(set_2))  # Output: {1, 2}
print(set_1 - set_2)  # Output: {1, 2}

# Symmetric difference: Elements in set_1 or set_2, but not in both.
set_1 = {1, 2, 3, 4}
set_2 = {3, 4, 5, 6}
print(set_1.symmetric_difference(set_2))  # Output: {1, 2, 5, 6}
print(set_1 ^ set_2)  # Output: {1, 2, 5, 6}
# set_1 not mutatedby ^

# Check if a set is a subset of another.
set_1 = {1, 2}
set_2 = {1, 2, 3}
print(set_1.issubset(set_2))  # Output: True
print(set_2.issubset(set_1))  # Output: False

# Check if a set is a superset of another.
set_1 = {1, 2}
set_2 = {1, 2, 3}
print(set_1.issuperset(set_2))  # Output: False
print(set_2.issuperset(set_1))  # Output: True

# Add one element to set using add() methd.
numbers = {1, 2, 3}
numbers.add(4)  # <- takes a single hashable element.
print(numbers)  # Output: {1, 2, 3, 4}

# Add multiple elements to set using update() methd.
numbers = {1, 2, 3}
numbers.update([4, 5])  # <- takes any iterable.
print(numbers)  # Output: {1, 2, 3, 4, 5}

# Remove elements using remove() method.
numbers = {1, 2, 3}
numbers.remove(1)
print(numbers)  # Output: {2, 3}
numbers.remove(5)  # Raises KeyError if element is not present.

# Remove elements using discard element.
numbers = {1, 2, 3}
numbers.discard(1)
print(numbers)  # Output: {2, 3}
numbers.discard(5)
# Does not raise any error even
# if element is not present.

# Remove and get the last item in a set,
# sets are unordered so any element could get removed!
numbers = {1, 2, 3}
print(numbers.pop())  # Output: 3
print(numbers)  # Output: {1, 2}

# Empty set using clear()
numbers = {1, 2, 3}
numbers.clear()
print(numbers)  # Output: set()

# Delete set using del keyword
numbers = {1, 2}
del numbers


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
# ^ replaces .symmetric_difference but only operates on
# in set elements
# s is not mutated by ^

s = set('ank')
print(s ^ set('Rank'))
print(s.symmetric_difference(set('Rank')))
print(s)

# %% PYTHON - MATH  Triangle Quest

for i in range(1, int(input())):
    print(int((10**i-1)/9)*i)

# 5
# 1
# 22
# 333
# 4444

# %% PYTHON - MATH  Triangle Quest 2

for i in range(1, int(input())+1):
    # More than 2 lines will result in 0 score. Do not leave a blank line also
    # print(''.join(map(str, [x for x in range(1, i)] + \
    # [i-x for x in range(i)])))

    # works but string not allowed
    # or print(int(str(1)*i)**2)

    # print((sum([10**(x-1) for x in range(i)]))**2)

    # but only 1 for is allowed
    # print((sum([10**(x-1) for x in range(1, i+1)]))**2)

    # math solution
    print(int((10**i-1)/9)**2)
    # ((10**1-1)/ 9)**2 = 1**2  = 1
    # ((10**2-1)/ 9)**2 = 11**2 = 121
    # ((10**3-1)/ 9)**2 = (999/9)**2 = 111**2 = 12321
    # 1111**2
    # 11111**2 etc.

# %% PYTHON - MATH

import cmath

print(*cmath.polar(complex(input())), sep='\n')

# %% PYTHON - MATH -

import cmath

print(*cmath.polar(complex(list(map(int, input())), sep=' ')))
x = cmath.polar(complex(input()))

# %% PYTHON - ITERTOOLS - Reference
"""
https://realpython.com/python-itertools/


Infinite iterators:
count(10)        -> 10 11 12 ...
cycle('ABCD') -> A B C D A ...
repeat(10, 3) -> 10 10 10

Terminate on shortest input:
accumulate([1,2,3])               1 3 6
chain('abc', 'def')               a b c d e f
chain_from_iterable(['abc','def']) a b c d e f
compress('abcd', [1,0,1,0])      a c
dropwhile()

groupby(str)
islice()
starmap()
takewhile()
tee()
zip_longest()

Combinatorics:
permutations('ABCD', repeat=2)
combinations_with_replacement('ABCD', repeat=2)
combinations('ABCD', repeat=2)
product('ABCD', repeat=2)


Permutations are printed in a lexicographic sorted order. So, if the
input iterable is sorted, the permutation tuples will be produced in a
sorted order.
"""
# mylist = 'HACK 2'.split()

from itertools import permutations

mylist = input().split()
strings = list(mylist[0])
k = int(mylist[1])
strings.sort()
my_generator = permutations(strings, k)
for i, triple in enumerate(my_generator):
    print(''.join(triple))

print(f'# of {k}-length permutations of {"".join(mylist)}: {i+1}')
# permutations of 12345, 3 is 5x4x3=60

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
k = int(mylist[1])
strings.sort()
for i in combinations_with_replacement(strings, k):
    print(''.join(i))


# %% PYTHON - ITERTOOLS - Iterables and Iterators

from itertools import combinations

mylist = input().split()
strings = list(mylist[0])
k = int(mylist[1])
strings.sort()
for j in range(1, k + 1):
    for i in combinations(strings, j):
        print(''.join(i))

# %%

mylist = list(map(int, input().split()))
N = mylist[0]
M = mylist[1]

lists = []
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
    if 'a' in i:
        count += 1

print(count/tot)

# %% PYTHON - ITERTOOLS - Groupby
# sub-iterators grouped by value of key(v)
from itertools import groupby

out = []
for k, g in groupby('112333'):
    out.append(tuple((len(list(g)), int(k))))

print(*out)
# >>(2, 1) (1, 2) (3, 3)

print(*groupby('112333'))
# ('1', <itertools._grouper object at 0x00000270E084F6A0>)
# ('2', <itertools._grouper object at 0x00000270E084FD00>)
# ('3', <itertools._grouper object at 0x00000270E084F220>)
# %% PYTHON - ITERTOOLS - Groupby
L = [("a", 1), ('c', 'a'), ("a", 2), ("b", 3), ("b", 4)]
key_func = lambda x: x[0]

for key, group in groupby(L, key_func):
    # groupby doesn't rearrange keys,i.e., it will only group consecutive
    # items in the sequence
    print(key + " :", list(group))

"""
a : [('a', 1)]
c : [('c', 'a')]
a : [('a', 2)]
b : [('b', 3), ('b', 4)]
"""

# %% PYTHON - ITERTOOLS - max product
# max product, done with generator object
# product(A, B) returns the same as ((x,y) for x in A for y in B)
from itertools import product

# K, M = map(int, input().split())
K = 3
M = 1000
# M = used for modulus operator below
# K = # lists from which we need to choose one value that will max the
# sum of squares

# we want to use product to create all combinations from the K different
# lists;  then we can choose the max yielding combination


def sum_eles(combo):
    return sum(combo) % M


def square(arr):
    return [x**2 for x in arr]


inputs = ['2 5 4', '3 7 8 9', '5 7 8 9 10']
k_lists = (square(list(map(int, inputs[k].split()))[1:]) for k in range(K))
combos = product(*k_lists)
# we need to unpack because we need as many arguments to product as
# elements in each tuple
# product(k_lists) = product([1]) is going go generate only one output
# because there is only one argument
# product(a,b,c) will generate combinations of each elements of a, b, c
# with one another
results = list(map(sum_eles, combos))
print(max(results))

# for inputs:
# 3 1000
# 2 5 4
# 3 7 8 9
# 5 5 7 8 9 10

# correct answer is 206, but there is some deprecated code here

# %% PYTHON - ITERTOOLS - itertools
from itertools import product

x = [1, 2]
y = [3, 4]
print(*product(x, y))  # unpack the product iterable
print(*zip(x, y))
# (1, 3) (2, 4)

# unpacking the contents
print(*(x*y for x, y in zip(x, y)))
# 3 8

# see this site for explanation of unpacking a list into fstrnig
# /stackoverflow.com/questions/42756537/
# f-string-syntax-for-unpacking-a-list-with-brace-suppression
# =============================================================================
# Examples
# =============================================================================
print('Tuple products:', *(x*y for x, y in zip(x, y)))
# >Tuple products: (3, 8)

# without the unpacking, returns generator object
print('Tuple products: {}'.format((x*y for x, y in zip(x, y))))
# >Tuple products: <generator object <genexpr> at 0x00000270E0825D60>

# with only one unpacking, returns just first generated product
print('Tuple products: {}'.format(*(x*y for x, y in zip(x, y))))
# > Tuple products: 3

# therefore, need to multiply the string by the length
print(('Tuple products:' + ' {}' * len(x)).format(*(x*y for x, y in zip(x, y))))
# Tuple products: 3 8

# or could just add another {}
print('Tuple products:' + ' {} {}'.format(*(x*y for x, y in zip(x, y))))
# >Tuple products: 3 8

my_list = [(1, 3), (2, 4)]
print('Tuple products:' + ' {} {}'.format(*my_list))
# >Tuple products: (1, 3) (2, 4)

# with f-string, need a comma following or there will be error
print(f'Tuple products: {*my_list}')
#                        ^
# SyntaxError: can't use starred expression here

print(f'Tuple products: {my_list}')
# >Tuple products: [(1, 3), (2, 4)]
print(f'Tuple products: {my_list,}')
# >Tuple products: ([(1, 3), (2, 4)],)

# we will get ()'s surrounding contents of list
print(f'Tuple products: {*my_list,}')
#  Tuple products: ((1, 3), (2, 4))


# trailing comma allows unpacking of the product in fstring
print(f'Tuple products: {*(x*y for x,y in zip(x, y)),}')
# >Tuple products: (3, 8)

# without comment generates error
print(f'Tuple products: {*(x*y for x,y in zip(x, y))}')
# >     print(f'Tuple products: {*(x*y for x, y in zip(x, y))}')
#                                ^
# SyntaxError: can't use starred expression here

# generator object
gen = ((x, y) for x in [1, 2] for y in [3, 4])
for i in gen:
    print(i)

# %%
n = int(input())
print(all([int(input().split()) > 0 for _ in range(n)]))

# =============================================================================
# %% PYTHON - COLLECTIONS - Word Order
# =============================================================================
"""Reference.

This module implements specialized container datatypes providing
alternatives to Python’s general purpose built-in containers,
dict, list, set, and tuple.

namedtuple()  factory function for creating tuple subclasses
              with named fields
deque         list-like container with fast appends and pops on either
                end
ChainMap      dict-like class for creating a single view of multiple
               mappings
Counter       dict subclass for counting hashable objects
OrderedDict   dict subclass that remembers the order entries were added

defaultdict   dict subclass that calls a factory function to supply
              missing values
UserDict      wrapper around dictionary objects for easier dict
              subclassing
UserList      wrapper around list objects for easier list subclassing
UserString    wrapper around string objects for easier string
              subclassing

"""
# completed all 7 test cases 6/24/2020
# could use ordereddictionary as well

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

# %% # PYTHON - COLLECTIONS - Word Order

from collections import OrderedDict

n = int(input())
words = OrderedDict()

for i in range(n):
    word = input()
    if word in words:
        words[word] += 1
    else:
        words[word] = 1

out = list(map(str, words.values()))
print(len(out))
print(' '.join(out))


# %% # PYTHON - COLLECTIONS - Piling Up!
# USING deque is better way

from collections import deque

cases = int(input())

for c in range(cases):

    ncubes = int(input())
    cubes = deque(map(int, input().split()))
    left_index = 0

    prior = 2**31

    result = 'Yes'
    while len(cubes) > 0:
        if cubes[0] >= cubes[-1] and cubes[0] <= prior:
            prior = cubes.popleft()
        elif cubes[-1] > cubes[0] and cubes[-1] <= prior:
            prior = cubes.pop()
        else:
            result = 'No'
            break
    print(result)


# %% # PYTHON - COLLECTIONS - Piling Up!
# done 12/4/2020
# solution above using deque data structure is better

cases = int(input())

for c in range(cases):

    ncubes = int(input())
    cubes = list(map(int, input().split()))
    left_index = 0

    for i in range(1, ncubes):
        if cubes[i] <= cubes[i-1]:
            left_index = i
        else:
            break

    result = 'Yes'
    for j in range(left_index+1, ncubes):
        if cubes[j] < cubes[j-1]:
            result = 'No'
            break

    print(result)
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

x = pd.DataFrame([lookup.keys(), lookup.values()])
x = x.transpose()
x = x.sort_values(by=[1, 0], ascending=[False, True]).reset_index(drop=True)

for i in range(3):
    print(x.iloc[i, 0], x.iloc[i, 1])

# %% # PYTHON - COLLECTIONS - Company Logo --- suggested solution

from collections import Counter

if __name__ == '__main__':
    s = input()

    [print(*c) for c in Counter(sorted(s)).most_common(3)]

# %% Company logo practice
from collections import Counter

z = Counter()
x = 'hellfireehi'
y = 'akxlljookbbccc'
[print(*c) for c in Counter(sorted(x)).most_common(3)]
[print(*c) for c in Counter(sorted(y)).items()]

# %%
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
    students = namedtuple('student', fields)
    field1, field2, field3, field4 = input().split()
    student = students(field1, field2, field3, field4)
    total += int(student.MARKS)
print('{:.2f}'.format(total/N))

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

# =============================================================================
# PYTHON - TIME DATE - calendar
# =============================================================================
# %% PYTHON - TIME DATE - calendar
import calendar

print(calendar.TextCalendar(firstweekday=6).formatyear(2015))
month, day, year = map(int, input().split())
d = calendar.weekday(year, month, day)
print(calendar.day_name[d].upper())

# %% PYTHON - TIME DATE - Timedelta
# input in format
# t1 = 'Sun 10 May 2015 13:54:36 -0700'

"""2
Sun 10 May 2015 13:54:36 -0700
Sun 10 May 2015 13:54:36 -0000
Sat 02 May 2015 19:54:36 +0530
Fri 01 May 2015 13:54:36 -0000
"""

import math
import os
import random
import re
import sys
from datetime import timedelta
from datetime import datetime

# Complete the time_delta function below.


def time_delta(t1, t2):
    """time_delta."""
    formt = '%a %d %b %Y %H:%M:%S %z'
    time1 = datetime.strptime(t1, formt)
    time2 = datetime.strptime(t2, formt)
    diff = time1 - time2
    return int(abs(timedelta.total_seconds(diff)))


if __name__ == '__main__':
    #    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())

    for t_itr in range(t):
        t1 = input()
        t2 = input()
        delta = time_delta(t1, t2)
        print(delta)
        # fptr.write(delta + '\n')
        # fptr.close()

# =============================================================================
# %%  PYTHON - ERRORS AND EXCEPTIONS - Exceptions
# =============================================================================
"""ONLY 2 EXERCISES IN ERRORS/EXCEPTIONS"""

x = int(input())
for i in range(x):
    try:
        a, b = map(int, input().split())
        print(a//b)
    except Exception as e:
        # a best practice is to specific the type of Exception; having
        # general error is often a way to leave loopholes in code
        # that will allow errors in execution that are difficult to
        # debug; this program is simple but we might want to anticipate
        # different types of error and take different actions for each
        # type of error

        print("Try again:", e)

        # except ZeroDivisionError as e:
        # except ValueError as e:
# %% PYTHON - ERRORS AND EXCEPTIONS - My example

flag = 0
while flag==0:
    try:
        a, b = map(int, input().split())
        print(f'Answer is {a/b}')
        flag = 1
    except Exception as e:
        print(f'Try again: {e}')
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

# =============================================================================
# %% PYTHON - BUILTINS - Classes: Dealing with Complex Numbers
# =============================================================================

import math


class Complex(object):
    """Complex."""

    def __init__(self, real, imaginary):
        self.real = real
        self.imaginary = imaginary

    def __add__(self, no):
        """Docstring."""
        real = self.real + no.real
        imaginary = self.imaginary + no.imaginary
        return Complex(real, imaginary)

    def __sub__(self, no):
        """Docstring."""
        real = self.real - no.real
        imaginary = self.imaginary - no.imaginary
        return Complex(real, imaginary)

    def __mul__(self, no):
        """Docstring."""
        real = self.real * no.real - self.imaginary * no.imaginary
        imaginary = self.imaginary * no.real + self.real * no.imaginary
        return Complex(real, imaginary)

    def __truediv__(self, no):
        """Docstring."""
        coeff = 1/(no.real**2 + no.imaginary**2)
        real = coeff * (self.real * no.real + self.imaginary * no.imaginary)
        imaginary = coeff * (-self.real * no.imaginary +
                             self.imaginary * no.real)
        return Complex(real, imaginary)

    def mod(self):
        """Docstring."""
        mod = ((self.real**2)+(self.imaginary**2))**0.5
        return Complex(mod, 0)

    def __str__(self):
        """Docstring."""
        if self.imaginary == 0:
            result = "%.2f+0.00i" % (self.real)
        elif self.real == 0:
            if self.imaginary >= 0:
                result = "0.00+%.2fi" % (self.imaginary)
            else:
                result = "0.00-%.2fi" % (abs(self.imaginary))
        elif self.imaginary > 0:
            result = "%.2f+%.2fi" % (self.real, self.imaginary)
        else:
            result = "%.2f-%.2fi" % (self.real, abs(self.imaginary))
        return result


if __name__ == '__main__':
    c = map(float, input().split())
    d = map(float, input().split())
    x = Complex(*c)
    y = Complex(*d)
    print(*map(str, [x+y, x-y, x*y, x/y, x.mod(), y.mod()]), sep='\n')

# %% PYTHON - BUILT INS - any and all
# - are all integers in array positive?
# - are any of them palindromes, i.e., reflect around vertical
# axis in middle of number?

n = int(input())
ns = list(map(int, input().split()))
palin = any([str(x)[0:len(str(x)) // 2] == str(x)[-1:len(str(x))//2 - 1: -1]
             for x in ns])
print(bool(all([x > 0 for x in ns])*palin))

# %% PYTHON - BUILT INS - sorting
# order should be lowers, uppers, odds, evens

s = sorted(input())

lowers = []
uppers = []
odds = []
evens = []

for letter in s:
    try:
        if int(letter) % 2 == 1:
            odds.append(letter)
        elif int(letter) % 2 == 0:
            evens.append(letter)
    except Exception:
        if letter.islower():
            lowers.append(letter)
        elif letter.isupper():
            uppers.append(letter)

print(''.join(lowers) + ''.join(uppers) + ''.join(odds) + ''.join(evens))

# %% PYTHON - BUILTINS - Athlete Sort
# this works with pandas but site doesn't allow its use
import pandas as pd

N, M = list(map(int, input().split()))
df = pd.DataFrame()
for i in range(N):
    df = pd.concat([df, pd.Series(list(map(int, input().split())))], axis=1)

dft = df.T
dft.columns = list(range(M))
dft.index = list(range(N))
K = int(input())
dft.sort_values(by=K)

for i in range(N):
    print(' '.join(map(str, list(dft.values[i]))))


# %% PYTHON - BUILTINS - Athlete Sort
# solution

N, M = map(int, input().split())
rows = [input() for _ in range(N)]
K = int(input())

for row in sorted(rows, key=lambda row: int(row.split()[K])):
    print(row)

# =============================================================================
# %% PYTHON - PYTHON FUNCTIONALS - map and lambda
# =============================================================================


def fib(n):
    """fib."""
    if n == 0:
        mylist = []
    elif n == 1:
        mylist = [0]
    elif n == 2:
        mylist = [0, 1]
    elif n == 3:
        mylist = [0, 1, 1]
    else:
        mylist = [0, 1, 1]
        for i in range(n-3):
            mylist.append(mylist[-2] + mylist[-1])
    return mylist


print(list(map(lambda x: x**3, fib(5))))

# %% PYTHON - PYTHON FUNCTIONALS - Validate Email Addresses With Filter
"""
You are given an integer followed by email addresses. Your task is to
print a list containing only valid
email addresses in lexicographical order.

Valid email addresses must follow these rules:

    It must have the username@websitename.extension format type.
    The username can only contain letters, digits, dashes and underscore
    The website name can only have letters and digits.
    The maximum length of the extension is
"""
import re


def fun(s):
    """Docstring."""
    try:
        username, siteext = s.split('@')
        site, ext = siteext.split('.')

        # print(username)
        # print(site, ext)

        valid_ext = len(ext) <= 3
        valid_user = False
        valid_site = False
        # print(re.search(r'[^a-zA-Z0-9-_]', username))
        # print(re.search(r'[^a-zA-Z0-9]', site))

        if re.search(r'[^a-zA-Z0-9-_]', username) is None:
            if len(username) > 0:
                valid_user = True
        if re.search(r'[^a-zA-Z0-9]', site) is None:
            valid_site = True

        # print(valid_ext, valid_user, valid_site)
        return valid_ext & valid_user & valid_site
        # return True if s is a valid email, else return False

    except Exception:
        return False


def filter_mail(emails):
    """Docstring."""
    return list(filter(fun, emails))


if __name__ == '__main__':
    n = int(input())
    emails = []
    for _ in range(n):
        emails.append(input())

filtered_emails = filter_mail(emails)
filtered_emails.sort()
print(filtered_emails)
# %%  PYTHON - PYTHON FUNCTIONALS - Reduce Function
from functools import reduce
from fractions import gcd
import operator

# print(reduce(lambda x, y: x + y, [1, 2, 3, 4]))
# print(reduce(math.gcd,[2,4,8]))

n = int(input())
nums = []
dens = []
for i in range(n):
    numden = list(map(int, input().split()))
    nums.append(numden[0])
    dens.append(numden[1])

numer = math.prod(nums)
denom = math.prod(dens)
gc = math.gcd(numer, denom)
print(int(numer/gc), int(denom/gc))

""" solution

def product(fracs):
    t = reduce(operator.mul , fracs)
    return t.numerator, t.denominator
"""

# =============================================================================
# %% PYTHON - REGEX AND PARSING - re.split
# note that there is separate section below on just Regex with 45
# drills; the exercises that follow are under Python drills (17)
# =============================================================================
import re

x = '100,000'
print('\n'.join(re.split('[,.]', x)))

pat = re.compile(r'[,.]')
print("Second try:\n" + '\n'.join(re.split(pat, x)))

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
    """Docstring."""

    def handle_starttag(self, tag, attrs):
        """Docstring."""
        print("Start :", tag)
        for name, value in attrs:
            try:
                print("-> {0} > {1}".format(name, value))
            except Exception:
                print("-> {0} > None".format(name))

    def handle_endtag(self, tag):
        """Docstring."""
        print("End   :", tag)
        # very important that we have 3 spaces after 'End' so that colons
        # line up for after Start, End, and Empty.

    def handle_startendtag(self, tag, attrs):
        """Docstring."""
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

# %% PYTHON - REGEX AND PARSING - HTML Parser - Part 2
# I should read howHTMLParser works, because this exercise itself isn't
# that useful or informative

from html.parser import HTMLParser


class MyHTMLParser(HTMLParser):
    """Docstring."""

    def handle_comment(self, data):
        """Docstring."""
        num_lines = len(data.split('\n'))
        if num_lines > 1:
            print(">>> Multi-line Comment")
            print(data)
        else:
            print(">>> Single-line Comment")
            print(data)

    def handle_data(self, data):
        """Docstring."""
        if data.strip():  # space or \n will evaluate as False
            print('>>> Data')
            print(data)


html = ""
N = int(input())
for i in range(N):
    html += input().rstrip()
    html += '\n'  # so we end up feeding all N lines to parser at once

parser = MyHTMLParser()
parser.feed(html)
parser.close()


# %% leftover from prior HTML Parser - Part 2
# didn't need any of this - HTML parser module apparently already knows
# how to recognize comment, ml comment, or data
import re

multi_comment = """<!--[if IE 9]>IE9-specific content
<![endif]-->
"""
one_data = '<div> Welcome to HackerRank</div>'
ol_comment = '<!--[if IE 9]>IE9-specific content<![endif]-->'

single_line_comment = r'<!--(.*)-->'
multi_line_comment = r''
data_pattern = r'<\w+>(.*)(?=<[/]\w+>)'


# %%  PYTHON - REGEX AND PARSING - Detect HTML Tags, Attributes and
# Attribute Values

# cannot seem to solve all cases due to comments i.e.,  <!-- -->
# the tough problem is how to exclude

import re

N = int(input())
html = ''
for i in range(N):
    html = html + input()

html = input()
html = re.sub(r'<!.+-->', r'', html, re.DOTALL)
tags_in_line = re.findall(r"<((\w+).*?)>", html, re.M | re.DOTALL)

for tag in tags_in_line:
    attrs_contents = re.findall(r'\s(\w+)="([/\w.-]*)"', tag[0])
    print(tag[1])
    for item in attrs_contents:
        print('-> ' + str(item[0]) + ' > ' + str(item[1]))
"""
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for attr in attrs:
            print(f'-> {attr[0]} > {attr[1]}')
string = ''
num = int(input())
for i in range(num):
    N = input()
    string = string + N
parser = MyHTMLParser()
parser.feed(string)
"""


html= """<!--[if !IE 6]><!-->
  <link rel="stylesheet" type="text/css" media="screen, projection" href="REGULAR-STYLESHEET.css" />
<!--<![endif]-->

<!--[if gte IE 7]>
  <link rel="stylesheet" type="text/css" media="screen, projection" href="REGULAR-STYLESHEET.css" />
<![endif]-->

<!--[if lte IE 6]>
  <link rel="stylesheet" type="text/css" media="screen, projection" href="http://universal-ie6-css.googlecode.com/files/ie6.0.3.css" />
<![endif]-->"""

# %% PYTHON - REGEX AND PARSING - check if input is a float
import re

for _ in range(int(input())):
    print(bool(re.match(r'^[-+]?[0-9]*\.[0-9]+$', input())))

# %% PYTHON - REGEX AND PARSING - findall, finditer
import re

s = input()
# this worksfortest cases on Hackerrank,but may allow nonconsonant letters
pat = r'(?<=[^AEIOUaeiou])([aeiouAEIOU]{2,})(?=[^AEIOUaeiou])'
pat2 = r'(?<=[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm])([aeiouAEIOU]{2,})(?=[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm])'
matches = list(map(lambda x: x.group(), re.finditer(pat, s)))
if matches == []:
    print(-1)
else:
    for i, mat in enumerate(matches):
        print(mat)

# %% PYTHON - REGEX AND PARSING - findall, finditer

# our 'not vowel' patterndoesn'tfilter out the first ee like pat1 does
s = 'rabcdeefgyYhFjkIoomnpOeorteeeeet'
s2 = 'rabcd3eefgyYhFjkIoomnpOeorteeeeet'
list(map(lambda x: x.group(), re.finditer(pat, s2)))
# ['ee', 'Ioo', 'Oeo', 'eeeee']
list(map(lambda x: x.group(), re.finditer(pat2, s2)))
# ['Ioo', 'Oeo', 'eeeee']

matches2 = list(map(lambda x: x.group(0), re.finditer(pat, s)))
matches3 = list(map(lambda x: x.groups(), re.finditer(pat, s)))
print(matches2)
# ['ee', 'Ioo', 'Oeo', 'eeeee']

print(matches3)
# [('ee',), ('Ioo',), ('Oeo',), ('eeeee',)]

print(list(re.finditer(pat, s)))

# %% PYTHON - REGEX AND PARSING - .group(). .groups(), .groupdict()
"""
>>> import re
>>> m = re.match(r'(\w+)@(\w+)\.(\w+)','username@hackerrank.com')

group()
returns one or more subgroups of the match.
>>> m.group(0)       # The entire match
'username@hackerrank.com'
>>> m.group(1)       # The first parenthesized subgroup.
'username'
>>> m.group(2)       # The second parenthesized subgroup.
'hackerrank'
>>> m.group(3)       # The third parenthesized subgroup.
'com'
>>> m.group(1, 2, 3)   # Multiple arguments give us a tuple.
('username', 'hackerrank', 'com')


groups()
returns a tuple containing all the subgroups of the match.
>>> m.groups()  # three groups in pattern means groups will have 3 items
('username', 'hackerrank', 'com')

groupdict()
returns a dictionary containing all the named subgroups of the match,
keyed by the subgroup name.

>>> m = re.match(r'(?P<user>\w+)@(?P<website>\w+)\.(?P<extension>\w+)',
                 'myname@hackerrank.com')
>>> m.groupdict()
{'website': 'hackerrank', 'user': 'myname', 'exten
"""

import re

#s = input()
s = 'a235bbcddabbok33'
m = re.search(r'(?P<first>[A-Za-z0-9])\1+', s)  # two consec appearances
print(f'string\t\t: \t\t{s:10}')
print(f'.group(1)\t: \t\t{m.group(1):10}')
print(f'.group()\t: \t\t{m.groups()}')
print(f'.groupdict(): \t\t{m.groupdict()}')
"""
string		: 		a235bbcddabbok33
.group(1)	: 		b
.group()	: 		('b',)
.groupdict(): 		{'first': 'b'}
"""

# %% PYTHON - REGEX AND PARSING - Validating Postal Codes
# hard
import re

regex_integer_in_range = r"[1-9][0-9]{5}$"
regex_alternating_repetitive_digit_pair = r"(\d)(?=\d\1)"


def pat_match(str):
    return (bool(re.match(regex_integer_in_range, str)) and
            len(re.findall(regex_alternating_repetitive_digit_pair, str)) < 2)


# 7 Test cases, 0 through 6
test_strings = ['110000', '111456', 'abcdef', '101201', '542361', '4542867',
                '137370']

# 111456 is True because the first one is the match. It doesn't matter that
# the in between digit is the same. So the pattern is matched by the first 1,
# any digit, and the repeated 1.
answers = [pat_match(s) for s in test_strings]
print(answers)
# [False, True, False, True, True, False, False]

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

# %% PYTHON - REGEX AND PARSING - Validating Credit Card Numbers 2
# rewrote from scratch for practice; this works

import re


N = int(input())

for i in range(N):
    pat1 = r'[456](\d){3}-(\d){4}-(\d){4}-(\d){4}'
    pat2 = r'[456](\d){15}'
    number = input()

    try:
        assert len(number) == 16 or len(number) == 19
        if len(number) == 16:
            assert re.match(pat2, number)
        elif len(number) == 19:
            assert re.match(pat1, number)
        assert not re.search(r'(\d)\1\1\1', ''.join(number.split('-')))

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
    """Docstring."""
    return ' and '


def my_filter2(match):
    """Docstring."""
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
# >1 2 3 and 2d bbd
# %% PYTHON - REGEX AND PARSING - Validating and Parsing Email Addresses
"""
2
DEXTER <dexter@hotmail.com>
VIRUS <virus!@variable.:p>
"""
import email.utils
import re

for i in range(int(input())):
    t = email.utils.parseaddr(input())
    # creates tuple with two items
    # it looks like the name is not checked, and the problem does
    # in fact NOT
    # say anything about the name, only the user; so it was pointless
    # to do
    # prior match on name in section above;  some names are in fact
    # not user  \w includes _name <user@email.com>
    # re.match automatically starts at beginning, so any characters
    # at front that are not alphabetic causes False
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

        # must not have non alphanum
        assert not re.search(r'[^a-zA-Z0-9]', u)

        # \1 means repeat of prior match
        assert not re.search(r'(.)\1', u)

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
#  { #fff returns #fff, not (#fff, fff) because grouping with # is
# optional
# when theres a subgroup
#  { #fff123 returns #fff123, not (#fff123, 123)
# the {1,2} outside of () group means you could repeat that (), which is
# why the second three hex digits wouldn't need #

pat2 = r'(?<!^)(#([\da-fA-F]{3}){1,2})'  # { #fff would return
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

# %% PYTHON - 2 XML
import sys
import xml.etree.ElementTree as etree


def get_attr_number(node):
    """Docstring."""
    return etree.tostring(node).count(b'=')    # your code goes here


if __name__ == '__main__':
    # sys.stdin.readline()
    # xml = sys.stdin.read()
    xml = input()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()

    print(get_attr_number(root))

# %% PYTHON -  XML2 - Find the Maximum Depth
import xml.etree.ElementTree as etree

maxdepth = -1


def depth(elem, level):
    """Docstring."""
    global maxdepth
    if (level == maxdepth):
        maxdepth += 1

    for child in elem:
        depth(child, level + 1)


if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml = xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)

# %% PYTHON - Closures and Decorators - Standardize Mobile Number
# Using Decorators


def wrapper(f):
    """Docstring."""
    def phone(list_n):
        f(["+91 "+c[-10:-5]+" "+c[-5:] for c in list_n])
    return phone


@wrapper
def sort_phone(list_nums):
    """Docstring."""
    print(*sorted(list_nums), sep='\n')


if __name__ == '__main__':
    li = [input() for _ in range(int(input()))]
    sort_phone(li)

# %% PYTHON - Closures and Decorators - Decorators 2 - Name Directory


def person_lister(f):
    """Docstring."""
    def inner(people):
        # complete the function
        return map(f, sorted(people, key=lambda x: int(x[2])))
    return inner


@person_lister
def name_format(pers):
    """Docstring."""
    return ("Mr. " if pers[3] == "M" else "Ms. ") + pers[0] + " " + pers[1]


if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')

"""
Robert Bustle 32 M
Robert Bux 32 M
Mike Thomson 20 M

For sorting a nested list based on some parameter, you can use the
itemgetter library. You can read more about it here.
"""
# %% PYTHON - Closures and Decorators - Decorators 2 -
#  Name Directory Try 2
# we use map iterator above because it disappears after iteration; if
# we want to use
# memory to store elements, we need to use list


def person_lister(f):
    """Docstring."""

    def inner(people):
        # complete the function
        return list(map(f, sorted(people, key=lambda x: int(x[2]))))
    return inner


@person_lister
def name_format(pers):
    """Docstring."""
    return ("Mr. " if pers[3] == "M" else "Ms. ") \
            + pers[0] + " " + pers[1]


if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')

# since we didn't store the list in a variable, it doesn't matter
# %% Python - Numpy

# The reshape tool gives a new shape to an array without changing its
# data. It creates a new array and does not modify the original array
# itself.
import numpy

my_array = numpy.array([1, 2, 3, 4, 5, 6])
print(numpy.reshape(my_array, (3, 2)))


# Linear Algebra
# The linalg.det tool computes the determinant of an array.

print(numpy.linalg.det([[1, 2], [2, 1]]))       # Output : -3.0
print(numpy.linalg.det([[1.1, 1.1], [1.1, 1.2]]))  # Output : -3.0

# The linalg.eig computes the eigenvalues and right eigenvectors of
# square array.
vals, vecs = numpy.linalg.eig([[1, 2], [2, 1]])
print(vals)  # Output : [ 3. -1.]
print(vecs)  # Output : [[ 0.70710678 -0.70710678]
#            #          [ 0.70710678  0.70710678]]

A = numpy.array([1, 2])
B = numpy.array([3, 4])
print(numpy.dot(A, B))    # Output : 11

A = numpy.array([1, 2])
B = numpy.array([3, 4])
print(numpy.cross(A, B))  # Output : -2

x = [1, 2]
y = [3, 4]
A = []
A.append(x)
A.append(y)
A = numpy.matrix(A)
print(numpy.dot(A, A))

# =============================================================================
# # polynomials
# =============================================================================
# The poly tool returns the coefficients of a polynomial with the given
# sequence of roots.

print(numpy.poly([-1, 1, 1, 10]))
# Output : [  1 -11   9  11 -10]

# The roots tool returns the roots of a polynomial with the given
# coefficients.

print(numpy.roots([1, 0, -1]))          #Output : [-1.  1.]

# The polyint tool returns an antiderivative (indefinite integral) of a
# polynomial.

print(numpy.polyint([1, 1, 1]))
# Output : [ 0.33333333  0.5         1.          0.        ]

# The polyder tool returns the derivative of the specified order of a
#   polynomial.

print(numpy.polyder([1, 1, 1, 1]))
# Output : [3 2 1]

# The polyval tool evaluates the polynomial at specific value.

print(numpy.polyval([1, -2, 0, 2], 4))
# Output : 34

# The polyfit tool fits a polynomial of a specified order to a set of
# data using a least-squares approach.
print(numpy.polyfit([0,1,-1, 2, -2], [0, 1, 1, 4, 4], 2))
# Output : [  1.00000000e+00   0.00000000e+00  -3.97205465e-16]
# The functions polyadd, polysub, polymul, and polydiv also handle
# proper addition, subtraction, multiplication, and division of polynomial
# coefficients, respectively.

# %%
n = int(input())

A = []
for i in range(2 * n):
    row = list(map(int, input().split()))
    A.append(row)

B = numpy.matrix(A[n:])
A = numpy.matrix(A[0:n])

print(numpy.dot(A, B))


p = list(map(float, input().split()))
x = float(input())
print(numpy.polyval(p, x))
# %% PYTHON - Debugging - Default arguments


class EvenStream(object):
    """Docstring."""

    def __init__(self):
        self.current = 0

    def get_next(self):
        """Docstring."""
        to_return = self.current
        self.current += 2
        return to_return


class OddStream(object):
    """Docstring."""

    def __init__(self):
        self.current = 1

    def get_next(self):
        """Docstring."""
        to_return = self.current
        self.current += 2
        return to_return


def print_from_stream(n, stream=EvenStream()):
    """Docstring."""
    stream.__init__()   # this is the key line to fix error; need to
    # init the Stream object, because we were just passed it as arg or
    # as dafault
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
# =============================================================================
# %% HACKERRANK SQL SKILLS - 58 TOTAL EXERCISES
# =============================================================================
"""
Topics covered in 58 questions on hackerrank, only some examples
included as reference in this file

# Qs Topic
---- ------------
- 20 Basic Select            x
-  5 Advanced Select         x
- 17 Aggregation             x
-  8 Basic Join              x
-  5 Advanced Join           3
-  3 Alternative Queries     1
====
54 of 58 done
"""

# %% HACKERRANK SQL - SQL Project Planning
"""
SET sql_mode = '';
SELECT Start_Date, End_Date
FROM
    (SELECT Start_Date FROM Projects WHERE Start_Date NOT IN (SELECT End_Date FROM Projects)) a,
    (SELECT End_Date FROM Projects WHERE End_Date NOT IN (SELECT Start_Date FROM Projects)) b
WHERE Start_Date < End_Date
GROUP BY Start_Date
ORDER BY DATEDIFF(End_Date, Start_Date), Start_Date


# OR

SELECT Start_Date, MIN(End_Date)
FROM
# Choose start dates that are not end dates of other projects
#     (if a start date is an end date, it is part of the samee project) */
    (SELECT Start_Date FROM Projects WHERE Start_Date NOT IN
     (SELECT End_Date FROM Projects)) a,
# Choose end dates that are not end dates of other projects */
    (SELECT end_date FROM PROJECTS WHERE end_date NOT IN
     (SELECT start_date FROM PROJECTS)) b
# At this point, we should have a list of start dates and end dates that
# don't necessarily correspond with each other */
# This makes sure we only choose end dates that fall after the start date,
#and choosing the MIN means for the particular start_date, we get the closest
# end date that does not coincide with the start of another task */
where start_date < end_date
GROUP BY start_date
ORDER BY datediff(start_date, MIN(end_date)) DESC, start_date
# %% HACKERRANK SQL - Ollivander's Inventory
SELECT
    w.id,
    wp.age,
    w.coins_needed,
    w.power
FROM Wands w
JOIN Wands_Property wp
    ON w.code = wp.code
WHERE wp.is_evil=0
AND w.coins_needed =
    # this is the key part
    (select
         min(w1.coins_needed)
    from Wands as w1
    join Wands_Property as p1
        on (w1.code = p1.code)
    where w1.power = w.power
        and p1.age = wp.age)
ORDER BY w.power DESC, wp.age DESC;
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
# %% HACKERRANK SQL - Interviews
""" ANSWER

SELECT
    con.contest_id,
    con.hacker_id,
    con.name,
    SUM(total_submissions),
    SUM(total_accepted_submissions),
    SUM(total_views),
    SUM(total_unique_views)
FROM Contests con
JOIN Colleges col on con.contest_id = col.contest_id
JOIN Challenges cha on  col.college_id = cha.college_id
LEFT JOIN
    (SELECT
        challenge_id,
        SUM(total_views) AS total_views,
        SUM(total_unique_views) AS total_unique_views
    FROM View_Stats
    GROUP BY challenge_id) vs
    ON cha.challenge_id = vs.challenge_id
LEFT JOIN
    (SELECT
        challenge_id,
        SUM(total_submissions) AS total_submissions,
        SUM(total_accepted_submissions) AS total_accepted_submissions
    FROM Submission_Stats
    GROUP BY challenge_id) ss
    ON cha.challenge_id = ss.challenge_id

GROUP BY con.contest_id, con.hacker_id, con.name
HAVING SUM(total_submissions)!=0 or
       SUM(total_accepted_submissions)!=0 or
       SUM(total_views)!=0 or
       SUM(total_unique_views)!=0
ORDER BY contest_id;
"""

# **end sql
# =============================================================================
# %%  HACKERRANK - REGEX DRILLS - 47 total
# =============================================================================
"""
Course Outline (not all problems in this outline are included in this file)
---------------------------------------------------------------------------
- 6 Introduction            DONE
- 3 Character Class         DONE
- 5 Repetitions             DONE
- 3 Grouping and Capturing  DONE
- 4 Backreferences         -2 (2 problems not supported with Python)
- 4 Assertions              DONE
- 22 Applications           DONE
===
Finished 45 of 47

Other References:
https://docs.python.org/3/library/re.html#re.MULTILINE


"""

# %% REGEX - Introduction - start() and end()
# https://docs.python.org/3/library/re.html
# https://docs.python.org/3/howto/regex.html
# https://www.oreilly.com/library/view/mastering-regular-expressions/0596528124/
# Return the indices of the start and end of the substring matched.
# by the group.
# """ Notes
# The dot (.) matches anything (except for a newline).
# \d digit
# \D non-digit
# \s whitespace [ \r\n\t\f ].
# \S  nonwhitespace
# \w word (alphanumeric and numbers and _)
# \W non word
# ^ start
# $ end
# [] only matches one character []
# [^] matches any character NOT in[]
# + one ore more
# * 0 or more
# () captureing group
# (?:) noncapturing
"""Here, b? is optional and matches nothing.
Thus, (b?) is successfully matched and capture nothing.
o is matched with o and \1 successfully matches the nothing captured by the
group.

(b?)o\1 matches 'o' because the first group did match nothing, and so didthe \1
(b)?o\1 does not match 'o', because (b)? didn't even participate, so \1
    cannot match
c(?=o) matches 2nd o in chocolate - POSITive lookahead
c(?!o) matches 1st o in chocolate - negative loohahead
(?<=[a-z])[aeiou] matches e in he10 because h comes before - pos lookbehind
(?<![a-z])[aeiou] matches 0 in he10 because 1 comes before, not a letter
    -- neg lookbehind
re.MULTILINE mode at end of expression counts \n as edge for $ as well vs.only
at end of string for $ when re.MULTILINE is notinlcuded
"""


prog = re.compile(r'[a-z]', flags=0)
prog
# Out[30]: re.compile(r'[a-z]', re.UNICODE)
prog.flags
# 32
prog.groups
# 0
prog = re.compile(r'([a-z])', flags=0)
prog.groups
# 1
prog.pattern
# Out[40]: '([a-z])'

result = prog.match(string)
#    is equivalent to
result = re.match(pattern, string)
#    using re.compile() allows reuse of regex object

re.search(pattern, string, flags=0)
# find first location where matches

# re.match - will only match at beginning of string

# re.DOTALL
#    Make the '.' special character match any character at all,
# including a # newline; without this, '.' will match anything except
# a newline.
# Corresponds to the inline flag (?s).

# %% REGEX
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
# m.end() is 1 greater than index of match ending, so if 0-5 match,
# m.end() is 6
# %% REGEX - Introduction - search for start/end index of matches

import re

my_string = input()
pat = input()
index = 0

if re.search(pat, my_string):  # stops looking after find first match
    while index + len(pat) < len(my_string):
        m = re.search(pat, my_string[index:])  # begins search with new index
        if m is None:
            break
        else:
            print(f'Match start, end: '
                  f'({index + m.start()}, {index + m.end()-1})')
            index += m.start() + 1
else:
    print('No matches found (-1, -1)')


# %%  REGEX - pattern match
import re
import sys

regex_pattern = r"...\....\....\...."  # must have 'r'.
test_string = input()
match = re.match(regex_pattern, test_string) is not None
# True if there's match, otherwise false

print(str(match).lower())  # prints 'true' or 'false'

# %%  REGEX - pattern match

"""
pattern below didn't work because 2 of the test cases out of 7 had
7 character length, unlike the specification below

with the following conditions:

must be of length: 6
First character: 1, 2 or 3
Second character: 1, 2 or 0
Third character: x, s or 0
Fourth character: 3, 0 , A or a
Fifth character: x, s or u
Sixth character: . or ,  don't need excape \ inside []
"""
Regex_Pattern = r'[1-3][0-2][x,s,0][3,0,A,a][x,s,u][.,]'

print(bool(re.match(r'[abc][1-3]', 'a5')))  # False
print(bool(re.match(r'[abc][1-3]', 'a3')))  # True

# %%  REGEX - excluding

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
# $ boundary matcher matches occurrence of a character/character class
# or group at the end of a line.

# \b assert position at a word boundary.
# 3 positions for \b
# Before the first character in the string, if the first character is a
# word character.
# Between two characters in the string, where one is a word character
# and the other is not a word character.
# After the last character in the string, if the last character is a
# word character.

# () groups part of regex together
# (?:) can be part of match that is optional

# (Al|Bo) matches either Al or Bo

# (\d)\1: It can match 00, 11, 22, 33, 44, 55, 66, 77, 88
# or 99.
# This tool (\1 references the first capturing group) matches the same
# text as previously matched by the capturing group.


r'([a-z])([\w])([\s])\1\2\3'
# lowercase, word, whitespace, then repeat 1st (lowercase), then word,
# whitespace

# %% REGEX - Parse for digits
rp = r'^\d{2}(-?)(\d{2}\1){2}\d{2}$'

print(bool(re.search(rp, '12-34-56-78')))


# %% REGEX - Lookahead and behind
# The positive lookahead (?=) asserts regex_1 to be immediately followed
# by regex_2. The lookahead is excluded from the match. Does not return
# matches of regex_2. The lookahead only asserts whether a match is
# possible or not.
rp = r'o(?=oo)'
print(re.search(rp, 'xxoooxx'), bool(re.search(rp, 'xxoooxx')))

# (?<=regex_2)regex_1
rp = r'(?<=xx)o'
print(re.search(rp, 'xxo'), bool(re.search(rp, 'xxo')))

# The positive lookbehind (?<=) asserts regex_1 to be immediately
# preceded by regex_2. Lookbehind is excluded from the match
# (do not consume matches of regex_2), but only assert whether match is
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
# this is one match,   this is second match for word, space, ,
# or . or /
for i in range(n):
    text = input()
    # text = html.loc[i].values[0]
    alllinks = re.findall(pat, text)
    for link, title in alllinks:
        print(link.strip() + ',' + title.strip())


# %% Regex - Test HTML
# testing how the groups are addressed in tuples
import re

pat = r'<a href="(.*?)".*?>([\w ,./]*)(?=</)'

text = '<p><a href="http://www.quackit.com/html/tutorial/html_links.'
text = text + 'cfm">Example Link</a></p>'
all = re.findall(pat, text)
print(all[0][0] + ',' + all[0][1])

# %% REGEX - APPLICATIONS - The British and American Style of Spelling
"""
DONE= 11/30/2020

Input
number N
N lines of text
number T
T words in American Style spelling i.e., ending in ze vs. British se
"""
import re

N = int(input())
text = []
for i in range(N):
    text.append(input())
all_text = '\n'.join(text)

T = int(input())

for j in range(T):
    word_american = input()
    pat = word_american[:-2] + '(?:se|ze)'
    print(len(re.findall(pat, all_text)))


# %% REGEX - APPLICATIONS - Find a Word
"""
text = 'foo bar (foo) bar foo-bar foo_bar foo\'bar bar-foo bar, foo.'
text = 'colourfuture saturday.future face_future(anxiety obtain.future '
text = text + 'surroundings\'futurerefrigerator alone)futurecomparison '
text = text + 'wine-future,tight futureimpatient '
text = text + 'bodyfuture excite(future(grandfather'
"""

import re

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

A new identification number is given for every Citizen of the Country
Utopia and it has the following format.

- The string must begin with between 0-3 (inclusive) lowercase letters.
- Immediately following the letters, there must be a sequence of digits
    (0-9). The length of this segment must be between 2 and 8, both
    inclusive.
- Immediately following the numbers, there must be atleast 3 uppercase
    letters.
2
abc012333ABCDEEEE
0123AB

Sample Output

VALID
INVALID


3c0123ABC
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


# %% REGEX - APPLICATIONS - Find a Sub-word

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

pat1 = r'^([_.])\d+([a-zA-Z]){0,}_?$'
for _ in range(int(input())):
    u = input()

    try:
        x = re.match(pat1, u)
        assert x  # x evaluates as True
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
    pat1 = r'^([_.])\d+([a-zA-Z]){0,}_?$'
    # inside the [], . means ., no escape \ needed
    pat2 = r'^([._])\d+([a-zA-Z]){0,}_?$'
    pat3 = r'^(_.)\d+([a-zA-Z]){0,}_?$'
    # pat3 catches '_,0898989811abced_' because . is any
    # character, not just .; if we want to specify ., we
    # need \.
    pat4 = r'^(._)\d+([a-zA-Z]){0,}_?$'
    pat5 = r'^(_\.)\d+([a-zA-Z]){0,}_?$'
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
languages = 'C:CPP:JAVA:PYTHON:PERL:PHP:RUBY:CSHARP:HASKELL:'
languages = languages + 'CLOJURE:BASH:SCALA:ERLANG:CLISP:LUA:'
languages = languages + 'BRAINFUCK:JAVASCRIPT:GO:D:OCAML:R:PASCAL:'
languages = languages + 'SBCL:DART:GROOVY:OBJECTIVEC'
languages = languages.split(':')

n = int(input())
for i in range(n):
    code, language = input().split()
    # code is a code snippet, but is not used in this exercise
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
    answer = 'CountryCode=' + x[0] + ',' +\
             'LocalAreaCode=' + x[1] +', ' +\
             'Number=' + y[0]
    print(answer)

# %% REGEX - APPLICATIONS - Split the phone numbers

    x = re.findall(r'(\d{1,3})[ -](\d{1,3})[ -](\d{4,10})$', num)
    answer = 'CountryCode=' + x[0][0] + ',LocalAreaCode='
    + x[0][1] + ',Number=' + x[0][2]

    print(answer)

# %% REGEX - APPLICATIONS - Detecting Valid Latitude and Longitude Pairs
"""Description.

Given a line of text which possibly contains the latitude and longitude
of a point, can you use regular expressions to identify the latitude
and longitude referred to (if any)?

Input Format
The first line contains an integer N, which is the number of tests to
follow.
This is followed by N lines of text. Each line contains a pair of
co-ordinates which possibly indicate the latitude and longitude of a
place.

Constraints
1 <= N <= 100
The latitude and longitude, if present will always appear in the form
of (X, Y) where X and Y are decimal numbers.
For a valid (latitude, longitude) pair:
-90<=X<=+90 and -180<=Y<=180.
They will not contain any symbols for degrees or radians or N/S/E/W.
There may or may not be a +/- sign preceding X or Y.
There will be a space between Y and the comma before it.
There will be no space between X and the preceding left-bracket, or
between Y and the following right-bracket.
There will be no unnecessary zeros (0) before X or Y.

Output Format
"Valid" where X and Y are the latitude and longitude which you found
to be a valid (latitude, Longitude) pair.
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
# text = 'kj342lal;;hackerrank8a  asdf hackerrank kal8'

import re

N = int(input())
count = 0
for i in range(N):
    text = input().lower()
    if re.match('.*hackerrank.*', text):
        count += 1
print(count)

# %% REGEX - APPLICATIONS - UK and US: Part 2
# not sure why the \b doesn't work

import re

n = int(input())

text = []
for i in range(n):
    line = input()
    text.append(line)

t = int(input())

total_text = '\n'.join(text)

for i in range(t):
    word = input()
    US_word = word.replace('our', 'or')
    pat = '(' + word + '\\b|' + US_word + '\\b)'
    # need word boundaries so that 'savoury' looks different
    # from 'savour'
    print(len(re.findall(pat, total_text)))

# %% REGEX - APPLICATIONS - HTML
import re
from collections import defaultdict

inputs = []
for _ in range(int(input())):
    inputs.append(input())

string = ''.join(inputs)

tags = re.findall(r'<(\w+)*>', string)

tag_attributes = defaultdict(list)
for tag in tags:
    tag_, attr = tag
    tag_attributes[tag_].extend(re.findall(r'(\w+)=[\'\"]', attr))

for tag, attr in sorted(tag_attributes.items()):
    print(':'.join([tag, ','.join(sorted(set(attr)))]))
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


# %% REGEX - APPLICATIONS - Detect HTML Attributes
import re

d = {}
N = int(input())
for i in range(N):
    tags_in_line = re.findall(r"<((\w+).*?)>", input())
    # re.findall here returns tuples for each opening tag, once for the full
    # contents of the opening <>, and another time in non-greedy fashion

    for tag in tags_in_line:
        attrs = [x.group(1) for x in re.finditer(r"\s(\w+)=", tag[0])]
        # or
        # attrs = re.findall(r"\s(\w+)=", tag[0])

        # tag[0] is the opening tag and all attributes that follow
        # tag[1] is the closing tag

        if tag[1] in d:
            d[tag[1]].update(attrs)
            # update adds the missing values to the
            # if the key is already present
        else:
            d[tag[1]] = set(attrs)

print(*sorted(["{}:{}".format(k, ",".join(sorted(v))) for k, v in d.items()]),
      sep="\n")

# %% REGEX - APPLICATIONS - Detect HTML Tags
import re

N = int(input())
text = []
for i in range(N):
    text.append(input())
alltext = "\n".join(text)
tags = re.findall(r'<\s*(\w+).*?>', alltext)
set_tags = list(set(tags))
set_tags.sort()
print(";".join(set_tags))

# %% REGEX - APPLICATIONS - Detect the Email Addresses
"""
solved 11/25/2020

You will be provided with a block of text, spanning not more than hundred
lines. Your task is to find the unique e-mail addresses present in the text.
You could use Regular Expressions to simplify your task. And remember that
the "@" sign can be used for a variety of purposes!

Input Format

The first line contains an integer N (N<=100), which is the number of lines
 present in the text fragment which follows.
From the second line, begins the text fragment (of N lines) in which you need
 to search for e-mail addresses.

Output Format

All the unique e-mail addresses detected by you, in one line, in
lexicographical order, with a semi-colon as the delimiter.


How to use findall with noncapturing group

noncapturing group is a group that doesn't create text separate from full match

name#some.website.co.in
           Regex     Output by re.findall()
r'#\w+(\.\w+)*'      ['.in']
r'(#\w+(\.\w+)*)'    [('#some.website.co.in', '.in')]
r'#\w+(?:\.\w+)*'    ['#some.website.co.in']
"""

import re

N = int(input())

names = []
sites = []
for i in range(N):
    line = input()
    names = names + re.findall(r'(\w+(?:\.\w+)*)@', line)
    sites = sites + re.findall(r'@(\w+(?:\.\w+)*)', line)
emails_unique = []

for i in range(len(names)):
    emails_unique = emails_unique + [names[i] + '@' + sites[i]]
emails_unique = list(set(emails_unique))
emails_unique.sort()
print(';'.join(emails_unique))


# %% REGEX - APPLICATIONS - Build a Stack Exchange Scraper
import re, sys

# stack = sys.stdin.read()
stack = input()

about = re.findall(r'(?<=question-hyperlink.>)\w+.+?(?=<)', stack)
ids = re.findall(r'(?<=question-summary-)\d+', stack)
times = re.findall(r'(?<=relativetime.>)\w+.+?(?=<)', stack)

# print(ids, about, times)

for i in range(len(ids)):
    print(ids[i] + ';' + about[i] + ';' + times[i])

results = re.findall(r'question-summary-(\w\w\w\w\w)".\
                     *?class="question-hyperlink">(.+?)\
                     </a>.*?class=\"relativetime\">(.+?)\
                     </span>', stack, re.DOTALL)
"""

#  Out[78]:
# [('80407', 'about power supply of opertional amplifier', '11 hours ago'),
# ('80405', '5V Regulator Power Dissipation', '11 hours ago')]

for result in results:
    print(';'.join(result))

"""
# %% REGEX - APPLICATIONS - Detect Domain Name
"""
<li id="cite_note-1"><span class="mw-cite-backlink"><b>^ ["Train (noun)"]
(http://www.askoxford.com/concise_oed/train?view=uk). <i>
(definition – Compact OED)</i>. Oxford University

(http://ww2.abc.com/concise_oed/train?view=uk). <i>
(http://abc.123.com/concise_oed/train?view=uk). <i>


"""
# NOTE - DASH is not part of \w, so weneeded to use [\w-] to include the dash
# as a possible character in domain name

import re

N = int(input())
html = ''
for i in range(N):
    html = html + '\n' + input()

dom_names = re.findall(r'(?:https*://)(?:www[.]|ww2[.])*([\w-]+(?:\.[\w-]+)*\.[A-Za-z0-9]+)(?=[_/])*', html)
# remember lookbehind is fixed width, so betterto use noncapturing groups when
# could be different widths like http or https

for i in range(len(dom_names)):
    if re.match(r'(www|ww2)', dom_names[i]):
        dom_names[i] = dom_names[i][4:]
print(';'.join(sorted(list(set(dom_names)))))
# %% REGEX - APPLICATIONS - IP Address Validation


N = int(input())

for i in range(N):
    text = input()

    if '.' in text:
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
# %% REGEX - APPLICATIONS - Building a Smart IDE: Identifying comments
# try 1
# example code:
"""
// this is a single line comment
x = 1; // a single line comment after code

/* This is one way of writing comments */"""
# code=
"""/* This is a multiline
   comment. These can often
   be useful*/
"""
import re
import sys


code = input()
code = sys.stdin.read()
single_line_pat = r'([/]{2}.*)'
line_pat = r'/\*.*\*/'
ml_pat = r'/\*(?:.*\n){1,}(?:.*){1,}(?=\n)'

both = r'([/]{2}.*)|(/\*.*\*/)'
both_ml = r'([/]{2}.*)|(/\*.*\*/)|(/\*(?:.*\n){1,}(?:.*){1,}\*/(?=\n))'

"""sin_comments = re.findall(single_line_pat , code)
l_comments = re.findall(line_pat, code)
ml_comments = re.findall(both_ml, code)
comments = re.findall(both, code)

print('--------\n')
print('Results')
print('--------\n')
"""

matches = list(map(lambda x: x.group(), re.finditer(both_ml, code)))
for i in matches:
    mat = i.split('\n')
    for j in mat:
        print(j.strip())

# print('Single line comments:', sin_comments)
# print('Multi-line comments:', l_comments)
# print('All comments:', comments)


# %% REGEX - APPLICATIONS - Building a Smart IDE: Identifying comments
# CORRECT

import re
import sys

pat = r'(/\*.*?\*/|//.*?$)'
txt = sys.stdin.read()
# re.sub() for Testcase #4: others will just work with comment
matches = re.findall(pat, txt, re.DOTALL | re.MULTILINE)
# DOTALL and Multiline take care of all the messinessI was tryingto fix
# in my first try at this

print("\n".join(re.sub('\n\s+', '\n', comment) for comment in matches))

# %% REGEX - APPLICATIONS - Building a Smart IDE: Programming Language
# Detection
"""
C code:
    /*  */         are comments
    #include<   >   for importing DIFFERENT FOROM JAVA - NO ; AT END
    {}            for code segments
    ;              to end lines of code

Java code:
    /*  */         are comments
    ;              to end lines
    {}             to encapsulate code
    import javax;  to import DIFFERENT FROM C

Python code:
    #              are comments
                   to end lines; DIFFERENT FROM C/JAVA
    :             to encapsulate code
    import xx     slightly DIFFERENT FROM C/JAVA

"""
import re
import sys

txt = sys.stdin.read()
# txt = input()

cjava_pattern = r'(/\*.*?\*/|;.*?})'
c_pattern = r'#include<'
java_pattern = r'import \w+;?'
# python_pat = r'#|:\n'

find_cjava = re.findall(cjava_pattern, txt, re.DOTALL | re.MULTILINE)
find_c = re.findall(c_pattern, txt)
find_java = re.findall(java_pattern, txt)
# find_python = re.findall(python_pat, txt)

if find_cjava != []:
    # try c vs. java
    if find_c != []:
        print('C')
    elif find_java != []:
        print('Java')
    else:
        print('C or Java??')
else:
    print('Python')


# %% REGEX - APPLICATIONS - Valid PAN format
import re

N = int(input())
for i in range(N):
    pan = input()
    try:
        assert len(pan) == 10
        assert re.match('^[A-Z]{5}', pan[0:5])
        assert re.match('^[0-9]{4}', pan[5:9])
        assert re.match('^[A-Z]', pan[9])
        print('YES')
    except Exception:
        print('NO')


# %% ARTIFICIAL INTELLIGENCE The Best Aptitude Test
# https://www.hackerrank.com/challenges/the-best-aptitude-test/problem
# 1 <= T <= 10
# 4 <= N <= 100
# 0.0 <= k <= 10.0, where k is the GPA of every student.
# 0.0 <= s <= 100.0, where s is the score of every student in any of the 5
# aptitude tests.
# Enter your code here. Read input from STDIN. Print output to STDOUT


def get_gpa(n):
    """Docstring."""
    # n GPAs for first year, one per student
    gpas_ = input().split(' ')
    gpas = []
    for i in range(n):
        gpas.append(float(gpas_[i])*10)
    return gpas


def get_scores(n):
    """Docstring."""
    scores_ = []
    scores = []
    for i in range(5):
        scores_.append(input().split(' '))
        scores.append([])
        for j in range(n):
            scores[i].append(float(scores_[i][j]))
    return scores


def get_relscore(gpas, scores):
    """Docstring."""
    correct = 0
    for i in range(1, len(gpas)):
        if scores[i] > scores[i - 1]:
            correct += 1
    return correct


def get_error(gpas, scores):
    """Docstring."""
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

# %% MISCELLANEOUS PYTHON - Flipping bits
#  given decimal # n, the following returns the decimal
# that would flip all the bits of its 32 bit representation
# i.e., binary math of 11111111 11111111 11111111 11111111 - int(format(n, 'b')
# then turned back to decimal


def flippingBits(n):
    """Docstring."""
    return 2**32-1-n


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
    """Docstring."""
    ans = 0
    for i in x:
        ans += (avg - i) ** 2

    return math.sqrt(ans / (n - 1))


def cof(u, v, avg_u, avg_v, std_u, std_v, n):
    """Docstring."""
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
