"""Created on Wed Feb 10 08:02:04 2021.

Python Tricks and Interview Tips
From emails I get from Dan at Real Python <info@realpython.com>

I. Tips for Interviews
How to Stand Out in a Python Coding Interview
March 27, 2019
https://realpython.com/python-coding-interview-tips/
Course: How to Stand Out in a Python Coding Interview (Overview)
https://realpython.com/courses/python-coding-interviews-tips-best-practices/
https://realpython.com/lessons/python-coding-interview-tips-overview/
https://www.studytonight.com/data-structures/queue-data-structure#:~:text=Queue%20can%20be%20implemented%20using,index%20of%20array%20from%200%20).

    A. Use built-in functions done 2/12/21
    x    - debug with breakpoint
    x    - iterate with enumerate instead of range
    x    - list comprehension instead of map / filter
    x    - don't use list comprehension unless need to use memory again
    x    - other PEP / idioms / best practices
    x    - sorted to sort complex lists
    x    - use of slice
    x    - iter, next
    x    - f-strings for string formatting
    x    - zip
    x    - getattr, setattr, etc.
         - bytearray() mutable
         - np.arange()

    B. Use data structures
    x    - use sets for storing unique values
         - review big O() for set operations
    x    - save memory with generators instead of lists
    x    - .get() and setdefault for one-off default values for keys
            in dict
    x    - defaultdict for missing keys
    x    - deque
    x    - typed arrays
         - namedtuple
         - struct.Struct for C types
         - dataclass for writing classes with less code

    C. Use standard library
    x    - Counter
    x    - Permutations and Combinations with itertools
    x    - string constants
         - doctest Module and assert Statements 07:42
         - functools (lru_cache, cached_property(func), cache,
                      partial(), reduce())
         - := expression

    D. Type checking course
        - Dynamic vs. Static typing
        - Duck typing
        - Type hinting
        - Type checking with mypy
        - Pros/cons of type hints

        - annotations
        - type comments
        - practice

    E. Decimal module / rounding

II. Debugging

III. Tricks - unscheduled, via email
    x    - timeit (for small snippets of code)
    x        - use time.time() time stamps for code that takes at least
              0.1 secs to run)
    x        - dictionaries as case/switch statements 3/9/2021

IV. Interviews questions
        - Easy
        - Interview
        - Hard

V. How to contribute
    a. to contribute to CPython, you can:
        - Fix code bugs
        - Write unit tests for functions in the standard library
        - Write documentation for functions in the standard library
    B. Write documentation for the Python Developer’s Guide
        https://devguide.python.org/

        learn RESTRUCTURED TEXT and Sphinx
        https://devguide.python.org/documenting/?highlight=restructuredtext#restructuredtext-primer


@author: Trader
"""

# %% I. Tips for Interviews - A. Use built-in functions - breakpoint()
# insert breakpoint() into code and can interact/inspect variables at
# that point. use 'next' to advance a step in program

# By default, sys.breakpointhook() calls pdb.set_trace() function.
# So at the very least, using breakpoint() provide convenience in using a
# debugger because we don’t have to explicitly import pdb module.
# https://www.journaldev.com/22695/python-breakpoint#:~:text=Python%20breakpoint()%20function%20is,and%20runs%20the%20program%20normally.

# c to continue
# next to continue

# Disable Breakpoint:
# $PYTHONBREAKPOINT=0 python3.7 python_breakpoint_examples.py

# Using Other Debugger (for example web-pdb):
# $PYTHONBREAKPOINT=web_pdb.set_trace python3.7 python_breakpoint_examples.py

# %% I. Tips for Interviews - A. Use built-in functions - enumerate()
numbers = [45, 22, 14, 65, 97, 72]

for i, num in enumerate(numbers):
    print('-' * 4, '\n i=', i, 'num=', num)
    if num % 3 == 0 and num % 5 == 0:
        print(numbers[i], 'being changed to')
        numbers[i] = 'fizzbuzz'
        print(numbers[i])

    elif num % 3 == 0:
        print(numbers[i], 'being changed to')
        numbers[i] = 'fizz'
        print(numbers[i])
    elif num % 5 == 0:
        print(numbers[i], 'being changed to')
        numbers[i] = 'buzz'
        print(numbers[i])
    else:
        print('no changes')

print(numbers)

numbers = [45, 22, 14, 65, 97, 72]
for i, num in enumerate(numbers, start=52):
    print(i, num)

# example of enumerating through a dictionary?
x = {'a': 1, 'd': 3, 'b': 2}
for i, key in enumerate(x):
    print('i=', i, key, ':', x[key])
    # seems like the keys are numbered the way they're entered


grocery = ['bread', 'milk', 'butter']
enumerateGrocery = enumerate(grocery)
print(list(enumerateGrocery))  # convert to list

# %% I. Tips for Interviews - A. Builtins - Use generators to save memory
# Bad:
# needlessly allocates a list of all (gpa, name) entires in memory
valedictorian = max([(student.gpa, student.name) for student in graduates])

# Good:
valedictorian = max((student.gpa, student.name) for student in graduates)
# Use list comprehensions when you really need to create a second list,
# for example if you need to use the result multiple times.

# If your logic is too complicated for a short list comprehension or
# generator expression, consider using a generator function instead of
# returning a list.

# Never use a list comprehension just for its side effects.
# Bad:

[print(x) for x in sequence]

# Good:
for x in sequence:
    print(x)


def make_batches(items, batch_size):
    """
    >>> list(make_batches([1, 2, 3, 4, 5], batch_size=3)).

    [[1, 2, 3], [4, 5]]
    """
    current_batch = []
    for item in items:
        current_batch.append(item)
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
    yield current_batch


# Never remove items from a list while you are iterating through it.
# Bad:
# Filter elements greater than 4
a = [3, 4, 5]
for i in a:
    if i > 4:
        a.remove(i)

# Don’t make multiple passes through the list.
while i in a:
    a.remove(i)

# Good:
# Use a list comprehension or generator expression.
# comprehensions create a new list object
filtered_values = [value for value in sequence if value != x]
# generators don't create another list
filtered_values = (value for value in sequence if value != x)

# Modifying the values in a list
# Remember that assignment never creates a new object. If two or more
# variables refer to the same list, changing one of them changes them all.
# Add three to all list members.
# Bad:
a = [3, 4, 5]
b = a                     # a and b refer to the same list object

for i in range(len(a)):
    a[i] += 3             # b[i] also changes


# Good:
# It’s safer to create a new list object and leave the original alone.

a = [3, 4, 5]
b = a

# assign the variable "a" to a new list without changing "b"
a = [i + 3 for i in a]


# use enumerate() keep a count of your place in the list.
a = [3, 4, 5]
for i, item in enumerate(a):
    print(i, item)
# prints
# 0 3
# 1 4
# 2 5

# Read From a File
# Use the with open syntax to read from files. This will automatically
# close files for you.

# Bad:
f = open('file.txt')
a = f.read()
print(a)
f.close()
#
# Good:
with open('file.txt') as f:
    for line in f:
        print(line)
# The with statement is better because it will ensure you always close the
# file, even if an exception is raised inside the with block.


x = iter('abc')
next(x)
'a'

x = [i for i in range(10)]
x[slice(1, 3)]
# [1, 2]

def fibonacci():
    a, b = 0, 1
    while True:
        yield b
        a, b = b, a + b


fib = fibonacci()
next(fib)
1
next(fib)
1
next(fib)
2
[next(fib) for i in range(10)]
[3, 5, 8, 13, 21, 34, 55, 89, 144, 233]

# %% I. Tips for Interviews - A. Use built-in functions -
# - list comprehension vs. map() and filter()
numbers = [4, 2, 1, 6, 9, 7]


def square(x):
    return x * x


f = lambda x: x * x  # preferable to use def f(): instead of lambda
g = lambda x: bool(x % 2)


def is_odd(x):
    return bool(x % 2)


num_cols = 3
num_rows = 2
grid = [[0 for _ in range(num_cols)] for _ in range(num_rows)]


print('using map:\t\t\t\t\t', list(map(square, numbers)))
print('using map with lambda:\t\t', list(map(lambda x: x * x, numbers)))
print('using list comprehension:\t', [square(x) for x in numbers])
print('using list comp and lambda\t', [f(x) for x in numbers])
print('using list comp and lambda\t', [(lambda x: x * x)(x) for x in numbers])
print()
print('using filter:\t\t\t\t ', list(filter(is_odd, numbers)))
print('using list comprehension:\t ', [x for x in numbers if is_odd(x)])
print('using list comp and lambda\t ', [x for x in numbers if g(x)])
print('using list comp and lambda2:', [x for x in numbers if
                                       (lambda x: bool(x % 2))(x)])

# %% I. Tips for Interviews - A. Use built-in functions -
# sorted to sort complex lists
sorted([6, 5, 3, 7, 2, 4, 1])
# [1, 2, 3, 4, 5, 6, 7]
animals = ['cat', 'dog', 'cheetah', 'rhino', 'bear']


sorted(animals, reverse=True)  # sorted doesn't mutate the list
# ['rhino', 'dog', 'cheetah', 'cat', 'bear]

animals = [{'type': 'penguin', 'name': 'Stephanie', 'age': 8},
           {'type': 'elephant', 'name': 'Devon', 'age': 3},
           {'type': 'puma', 'name': 'Moe', 'age': 5}]
out = sorted(animals, key=lambda animal: animal['age'])
# returns in ascending order of age

xs = {'a': 4, 'b': 3, 'c': 2, 'd': 1}
sorted(xs.items(), key=lambda x: x[1])
# [('d', 1), ('c', 2), ('b', 3), ('a', 4)]

import operator
y = sorted(xs.items(), key=operator.itemgetter(1))  # y is a list

print(xs)
print(y)
# [('d', 1), ('c', 2), ('b', 3), ('a', 4)]

animals.sort(key=lambda animal: animal['age'])

# %% I. Tips for Interviews - A. Use built-in functions - attrs


class Person:
    age = 23
    name = 'Adam'
    height = None


person = Person()

print('Person has age?:', hasattr(person, 'age'))
print('Person has salary?:', hasattr(person, 'salary'))
print('Person age is:', getattr(person, 'age'))
print('Person name is:', getattr(person, 'name'))
print('Person height is:', getattr(person, 'height'))  # raises AttributeError
setattr(person, 'height', 160)  # doesn't work if not defined in class
print('Person height is:', getattr(person, 'height'))  # raises AttributeError
delattr(person, 'height')
print('Person height is:', getattr(person, 'height'))  # raises AttributeError
person.age
person.name
person.height


# %% I. Tips for Interviews - A. Use built-in functions - fstrings x format
"""
study the formatting codes

https://realpython.com/python-string-formatting/

Python String Formatting Rule of Thumb:
- If your format strings are user-supplied, use Template Strings (#4)
to avoid security issues.
- Otherwise, use Literal String Interpolation/f-Strings #3 if you’re on
Python 3.6+, and “New Style” str.format #2 if you’re not.


1. “Old Style” String Formatting (% Operator)
- still works, but others are preferable because this gets messy after
    a few variables

https://docs.python.org/3/library/stdtypes.html#old-string-formatting
'Hey %(name)s, it is a 0x%(errno)x error!' % {"name": name, "errno": errno }

2. “New Style” String Formatting (str.format)
Custom String Formatting
- better than #1 using %, but still gets a little verbose with more
    variables

https://docs.python.org/3/library/string.html#string-formatting
https://docs.python.org/3/library/string.html#format-specification-mini-language
- Format specifications are used within replacement fields contained
    within a format string to define how individual values are presented
    (see Format String Syntax and Formatted string literals).
- They can also be passed directly to the built-in format() function.

3. String Interpolation / f-Strings (Python 3.6+)
https://realpython.com/python-f-strings/
https://docs.python.org/3/reference/lexical_analysis.html#f-strings
Because f-strings are evaluated at runtime, you can put any and
all valid Python expressions in them

4. Template Strings (Standard Library)
- a simpler and less powerful mechanism, but in some cases is best
- best when handling formatted strings generated by users of program
    because if a malicious user can supply a format string, they can
    potentially leak secret keys and other sensitive information
    (it’s possible for format strings to access arbitrary variables
     in your program.)
- fast

templ_string = 'Hey $name, there is a $error error!'
Template(templ_string).substitute(name=name, error=hex(errno))

user_input = '${error.__init__.__globals__[SECRET]}'
>>> Template(user_input).substitute(error=err)
"""


s = input('Enter input here-->')
# NOTE: id() gives unique id to an object. value is location in memory
# using '%f %d %s' % () notation
print('s: %s \t id(s): %d \t hash(s): %.2f' % (s, id(s), hash(s)))


# using '{}'.format()
age = 30
name = 'phil'
"Hello, {1}. You are {0}.".format(age, name)
person = {'name': 'Eric', 'age': 74}
"Hello, {name}. You are {age}.".format(name=person['name'], age=person['age'])
"Hello, {name}. You are {age}.".format(**person)

print('s: {} \t id(s): {} \t hash(s):'.format(s, id(s)), hash(s))

# fstrings
a = 2
b = 3
errno = 10
f'Five plus ten is {a + b} and not {2 * (a + b)}.'
f'Hello, {name}!'
f"Hey {name}, there's a {errno: #x} error!"
f"{name.lower()} is funny."
# Multi line f string
name = "Eric"
profession = "comedian"
affiliation = "Monty Python"
message = (f"Hi {name}. "
           f"You are a {profession}. "
           f"You were in {affiliation}.")

# this avoids newline characters; cannot break up fstring with comment
message = f"Hi {name}. " \
          f"You are a {profession}. " \
          f"You were in {affiliation}."

# f-string expression part cannot include '#' or backslash
# must use different quotes at beginning ofstring than in keys of a dict
# f"The comedian is {comedian['name']}, aged {comedian['age']}."

# needs double braces to show in expression
f'{{70 + 4}}'
# '{70 + 4}'
f'{70 + 4}'
# '74'
# can use even with class definitions
# new_comedian = Comedian("Eric", "Idle", "74")
# f"{new_comedian}"  # include __repr__() in class definition

f"The \"comedian\" is {name}, aged {age}."
# or can use different tupe of quotes on inside of
f"This is valid {'usage'} of different quote marks, no escape needed."


def get_name_and_decades(name, age):
    return f"My name is {name} and I'm {age / 10:.5f} decades old."


print(get_name_and_decades("Maria", 31))
# My name is Maria and I'm 3.10000 decades old.

# integers
print(format(-10, '+'))  # with sign
print(format(15, 'b'))  # binary
print(format(15, 'x'))  # hex lower
print(format(15, 'X'))  # hex upper

# float
print(format(.2, '%'))           # 20.000%
print(format(10.5, 'e'))         # 1-10 e power of 10
print(format(10.5345678, 'f'))
print(format(10.5345678, 'F'))   # no difference
print(format(10.5, 'F'))

x = 'abcda'
x.count('a')  # 2
y = list(x)
y.count('a')  # 2
x.replace('a', 'z')      # doesn't change x unless reassigned as below
x = x.replace('a', 'z')  # doesn't change x


# can use unpacking on dictionary in fstring
person = {'name': 'Eric', 'age': 74}
"Hello, {name}. You are {age}.".format(**person)


# Format Specification Mini-Language
# https://docs.python.org/3/library/string.html#formatspec
width = 5

for num in range(0, 15):
    for base in 'dXob':
        print('{0:{width}{base}}'.format(num, base=base, width=width), end=' ')
    print()

x = '{2}, {1}, {0}'.format('a', 'b', 'c')
y = '{2}, {1}, {0}'.format(*'abc')      # unpacking argument sequence
z = '{0}{1}{0}'.format('abra', 'cad')   # args' indices can be repeated
print(x, '\t\t', y, '\t\t', z)
print('{:*<30}'.format('left aligned'))
print('{:*>30}'.format('right aligned'))
print('{:*^30}'.format('centered'))

print("int: {0:d};  hex: {0:x};  oct: {0:o};  bin: {0:b}".format(42))
print("int: {0:d};  hex: {0:#x};  oct: {0:#o};  bin: {0:#b}".format(42))
print('{:,}'.format(1234567890))
print('Correct answers: {:.2%}'.format(15/100))

# %% I. Tips for Interviews - A. Use built-ins - bytes() and bytearray()


# %% I. Tips for Interviews - A. Use built-in functions - Miscellaneous

divmod(5, 2) == (5 // 2, 5 % 2)
chr(ord('a')) == 'a'
ord(chr(97)) == 97
eval('3+2')
print(eval('3+2'))
z = {}
isinstance(z, dict)  # True
a = [1, 2, 3, 4, 5, 6]
sl = slice(1, 5, 1)
print(a[sl], 'slice of a using slice object')
print(a[1:5], 'slice of a')
pow(5, 2)  # 25 = 5**2

x = iter('abc')
# the follwing iterates through x, leaving nothing left in next(x)...
y = reversed(list(x))
# ...which is why we need to reset x to a new iterator
x = iter('abc')
print('going through x \n', next(x))  # a
print(next(x))  # b
print(next(x))  # c
# print(next(x))  # StopIteration
print('now through y\n', next(y))  # c
print(next(y))  # b
print(next(y))  # a
# print(next(y))  # StopIteration

# print(globals())
# locals()
x = [1, 2]
print('repr(x)=', repr(x))
z = {'a': 1}
print('repr(z)=', repr(z))

x = [1, 2, -5, 3]
max(x, key=lambda x: x * x)


# finds which elements of x are odd
[(lambda x: x % 2 == 1)(num) for num in x]


# %% How to sort a Python dict by value
# Feb 10, 2021
# (== get a representation sorted by value)

xs = {'a': 4, 'b': 3, 'c': 2, 'd': 1}
sorted(xs.items(), key=lambda x: x[1])
# [('d', 1), ('c', 2), ('b', 3), ('a', 4)]

# Or:

import operator
y = sorted(xs.items(), key=operator.itemgetter(1))  # y is a list

print(xs)
print(y)
# [('d', 1), ('c', 2), ('b', 3), ('a', 4)]

# %% A. Use built-in functions done 2/12/21 - zip
# zip() function returns a zip object, which is an iterator of tuples
# where the first item in each passed iterator is paired together, and
# then the second item in each passed iterator are paired together etc.

# If the passed iterators have different lengths, the iterator with the
# least items decides the length of the new iterator.

x = [1, 2, 3]
y = [4, 5, 6]
print([zip(x, y)])
print(*zip(x, y))
print([*zip(x, y)])

for tup in zip(x, y):
    print(tup)

for x, y in zip(x, y):
    print(x, y)


# %% I. Tips for Interviews - A. numpy.arange()
# np.arange is a function, range is an object

import numpy as np

np.arange(start=1, stop=10, step=1)
np.arange(1, 10)
np.arange(10)  # default start is 0
np.arange(1.0, 10, step=1)
np.arange(1.0, 10.1, step=1)
np.arange(1.0, 10.1, step=11)
np.arange(1, -10, step=-1)

# different data types
x = np.arange(10)
x.itemsize  # Bytes
x.dtype
np.arange(23.0, 51, .98).dtype

type(x)
2**x
x**2
x / 3
np.sin(x)
x.shape
x.reshape(2, 5)
x

type(np.arange(10))  # returns an array

type(range(10))
# a Class range; needs parameters with integers,
# no float as parameters
list(range(10))
# np.arange works faster than range, though
# and better than range for manipulation of array; also allows floating
# data

# range saves memeory because it is a generator, vs. an object

np.linspace(1, 100, num=33)  # includes endpoint unless False
np.linspace(1, 100, num=33, endpoint=False)  # includes endpoint unless False
np.logspace(1, 100, num=33, endpoint=False)
np.geomspace(1, 100, num=33, endpoint=False)
x = [1, 2, 3, 4]
y = [1, 2, 3, 4]
z = np.meshgrid(x, y)
z


# %% B Use data structures - use sets
# The difference in lookup time means that the time complexity for adding
# to a set grows at a rate of O(N), which is much better than the O(N²)
# from the second approach in most cases.


def count_unique(s):
    n_unique = len(set(s))
    print(n_unique)
    return n_unique


count_unique('abcdabcd')

set_comprehension = {x * x for x in range(10)}

squares = {x: x * x for x in range(6)}
# %% B Use data structures - Save memory with Generators
# don't use []s, which creates a list comprehension
from itertools import accumulate
import sys

# this is the memory intensive approach
z = [i * i for i in range(1, 1001)]
print(sum(z))  # 333833500
print('size of list comp', sys.getsizeof(z))  # 9016

cum = 0
for i in range(1, 1001):
    cum = cum + i * i
print(cum)


x = (i * i for i in range(1, 1001))
print('size of generator', sys.getsizeof(x))  # 112
for i, val in enumerate(accumulate(x)):
    pass
    # print(f'{i+1:>2} {(i+1)**2:>4} {val: >5}')
print('size of generator', sys.getsizeof(x))  # 112

# =============================================================================
# Generator Functions
# =============================================================================


def f():
    yield 1
    yield 2
    yield 3


g = f()

print('Generator function call', next(g))
print('Generator function call', next(g))
print('Generator function call', next(g))
g2 = f()
print(list(g2))

# %% B. Use data structures - dict comprehension

x = [('dog', 'Cacau'), ('cat', 'jojo')]
pets = {species: name for species, name in x}

# %% B. Use data structures - chainMap, MappingProxy
# map allows search across all keys in the dictionaries
# responds with answer from frist key found, in case there are multiple
# keys

# insertion / deletions better done to original dictionaries; chian better for
# just reading from the multiple dict's, because insertion is added to
# first dict in chain
from collections import ChainMap
from types import MappingProxyType


adict = {'a': 1, 'b': 2}
bdict = dict(c=3, a=4)
chain = ChainMap(adict, bdict)

chain['a']
chain['b']


# MappingProxy is just read_only wrapper of dictionary
read_only = MappingProxyType(adict)
# cannot add, or remove from read_only; need to go to original
# adict to make changes

# When to use what?
# start with dict built-in for newer versions, as it is already ordered
# others, only use for specific cases;
# defaultdict is a good choice when want a list as default value
# MappingProxyType very rarely used


# %% B. Use data structures - collections.OrderedDict,
# =============================================================================
# normal dict is now (post Python 3.8) very similar to OrderedDict
# but OrderedDict is good to explicitly show that order is important
# a few methods only apply to OrderedDict
# reversed()
# .move_to_end() moves a key to last position
# .popitem(last=True) can pop first or last item
# two dicts are same if keys/values are same, even if order is not
# two dict of OrderedDict are only same if keys/values are same
# AND in same order
# =============================================================================
from collections import OrderedDict

c = OrderedDict(one=1, two=2, three=3)
d = OrderedDict(one=1, two=2, three=3)
e = OrderedDict(one=1, two=2)
f = OrderedDict(two=2, one=1)
g = dict(one=1, two=2)
h = dict(two=2, one=1)
print(c == d)  # True
print(e == f)  # False
print(g == h)  # True

list(reversed(d))  # reverses the keys, by default
list(reversed(d.keys()))  # reverses the keys
list(reversed(d.values()))  # reverses the values
list(reversed(d.items()))  # reverses the dictionary
d.popitem(last=False)  # first item popped
d.popitem(last=True)  # last item popped

# %% B. Use data structures - .get() and setdefault
# .get() and .setdefault() work well when you’re setting default for
# single key default dict even better
cowboy = {'age': 32, 'horse': 'mustang', 'hat_size': 'large'}

name = cowboy.get('name', 'The Man with No Name')
# this doesn't set name field in cowboy, though...nothing has changed
# so if we want to maybe access name again in the future, we need :
name = cowboy.setdefault('name', 'The Man with No Name')

from collections import defaultdict


student_grades = defaultdict(lambda: [])
# can pass a second argument as initial dict; if none, initializes as {}

# student_grades = defaultdict(list)  # equivalent to prior line

# student_grades = defaultdict(lambda: 0)
# this wouldn't work because of append operation in loop below, but is
# way to make the default a constant

grades = [('elliot', 91),
          ('neelam', 98),
          ('bianca', 81),
          ('elliot', 88)
          ]

for name, grade in grades:
    student_grades[name].append(grade)

# %% B. Use data structures - Deque and named tuple
from collections import namedtuple
from collections import deque


class TicketQueue(object):
    def __init__(self):
        self.deque = deque()

    def add_person(self, name):
        self.deque.append(name)
        print(f'{name} has been added to the queue')

    def service_person(self):
        name = self.deque.popleft()
        print(f'{name} has been serviced')

    def bypass_queue(self, name):
        self.deque.appendleft(name)
        print(f'{name} has bypassed the queue')


class Car:
    def __init__(self, color, make, model, mileage):
        self.color = color


Car = namedtuple('Car', 'color make model mileage')
Car = namedtuple('Car', ['color', 'make', 'model', 'mileage'])
mycar = Car('black', 'BMW', 'M3', 5)
# =============================================================================
# see documentation for methods on namedtuple and deque()
# =============================================================================


# %% B. Use data structures - Typed Arrays
"""fs
Code    C Type              Python Type     Min Size (Bytes)
‘b’     signed char         int             1
‘B’     unsigned char       int             1
‘u’     wchar_t Unicode     character       2*
‘h’     signed short        int             2
‘H’     unsigned short      int             2
‘i’     signed int          int             2
‘I’     unsigned int        int             2
‘l’     signed long         int             4
‘L’     unsigned long       int             4
‘q’     signed long long    int             8
‘Q’     unsigned long long  int             8
‘f’     float               float           4
‘d’     double              float           8
"""
from array import array

# unsigned int
a = array('I', [2**31])     # no error
a = array('I', [2**33])     # OverflowError:

# unsigned long
a = array('I', [2**31])     # no error
a = array('L', [2**32])     # OverflowError:

# unsigned long long
a = array('Q', [2**32])     # no error
a = array('Q', [2**63])     # no error
a = array('Q', [2**64])     # OverflowError:

# %% C. Use standard library - Counter

from collections import Counter

x = Counter('aabcddd')
y = Counter('bccd')
# Counter({'a': 2, 'b': 1, 'c': 1, 'd': 3})
print(sorted(x.elements()))
print(x.most_common(2))
x.update('abc')   # adds to counts of x
x.subtract('abc')  # removes count of string from x

sum(list(x.values()))  # total of all counts
# x.clear()
list(x)   # list of unique elements
+x  # removezero and neg counts

print('x - y', x - y)  # just positive counts
print('x + y', x + y)


# %% C. Use standard library -  Permutatinos and Combinations

import itertools as it
friends = ['Monique', 'Ashish', 'Devon', 'Bernie']

x = list(it.permutations(friends, r=2))
y = list(it.combinations(friends, r=2))
print(f'There are {len(x)} permutations: \n{x}\n')
print(f'There are {len(y)} combinations: \n{y}')

all_ones = it.repeat(1)
five_ones = it.repeat(1, times=5)
print(list(five_ones))
print(list(map(pow, range(10), it.repeat(2))))
print(next(all_ones))
alter_ones = it.cycle([1, -1])  # infinite series generator of 1, -1, 1, ...

# %% C. Use standard library -  functools module
"""
functools  - Higher-order functions and operations on callable objects

@functools.cache(user_function)
@functools.lru_cache(user_function)
@functools.lru_cache(maxsize=128, typed=False)
@functools.cached_property(func)

functools.partial(func, /, *args, **keywords)
functools.partialmethod(func, /, *args, **keywords)
functools.reduce(function, iterable[, initializer])

there are more at
https://docs.python.org/3/library/functools.html
"""

from functools import reduce
from functools import cached_property
from functools import lru_cache

reduce(lambda x, y: x + y, [1, 2, 3, 4])
reduce(lambda x, y: x * y, [1, 2, 3, 4], 10)


class Data:
    def __init__(self, n):
        self.n = n

    @cached_property
    def f(self):
        total = 0
        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    total += i + j + k
        return total


@lru_cache
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)


print(fib(4))


@lru_cache
def factorial(n):
    return n * factorial(n-1) if n else 1


###############################################################################
# %% Appendix B: Rough code translations for comprehensions
###############################################################################

def f():
    a = [EXPR for VAR in ITERABLE]

# Translation (let's not worry about name conflicts):

def f():
    def genexpr(iterator):
        for VAR in iterator:
            yield EXPR
    a = list(genexpr(iter(ITERABLE)))

# Let's add a simple assignment expression.

###############################################################################
# Original code:
def f():
    a = [TARGET := EXPR for VAR in ITERABLE]

# Translation:

def f():
    if False:
        TARGET = None  # Dead code to ensure TARGET is a local variable
    def genexpr(iterator):
        nonlocal TARGET
        for VAR in iterator:
            TARGET = EXPR
            yield TARGET
    a = list(genexpr(iter(ITERABLE)))

###############################################################################
# Let's add a global TARGET declaration in f().
# Original code:

def f():
    global TARGET
    a = [TARGET := EXPR for VAR in ITERABLE]
# Translation:

def f():
    global TARGET
    def genexpr(iterator):
        global TARGET
        for VAR in iterator:
            TARGET = EXPR
            yield TARGET
    a = list(genexpr(iter(ITERABLE)))


###############################################################################
# Or instead let's add a nonlocal TARGET declaration in f().
# Original code:

def g():
    TARGET = ...
    def f():
        nonlocal TARGET
        a = [TARGET := EXPR for VAR in ITERABLE]

# Translation:

def g():
    TARGET = ...
    def f():
        nonlocal TARGET
        def genexpr(iterator):
            nonlocal TARGET
            for VAR in iterator:
                TARGET = EXPR
                yield TARGET
        a = list(genexpr(iter(ITERABLE)))

###############################################################################
# Finally, let's nest two comprehensions.
# Original code:


def f():
    a = [[TARGET := i for i in range(3)] for j in range(2)]
    # I.e., a = [[0, 1, 2], [0, 1, 2]]
    print(TARGET)  # prints 2

# Translation:


def f():
    if False:
        TARGET = None

    def outer_genexpr(outer_iterator):
        nonlocal TARGET

        def inner_generator(inner_iterator):
            nonlocal TARGET
            for i in inner_iterator:
                TARGET = i
                yield i
        for j in outer_iterator:
            yield list(inner_generator(range(3)))
    a = list(outer_genexpr(range(2)))
    print(TARGET)


###############################################################################
# %%

# print(factorial(10))  # no previously cached result, makes
# 11 recursive calls 3628800
factorial(5)  # just looks up cached value result
# 120
# factorial(12)  # makes two new recursive calls, the other 10 are
# cached
# 479001600
print(factorial.cache_info())

# %% C doctest module


class A:
    def f(self):
        """Print message.

        Returns
        -------
        None.
        >>> a = A()
        >>> a.f()
        Hello world
        'Hello world'
        """
        print('Hello world')
        return 'Hello World'

    @property
    def error(self):
        """Make error.

        This function just errors.

        >>> A().error
        Traceback (mostrecent call last):
        ...
        Exception: I am an error
        """
        raise Exception('I am an error')


# %%
a = A()
a.f()
A().f()
A().error()
# a.error()
# %% C String constants
import string

print(string.ascii_letters)
print(string.ascii_lowercase)  # 'abcdefghijklmnopqrstuvwxyz'
print(string.ascii_uppercase)  # 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
print(string.digits)           # '0123456789'
print(string.punctuation)      # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~.
a = string.whitespace          # space, tab, linefeed, return, formfeed,verttab
b = string.printable           # digits, ascii_letters, punct, and wspace.

print(string.hexdigits)        # '0123456789abcdefABCDEF'
print(string.octdigits)        # '01234567'
a                              # ' \t\n\r\x0b\x0c'
b  # '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()'
# cont.  '*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c'
print(f'There are {len(b)} printables in {b[:-1]}'
      f'\nthe last character is clear screen, which is omitted')
print(f'There are {len(a)} whitespace characters')

# %% E. Type checking
# mypy package does the actual checking


def greet(name: str) -> str:
    return 'Hello' + name


def headline(text, align=True):
    if align:
        return f"{text.title()}\n{'-' * len(text)}"
    else:
        return f"{text.title()} ".center(50, 'o')


print(headline(' ok', align=False))


def headline2(text: str, align: bool = True) -> str:
    # not enforced on its own
    if align:
        return f"{text.title()}\n{'-' * len(text)}"
    else:
        return f"{text.title()} ".center(50, 'o')


print(headline2('use pycharm', align='center'))
# can run 'mypy typechecking.py' from terminal to see if there
# are errors; no automatic type checking in Spyder

# %%  Annotations - mypy will infer var types
# type checking for python 3 and beyond


import math


def circum(radius: float) -> float:
    return 2 * math.pi * radius


circum.__annotations__
circum(2)


def circum2(radius: float) -> float:
    return 2 * pi * radius


# variable annotations
pi: float = 3.142

__annotations__
# in console, will be stored in module level annotations dictionary
# >> {'pi': float}


def func(arg):
    # type: str -> str
    my_var = 42  # type: int (add comment on same line)
    # comments do not appear in __annotations__

# %%  Annotations - example code
# game.py

import random
from typing import Dict, List, Tuple

Card = Tuple[str, str]
Deck = List[Card]
SUITS = "♠ ♡ ♢ ♣".split()
RANKS = "2 3 4 5 6 7 8 9 10 J Q K A".split()


def create_deck(shuffle: bool = False) -> Deck:
    """Create a new deck of 52 cards."""
    deck = [(s, r) for r in RANKS for s in SUITS]
    if shuffle:
        random.shuffle(deck)
    return deck


def deal_hands(deck: Deck) -> Tuple[Deck, Deck, Deck, Deck]:
    """Deal the cards in the deck into four hands."""
    return (deck[0::4], deck[1::4], deck[2::4], deck[3::4])


def play():
    """Play a 4-player card game."""
    deck = create_deck(shuffle=True)
    names = "P1 P2 P3 P4".split()
    hands = {n: h for n, h in zip(names, deal_hands(deck))}

    for name, cards in hands.items():
        card_str = " ".join(f"{s}{r}" for (s, r) in cards)
        print(f"{name}: {card_str}")


if __name__ == "__main__":
    play()

# %% F. Decimal module / rounding - Using Decimal module

"""
to change default rounding strategy.
decimal.getcontext().rounding = <flag>
Flag                       Rounding Strategy
-----                      --------------------
decimal.ROUND_CEILING      Rounding up (not symmetric around 0)
decimal.ROUND_FLOOR	 	   Rounding down (not symmetric around 0)
decimal.ROUND_DOWN	   	   Truncation (symmetric)
decimal.ROUND_UP	       Rounding away from zero (symmetric)
decimal.ROUND_HALF_UP	   Rounding half away from zero
decimal.ROUND_HALF_DOWN    Rounding half towards zero
decimal.ROUND_HALF_EVEN	   Rounding half to even
decimal.ROUND_05UP	       Rounding up and rounding towards zero

Best practices
 - to store at least two or three more decimal places of precision than
 needed
- when you compute the daily average temperature, you should calculate
it to the full precision available and round the final answer
- when in doubt, use round ties to even (default in python, numpy, and
                                         pandas)
"""
import decimal
from decimal import Decimal

decimal.getcontext().rounding = 'ROUND_FLOOR'

# this means at most 3 sig figs;
decimal.getcontext().prec = 3

# so we get error if we try to turn 200.52 to same # decimal pts at 1.0
# print(Decimal('201.52').quantize(Decimal('1.0')))
# but 0 decimals will correctly print 200
print(Decimal('200.52').quantize(Decimal('1')))

decimal.getcontext().prec = 4
# now this correctly prints 201.5, because now we can have 4 sig figs
print(Decimal('201.52').quantize(Decimal('1.0')))
# but this yields error
# print(Decimal('201.52').quantize(Decimal('1.00')))

# and this prints 21.52
print(Decimal('21.526').quantize(Decimal('1.00')))

# and this -21.97
print(Decimal('-2.961').quantize(Decimal('1.00')))

# =============================================================================
# # data = [0.15, -1.45, 3.65, -7.05, 2.45]
# If you round every number in data to one decimal place using the
# “round half up” strategy, which of the following rounding biases is
# introduced?
# Round towards positive infinity bias
# =============================================================================

# %% F. Decimal module / rounding
# ROUNDING - QUIZ
# Write the code for a Python function round_half_towards_zero that
#  takes a number n and a keyword argument decimals that defaults to 0,
#  and returns the value of n rounded to decimals decimal places, where
#  ties are rounded towards zero.

# You can assume that the math module has been imported and a function
# called round_half_down() exists that takes two arguments—a number n
# and a keyword argument decimals—and returns the number n rounded to
# decimals decimal places, with ties rounded down.


def round_half_towards_zero(n, decimals=0):
    return math.copysign(1, n) * round_half_down(abs(n), decimals)


def round_half_towards_zero2(n, decimals=0):
    rounded_abs = round_half_down(abs(n), decimals)
    return math.copysign(rounded_abs, n)


def round_half_towards_zero3(n, decimals=0):
    # Also acceptable:
    sign = 1 if n >= 0 else -1
    rounded_abs = round_half_down(abs(n), decimals)
    return sign * rounded_abs


def round_half_towards_zero4(n, decimals=0):
    # Without `round_half_down()`:
    sign = 1 if n >= 0 else -1
    multiplier = 10 ** decimals
    rounded_abs = math.ceil(abs(n) * multiplier - 0.5) / multiplier
    return sign * rounded_abs

# %% F. Decimal module / rounding
# Rounding for numpy arrays,  for pandas and dataframes

import numpy as np
import pandas as pd

np.random.seed(444)
data = np.random.randn(3, 4)
np.around(data, decimals=3)
np.ceil(data)
np.floor(data)
np.trunc(data)
np.rint(data)  # rounding half to even” strategy


def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    # Replace math.floor with np.floor
    return np.floor(n * multiplier + 0.5) / multiplier


series = pd.Series(np.random.randn(4))
series.round(2)
df = pd.DataFrame(np.random.randn(3, 3), columns=["A", "B", "C"])
df.round(3)
df.round({"A": 1, "B": 2, "C": 3})
decimals = pd.Series([1, 2, 3], index=["A", "B", "C"])
df.round(decimals)
# np function works on df
np.floor(df)

# %% Unittest
# https://docs.python.org/3/library/unittest.html

import unittest


class TestSum(unittest.TestCase):

    def test_sum(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

    def test_sum_tuple(self):
        self.assertEqual(sum((1, 2, 2)), 6, "Should be 6")


if __name__ == '__main__':
    unittest.main()

# %% II. Debugging
# to enter debugger automatically, for programs you don't have
# write access to

# By default, sys.breakpointhook() calls pdb.set_trace() function.
# So at the very least, using breakpoint() provide convenience in using
# a debugger because we don’t have to explicitly import pdb module.
# https://www.journaldev.com/22695/python-breakpoint#:~:text=Python%20breakpoint()%20function%20is,and%20runs%20the%20program%20normally.

# Disable Breakpoint:
# $PYTHONBREAKPOINT=0 python3.7 python_breakpoint_examples.py

# Using Other Debugger (for example web-pdb):
# $PYTHONBREAKPOINT=web_pdb.set_trace python3.7 python_breakpoint_examples.py

# pdbpp even better
# sticky shows lines as you go through


"""
List of commands
----------------
@pbp.hideframe decorator would hide function it decorates from
stacktrace

<enter> repeats last command entered
enable1 or disable1 to disable 1st

a     will show us the arguments passed into a function

b:    With no arguments, list all breaks.
      With a line number argument, set
      breakpoint at this line in the current file.
      # b command, the module name and line number, to set a breakpoint:
# $ b util:5

# sets conditional breakpoints with arguments to function called
b util.get_path, not filename.startswith('/')

# When you’re setting the breakpoint with a function name rather than a
#  line number, note that the expression should use only function
#  arguments or global variables that are available at the time the
#  function is entered. Otherwise, the breakpoint will stop execution
#  in the function regardless of the expression’s value.

c:    Continue execution and only stop when a breakpoint is encountered.
d:    Move the current frame count (default one) levels down in the stack
      trace (to a newer frame).
display char
This will display the value of the char variable each time it changes.
If char doesn’t change on the next execution, then it will not be
 displayed

 The display command creates a watchlist. We can add more variables by
 running display again with a new variable name or expression. Running
 display without additional arguments will show us our entire
 watchlist. undisplay will clear the watchlist. If we pass in a
 name, then it will remove that variable from the watch list.


e          list of breakpoints
h:         See a list of available commands.
h <topic>: Show help for a command or topic.
h pdb:     Show the full pdb documentation.

l:    List source code for the current file. Without arguments, list 11
      lines around the current line or continue the previous listing
l.    goes back to beginning
ll:   List the whole source code for the current function or frame.
n:    Continue execution until the next line in the current function
         is reached or it returns.
p:    Print the value of an expression.
pp:   Pretty-print the value of an expression.
q:    Quit the debugger and exit.

s:    Execute the current line and stop at the first possible opportuty
      (either in a function that is called or in the current function).
unt:  Continue execution until the line with a number greater than the
      current one is reached. With a line number argument, continue
      execution
      until a line with a number greater or equal to that is reached.
The only difference is that this is not logically executed, and so the
 unt command will iterate through entire loops automatically, instead
 of just moving forward one iteration. If you do supply a line number,
 unt will act like the c command, except you’re telling it where to
 stop.


u:    Move the current frame count (default one) levels up in the stack
      trace (to an older frame).

w:    Print a stack trace, with the most recent frame at the bottom. An
      arrow indicates the current frame, which determines the context of
      most commands.
"""

# %% Bitwise operators
"""
https://blog.tarkalabs.com/real-world-uses-of-bitwise-operators-c41429df507f
&
4&2 = 0
3&1 = 1

|

^ xor
~ bitwise not makes numbers negative because they're stored as 2's complement
~2 = -3, or ~b00000010 => 11111101 = -3
~3 = -4     ~b00000011 => 11111100 = -4

<< left shift
---------------
5<<2 prints 20 5 x 2**2 = 5*4 = 20
5<<1 = 5*2 = 10
 <<2 means *2**2 = *4
 <<3 menas *2**3 = *8

>> right shift (least sig figs are dropped)
------------
4>>4 = 0
 >> means divide by 2**4
16>>4 = 1 because 16/16 = 1
16>>1 = 16/2 = 8
16>>2 = 16/4 = 4

x = 22
x>>=2  # divide by 2, set x to the new value


# caluclate 2**5
1 << 5

PRACTICAL USES
--------------
2. Checking for Odd and Even Numbers

CHECK FOR ODD #
(x&1) = 1 if odd

CHECK FOR EVEN #
not(x&1) = True if x even, as (x&1 would be 0)


3. Swapping Variables Without Using a Third Variable
a=0
b = 1
print('a', a, ', b', b)
a ^= b
b ^= a
a ^= b
print('a', a, ', b', b)
# now b = 0, a= 1

# switches a and b after third operation


4. Converting text casing (Lowercase & Uppercase) (might only be C)
text1 = 'UPPERCASE' # not sure how to apply simply to a whoel string,
text2 = 'lowercase'

print(chr(ord('a') & ord('_')))
print(chr(ord('A') | ord(' ')))

''.join(list(map(chr, list(map(ord, 'text')))))
‘_’ char as the right operand to convert each character in the string
to uppercase


to a lowercase character, using the bitwise OR operator with the
space ASCII character as the right operand.

5. check if number is a power of 2

"""
# %% III. Tricks - Argument unpacking
# Function argument unpacking


def myfunc(x, y, z):
    print(x, y, z)


tuple_vec = (1, 0, 1)
dict_vec = {'x': 1, 'y': 0, 'z': 1}

myfunc(*tuple_vec)
# 1, 0, 1

myfunc(**dict_vec)
# 1, 0, 1

# %% III. Tricks - collections.Counter to find most common elements
import collections
c = collections.Counter('helloworld')

c
# Counter({'l': 3, 'o': 2, 'e': 1, 'd': 1, 'h': 1, 'r': 1, 'w': 1})

c.most_common(3)
# [('l', 3), ('o', 2), ('e', 1)]

# %% III. Tricks -  Functions are first-class citizens in Python:
"""
1) can be passed as arguments to other functions,
2) returned as values from other functions, and
3) assigned to variables and stored in data structures.

"""


def myfunc(a, b):
    return a + b


funcs = [myfunc]
funcs[0]
# <function myfunc at 0x107012230>
funcs[0](2, 3)
# 5

# %% III. Tricks -  "is" vs "=="

a = [1, 2, 3]
b = a

a is b
# True
a == b
# True

c = list(a)

a == c
# True
a is c
# False

# • "is" expressions evaluate to True if two
#   variables point to the same object

# • "==" evaluates to True if the objects
#   referred to by the variables are equal

# %% III. Tricks -  Why Python is Great: Namedtuples
# Using namedtuple is way shorter than
# defining a class manually:
from collections import namedtuple
Car = namedtuple('Car', 'color mileage')

# Our new "Car" class works as expected:
my_car = Car('red', 3812.4)
my_car.color
# 'red'
my_car.mileage
# 3812.4

# We get a nice string repr for free:
my_car
# Car(color='red' , mileage=3812.4)

# Like tuples, namedtuples are immutable:
my_car.color = 'blue'
# AttributeError: "can't set attribute"


# %% III. Tricks -  The get() method on dicts
# and its "default" argument

name_for_userid = {
    382: "Alice",
    590: "Bob",
    951: "Dilbert"
}


def greeting(userid):
    return "Hi %s!" % name_for_userid.get(userid, "there")


greeting(382)
# "Hi Alice!"

greeting(333333)
# "Hi there!"

# %% III. Tricks -  Different ways to test multiple
# flags at once in Python
x, y, z = 0, 1, 0

if x == 1 or y == 1 or z == 1:
    print('passed')

if 1 in (x, y, z):
    print('passed')

# These only test for truthiness:
if x or y or z:
    print('passed')

if any((x, y, z)):
    print('passed')
# %% III. Tricks -  timeit
import timeit

timeit.timeit('sum([n for n in range(1000)])', number=10_000)
# it doesn't seem to work with user defined functions like fib()

# %% III. Tricks - Permutations
import itertools


for p in itertools.permutations('ABCD'):
    print(p)

# %% III. Tricks -  Dicts can be used to emulate switch/case statements

# Because Python has first-class functions they can
# be used to emulate switch/case statements

def dispatch_if(operator, x, y):
    if operator == 'add':
        return x + y
    elif operator == 'sub':
        return x - y
    elif operator == 'mul':
        return x * y
    elif operator == 'div':
        return x / y
    else:
        return None


def dispatch_dict(operator, x, y):
    x = {
        'add': lambda: x + y,
        'sub': lambda: x - y,
        'mul': lambda: x * y,
        'div': lambda: x / y,
    }.get(operator, lambda: None)()
    print(x)
    return x


"""
"""
dispatch_if('mul', 2, 8)
# 16
dispatch_dict('mul', 2, 8)
# 16
dispatch_if('unknown', 2, 8)
# None
print(dispatch_dict('unknown', 2, 8))
# None

# %% III. Tricks -  @classmethod vs @staticmethod vs "plain" methods
# 4/26/2021
# What's the difference?


class MyClass:
    def method(self):
        """
        Instance methods need a class instance and
        can access the instance through `self`.
        """
        return 'instance method called', self

    @classmethod
    def classmethod(cls):
        """
        Class methods don't need a class instance.
        They can't access the instance (self) but
        they have access to the class itself via `cls`.
        """
        return 'class method called', cls

    @staticmethod
    def staticmethod():
        """
        Static methods don't have access to `cls` or `self`.
        They work like regular functions but belong to
        the class's namespace.
        """
        return 'static method called'


# All methods types can be called on a class instance:
obj = MyClass()
obj.method()
# ('instance method called', <MyClass instance at 0x1019381b8>)

obj.classmethod()
# ('class method called', <class MyClass at 0x101a2f4c8>)
obj.staticmethod()
# 'static method called'

# Calling instance methods fails if we only have the class object:
MyClass.classmethod()
# ('class method called', <class MyClass at 0x101a2f4c8>)
MyClass.staticmethod()
# 'static method called'
MyClass.method()
# TypeError:
# "unbound method method() must be called with MyClass "
# "instance as first argument (got nothing instead)"


# %% III. Tricks - Python's list slice syntax can be used without indices
# for a few fun and useful things:

# You can clear all elements from a list:
lst = [1, 2, 3, 4, 5]
del lst[:]
lst
# []

# You can replace all elements of a list
# without creating a new list object:
a = lst
lst[:] = [7, 8, 9]
lst
# [7, 8, 9]
a
# [7, 8, 9]
a is lst
# True

# You can also create a (shallow) copy of a list:
b = lst[:]
b
# [7, 8, 9]
b is lst
# False


# %% Tricks - Python's built-in "dis" module to disassemble functions
# and inspect their CPython VM bytecode:

def greet(name):
    return 'Hello, ' + name + '!'


greet('Dan')
# 'Hello, Dan!'

import dis
dis.dis(greet)
# 2   0 LOAD_CONST     1 ('Hello, ')
#     2 LOAD_FAST      0 (name)
#     4 BINARY_ADD
#     6 LOAD_CONST     2 ('!')
#     8 BINARY_ADD
#    10 RETURN_VALUE

# %% IV. Interviews questions - Easy
# =============================================================================
from random import randint


def majority_element_indexes(lst):
    """
    Return a list of the indexes of the majority element.
    Majority element is the element that appears more than
    floor(n / 2) times.

    at powershell prompt in working directory:
    python -m doctest interview.py

    Verbose option:
    python -m doctest -v interview.py

    If there is no majority element, return []
    >>> majority_element_indexes([1, 1, 2])
    [0, 1]
    >>> majority_element_indexes([1, 2])
    []
    >>> majority_element_indexes([])
    []
    """
    n = len(lst)
    s_lst = lst[:]
    s_lst.sort()

    def find_ind(lst, el):
        indices = []
        for i, value in enumerate(lst):
            if value == el:
                indices.append(i)
        return indices

    # if even e.g., 10, len middle index is 4 and 5
    # and lst[5]==lst[0] or lst[4] == lst[9]

    # if odd # items, then middle index e.g., n=11, then lst[5] should
    # be same as lst[10] or lst[0]
    # no need to be so verbose; if we use counter, we do not need all
    # the elifs

    if n == 1:
        return [0]
    elif n == 2 and lst[0] == lst[1]:
        return [0, 1]
    elif (n != 2) and (n % 2 == 0) and ((s_lst[n // 2] == s_lst[-1])):
        major_el = s_lst[-1]
        return find_ind(lst, major_el)
    elif (n != 2) and (n % 2 == 0) and (s_lst[n // 2 + 1] == s_lst[0]):
        major_el = s_lst[0]
        return find_ind(lst, major_el)
    elif (n != 2) and (n % 2 == 1) and (s_lst[(n - 1) // 2] == s_lst[-1]):
        major_el = s_lst[-1]
        return find_ind(lst, major_el)
    elif (n != 2) and (n % 2 == 1) and (s_lst[(n - 1) // 2] == s_lst[0]):
        major_el = s_lst[0]
        return find_ind(lst, major_el)
    else:
        return []


import time

start = time.time()
mylist = [randint(1, 10) for x in range(100000)]
print(majority_element_indexes(mylist))
end = time.time()
print(end - start)


majority_element_indexes([1, 1, 2])

# %% IV. Interviews questions - Easy, Try 2
from collections import Counter
import time
from random import randint


def majority_element_indexes2(lst):
    """Return a list of the indexes of the majority element.

    Majority element is the element that appears more than
    floor(n / 2) times.

    at powershell prompt in working directory:
    python -m doctest interview.py

    If there is no majority element, return []
    >>> majority_element_indexes2([1, 1, 2])
    [0, 1]
    >>> majority_element_indexes2([1, 2])
    []
    >>> majority_element_indexes2([])
    []
    """
    from collections import Counter

    ctr = Counter(lst)
    n = len(lst)
    majel = None
    for key, cnt in ctr.items():
        if cnt > n // 2:
            majel = key

    if majel is None:
        return []
    else:
        return [i for i, val in enumerate(lst) if val == majel]


print(majority_element_indexes2([1, 1, 2]))

# %% IV. Interviews questions - Medium
from itertools import groupby
from collections import Counter


def keypad_string(keys):
    """Return string from keypad.

    Given a string consisting of 0-9,
    find the string that is created using
    a standard phone keypad
    | 1        | 2 (abc) | 3 (def)  |
    | 4 (ghi)  | 5 (jkl) | 6 (mno)  |
    | 7 (pqrs) | 8 (tuv) | 9 (wxyz) |
    |     *    | 0 ( )   |     #    |
    You can ignore 1, and 0 corresponds to space
    >>> keypad_string("12345")
    'adgj'
    >>> keypad_string("4433555555666")
    'hello'
    >>> keypad_string("2022")
    'a b'
    >>> keypad_string("")
    ''
    >>> keypad_string("111")
    ''
    """
    words = [(num, len(list(g))) for num, g in groupby(keys)
             if num not in set('*#1')]  # O(n) to form the groups
    # the grouper object is generator, so the groups disappear as
    # we iterate through the generator
    lookup = {'2': 'abc', '3': 'def',
              '4': 'ghi', '5': 'jkl', '6': 'mno',
              '7': 'pqrs', '8': 'tuv', '9': 'wxyz',
              '0': ' '}
    output = ''
    for (num, length) in words:  # O(n) worst case, every group 1 digit
        count = length // len(lookup.get(num, ''))
        last = length % len(lookup.get(num, ''))
        # We use the rule that the last letter for each key is the
        # priority; i.e., if we get '222', that means 'c' instead of
        # 'ab' or 'aaa' or 'ba'; '2222' would be 'ca', not 'bb' or
        # 'aaaa' or 'ac'
        output = (output
                  + count * lookup.get(num, [])[-1]  # line 2
                  # when last letter of group is chosen

                  + bool(last != 0) * lookup.get(num, '')[last - 1]
                  # need a boolean so that when last == 0 we don't
                  # repeat last letter of group.  Number of appearances
                  # of last letter is chosen by Line 2 above
                      )
    return output


print("4433555555666:", keypad_string("4433555555666"))
print("2022:", keypad_string("2022"))
print("<blank>:", keypad_string(""))
print("111:", keypad_string("111"))


# %% IV. Interviews questions - Difficult


class Link:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

    def __str__(self):
        if not self.next:
            return f"Link({self.val})"
        return f"Link({self.val}, {self.next})"


def merge_k_linked_lists(linked_lists):
    """Merge k sorted linked lists into one sorted linked list.

    requires insertion in existing listing or append of one list to
    end of antoerh in these two cases; may require instantiation of Link
    object that doesn't exist to form junction
    - second example requires two instantiations and moving next to end

    >>> print(merge_k_linked_lists([Link(1, Link(2)),
                                    Link(3, Link(4))]))
    Link(1, Link(2, Link(3, Link(4))))

    >>> print(merge_k_linked_lists([Link(1, Link(2)),
                                    Link(2, Link(4)),
                                    Link(3, Link(3))]))
    Link(1, Link(2, Link(2, Link(3, Link(3, Link(4))))))
    """

a = Link(3, 4)
b = Link(5)

# %% Sphinx
# Among its features are the following:
# - Output formats: HTML
# (including derivative formats such as HTML Help, Epub and Qt Help),
#  plain text, manual pages and LaTeX or direct PDF output using rst2pdf
# - Extensive cross-references: semantic markup and automatic links for
#     functions, classes, glossary terms and similar pieces of information
# - Hierarchical structure: easy definition of a document tree, with
#     automatic links to siblings, parents and children
# - Automatic indices: general index as well as a module index
# - Code handling: automatic highlighting using the Pygments highlighter
# - Flexible HTML output using the Jinja 2 templating engine
# - Various extensions are available, e.g. for automatic testing of
# snippets and inclusion of appropriately formatted docstrings
# - Setuptools integration
#
# =============================================================================
# Python Developer’s Guide issues
# https://github.com/python/devguide/issues
# - very few easy fixes recently as of April 2021 available to work on
#
# python.org issues
# https://github.com/python/pythondotorg/issues
#
# If you want to contribute to CPython, which is what most people
# mean when they say “Python,” then you’ll need to create an account
# at Python’s bug tracker, which is called BPO because it’s at
# bugs.python.org. You can register yourself as a user by going to
# User → Register in the menu on the left.
#
# The information in the default view can be a lot to take in since
# it shows issues raised by users as well as issues raised by core
# developers, which may already have been fixed. Fortunately, you
# can filter this list to find exactly what you’re looking for.
#
# To filter the list, start by logging in and then go to
# Your Queries → Edit in the menu on the left.
# You’ll get a list of queries that you can leave in or leave out
# =============================================================================













































