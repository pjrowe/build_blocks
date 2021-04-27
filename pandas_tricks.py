# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 19:24:56 2021

Source:
    Data Science and Pandas Tricks Tutorials/Courses from realpython.com

@author: Trader
"""
# %% Pandas DataFrames 101
# Importing CSV Data Into a Pandas DataFrame 02:16
# Slicing and Dicing a Pandas DataFrame 01:15
# Mapping and Analyzing a Data Set in Pandas 08:20
# Working With groupby() in Pandas 04:47
# Plotting a DataFrame 01:58

import vincent  # not available standard in Anaconda
import pandas as pd
from pandas import DataFrame, Series

vincent.core.initialize_notebook()
pd.set_option('display.max_columns', None)


del data['points']  # delete column
temp = data[['MP', 'FG', 'FGA']] # slices of df
# this code was presented in Jupyter notebook

import time
import datetime


def str_to_seconds(minutes):
    minutes = str(minutes)
    minutes = time.strptime(minutes, '%M:%S')
    return datetime.timedelta(minutes=minutes.tm_min, second=minutes.tm_sec).total_seconds()

temp['MP'] = temp['MP'].map(str_to_seconds)
# .map() takes a function as argument and puts result back in place

group_by_opp = data.groupby('Opp')

# %% Idiomatic Pandas: Tricks & Features You May Not Know
"""
 1- Configure Options and Settings at Interpreter Startup
 2- Make Toy Data Structures With Pandas’ Testing Module
 3- Take Advantage of Accessor Methods
 4- Create a DatetimeIndex from Component Columns
 5- Use Categorical Data to Save Time and Space

 6- Introspect Groupby Objects via Iteration
 7- Mapping Trick for Membership Binning
 8- Understand How Pandas Uses Boolean Operators
 9- Load Data from the Clipboard
10- Write Pandas Objects Directly to Compressed Format
"""
import pandas as pd


def start():
    """Configure options at startup of interpreter can save typing.

    From powerhsell:
        >>EXPORT PYTHONSTARTUP='pandas_tricks.'
        # THIS DOESN'T seem to work for PC...check how to run before
        Anaconda started

    From powerhsell:
        >>python pandas_tricks.py

    """
    options = {
        'display': {
            'max_columns': None,
            'max_colwidth': 25,
            'expand_frame_repr': False,
            'max_rows': 14,
            'max_seq_items': 50,
            'precision': 4,
            'show_dimensions': False
        }
    }

    for category, option in options.items():
        for op, value in option.items():
            pd.set_option(f'{category}.{op}', value)


if __name__ == '__main__':
    start()
    print(pd.get_option('display.max_rows'))
    del start


# %% Make Toy Data Structures With Pandas’ Testing Module

import pandas.util.testing as tm
tm.N, tm.K = 15, 3  # rows, columns
import numpy as np
np.random.seed(444)
tm.makeTimeDataFrame(freq='M').head()  # monthly index
tm.makeDataFrame().head()  # generic dataframe, random string as index

[i for i in dir(tm) if i.startswith('make')]  # list of all the types

['makeBoolIndex',
 'makeCategoricalIndex',
 'makeCustomDataframe',
 'makeCustomIndex',
 'makeDataFrame',
 'makeDateIndex',
 'makeFloatIndex',
 'makeFloatSeries',
 'makeIntIndex',
 'makeIntervalIndex',
 'makeMissingDataframe',
 'makeMixedDataFrame',
 'makeMultiIndex',
 'makeObjectSeries',
 'makePeriodFrame',
 'makePeriodIndex',
 'makePeriodSeries',
 'makeRangeIndex',
 'makeStringIndex',
 'makeStringSeries',
 'makeTimeDataFrame',
 'makeTimeSeries',
 'makeTimedeltaIndex',
 'makeUIntIndex',
 'makeUnicodeIndex']

# %% 3. Take Advantage of Accessor Methods

# Accessor is somewhat like a getter (although getters and setters are
# used infrequently in Python). A Pandas accessor as a property that
# serves as an interface to additional methods.  There are 4
import pandas as pd


pd.Series._accessors
# {'cat', 'str', 'dt', 'sparse'}
# .cat is for categorical data,
# .str is for string (object) data
# .dt is for datetime-like data
# .sparse for sparse matrix

# Pandas string methods are vectorized, meaning that they operate on
# the entire array without an explicit for-loop:
addr = pd.Series(['Washington, D.C. 20003',
                  'Brooklyn, NY 11211-1755',
                  'Omaha, NE 68154',
                  'Pittsburgh, PA 15211'
                  ])

addr.str.upper()
addr.str.count(r'\d')  # 5 or 9-digit
regex = (r'(?P<city>[A-Za-z ]+), '      # One or more letters
         r'(?P<state>[A-Z]{2}) '        # 2 capital letters
         r'(?P<zip>\d{5}(?:-\d{4})?)')  # Optional 4-digit extension

addr.str.replace('.', '').str.extract(regex)
# .str.extract(), .str is the accessor,
# .str.extract() is an accessor method
# .str maps to StringMethods.
# .dt maps to CombinedDatetimelikeProperties.
# .cat routes to CategoricalAccessor.
"""
These standalone classes are then “attached” to the Series class
using a CachedAccessor. It is when the classes are wrapped in
CachedAccessor that a bit of magic happens.

CachedAccessor is inspired by a “cached property” design: a property
is only computed once per instance and then replaced by an ordinary
attribute. It does this by overloading the .__get__() method, which is
part of Python’s descriptor protocol.

Note: If you’d like to read more about the internals of how this
works, see the Python Descriptor HOWTO and this post on the cached
property design. Python 3 also introduced functools.lru_cache(),
which offers similar functionality. There are examples all over
the place of this pattern, such as in the aiohttp package.


The second accessor, .dt, is for datetime-like data.
It technically belongs to Pandas’ DatetimeIndex, and if called on a
Series, it is converted to a DatetimeIndex first:
"""
daterng = pd.Series(pd.date_range('2017', periods=9, freq='Q'))
daterng
# 0   2017-03-31
# 1   2017-06-30
# 2   2017-09-30
# 3   2017-12-31
# 4   2018-03-31
# 5   2018-06-30
# 6   2018-09-30
# 7   2018-12-31
# 8   2019-03-31
# dtype: datetime64[ns]

daterng.dt.day_name()
# 0      Friday
# 1      Friday
# 2    Saturday
# 3      Sunday
# 4    Saturday

# Second-half of year only
daterng[daterng.dt.quarter > 2]
# 2   2017-09-30
# 3   2017-12-31
# ...
daterng[daterng.dt.is_year_end]
# 3   2017-12-31
# 7   2018-12-31

# %% 4. Create a DatetimeIndex From Component Columns
# Speaking of datetime-like data, as in daterng above, it’s possible to
#  create a Pandas DatetimeIndex from multiple component columns that
#  together form a date or datetime:

from itertools import product
datecols = ['year', 'month', 'day']

df = pd.DataFrame(list(product([2017, 2016], [1, 2], [1, 2, 3])),
                  columns=datecols)
df['data'] = np.random.randn(len(df))
df
#     year  month  day    data
# 0   2017      1    1 -0.0767
# 1   2017      1    2 -1.2798
# 2   2017      1    3  0.4032
# 3   2017      2    1  1.2377
# 4   2017      2    2 -0.2060
# 5   2017      2    3  0.6187
# 6   2016      1    1  2.3786
# 7   2016      1    2 -0.4730
# 8   2016      1    3 -2.1505
# 9   2016      2    1 -0.6340
# 10  2016      2    2  0.7964
# 11  2016      2    3  0.0005

df.index = pd.to_datetime(df[datecols])
df.head()
#             year  month  day    data
# 2017-01-01  2017      1    1 -0.0767
# 2017-01-02  2017      1    2 -1.2798
# 2017-01-03  2017      1    3  0.4032
# 2017-02-01  2017      2    1  1.2377
# 2017-02-02  2017      2    2 -0.2060
# Finally, can drop the old individual columns and convert to Series:

df = df.drop(datecols, axis=1).squeeze()
df.head()
# 2017-01-01   -0.0767
# 2017-01-02   -1.2798
# 2017-01-03    0.4032
# 2017-02-01    1.2377
# 2017-02-02   -0.2060
# Name: data, dtype: float64

df.index.dtype_str
# 'datetime64[ns]
# The intuition behind passing a DataFrame is that a DataFrame resembles
# a Python dictionary where the column names are keys, and the
# individual columns (Series) are the dictionary values.
# That’s why pd.to_datetime(df[datecols].to_dict(orient='list'))
# would also work in this case. This mirrors the construction of
# Python’s datetime.datetime, where you pass keyword arguments such
# as datetime.datetime(year=2000, month=1, day=15, hour=10).
# %% 5. Use Categorical Data to Save on Time and Space

# %%6. Introspect Groupby Objects via Iteration
# When you call df.groupby('x'), the resulting Pandas groupby
# objects can be a bit opaque. This object is lazily instantiated and
# doesn’t have any meaningful representation on its own.

# You can demonstrate with the abalone dataset from example 1:

abalone['ring_quartile'] = pd.qcut(abalone.rings, q=4, labels=range(1, 5))
grouped = abalone.groupby('ring_quartile')

grouped
# <pandas.core.groupby.groupby.DataFrameGroupBy object at 0x11c1169b0>
# Alright, now you have a groupby object, but what is this thing,
# and how do I see it?

# Before you call something like grouped.apply(func), you can take
# advantage of the fact that groupby objects are iterable:

help(grouped.__iter__)

#         Groupby iterator

#         Returns
#         -------
#         Generator yielding sequence of (name, subsetted object)
#         for each group
# Each “thing” yielded by grouped.__iter__() is a tuple of
# (name, subsetted object), where name is the value of the column on
# which you’re grouping, and subsetted object is a DataFrame that is a
# subset of the original DataFrame based on whatever grouping condition
# you specify. That is, the data gets chunked by group:


for idx, frame in grouped:
    print(f'Ring quartile: {idx}')
    print('-' * 16)
    print(frame.nlargest(3, 'weight'), end='\n\n')

# Ring quartile: 1
# ----------------
#      sex  length   diam  height  weight  rings ring_quartile
# 2619   M   0.690  0.540   0.185  1.7100      8             1
# 1044   M   0.690  0.525   0.175  1.7005      8             1
# 1026   M   0.645  0.520   0.175  1.5610      8             1

# Ring quartile: 2
# ----------------
#      sex  length  diam  height  weight  rings ring_quartile
# 2811   M   0.725  0.57   0.190  2.3305      9             2
# 1426   F   0.745  0.57   0.215  2.2500      9             2
# 1821   F   0.720  0.55   0.195  2.0730      9             2

# Ring quartile: 3
# ----------------
#      sex  length  diam  height  weight  rings ring_quartile
# 1209   F   0.780  0.63   0.215   2.657     11             3
# 1051   F   0.735  0.60   0.220   2.555     11             3
# 3715   M   0.780  0.60   0.210   2.548     11             3

# Ring quartile: 4
# ----------------
#      sex  length   diam  height  weight  rings ring_quartile
# 891    M   0.730  0.595    0.23  2.8255     17             4
# 1763   M   0.775  0.630    0.25  2.7795     12             4
# 165    M   0.725  0.570    0.19  2.5500     14             4
# Relatedly, a groupby object also has .groups and a group-getter,
# .get_group():

grouped.groups.keys()
dict_keys([1, 2, 3, 4])

grouped.get_group(2).head()
#    sex  length   diam  height  weight  rings ring_quartile
# 2    F   0.530  0.420   0.135  0.6770      9             2
# 8    M   0.475  0.370   0.125  0.5095      9             2
# 19   M   0.450  0.320   0.100  0.3810      9             2
# 23   F   0.550  0.415   0.135  0.7635      9             2
# 39   M   0.355  0.290   0.090  0.3275      9             2
# This can help you be a little more confident that the operation
# you’re performing is the one you want:

grouped['height', 'weight'].agg(['mean', 'median'])
#                height         weight
#                  mean median    mean  median
# ring_quartile
# 1              0.1066  0.105  0.4324  0.3685
# 2              0.1427  0.145  0.8520  0.8440
# 3              0.1572  0.155  1.0669  1.0645
# 4              0.1648  0.165  1.1149  1.0655
# No matter what calculation you perform on grouped, be it a single
# Pandas method or custom-built function, each of these “sub-frames” is
# passed one-by-one as an argument to that callable. This is where the
# term “split-apply-combine” comes from: break the data up by groups,
# perform a per-group calculation, and recombine in some aggregated
# fashion.

# If you’re having trouble visualizing exactly what the groups will
# actually look like, simply iterating over them and printing a few can
# be tremendously useful.

# %% 7. Use This Mapping Trick for Membership Binning

# Series and a corresponding “mapping table” where each value belongs
# to a multi-member group, or to no groups at all:

countries = pd.Series([
    'United States',
    'Canada',
    'Mexico',
    'Belgium',
    'United Kingdom',
    'Thailand'])

groups = {
    'North America': ('United States', 'Canada', 'Mexico', 'Greenland'),
    'Europe': ('France', 'Germany', 'United Kingdom', 'Belgium')}

# In other words, you need to map countries to the following result:

# 0    North America
# 1    North America
# 2    North America
# 3           Europe
# 4           Europe
# 5            other
# dtype: object
# What you need here is a function similar to Pandas’ pd.cut(), but
# for binning based on categorical membership. You can use
# pd.Series.map(), which you already saw in example #5, to mimic this:

from typing import Any


def membership_map(s: pd.Series, groups: dict,
                   fillvalue: Any = -1) -> pd.Series:
    # Reverse & expand the dictionary key-value pairs
    groups = {x: k for k, v in groups.items() for x in v}
    return s.map(groups).fillna(fillvalue)
    # This should be significantly faster than a nested Python
    # loop through groups for each country in countries.


# Here’s a test drive:

membership_map(countries, groups, fillvalue='other')
# 0    North America
# 1    North America
# 2    North America
# 3           Europe
# 4           Europe
# 5            other
# dtype: object
# Let’s break down what’s going on here. (Sidenote: this is a great
# place to step into a function’s scope with Python’s debugger, pdb,
# to inspect what variables are local to the function.)

# The objective is to map each group in groups to an integer.
# However, Series.map() will not recognize 'ab'—it needs the
# broken-out version with each character from each group mapped to an
# integer. This is what the dictionary comprehension is doing:

groups = dict(enumerate(('ab', 'cd', 'xyz')))
{x: k for k, v in groups.items() for x in v}
{'a': 0, 'b': 0, 'c': 1, 'd': 1, 'x': 2, 'y': 2, 'z': 2}
# This dictionary can be passed to s.map() to map or “translate” its
# values to their corresponding group indices.

# %% 8. Understand How Pandas Uses Boolean Operators
# and, not, or
# have lower precedence than arithmetic operators
# <, <=, >, >=, !=, ==

# Evaluates to "False and True"
4 < 3 and 5 > 4
False

# Evaluates to 4 < 5 > 4
4 < (3 and 5) > 4
True

# 3 and 5 evaluates to 5 because of short-circuit evaluation:
# “The return value of a short-circuit operator is the last evaluated
# argument.”
# pandas 3 and 5 evaluates to 5 because of short-circuit evaluation:

# Pandas and NumPy do not use and, or, or not.
# Instead, it uses &, |, and ~
# higher (rather than lower) precedence than arithmetic operators

pd.Series([True, True, False]) & pd.Series([True, False, False])
# 0     True
# 1    False
# 2    False
# dtype: bool

s = pd.Series(range(10))
s % 2 == 0 & s > 3                      # Same as above, original expression
(s % 2) == 0 & s > 3                    # Modulo is most tightly binding here
(s % 2) == (0 & s) > 3                  # Bitwise-and is second-most-binding
(s % 2) == (0 & s) and (0 & s) > 3      # Expand the statement
((s % 2) == (0 & s)) and ((0 & s) > 3)  # The `and` operator is least-binding
# results in ValueError, because truth of a series is ambiguous
x < y <= z
# is equivalent to
x < y and y <= z

left = (s % 2) == (0 & s)
right = (0 & s) > 3
left and right  # This will raise the same ValueError

# SOLUTION: USE PARENTHESES IF ARITHMETIC OPERATOR IS INVOLVED,
# AND PANDAS WILL DO ELEMENTWISE COMPARISON
(s % 2 == 0) & (s > 3)

# %% 9. Read in data from clipboard (and not a file)
"""
Example:  select and copy text below, then run this cell
could also select cells in excel; same result

a   b           c       d
0   1           inf     1/1/00
2   7.389056099 N/A     5-Jan-13
4   54.59815003 nan     7/24/18
6   403.4287935 None    NaT

"""
df = pd.read_clipboard(na_values=[None], parse_dates=['d'])
df
df.dtypes

"""
if the following cells are in excel, running command below results in :
    output below
a	b	c
1	2	3
"""
df = pd.read_clipboard(na_values=[None])
df
"""

Out[29]:
   a  b  c
0  1  2  3

"""


# %% 10. Write Pandas Objects Directly to Compressed Format
# gzip, bz2, zip, or xz compression
import os.path

url = ('https://archive.ics.uci.edu/ml/'
       'machine-learning-databases/abalone/abalone.data')
cols = ['sex', 'length', 'diam', 'height', 'weight', 'rings']
abalone = pd.read_csv(url, usecols=[0, 1, 2, 3, 4, 8], names=cols)
abalone.to_json('df.json.gz', orient='records',
                lines=True, compression='gzip')
# In this case, the size difference is about 10x:

abalone.to_json('df.json', orient='records', lines=True)
os.path.getsize('df.json') / os.path.getsize('df.json.gz')
# 9.9

# %% Dataframes

df.memory_usage()
