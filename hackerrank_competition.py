# -*- coding: utf-8 -*-
"""
Created on Mon April 26, 2021

@author: Trader
"""

# %% Project Euler #244: Sliders
# Letter	Code
from itertools import permutations

directions = {'l': 76, 'r': 82, 'u': 85, 'd': 68}
n = int(input())


def build_grid(n):
    matrix = ['x' * n for i in range(n)]
    for j in range(n):
        matrix[j] = input()
    return matrix

def distance(start, end):
    for row in range(n):
        if 'W' in start[row]:
            start_row = row
            start_col = start[row].index('W')
        if 'W' in end[row]:
            end_row = row
            end_col = end[row].index('W')
    dist_rows = end_row - start_row
    dist_cols = end_col - start_col
    if dist_cols > 0:
        cols = 'l' * dist_cols
    elif dist_cols < 0:
        cols = 'r' * dist_cols
    else:
        cols = ''

    if dist_rows > 0:
        rows = 'u' * dist_cols
    elif dist_rows < 0:
        rows = 'd' * dist_cols
    else:
        rows = ''

    print(rows + cols)
    return rows + cols


start = build_grid(n)
end = build_grid(n)
perms = list(permutations(distance(start, end), 2))

total = 0
for perm in perms:
    chk = 0
    for item in perm:
        chk = (chk * 243 + directions[item]) % 100_000_007
        total += chk


print(total)
