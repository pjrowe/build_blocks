"""Hackerrank Practice.
May 2, 2021

Artificial Intelligence - 150 exercises
- many are Hard / Expert
=============================================================================
11 Bot Building               0 / 11  4 easy, rest hard+
 5  A* Search                 0 /  5  one easy
10 Alpha Beta Pruning         0 / 10  Hard advanced
14 Combinatorial Search       0 / 14
 6 Games                      0 /  6 no easy
/// 46

25 Statistics and Machine L    6 / 25
24 Digital Image Analysis      4 / 24
25 Natural Lang Processing     0 / 25
30 Probability & Statistics   30 / 30
///104
-------                       --------
150                           40 / 150


"""

# %% Artificial Intelligence - Statistics and Machine Learning -
# The Best Aptitude Test

# Hackerrank Python The Best Aptitude Test
# https://www.hackerrank.com/challenges/the-best-aptitude-test/problem
# 1 <= T <= 10
# 4 <= N <= 100
# 0.0 <= k <= 10.0, where k is the GPA of every student.
# 0.0 <= s <= 100.0, where s is the score of every student in any of
# the 5 aptitude tests.


def get_gpa(n):
    # n GPAs for first year, one per student
    gpas_ = input().split(' ')
    gpas = []
    for i in range(n):
        gpas.append(float(gpas_[i]) * 10)
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
        error += (gpas[i] - scores[i])**2
    return error


if __name__ == '__main__':

    t = int(input())  # test cases

    for i in range(t):
        n = int(input())  # # of students, each case
        gpas = get_gpa(n)
        scores = get_scores(n)

        best_test = 0
        correct = get_relscore(gpas, scores[0])

        for j in range(1, 5):
            next_correct = get_relscore(gpas, scores[j])
            if next_correct > correct:
                best_test = j
                correct = next_correct

        print(best_test + 1)

# 1
# 5
# 7.5 7.7 7.9 8.1 8.3
# 10 30 20 40 50
# 11 9 5 19 29
# 21 9 15 19 39
# 91 9 75 19 89
# 81 99 55 59 89
# correct answer = 1

# %% Myroll ??
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import dirichlet
from scipy.stats import beta
import pandas as pd

mydie = np.array([0.05, 0.25, 0.2, 0.4, .1, 0.1])


def myroll(x, die):
    roll = []
    for i in range(x):
        z = [int(z) for z in pd.Series(6 * np.random.random())]
        roll.append(z)

    rolls = pd.Series(roll)
    tally = pd.DataFrame(rolls.value_counts(), columns=['#'])
    tally['%'] = pd.Series(rolls.value_counts() / x)
    return tally


tally = myroll(100, mydie)
print(tally)
