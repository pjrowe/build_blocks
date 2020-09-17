"""Reusing /Alter Code from Model Tuning, RF vs. bayesian.

Sept 2020
Bayesian optimization should focus on higher scoring hyperparameter values
based on the surrogate model of the objective function it constructs, but
random search of the hyperparams could stumble on a better scoring set of
hyperparams in fewer iterations, by luck.

OUTLINE
-------
Distribution of scores
    Overall distribution
    Score versus the iteration (did scores improve as search progressed)
Distribution of hyperparameters
    Overall distribution including the hyperparameter grid for a reference
    Hyperparameters versus iteration to look at evolution of values
Hyperparameter values versus the score
    Do scores improve with certain values of hyperparameters (correlations)
    3D plots looking at effects of 2 hyperparameters at a time on the score
Additional Plots
    Time to run each evaluation for Bayesian optimization
    Correlation heatmaps of hyperparameters with score

Reference
---------
https://www.kaggle.com/willkoehrsen/automated-model-tuning
local drive: \kaggle\build_blocks\automated-model-tuning.ipynb

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import ast
# from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression


# %% FUNCTIONS

def process(results):
    """Process results into a dataframe with one column per hyperparameter."""
    results = results.copy()
    results['hyperparameters'] = results['hyperparameters'].map(eval)
    # or .map(ast.literal_eval), but need to import ast

    # Sort with best values on top
    results = results.sort_values('score',
                                  ascending=False).reset_index(drop=True)

    # Create dataframe of hyperparameters
    hyp_df = pd.DataFrame(columns=list(results.loc[0,
                                                   'hyperparameters'].keys()))

    # Iterate through each set of hyperparameters that were evaluated
    for i, hyp in enumerate(results['hyperparameters']):
        hyp_df = hyp_df.append(pd.DataFrame(hyp, index=[0]),
                               ignore_index=True, sort=True)

    # Put the iteration and score in the hyperparameter dataframe
    hyp_df['iteration'] = results['iteration']
    hyp_df['score'] = results['score']

    return hyp_df


def plot_hyp_dist(hyp, max):
    """Plot distribution of hyp with best values of hyp as vertical line."""
    plt.figure(figsize=(20, 8))
    plt.rcParams['font.size'] = 18

    # Density plots of the learning rate distributions
    # Sampling dist isright top of random search
    sns.kdeplot(param_grid[hyp], label='Sampling Distribution',
                lw=4, color='k')
    sns.kdeplot(random_hyp[hyp], label='Random Search',
                lw=4, color='blue')
    sns.kdeplot(opt_hyp[hyp], label='Bayesian', lw=4, color='green')
    plt.vlines([best_random_hyp[hyp]], label='Best Random',
               ymin=0.0, ymax=max,
               linestyles='--', lw=4, colors=['blue'])
    plt.vlines([best_opt_hyp[hyp]], label='Best Bayesian',
               ymin=0.0, ymax=max, linestyles='--', lw=4, colors=['green'])
    plt.legend()
    plt.xlabel(hyp)
    plt.ylabel('Density')
    plt.title('{} Distribution'.format(hyp))

    print('\n{:.5f} Best {}, random search'.format(best_random_hyp[hyp], hyp))
    print('{:.5f} Best {}, Bayesian'.format(best_opt_hyp[hyp], hyp))


def plot_vs_iteration(data1, h1, h2, *h):
    """2D plots of 2 or more hyperparameters vs. iteration.

    Note: only useful for bayesian optimizations, not for random search of
    optimal hyperparameters
    """
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    i = 0

    # Plot of four hyperparameters
    for i, hyper in enumerate([h1, h2, *h]):
        data1[hyper] = data1[hyper].astype(float)
        # Scatterplot
        sns.regplot('iteration', hyper, data=data1, ax=axs[i])
        axs[i].scatter(best_opt_hyp['iteration'], best_opt_hyp[hyper],
                       marker='*', s=200, c='k')
        axs[i].set(xlabel='Iteration', ylabel='{}'.format(hyper),
                   title='{} over Search'.format(hyper))
    plt.tight_layout()


def plot_3d(x, y, z, df, cmap=plt.cm.seismic_r):
    """3D scatterplot of data in df."""
    plt.rcParams['font.size'] = 18
    plt.style.use('fivethirtyeight')
    plt.rcParams['axes.labelpad'] = 14
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df[x], df[y], df[z], c=df[z], cmap=cmap, s=40)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    plt.title('{} as function of {} and {}'.format(z, x, y), size=18)


def plot_2d(x, y, df, cmap=plt.cm.seismic_r):
    """2D scatterplot of data in df."""
    plt.rcParams['font.size'] = 18
    plt.style.use('fivethirtyeight')
    plt.rcParams['axes.labelpad'] = 14
    plt.figure(figsize=(8, 7))
    plt.plot(df[x], df[y], 'ro')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('{} vs {}'.format(y, x))


def plot_vs_score(h1, h2, h3, h4, data1, data2):
    """2D plots of 4 hyperparameters for 2 dataframes of optimizations."""
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    i = 0

    # Plot of four hyperparameters
    for i, hyper in enumerate([h1, h2, h3, h4]):
        data1[hyper] = data1[hyper].astype(float)
        # Scatterplot
        sns.regplot(hyper, 'score', data=data1, ax=axs[i], color='b',
                    scatter_kws={'alpha': 0.6})
        axs[i].scatter(best_random_hyp[hyper], best_random_hyp['score'],
                       marker='*', s=200, c='b', edgecolor='k')
        axs[i].set(xlabel='{}'.format(hyper), ylabel='Score',
                   title='Score vs {}'.format(hyper))

        data2[hyper] = data2[hyper].astype(float)
        # Scatterplot
        sns.regplot(hyper, 'score', data=data2, ax=axs[i], color='g',
                    scatter_kws={'alpha': 0.6})
        axs[i].scatter(best_opt_hyp[hyper], best_opt_hyp['score'],
                       marker='*', s=200, c='g', edgecolor='k')
    plt.legend()
    plt.tight_layout()


# %% DATA
random = pd.read_csv('random_search_simple.csv')
random = random.sort_values('score', ascending=False).reset_index()
opt = pd.read_csv('bayesian_trials_simple.csv')
opt = opt.sort_values('score', ascending=False).reset_index()

print('Best score from random search:         {:.5f} found on iteration: {}.'
      .format(random.loc[0, 'score'], random.loc[0, 'iteration']))
print('Best score from bayesian optimization: {:.5f} found on iteration: {}.'
      .format(opt.loc[0, 'score'], opt.loc[0, 'iteration']))

# %% create hypers df, the best params found for both searches

rand_hypers = np.transpose(pd.DataFrame(eval(random.hyperparameters[0]),
                                        index=['rand']))
opt_hypers = np.transpose(pd.DataFrame(eval(opt.hyperparameters[0]),
                                       index=['opt']))
hypers = opt_hypers.join(rand_hypers, how='left')
hypers = hypers.reset_index().sort_values('index', ascending=True)
hypers = hypers.set_index('index')
print(hypers)

# %% Kdeplot of scores across all the iterations of hyperparam search
# random search explores search space well and will probably find a higher-
# scoring set of values (if the space is not extremely high-dimensional)
# while Bayesian optimization focuses on values that yield higher scores.
plt.figure(figsize=(10, 6))
sns.kdeplot(opt['score'], label='Bayesian Opt')
sns.kdeplot(random['score'], label='Random Search')
plt.xlabel('Score (5 Fold Validation ROC AUC)')
plt.ylabel('Density')
plt.title('Random Search and Bayesian Optimization Results')

# %% Histograms of scores (look like kdeplot)

random['set'] = 'random'
opt['set'] = 'opt'
scores = random[['iteration', 'score', 'set']]
# not sure why it automatically chooses score column to sort on
scores = scores.append(opt[['set', 'score', 'iteration']], sort=True)
print(scores.head())

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.hist(random['score'], bins=20, color='blue', edgecolor='k')
plt.xlim((0.72, 0.80))
plt.xlabel('Score')
plt.ylabel("Count")
plt.title('Random Search Distribution of Scores')

plt.subplot(122)
plt.hist(opt['score'], bins=20, color='blue', edgecolor='k')
plt.xlim((0.72, 0.80))
plt.xlabel('Score')
plt.ylabel("Count")
plt.title('Bayes Opt Search Distribution of Scores')


# %% PLOT SCORE (ROC AUC) VS. ITERATION
aggs = ['mean', 'max', 'min', 'std', 'count']
scores.groupby('set')['score'].agg(aggs)
plt.rcParams['font.size'] = 16
best_random_score = random.loc[0, 'score']
best_random_iteration = random.loc[0, 'iteration']
best_opt_score = opt.loc[0, 'score']
best_opt_iteration = opt.loc[0, 'iteration']

sns.lmplot('iteration', 'score', hue='set', data=scores, size=8)
plt.scatter(best_random_iteration, best_random_score, marker='*', s=400,
            c='blue', edgecolor='k')
plt.scatter(best_opt_iteration, best_opt_score, marker='*', s=400, c='red',
            edgecolor='k')
plt.xlabel('Iteration')
plt.ylabel('ROC AUC')
plt.title("Validation ROC AUC versus Iteration")

# %% PROCESS DATAFRAMES OF ITERATIONS, TURNING HYPERPARAMETERS INTO COLUMNS
random_hyp = process(random)
opt_hyp = process(opt)
random_hyp.head()
print(random_hyp.columns)
print(opt_hyp.columns)

# %% define the hyperparameter grid
# that was used (the same ranges applied in both searches).

# metric, n_estimators, and verbose not here
param_grid = {
    'boosting_type': ['gbdt', 'goss', 'dart'],
    'colsample_bytree': list(np.linspace(0.6, 1, 10)),
    'is_unbalance': [True, False],
    'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base=10,
                                      num=1000)),
    'min_child_samples': list(range(20, 500, 5)),

    'num_leaves': list(range(20, 150)),
    'reg_alpha': list(np.linspace(0, 1)),
    'reg_lambda': list(np.linspace(0, 1)),
    'subsample_for_bin': list(range(20000, 300000, 20000)),
    'subsample': list(np.linspace(0.5, 1, 100))
}

best_random_hyp = hypers['rand']
best_opt_hyp = hypers['opt']
# %% Density plot and best values of each hyperparameters over range of values
plot_hyp_dist('learning_rate', 40)
plot_hyp_dist('min_child_samples', .008)
plot_hyp_dist('num_leaves', .02)
plot_hyp_dist('reg_alpha', 2)
plot_hyp_dist('reg_lambda', 2)

plot_hyp_dist('colsample_bytree', 6)
plot_hyp_dist('subsample_for_bin', .00001)
plot_hyp_dist('subsample', 4)

# %% Mean Score by boosting type
"""
In both search methods, the gbdt (gradient boosted decision tree) and
dart (dropout meets additive regression tree) do much better than goss
(gradient based one-sided sampling). gbdt does the best on average
(and for the max), so it might make sense to use that method in the future!
"""
rand_scores = random_hyp.groupby('boosting_type')['score'].agg(aggs)
opt_scores = opt_hyp.groupby('boosting_type')['score'].agg(aggs)

plt.figure(figsize=(16, 6))
plt.subplot(121)
rand_scores['mean'].plot.bar(color='b')
plt.ylabel('Score')
plt.title('Random Search Boosting Type Scores', size=14)

plt.subplot(122)
opt_scores['mean'].plot.bar(color='b')
plt.ylabel('Score')
plt.title('Bayesian Boosting Type Scores', size=14)

# %% Subsample hyperparameter under gbdt

plt.figure(figsize=(20, 8))
plt.rcParams['font.size'] = 18

# Density plots of the learning rate distributions
sns.kdeplot(param_grid['subsample'], label='Sampling Distribution', lw=4,
            color='k')
sns.kdeplot(random_hyp[random_hyp['boosting_type'] == 'gbdt']['subsample'],
            label='Random Search', lw=4, color='blue')
sns.kdeplot(opt_hyp[opt_hyp['boosting_type'] == 'gbdt']['subsample'],
            label='Bayesian', linewidth=4, color='green')
plt.vlines([best_random_hyp['subsample']],
           ymin=0.0, ymax=10.0, linestyles='--', lw=4, colors=['blue'])
plt.vlines([best_opt_hyp['subsample']],
           ymin=0.0, ymax=10.0, linestyles='--', lw=4, colors=['green'])
plt.legend()
plt.xlabel('Subsample')
plt.ylabel('Density')
plt.title('Subsample Distribution')

print('{:.5f} Best from random search'.format(best_random_hyp['subsample']))
print('{:.5f} Best from Bayesian'.format(best_opt_hyp['subsample']))

# %% Hyperparameters versus Iteration
# only makes sense for bayesian search, as random search will not show any
# trend by iteration
best_opt_hyp = opt_hyp.loc[0]
best_random_hyp = random_hyp.loc[0]

plot_vs_iteration(opt_hyp, 'colsample_bytree', 'learning_rate',
                  'min_child_samples', 'num_leaves')
plot_vs_iteration(opt_hyp, 'reg_alpha', 'reg_lambda', 'subsample_for_bin')

# %% Hyperparameters vs Score
# plots of a single hyperparameter versus the score cannot be given too
# emphasis because we are not changing one hyperparameter at a time. Therefore,
#  if there are trends, it might not be solely due to the single
# hyperparameter we show.
random_hyp['set'] = 'Random Search'
opt_hyp['set'] = 'Bayesian'

plot_vs_score('reg_alpha', 'reg_lambda', 'subsample_for_bin', 'subsample',
              random_hyp, opt_hyp)
plot_vs_score('colsample_bytree', 'learning_rate', 'min_child_samples',
              'num_leaves', random_hyp, opt_hyp)

# %% 3D plots - 2 Hyperparameters vs Score
# learning rate, number of estimators vs the score
# number of estimators was selected using early stopping for 100 rounds with
# 5-fold cross validation; number of estimators was not a hyperparameter in
# the grid that we searched over. Early stopping is a more efficient method
# of finding the best number of estimators than including it in a search
# (based on my limited experience)!

plot_3d('reg_alpha', 'reg_lambda', 'score', random_hyp)
plot_3d('reg_alpha', 'reg_lambda', 'score', opt_hyp)
plot_3d('learning_rate', 'n_estimators', 'score', random_hyp)
plot_3d('learning_rate', 'n_estimators', 'score', opt_hyp)

# %% 2D plots - Hyperparameters vs Score
# each individual decision trees contribution is lessened as the
# learning rate is decreased leading to a need for more decision trees in the
# ensemble.
plot_2d('learning_rate', 'n_estimators', random_hyp)
plot_2d('learning_rate', 'n_estimators', opt_hyp)
plot_2d('learning_rate', 'score', random_hyp)
plot_2d('learning_rate', 'score', opt_hyp)
plot_2d('n_estimators', 'score', random_hyp)
plot_2d('n_estimators', 'score', opt_hyp)

# %% Correlations for Random Search
random_hyp.corr()['score']
# pretty highly negative correlation between learning rate and score for
# all boosting types
opt_hyp[opt_hyp['boosting_type'] == 'gbdt'].corr()['score']['learning_rate']
# == 'gbdt': -0.982
opt_hyp[opt_hyp['boosting_type'] == 'dart'].corr()['score']
opt_hyp[opt_hyp['boosting_type'] == 'goss'].corr()['score']

# Heatmap of correlations
plt.figure(figsize=(12, 12))
sns.heatmap(random_hyp.corr().round(2), cmap=plt.cm.gist_heat_r, vmin=-1.0,
            annot=True, vmax=1.0)
plt.title('Correlation Heatmap')

# %% Meta-Machine Learning
# labeled set of data: the hyperparameter values and the resulting score.
# we can fit an estimator on top of the hyperparameter values and the scores
# in a supervised regression problem

# Create training data and labels
opt_hyp2 = opt_hyp.drop(columns=['metric', 'set', 'verbose'])
opt_hyp2['n_estimators'] = opt_hyp2['n_estimators'].astype(np.int32)
opt_hyp2['min_child_samples'] = opt_hyp2['min_child_samples'].astype(np.int32)
opt_hyp2['num_leaves'] = opt_hyp2['num_leaves'].astype(np.int32)
opt_hyp2['subsample_for_bin'] = opt_hyp2['subsample_for_bin'].astype(np.int32)
opt_hyp2 = pd.get_dummies(opt_hyp2)

train_labels = opt_hyp2.pop('score')
train = np.array(opt_hyp2.copy())

# Create the lasso regression with cv
lr = LinearRegression()
# Train on the data
lr.fit(train, train_labels)
x = list(opt_hyp.columns)
x_values = lr.coef_

coefs = {variable: coef for variable, coef in zip(x, x_values)}
coefs
