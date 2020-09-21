"""Using automated hyperparameter tuning to optimize a machine learning model.
Created on Thu Sep 17 10:25:05 2020

Bayesian Optimization and using the Tree Parzen Esimator of the Hyperopt
library to tune the hyperparameters of a gradient boosting machine.

The concept of Bayesian optimization is to  build a probability model of the
objective function, which is then used to choose the next hyperparameter values
based on previous results, rather than doing an uninformed, random, or
exhaustive grid search of the entire hyperparameter space. Thus, more time is
spent evaluating promising hyperparameter values.


Four Parts of Sequential Model Based Optimization (e.g., Bayes Optim)
--------------------------------------------------------------------
1 Objective Function: with hyperparameters as inputs, the OF must be minimized;
    example is Loss = 1-ROC AUC, calculated using cross validation functions
    over a range of hyperparameter inputs
2 Domain space: the range of input values (hyperparameters) to evaluate
3 Optimization Algorithm: the method used to construct the surrogate function
  and choose the next values to evaluate; surrogate functions approximate the
  OF, but is much easier to evaluate / calculate

  Examples:
      These libraries all use the Expected Improvement criterion (which
      is maximized) to select the next hyperparameters from the surrogate
      model; EI is proportional to (l/g)/(cl/g + (1-c)); so maximize l/g
      - Hyperopt library implements Tree Parzen Estimator algorithm (TPE)
          uses Bayes rule to represent p(y|x) as p(x|y)p(y)/p(x) in EI formula
      - Spearmint and MOE libraries use a Gaussian Process for the surrogate
      - SMAC uses a Random Forest regression.

4 Results: (score, value) pairs that the algorithm uses to build the surrogate
  function

Source: https://www.kaggle.com/willkoehrsen/automated-model-tuning

References
----------
1. Detailed technical
"Algorithms for Hyper-Parameter Optimization"
https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf
http://proceedings.mlr.press/v28/bergstra13.pdf

2. Conceptual explanation
- build a probability model of the objective function and use it to select the
most promising hyperparameters to evaluate in the true objective function.
DONE
https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f

"""
# %% IMPORT
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from hyperopt import STATUS_OK
from timeit import default_timer as timer
import datetime
import json
import ast
import pickle

from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin

plt.rcParams['font.size'] = 18

# %% FUNCTIONS


def objective(hyperparameters):
    """Objective function for GBM Hyperparameter Optimization.

    Writes a new line to `outfile` on every iteration
    """
    global ITERATION
    global OUTFILE_CSV

    ITERATION += 1

    # Using early stopping to find number of trees trained
    if 'n_estimators' in hyperparameters:
        del hyperparameters['n_estimators']

    # Retrieve the subsample
    subsample = hyperparameters['boosting_type'].get('subsample', 1.0)

    # Extract the boosting type and subsample to top level keys
    hyperparameters['boosting_type'] = hyperparameters['boosting_type']['boosting_type']
    hyperparameters['subsample'] = subsample

    if hyperparameters['boosting_type'] == 'dart':
        # dart doesn't work with early_stopping, so it may take too long
        # to complete on my laptop; I commented out dart as an option in
        # hyperparameters space for this reason
        early_stop = 0
        num_rounds = 1000
        print(hyperparameters['boosting_type'], early_stop, num_rounds)
    else:
        early_stop = 100
        num_rounds = 10000
        print(hyperparameters['boosting_type'], early_stop, num_rounds)

    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'subsample_for_bin',
                           'min_child_samples']:
        hyperparameters[parameter_name] = int(hyperparameters[parameter_name])

    start = timer()

    # Perform n_folds cross validation
    cv_results = lgb.cv(hyperparameters, train_set, num_boost_round=num_rounds,
                        nfold=N_FOLDS, early_stopping_rounds=early_stop,
                        metrics='auc', seed=50)
    run_time = timer() - start

    # Extract the best score
    best_score = cv_results['auc-mean'][-1]
    # Loss must be minimized
    loss = 1 - best_score
    # Boosting rounds that returned the highest cv score
    n_estimators = len(cv_results['auc-mean'])

    # Add the number of estimators to the hyperparameters
    hyperparameters['n_estimators'] = n_estimators

    # Write to the csv file ('a' means append)
    of_connection = open(OUT_FILE_CSV, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, hyperparameters['boosting_type'],
                     hyperparameters['n_estimators'], hyperparameters,
                     ITERATION, run_time, best_score])
    of_connection.close()

    # Dictionary with information for evaluation
    return {'loss': loss, 'hyperparameters': hyperparameters,
            'iteration': ITERATION, 'train_time': run_time,
            'status': STATUS_OK}


def evaluate(results, name):
    """Evaluate model on test data using hyperparameters in results.

    Return dataframe of hyperparameters
    """
    new_results = results.copy()
    # String to dictionary
    new_results['hyperparameters'] = new_results['hyperparameters'].map(ast.literal_eval)
    new_results = new_results.sort_values('score',
                                          ascending=False).reset_index(drop=True)

    # Print out cross validation high score
    text = '{:.5f}, iteration {} Highest Cross validation score from {}'
    print(text.format(new_results.loc[0, 'score'],
                      new_results.loc[0, 'iteration']), name)

    # Use best hyperparameters to create a model
    hyperparameters = new_results.loc[0, 'hyperparameters']
    model = lgb.LGBMClassifier(**hyperparameters)

    # Train and make predictions
    model.fit(train_features, train_labels)
    preds = model.predict_proba(test_features)[:, 1]
    print('{:.5f} ROC AUC from {} on test data'.format(roc_auc_score(test_labels, preds), name))
    # Create dataframe of hyperparameters
    hyp_df = pd.DataFrame(columns = list(new_results.loc[0, 'hyperparameters'].keys()))
    # Iterate through each set of hyperparameters that were evaluated
    for i, hyp in enumerate(new_results['hyperparameters']):
        hyp_df = hyp_df.append(pd.DataFrame(hyp, index=[0]),
                               ignore_index=True)

    # Put the iteration and score in the hyperparameter dataframe
    hyp_df['iteration'] = new_results['iteration']
    hyp_df['score'] = new_results['score']

    return hyp_df


def run_trial(n_steps, loadfile=None):
    """Run trial for first time or from prior run.

    Loadfile should be string of file name, ex: "my_model.hyperopt"
    """
    global ITERATION
    global OUT_FILE_CSV

    now = datetime.datetime.now()
    thedate = str(now.strftime('%Y_%m_%d'))
    headers = ['loss', 'boosting_type', 'n_estimators',
               'hyperparameters', 'iteration', 'runtime', 'score']

    if loadfile is None:
        n_max_evals = n_steps
        ITERATION = 0
        trials = Trials()

        # need to put in headers of new file
        OUT_FILE_prefix = 'bayesian_trials_' + str(n_max_evals) + '_' + thedate
        OUT_FILE_CSV = OUT_FILE_prefix + '.csv'
        of_connection = open(OUT_FILE_CSV, 'w')
        writer = csv.writer(of_connection)
        writer.writerow(headers)
        of_connection.close()
    else:
        trials = pickle.load(open(loadfile + ".hyperopt", "rb"))
        n_max_evals = n_steps + len(trials.trials)
        ITERATION = len(trials.trials)

        # need to copy output from prior trials into new output file
        OUT_FILE_prefix = 'bayesian_trials_' + str(n_max_evals) + '_' + thedate
        OUT_FILE_CSV = OUT_FILE_prefix + '.csv'
        df = pd.read_csv(loadfile + '.csv')
        df[headers].to_csv(OUT_FILE_CSV, index=False)

    best = fmin(fn=objective, space=space, algo=tpe.suggest,
                max_evals=n_max_evals, trials=trials)

    print("Best:", best)

    # clean up output file
    df = pd.read_csv(OUT_FILE_CSV)
    df[headers].to_csv(OUT_FILE_CSV, index=False)

    # Sort the trials with lowest loss (highest AUC) first
    trials_dict = sorted(trials.results, key=lambda x: x['loss'])
    print('\nFinished, best results')
    print(trials_dict[:1])

    # Save the trial results
    with open(OUT_FILE_prefix + '_results.json', 'w') as f:
        f.write(json.dumps(trials_dict))
    # save the trials object
    with open(OUT_FILE_prefix + ".hyperopt", "wb") as f:
        pickle.dump(trials, f)


# %% LOAD DATA
N_FOLDS = 5

features = pd.read_csv('application_train.csv')
features = features.sample(n=16000, random_state=42)
features = features.select_dtypes('number')

# Extract the labels
labels = np.array(features['TARGET'].astype(np.int32)).reshape((-1, ))
features = features.drop(columns=['TARGET', 'SK_ID_CURR'])

# Split into training and testing data
train_features, test_features, train_labels, test_labels =\
    train_test_split(features, labels, test_size=6000, random_state=42)
print('Train shape: ', train_features.shape)
print('Test shape: ', test_features.shape)
train_features.head()

# %% Baseline Model
# First we create a model with the default value of hyperparameters
#  and score it using cross validation with early stopping.
model = lgb.LGBMClassifier(random_state=50)
# Training set
train_set = lgb.Dataset(train_features, label=train_labels)
test_set = lgb.Dataset(test_features, label=test_labels)
# Default hyperparamters
hyperparameters = model.get_params()

# Using early stopping to determine number of estimators.
del hyperparameters['n_estimators']
# Perform cross validation with early stopping
cv_results = lgb.cv(hyperparameters, train_set, num_boost_round=10000,
                    nfold=N_FOLDS, metrics='auc',
                    early_stopping_rounds=100, verbose_eval=False, seed=42)
# Highest score
best = cv_results['auc-mean'][-1]
# Standard deviation of best score
best_std = cv_results['auc-stdv'][-1]

print('{:.5f} with std of {:.5f} Maximium ROC AUC in cross validation'
      .format(best, best_std))
print('{} Ideal number of iterations'.format(len(cv_results['auc-mean'])))

# Optimal number of esimators found in cv
model.n_estimators = len(cv_results['auc-mean'])

# Train and make predicions with model
model.fit(train_features, train_labels)
preds = model.predict_proba(test_features)[:, 1]
baseline_auc = roc_auc_score(test_labels, preds)
print('{:.5f} The baseline model ROC AUC on test set.'.format(baseline_auc))


# %% Entire Domain
# 10 hyperparameters to optimize.
space = {
    'boosting_type': hp.choice('boosting_type',
                               [{'boosting_type': 'gbdt', 'subsample':
                                 hp.uniform('gdbt_subsample', 0.5, 1)},
                                # AVOID DART UNTIL WE FIGURE OUT HOW TO RUN
                                # IN TIME EFFECTIVE MANNER; IS TOO SLOW, IE.,
                                # TAKES TOO LONG
                                # WITHOUT EARLY STOPPING ON MY LAPTOP
                                # {'boosting_type': 'dart', 'subsample':
                                #  hp.uniform('dart_subsample', 0.5, 1)},
                                {'boosting_type': 'goss', 'subsample': 1.0}]),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
    'is_unbalance': hp.choice('is_unbalance', [True, False]),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.5)),

    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'num_leaves': hp.quniform('num_leaves', 20, 150, 1),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'subsample_for_bin': hp.quniform('subsample_for_bin',
                                     20000, 300000, 20000)
}


# %% RUN TRIALS 1
# initial run
run_trial(3)

# %% RUN TRIALS 2
# followup run
run_trial(4, 'bayesian_trials_3_2020_09_20')

"""Code for loading hyperparameter search.

These files were downloaded. My computer would take too long to do the search.
Plot of distributions can be found in model_tuning.py

bayes_results = pd.read_csv('bayesian_trials_simple.csv')
bayes_results = bayes_results.sort_values('score', ascending=False).reset_index()
random_results = pd.read_csv('random_search_trials_simple.csv')
random_results = random_results.sort_values('score', ascending=False).reset_index()
random_results['loss'] = 1 - random_results['score']
bayes_params = evaluate(bayes_results, name='Bayesian')
random_params = evaluate(random_results, name='random')
"""