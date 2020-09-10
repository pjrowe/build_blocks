"""Implementation and Optimization and Evaluation of Random Forest Model.

Description
-----------
The following file contains code we can reuse as a model for other data.

References
----------
https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76.

/Dropbox/Programming/Coursera%20ml/johns%20hopkins/8%20Machine%20learning/Random%20Forest%20Tutorial.ipynb
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, \
    roc_auc_score, roc_curve


# %% FUNCTION DEFINITIONS---------


def evaluate_model(model_type, predictions, probs, train_predictions,
                   train_probs):
    """Compare machine learning model to baseline performance.

    Computes statistics and shows ROC curve.
    """
    baseline = {}
    baseline['recall'] = recall_score(test_labels,
                                      np.ones((len(test_labels))))
    baseline['precision'] = precision_score(test_labels,
                                            np.ones((len(test_labels))))
    baseline['roc'] = 0.5

    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels,
                                                 train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)

    results = {}
    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['roc'] = roc_auc_score(test_labels, probs)

    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)}')
        print(f'Train: {round(train_results[metric], 2)}')
        print(f'Test: {round(results[metric], 2)}\n')

    # Calculate false positive rates and true positive rates
    fill = np.ones(len(test_labels))
    base_fpr, base_tpr, _ = roc_curve(test_labels, fill)
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 16

    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label='Baseline')
    plt.plot(model_fpr, model_tpr, 'r', label=model_type + ' Model')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Oranges):
    """
    Print and plot the confusion matrix.

    Normalization can be applied by setting `normalize=True`.
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        # cm here is not in orientation I'm used to; here, predictions are
        # across top while actual conidition is in the rows
        # axis = 1 means we sum across columns
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)  # precision = positive predictive value
    recall = tp / (tp + fn)     # recall = sensitivity
    print(f'Test Recall is {round(recall, 2)}')
    print(f'Test Precision is {round(precision, 2)}')

    # Plot the confusion matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size=24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=14)
    plt.yticks(tick_marks, classes, size=14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size=18)
    plt.xlabel('Predicted label', size=18)


# %% LOAD DATA---------
RSEED = 50

# Load in data
# df = pd.read_csv('2015.csv')
# https://www.kaggle.com/cdc/behavioral-risk-factor-surveillance-system
df = pd.read_csv('2015.csv').sample(100000, random_state=RSEED)


df = df.select_dtypes('number')
df['_RFHLTH'] = df['_RFHLTH'].replace({2: 0})
df = df.loc[df['_RFHLTH'].isin([0, 1])].copy()
df = df.rename(columns={'_RFHLTH': 'label'})
# Remove columns with missing values
df = df.drop(columns=['POORHLTH', 'PHYSHLTH', 'GENHLTH', 'PAINACT2',
                      'QLMENTL2', 'QLSTRES2', 'QLHLTH2', 'HLTHPLN1',
                      'MENTHLTH'])

# Extract the labels
labels = np.array(df.pop('label'))
# 30% examples in test data
train, test, train_labels, test_labels = train_test_split(df,
                                                          labels,
                                                          stratify=labels,
                                                          test_size=0.3,
                                                          random_state=RSEED)
# %%  CLEAN DATA---------
# Imputation of missing values
train = train.fillna(train.mean())
test = test.fillna(test.mean())

# Features for feature importances
features = list(train.columns)
train.shape
test.shape

# %%  FIT MODEL ---------

model = RandomForestClassifier(n_estimators=100, random_state=RSEED,
                               max_features='sqrt', n_jobs=-1, verbose=1)
# Fit on training data
model.fit(train, train_labels)

n_nodes = []
max_depths = []

# Stats about the trees in random forest
for ind_tree in model.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)

print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average maximum depth {int(np.mean(max_depths))}\n')
print(round(pd.DataFrame(max_depths, columns=['Max Depth of Trees']).describe(), 1))
print(round(pd.DataFrame(n_nodes, columns=['# Nodes of Trees']).describe(), 1))

# %%  PREDICTIONS ---------

# Training predictions (to demonstrate overfitting)
train_rf_predictions = model.predict(train)
train_rf_probs = model.predict_proba(train)[:, 1]

# Testing predictions (to determine performance)
rf_predictions = model.predict(test)
rf_probs = model.predict_proba(test)[:, 1]

# %%  EVALUATE MODEL, PLOT ROC CURVE AND CONFUSION MATRIX ---------
# Plot formatting
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18

evaluate_model('Random Forest', rf_predictions, rf_probs,
               train_rf_predictions, train_rf_probs)
plt.savefig('roc_auc_curve.png')

# Confusion matrix
cm = confusion_matrix(test_labels, rf_predictions)
plot_confusion_matrix(cm, classes=['Poor Health', 'Good Health'],
                    #  normalize=True,
                      title='RandomForest Health Confusion Matrix')
plt.savefig('cm.png')

fi_model = pd.DataFrame({'feature': features, 'importance':
                         model.feature_importances_})
fi_model = fi_model.sort_values('importance', ascending=False)
fi_model.head(10)

# %%  TUNE MODEL ---------
# Hyperparameter grid
param_grid = {
    'n_estimators': np.linspace(10, 200).astype(int),  # default is 50 #'s
    'max_depth': [None] + list(np.linspace(3, 20).astype(int)),
    'max_features': ['auto', 'sqrt', None] + list(np.arange(0.5, 1, 0.1)),
    'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),
    'min_samples_split': [2, 5, 10],
    'bootstrap': [True, False]
}

# Estimator for use in random search
estimator = RandomForestClassifier(random_state=RSEED)

# Create the random search model
rs = RandomizedSearchCV(estimator, param_grid, n_jobs=-1,
                        scoring='roc_auc', cv=3,
                        n_iter=10, verbose=1, random_state=RSEED)
rs.fit(train, train_labels)

rs.best_params_
best_model = rs.best_estimator_

train_rf_predictions = best_model.predict(train)
train_rf_probs = best_model.predict_proba(train)[:, 1]
rf_predictions = best_model.predict(test)
rf_probs = best_model.predict_proba(test)[:, 1]

n_nodes = []
max_depths = []

for ind_tree in best_model.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)

print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average maximum depth {int(np.mean(max_depths))}')

analysis = pd.DataFrame(n_nodes).describe()
analysis['max_depths'] = pd.DataFrame(max_depths).describe()
# analysis=np.transpose(analysis)
analysis.columns = ['n_nodes', 'max_depth']
print(round(np.transpose(analysis), 1))

evaluate_model('Random Forest tuned', rf_predictions, rf_probs,
               train_rf_predictions, train_rf_probs)

cm = confusion_matrix(test_labels, rf_predictions)
plot_confusion_matrix(cm, classes=['Poor Health', 'Good Health'],
                      title='Health Confusion Matrix')

# there are 29 estimators (trees) in best_model; below we choose 2nd, in
# 1 position
estimator = best_model.estimators_[1]
