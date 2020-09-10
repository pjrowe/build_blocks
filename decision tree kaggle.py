"""Implementation and Optimization and Evaluation of Decision Tree Model.

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
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix
from collections import Counter

import itertools
from sklearn.metrics import precision_score, recall_score, roc_auc_score, \
    roc_curve
import matplotlib.pyplot as plt
from sklearn import tree

# %% Function definitions


def evaluate_model(predictions, probs, train_predictions, train_probs):
    """Compare machine learning model to baseline performance.

    Computes statistics and shows ROC curve.
    """
    baseline = {}
    baseline['recall'] = recall_score(test_labels,
                                      np.ones((len(test_labels))))
    baseline['precision'] = precision_score(test_labels,
                                            np.ones((len(test_labels))))
    baseline['roc'] = 0.5
    results = {}

    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['roc'] = roc_auc_score(test_labels, probs)

    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels,
                                                 train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)

    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)}')
        print('Test: {round(results[metric], 2)}')
        print('Train: {round(train_results[metric], 2)}')

    # Calculate false positive rates and true positive rates
    fill = np.ones(len(test_labels))
    base_fpr, base_tpr, _ = roc_curve(test_labels, fill)
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 16

    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label='baseline')
    plt.plot(model_fpr, model_tpr, 'r', label='model')
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
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

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


# %%
RSEED = 50

# Load in data
# df = pd.read_csv('2015.csv')
# https://www.kaggle.com/cdc/behavioral-risk-factor-surveillance-system
df = pd.read_csv('2015.csv').sample(100000, random_state=RSEED)


df = df.select_dtypes('number')
df['_RFHLTH'] = df['_RFHLTH'].replace({2: 0})
df = df.loc[df['_RFHLTH'].isin([0, 1])].copy()
df = df.rename(columns={'_RFHLTH': 'label'})
df['label'].value_counts()
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
# %%
# Imputation of missing values
train = train.fillna(train.mean())
test = test.fillna(test.mean())

# Features for feature importances
features = list(train.columns)

train.shape
test.shape

mytree = DecisionTreeClassifier(random_state=RSEED, max_depth=4)
mytree.fit(train, train_labels)
print(f'Decision tree has {mytree.tree_.node_count} nodes with maximum depth \
      {mytree.tree_.max_depth}.')
print(f'Model Accuracy: {mytree.score(train, train_labels)}')

train_probs = mytree.predict_proba(train)[:, 1]
# probabilities are 1 because this is infinite depth tree with perfect
# classification (overfitting)
probs = mytree.predict_proba(test)[:, 1]

train_predictions = mytree.predict(train)
predictions = mytree.predict(test)

print(f'Train ROC AUC Score: {roc_auc_score(train_labels, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(test_labels, probs)}')
print(f'Baseline ROC AUC: {roc_auc_score(test_labels, np.ones(len(test_labels)))}')

# %% -----------------------------------------------------------------

print(Counter(probs))
print(Counter(predictions))
evaluate_model(predictions, probs, train_predictions, train_probs)

# %%-----------------------------------------------------------------

cm = confusion_matrix(test_labels, predictions)
plot_confusion_matrix(cm, classes=['Poor Health', 'Good Health'],
                      title='Health Confusion Matrix')

fi = pd.DataFrame({'feature': features,
                   'importance': mytree.feature_importances_}).\
                    sort_values('importance', ascending=False)
print(fi.head())
# difficulty walking is 66% importance!!next xlosest is difficulty in physical
# activity at 9.4%

# not readable plot of tree; size of tree is too big even with only 4 deep
# because of all the columns; for big problems, probably better to rely on
# feature importances
tree.plot_tree(mytree)

