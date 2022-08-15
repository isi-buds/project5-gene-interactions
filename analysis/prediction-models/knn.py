import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from evaluation import kfold_log_loss, kfold_accuracy

data_path = ['data']

X = pd.read_table(os.path.join(*data_path, 'SyntheticData_FirstSet.txt'),
    delimiter='   ',
    header=None,
    dtype=float,
    engine='python')

y = pd.read_table(os.path.join(*data_path, 'membership.txt'),
    delimiter='   ',
    names=['motif'],
    dtype=int,
    engine='python')

train_index = y.groupby('motif').sample(frac=.8).index
test_index = ~y.index.isin(train_index)

y = y['motif'].values.reshape(-1)


def get_k(X, y, k_range=range(3, 200, 5)):
    score_k = []
    for k in k_range:
        knn = Pipeline([('Scaler', StandardScaler()), ('knn',KNeighborsClassifier(n_neighbors=k))])
        score = kfold_log_loss(knn, X=X, y=y)
        score_k.append([score, k])
    score_k = np.array(score_k)
    k = int(score_k[np.argmin(score_k[:, 0]), 1])
    return k


### Untransformed X

k = get_k(X, y)
knn = Pipeline([('Scaler', StandardScaler()), ('knn',KNeighborsClassifier(n_neighbors=k))])

logloss_1 = kfold_log_loss(knn, X=X, y=y)
print('Untransformed X\nLogLoss: %s' % logloss_1)

acc_1 = kfold_accuracy(knn, X=X, y=y)
print('Accuracy: %s' % acc_1)

### Tranformed X

X_m = pd.read_csv(os.path.join(*data_path, 'firstset_measures.csv')).drop('entropy', axis=1)

k = get_k(X_m, y)
knn = Pipeline([('Scaler', StandardScaler()), ('knn',KNeighborsClassifier(n_neighbors=k))])

logloss_2 = kfold_log_loss(knn, X=X_m, y=y)
print('\nTransformed X\nLogLoss: %s' % logloss_2)

acc_2 = kfold_accuracy(knn, X=X_m, y=y)
print('Accuracy: %s' % acc_2)