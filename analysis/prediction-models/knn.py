import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import GridSearchCV, train_test_split
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


def knn_eval(X, y, k):
    knn = Pipeline([('Scaler', StandardScaler()), ('knn',KNeighborsClassifier(n_neighbors=k))])
    logloss = kfold_log_loss(knn, X=X, y=y)
    acc = kfold_accuracy(knn, X=X, y=y) 
    return logloss, acc


class outcome_table:
    def __init__(self):
        self.table = pd.DataFrame(columns=['', 'Log Loss', 'Accuracy'])
    
    def add_eval(self, comment, eval_return):
        temp = pd.DataFrame([[comment, *eval_return]], columns=self.table.columns)
        self.table = pd.concat([self.table, temp])

    def eval(self, comment, X, y, k):
        eval_return = knn_eval(X, y, k)
        self.add_eval(comment, eval_return)

    def print(self):
        print(self.table.to_markdown())



scores = outcome_table()

### Untransformed X

k1 = get_k(X, y)
scores.eval('X', X, y, k1)

### Tranformed X

X_m = pd.read_csv(os.path.join(*data_path, 'firstset_measures.csv')).drop('entropy', axis=1)

k2 = get_k(X_m, y)
scores.eval('f(X)', X, y, k2)

### Second Set

data_2_m = pd.read_csv('data/secondset_measures.csv')
X_2m = data_2_m.iloc[:, 1:]
y_2m = data_2_m['motif']

'''
X_train, X_test, y_train, y_test = train_test_split(X_2m, y_2m, test_size=.2)
out = []
for k in range(3, 200, 5):
    for n in range(1, X_2m.shape[1] - 1):
        knn = Pipeline([('Scaler', StandardScaler()), ('knn',KNeighborsClassifier(n_neighbors=k))])
        sfs = SequentialFeatureSelector(knn, n_features_to_select=n)
        sfs.fit(X_train, y_train)
        mask = sfs.get_support()
        out.append([kfold_log_loss(knn, X_2m.loc[:, mask], y_2m), mask])
out = np.array(out)
print(int(out[np.argmin(out[:, 0]), 1]))  # [True, False, True, True, True, True, False, True, True, True]
'''
X_masked = X_2m.loc[:, [True, False, True, True, True, True, False, True, True, True]]
k3 = get_k(X_masked, y_2m)
scores.eval('Forward Selection', X_masked, y_2m, k3)

scores.print()

'''
|    |                   |   Log Loss |   Accuracy |
|---:|:------------------|-----------:|-----------:|
|  0 | X                 |   4.87771  |   0.0674   |
|  0 | f(X)              |   4.92302  |   0.0584   |
|  0 | Forward Selection |   0.501257 |   0.777361 |
'''
