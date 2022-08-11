# %%
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from evaluation import kfold_log_loss

data_path = ['data']
# %%
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

# %%
'''
parameters = dict(n_estimators = [100, 500, 1000],
                  max_depth = [2,4,10],
                  max_features = [.5, .7, .8])

GSRF = GridSearchCV(RandomForestClassifier(), parameters, n_jobs=-1)
GSRF.fit(X.loc[train_index], y.loc[train_index].values.reshape(-1))
print(GSRF.best_params_)  # {'max_depth': 10, 'max_features': 0.5, 'n_estimators': 100}
'''
rfc_params_1 = {'max_depth': 10, 'max_features': 0.5, 'n_estimators': 100}
rfc_1 = RandomForestClassifier(**rfc_params_1, n_jobs=-1)

logloss_1 = kfold_log_loss(rfc_1, X=X, y=y)  # 3.337138180175846
print('Random Forest with Untransformed X, Unreduced y\nLogLoss: %s' % logloss_1)

rfc_1.fit(X.loc[train_index], y[train_index])

acc_1 = accuracy_score(y[test_index], rfc_1.predict(X.loc[test_index]))  # 0.1334002006018054
print('Accuracy: %s' % acc_1)
# %% Reduced number of outcomes
motifs_of_interest = [57, 50, 32, 29, 53, 8, 44]
y_new = [ y_i for y_i in y if y_i in motifs_of_interest]
X_new_index = [ index for index, y_i in enumerate(y) if y_i in motifs_of_interest]



'''
parameters = dict(n_estimators = [100, 500, 1000],
                  max_depth = [2,4,10],
                  max_features = [.5, .7, .8])

GSRF = GridSearchCV(RandomForestClassifier(), parameters, n_jobs=-1)
GSRF.fit(X.loc[train_index], y_new[train_index])
print(GSRF.best_params_)  # 
'''
rfc_2 = RandomForestClassifier(**rfc_params_1, n_jobs=-1)

logloss_2 = kfold_log_loss(rfc_2, X=X, y=y_new)
print('\nReduced y\nLogLoss: %s' % logloss_2)

rfc_2.fit(X.loc[train_index], y[train_index])
acc_2 = accuracy_score(y[test_index], rfc_2.predict(X.loc[test_index]))  # 0.1334002006018054
print('Accuracy: %s' % acc_2)

# %% Transformed X
X_m = pd.read_csv(os.path.join(*data_path, 'firstset_measures.csv'))

'''
parameters = dict(n_estimators = [100, 500, 1000],
                  max_depth = [2,4,10],
                  max_features = [.5, .7, .8])

GSRF = GridSearchCV(RandomForestClassifier(), parameters, n_jobs=-1)
GSRF.fit(X_m.loc[train_index], y_new[train_index])
print(GSRF.best_params_)  # 
'''
