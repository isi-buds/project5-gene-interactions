# %%
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from evaluation import kfold_log_loss, kfold_accuracy

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

### Untransformed X, unreduced y

'''
parameters = dict(n_estimators = [100, 200, 500],
                  max_depth = [4, 10, 50],
                  max_features = [.5, .7, .8])

GSRF = GridSearchCV(RandomForestClassifier(), parameters, scoring='neg_log_loss', n_jobs=-1)
GSRF.fit(X.loc[train_index], y[train_index])
print(GSRF.best_params_)  # {'max_depth': 10, 'max_features': 0.5, 'n_estimators': 500}
'''
rfc_params_1 = {'max_depth': 10, 'max_features': 0.5, 'n_estimators': 500}
rfc_1 = RandomForestClassifier(**rfc_params_1, n_jobs=-1)

logloss_1 = kfold_log_loss(rfc_1, X=X, y=y) # 3.3081473849185734
print('Random Forest with Untransformed X, Unreduced y\nLogLoss: %s' % logloss_1)

rfc_1.fit(X.loc[train_index], y[train_index])

acc_1 = kfold_accuracy(rfc_1, X=X, y=y) 
print('Accuracy: %s' % acc_1) # 0.1366

# %% 

### Reduced y

motifs_of_interest = [57, 50, 32, 29, 53, 8, 44]
subset = [ index for index, y_i in enumerate(y) if y_i in motifs_of_interest]
red_train = [ index in train_index for index in subset ]
red_test = [ test_index[index] for index in subset ]

'''
parameters = dict(n_estimators = [100, 200, 500],
                  max_depth = [4, 10, 50],
                  max_features = [.5, .7, .8])

GSRF = GridSearchCV(RandomForestClassifier(), parameters, scoring='neg_log_loss', n_jobs=-1)
GSRF.fit(X.loc[subset].loc[red_train], y[subset][red_train])
print(GSRF.best_params_)  # {'max_depth': 10, 'max_features': 0.5, 'n_estimators': 200}
'''

rfc_params_2 = {'max_depth': 10, 'max_features': 0.5, 'n_estimators': 200}
rfc_2 = RandomForestClassifier(**rfc_params_2, n_jobs=-1)

logloss_2 = kfold_log_loss(rfc_2, X=X.loc[subset], y=y[subset])
print('\nReduced y\nLogLoss: %s' % logloss_2) # 1.1978847937607144

rfc_2.fit(X.loc[subset].loc[red_train], y[subset][red_train])

acc_2 = kfold_accuracy(rfc_2, X=X.loc[subset], y=y[subset])
print('Accuracy: %s' % acc_2) # 0.518631381983427

# %%

### Transformed X

X_m = pd.read_csv(os.path.join(*data_path, 'firstset_measures.csv'))

'''
parameters = dict(n_estimators = [100, 500, 1000],
                  max_depth = [2,4,10],
                  max_features = [.5, .7, .8])

GSRF = GridSearchCV(RandomForestClassifier(), parameters, scoring='neg_log_loss', n_jobs=-1)
GSRF.fit(X_m.loc[train_index], y[train_index])
print(GSRF.best_params_)  # {'max_depth': 10, 'max_features': 0.75, 'n_estimators': 1000}
'''

rfc_params_3 = {'max_depth': 10, 'max_features': 0.5, 'n_estimators': 1000}
rfc_3 = RandomForestClassifier(**rfc_params_3, n_jobs=-1)

logloss_3 = kfold_log_loss(rfc_3, X=X_m, y=y)  
print('\nTransformed X\nLogLoss: %s' % logloss_3) # 2.572978552599186

rfc_3.fit(X_m.loc[train_index], y[train_index])

acc_3 = kfold_accuracy(rfc_3, X=X_m, y=y)
print('Accuracy: %s' % acc_3) # 0.24980000000000002

# %%

### Transformed X, reduced y

'''
parameters = dict(n_estimators = [100, 500, 1000],
                  max_depth = [2,4,10],
                  max_features = [.5, .7, .8])

GSRF = GridSearchCV(RandomForestClassifier(), parameters, scoring='neg_log_loss', n_jobs=-1)
GSRF.fit(X.loc[subset].loc[red_train], y[subset][red_train])
print(GSRF.best_params_)  # {'max_depth': 10, 'max_features': 0.75, 'n_estimators': 1000}
'''

rfc_params_4 = {'max_depth': 10, 'max_features': 0.25, 'n_estimators': 500}
rfc_4 = RandomForestClassifier(**rfc_params_4, n_jobs=-1)

logloss_4 = kfold_log_loss(rfc_4, X=X_m.loc[subset], y=y[subset])
print('\nTransformed X, reduced y\nLogLoss: %s' % logloss_4) # 0.8139001082647404

rfc_4.fit(X_m.loc[subset].loc[red_train], y[subset][red_train])

acc_4 = kfold_accuracy(rfc_4, X=X_m.loc[subset], y=y[subset])
print('Accuracy: %s' % acc_4) # 0.673750334135258

# %%
