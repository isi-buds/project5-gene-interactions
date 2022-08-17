# %%
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from evaluation import kfold_log_loss, kfold_accuracy

data_path = ['data']


def get_params(X, y, params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    GSRF = GridSearchCV(RandomForestClassifier(), params, scoring='neg_log_loss', n_jobs=-1)
    GSRF.fit(X_train, y_train)
    return GSRF.best_params_


def rfc_eval(X, y, params):
    rfc = RandomForestClassifier(**params, n_jobs=-1)
    logloss = kfold_log_loss(rfc, X=X, y=y)
    acc = kfold_accuracy(rfc, X=X, y=y) 
    return logloss, acc
    

class outcome_table:
    def __init__(self):
        self.table = pd.DataFrame(columns=['', 'Log Loss', 'Accuracy'])
    
    def add_eval(self, comment, eval_return):
        temp = pd.DataFrame([[comment, *eval_return]], columns=self.table.columns)
        self.table = pd.concat([self.table, temp])

    def eval(self, comment, X, y, params):
        eval_return = rfc_eval(X, y, params)
        self.add_eval(comment, eval_return)

    def print(self):
        print(self.table.to_markdown())
    

scores = outcome_table()

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

y = y['motif'].values.reshape(-1)

# %%

### Untransformed X, unreduced y

'''
parameters = dict(n_estimators = [100, 200, 500],
                  max_depth = [4, 10, 50],
                  max_features = [.5, .7, .8])

print(get_params(X, y, parameters))  # {'max_depth': 10, 'max_features': 0.5, 'n_estimators': 500}
'''
rfc_params_0 = {'max_depth': 10, 'max_features': 0.5, 'n_estimators': 500}
scores.eval('Untransformed X, Unreduced y', X, y, rfc_params_0)

# %% 

### Reduced y

motifs_of_interest = [57, 50, 32, 29, 53, 8, 44]
subset = [ index for index, y_i in enumerate(y) if y_i in motifs_of_interest]

'''
parameters = dict(n_estimators = [100, 200, 500],
                  max_depth = [4, 10, 50],
                  max_features = [.5, .7, .8])

print(get_params(X.loc[subset], y[subset], parameters))  # {'max_depth': 10, 'max_features': 0.5, 'n_estimators': 200}
'''

rfc_params_1 = {'max_depth': 10, 'max_features': 0.5, 'n_estimators': 200}
scores.eval('Untransformed X, reduced y', X.loc[subset], y[subset], rfc_params_1)

# %%

### Transformed X

X_m = pd.read_csv(os.path.join(*data_path, 'firstset_measures.csv'))

'''
parameters = dict(n_estimators = [100, 500, 1000],
                  max_depth = [2,4,10],
                  max_features = [.25, .5, .75])

print(get_params(X_m, y, parameters))  # {'max_depth': 10, 'max_features': 0.75, 'n_estimators': 1000}
'''

rfc_params_2 = {'max_depth': 10, 'max_features': 0.5, 'n_estimators': 1000}
scores.eval('Transformed X, unreduced y', X_m, y, rfc_params_2)

# %%

### Transformed X, reduced y

'''
parameters = dict(n_estimators = [100, 500, 1000],
                  max_depth = [2,4,10],
                  max_features = [.25, .5, .75])

print(get_params(X_m.loc[subset], y[subset], parameters))  # {'max_depth': 10, 'max_features': 0.75, 'n_estimators': 1000}
'''

rfc_params_3 = {'max_depth': 10, 'max_features': 0.25, 'n_estimators': 500}
scores.eval('Transformed X, reduced y', X_m.loc[subset], y[subset], rfc_params_3)

# %%

### Second set

# Untransformed

data_2 = pd.read_csv('data/secondset-df.csv')
X_2 = data_2.iloc[:, 1:]
y_2 = data_2['motif']

'''
parameters = dict(n_estimators = [100, 500],
                  max_depth = [2,4,10],
                  max_features = [.1, .25, .5])

print(get_params(X_2, y_2, parameters))  # {'max_depth': 10, 'max_features': 0.25, 'n_estimators': 500}
'''

rfc_params_4 = {'max_depth': 10, 'max_features': 0.25, 'n_estimators': 500}
scores.eval('Second Set untransformed', X_2, y_2, rfc_params_4)

# %%

# Measures
data_2_m = pd.read_csv('data/secondset_measures.csv')
X_2m = data_2_m.iloc[:, 1:]
y_2m = data_2_m['motif']


'''
parameters = dict(n_estimators = [100, 500, 1000],
                  max_depth = [2,4,10],
                  max_features = [.25, .5, .75])

print(get_params(X_2m, y_2m, parameters))  # {'max_depth': 10, 'max_features': 0.5, 'n_estimators': 500}
'''

rfc_params_5 = {'max_depth': 10, 'max_features': 0.5, 'n_estimators': 500}
scores.eval('Second Set transformed', X_2m, y_2m, rfc_params_5)

# %%
scores.print()

'''
                             | Log Loss | Accuracy
-----------------------------|---------:|--------:
Untransformed X, unreduced y | 3.2901   | 0.1422  
Untransformed X, reduced y   | 1.2368   | 0.5322  
Transformed X, unreduced y   | 2.5697   | 0.2592  
Transformed X, reduced y     | 0.8240   | 0.6852  
Second Set untransformed     | 0.4969   | 0.8009  
Second Set transformed       | 0.2988   | 0.8544  
'''
