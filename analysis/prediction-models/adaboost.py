import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from evaluation import kfold_log_loss, kfold_accuracy

scores = pd.DataFrame(columns=['Comment', 'Log Loss', 'Accuracy'])

# 2nd set untransformed

data_2 = pd.read_csv('data/secondset-df.csv')
X_2 = data_2.iloc[:, 1:]
y_2 = data_2['motif']


'''
abc = []
for n in range(100, 500, 100):
    clf = AdaBoostClassifier(n_estimators=n)
    abc.append([kfold_log_loss(clf, X_2, y_2), n])
abc = np.array(abc)
n_est = int(abc[np.argmin(abc[:, 0]), 1])

print(n_est)
'''
ada_params = dict(n_estimators = 100,
                  learning_rate = .1)  # not optimized

kwargs_1 = {'model': AdaBoostClassifier(**ada_params),
            'X': X_2,
            'y': y_2}
scores.loc[scores.shape[0], :] = '2nd X', kfold_log_loss(**kwargs_1), kfold_accuracy(**kwargs_1)

# 2nd set transformed

data_2_m = pd.read_csv('data/secondset_measures.csv')
X_2m = data_2_m.iloc[:, 1:]
y_2m = data_2_m['motif']

'''
abc = []
for n in range(100, 500, 100):
    clf = AdaBoostClassifier(n_estimators=n)
    abc.append([kfold_log_loss(clf, X_2m, y_2m), n])
abc = np.array(abc)
n_est = int(abc[np.argmin(abc[:, 0]), 1])

print(n_est)
'''

kwargs_2 = {'model': AdaBoostClassifier(**ada_params),
            'X': X_2m,
            'y': y_2m}
scores.loc[scores.shape[0], :] = '2nd f(X)', kfold_log_loss(**kwargs_2), kfold_accuracy(**kwargs_2)

print(scores.to_markdown())

'''
|    | Comment   |   Log Loss |   Accuracy |
|---:|:----------|-----------:|-----------:|
|  0 | 2nd X     |    1.04099 |   0.755237 |
|  1 | 2nd f(X)  |    1.03125 |   0.739153 |
'''