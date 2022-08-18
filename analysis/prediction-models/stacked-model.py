import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from evaluation import kfold_log_loss, kfold_accuracy

scores = pd.DataFrame(columns=['Comment', 'Log Loss', 'Accuracy'])

data_2_m = pd.read_csv('data/secondset_measures.csv')
X_2m = data_2_m.iloc[:, 1:]
y_2m = data_2_m['motif']

class stacked:
    def __init__(self, rfc_2_params:dict):
        self.mlr_1_params = dict(solver = 'saga', max_iter = 200, multi_class = 'multinomial')
        self.rfc_1_params = {'max_depth': 10, 'max_features': 0.5, 'n_estimators': 500}
        self.knn_1_k = 73
        self.ada_1_params = {'n_estimators': 100,
                             'learning_rate': 0.1}
        self.rfc_2_params = rfc_2_params
    def get_input_2(self, X):
        '''
        Input for layer 2
        '''
        input_2 = np.hstack([m.predict_proba(X) for m in self.base])
        return input_2
    def fit(self, X, y):
        # layer 1
        self.mlr_1 = LogisticRegression(**self.mlr_1_params)
        self.rfc_1 = RandomForestClassifier(**self.rfc_1_params, n_jobs=-1)
        self.knn_1 = Pipeline([('Scaler', StandardScaler()), ('knn',KNeighborsClassifier(n_neighbors=self.knn_1_k))])
        self.ada_1 = AdaBoostClassifier(**self.ada_1_params)
        self.base = [self.mlr_1, self.rfc_1, self.knn_1, self.ada_1]
        # fit layer 1
        for m in self.base:
            m.fit(X, y)
        # layer 2
        input_2 = self.get_input_2(X)
        self.final_m = RandomForestClassifier(**self.rfc_2_params, n_jobs=-1)
        self.final_m.fit(input_2, y)
    def predict(self, X):
        input_2 = self.get_input_2(X)
        predict = self.final_m.predict(input_2)
        return predict
    def predict_proba(self, X):
        input_2 = self.get_input_2(X)
        predict_proba = self.final_m.predict_proba(input_2)
        return predict_proba

'''
abc = []
for n in range(100, 500, 100):
    model = stacked(ada_2_n=n)
    abc.append([kfold_log_loss(model, X_2m, y_2m), n])
abc = np.array(abc)
n_est = int(abc[np.argmin(abc[:, 0]), 1])
print(n_est)
'''
rfc_2_params = {'max_depth': 10, 'max_features': 0.5, 'n_estimators': 500}

kwargs_1 = {'model': stacked(rfc_2_params=rfc_2_params),
            'X': X_2m,
            'y': y_2m}
scores.loc[scores.shape[0], :] = '2nd X', kfold_log_loss(**kwargs_1), kfold_accuracy(**kwargs_1)

print(scores.to_markdown())
