import pandas as pd
import numpy as np
from fit_multi_logistic import fit_multi_log_model

dist_df = pd.read_csv('data/secondset-df.csv')
prob_df = pd.read_csv('data/secondset-prob-df.csv')

X_rep = dist_df.iloc[:, 1:]
y_rep = dist_df.iloc[:, 0]

X2 = np.full((3481, 1), 1)

fit_multi_log_model(X = X_rep, y = y_rep, solver = 'saga', max_iter = 200, 
                    multi_class = 'multinomial', penalty = 'l1', test_size = 0.33, 
                    X2 = X2)

print('\n ------------------------- \n')
X_prob = prob_df.iloc[:, 1:]
y_prob = prob_df.iloc[:, 0]

fit_multi_log_model(X = X_prob, y = y_prob, solver = 'saga', max_iter = 200, 
                    multi_class = 'multinomial', penalty = 'l1', test_size = 0.33, 
                    X2 = X2)

