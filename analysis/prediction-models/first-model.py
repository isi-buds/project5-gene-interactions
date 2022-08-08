# multinomial-log-model
import pandas as pd
import numpy as np
from fit_multi_logistic import fit_multi_log_model

data = pd.read_table('data/SyntheticData_FirstSet.txt', delimiter = '   ', 
                      header = None, engine = 'python')

motifs = pd.read_table('data/membership.txt', delimiter = '   ', 
                      header = None, dtype=int, engine='python')


# took out first column in attempt to solve multicollinearity
X = data.iloc[:, 1:] 
#print('X shape: ', X.shape)
y = motifs
#print('y shape: ', y.shape)

X2 = np.full((5000, 1), 1)

fit_multi_log_model(X = X, y = y, solver = 'sag', max_iter = 200, 
                    multi_class = 'multinomial', eval_type = 'Cross entropy',
                    penalty = 'l2', test_size = 0.33, X2 = X2)

