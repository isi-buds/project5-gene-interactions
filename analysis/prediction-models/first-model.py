# multinomial-log-model
import pandas as pd
import numpy as np
from fit_multi_logistic import fit_multi_log_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

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

#fit_multi_log_model(X = X, y = y, solver = 'sag', max_iter = 200, 
#                    multi_class = 'multinomial',
#                    penalty = 'l2', test_size = 0.33, X2 = X2)

print('\n ---------------------------- \n')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, 
                                                        random_state=42)

first_model = LogisticRegression(solver = 'sag', max_iter = 200,
                                           multi_class = 'multinomial')

first_model.fit(X_train, y_train)

confusion_matrix = confusion_matrix(y_test, first_model.predict(X_test))

cm_df = pd.DataFrame(confusion_matrix)
#cm_df = cm_df.rename(columns = motifs.values.tolist,
#xa                     index = motifs.values.tolist())

print(cm_df)
