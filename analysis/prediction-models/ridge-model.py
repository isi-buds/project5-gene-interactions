# L2 penalty
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_table('data/SyntheticData_FirstSet.txt', delimiter = '   ', 
                      header = None, engine = 'python')

motifs = pd.read_table('data/membership.txt', delimiter = '   ', 
                      header = None, dtype=int, engine='python')

X = data.iloc[:, 1:]
y = motifs

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

alphas = 10**np.linspace(10, -2, 100) * 0.5

ridgecv_reg = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', normalize = True)
ridgecv_reg.fit(X_train, y_train)

#print(ridgecv_reg.alpha_)

ridge_reg = Ridge(alpha = ridgecv_reg.alpha_, normalize = True)
ridge_reg.fit(X_train, y_train)

# get squared error loss
mse = mean_squared_error(y_test, ridge_reg.predict(X_test))

print('MSE: ', mse)

