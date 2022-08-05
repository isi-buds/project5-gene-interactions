# L2 penalty
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from evaluation import kfold_log_loss

data = pd.read_table('data/SyntheticData_FirstSet.txt', delimiter = '   ', 
                      header = None, engine = 'python')

motifs = pd.read_table('data/membership.txt', delimiter = '   ', 
                      header = None, dtype=int, engine='python')

X = data.iloc[:, 1:]
y = motifs

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

ridge_reg = Ridge(alpha = 1)

ridge_reg.fit(X_train, y_train)

preds = ridge_reg.predict(X_test)

print('predictions: \n', preds)

