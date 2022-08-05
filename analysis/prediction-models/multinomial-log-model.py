# multinomial-log-model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_table('data/SyntheticData_FirstSet.txt', delimiter = '   ', 
                      header = None, engine = 'python')

motifs = pd.read_table('data/membership.txt', delimiter = '   ', 
                      header = None, dtype=int, engine='python')

X = data.iloc[:, 1:]
y = motifs

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model1 = LogisticRegression(solver = 'sag', max_iter = 200,
                            multi_class = 'multinomial')

model1.fit(X_train, y_train.values.ravel())

preds = model1.predict(X_test)

