# multinomial-log-model
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

data = pd.read_table('data/SyntheticData_FirstSet.txt', delimiter = '   ', 
                      header = None, engine = 'python')

motifs = pd.read_table('data/membership.txt', delimiter = '   ', 
                      header = None, dtype=int, engine='python')

X = data.iloc[:, 1:]
#print('X shape: ', X.shape)
y = motifs
#print('y shape: ', y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model1 = LogisticRegression(solver = 'sag', max_iter = 200,
                            multi_class = 'multinomial')

model1.fit(X_train, y_train.values.ravel())

preds = model1.predict(X_test)

print('predictions: ', preds)

model1_probs = model1.predict_proba(X_test)
#print(model1_probs.shape)

train_accuracy = accuracy_score(y_train, model1.predict(X_train))
test_accuracy = accuracy_score(y_test, model1.predict(X_test))

print('Train accuracy: ', train_accuracy)
print('Test accuracy: ', test_accuracy)

baseline = np.full((1650, 81), 1/81)

cross_entropy = log_loss(y_test, model1_probs)

baseline_cross_entropy = log_loss(y_test, baseline)

print('Loss for model 1: ', cross_entropy)
print('Loss for baseline: ', baseline_cross_entropy)
