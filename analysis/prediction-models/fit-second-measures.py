import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from evaluation import kfold_log_loss
from sklearn.metrics import confusion_matrix

data_path = ['data']

second_measures = pd.read_csv('data/secondset_measures.csv')

prob_df = pd.read_csv('data/secondset-prob-df.csv')

X = second_measures

y = prob_df.iloc[:, 0]

X2 = np.full((3481, 1), 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, 
                                                        random_state=42)

measures2_model = LogisticRegression(solver = 'saga', max_iter = 200,
                                           multi_class = 'multinomial')

measures2_model.fit(X_train, y_train)

test_accuracy = accuracy_score(y_test, measures2_model.predict(X_test))

print('Test accuracy: ', test_accuracy)

five_fold_log_loss = kfold_log_loss(measures2_model, X, y)
print('Log loss: ', five_fold_log_loss)

print('\n ---------------------------- \n')

motifs = [57, 50, 32, 29, 53, 8, 44]

confusion_matrix = confusion_matrix(y_test, measures2_model.predict(X_test), 
                       labels = motifs)

cm_df = pd.DataFrame(confusion_matrix)
cm_df = cm_df.rename(columns = {0 : 57, 1 : 50, 2 : 32, 3 : 29,
                                4 : 53, 5 : 8, 6 : 44},
                     index = {0 : 57, 1 : 50, 2 : 32, 3 : 29,
                                4 : 53, 5 : 8, 6 : 44})

print(cm_df)

