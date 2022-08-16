import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from evaluation import kfold_log_loss
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


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

task = input('Evaluate model or plot confusion matrix? (evaluate/plot)')


if task == 'evaluate':

    test_accuracy = accuracy_score(y_test, measures2_model.predict(X_test))

    print('Test accuracy: ', test_accuracy)

    five_fold_log_loss = kfold_log_loss(measures2_model, X, y)
    print('Log loss: ', five_fold_log_loss)


if task == 'plot':

    print('\n ---------------------------- \n')

    motifs = [57, 50, 32, 29, 53, 8, 44]

    confusion_matrix = confusion_matrix(y_test, measures2_model.predict(X_test), 
                        labels = motifs)

    cm_df = pd.DataFrame(confusion_matrix)
    cm_df = cm_df.rename(columns = {0 : 57, 1 : 50, 2 : 32, 3 : 29,
                                    4 : 53, 5 : 8, 6 : 44},
                        index = {0 : 57, 1 : 50, 2 : 32, 3 : 29,
                                    4 : 53, 5 : 8, 6 : 44})

    cm_array = cm_df.to_numpy()

    wrong = []
    right = []
    row_index = 0

    for row in cm_array:

        num_wrong = 0
        col_index = 0

        for val in row:
            if col_index != row_index:
                num_wrong += val
            else:
                right.append(val)

            col_index += 1


        wrong.append(num_wrong)
        row_index += 1
    
    print(cm_df)
    
#----------------------------------------------
    X = ['1-1-11', '0010', '00-10', '0-1-10',
         '0110', '-11-10', '0100']
    X_axis = np.arange(len(X))

    plt.bar(X_axis - 0.2, right, 0.4, 
            label = '# Correct', color = 'lightblue')
    plt.bar(X_axis + 0.2, wrong, 0.4, 
            label = '# Wrong', color = 'steelblue')

    plt.xticks(X_axis, X)
    plt.xlabel('Motifs')
    plt.ylabel('# of Replicates')
    plt.title('Number of Correct and Wrong Predictions for Second Set Features Model')
    plt.legend()
    plt.show()




