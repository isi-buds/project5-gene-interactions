import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from evaluation import kfold_log_loss

# preconditions: y must be a 1d array or column or row vector,
# eval_type must be either 'Accuracy' or 'Cross entropy'
def fit_multi_log_model(X: pd.DataFrame, y: pd.DataFrame, solver: str, max_iter: int, 
                        multi_class: str, penalty: str,
                        test_size: float, X2) -> None:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, 
                                                        random_state=42)

    model = LogisticRegression(penalty = penalty, solver = solver,
                               max_iter = max_iter, multi_class = multi_class)
    
    model.fit(X_train, y_train.values.ravel())

    #preds = model.predict(X_test)

    #print('\nPredictions: ', preds)

    #model_probs = model.predict_proba(X_test)

    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))

    print('Train accuracy: ', train_accuracy)
    print('Test accuracy: ', test_accuracy)

    model_baseline = LogisticRegression(penalty = penalty, solver = solver, 
                                            max_iter = max_iter, multi_class = multi_class)

    five_fold_log_loss = kfold_log_loss(model, X, y)
    baseline_log_loss = kfold_log_loss(model_baseline, X2, y)
    print('Five fold log loss for model 1: ', five_fold_log_loss)
    print('Log loss for baseline model: ', baseline_log_loss)
