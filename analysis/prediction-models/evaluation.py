import numpy as np


def kfold_log_loss(model, X=None, y=None, n_splits=5):
    '''
    Returns the mean of a number (five by default) log-loss for the method.
        model: must have the methods fit(X,y) and predict_proba(X,y) to work
        X: endogenous variables or predictors
        y: exogenous variable or outcome
        n_splits: the number of times to find the log-loss. Default is five.
    '''
    from sklearn.model_selection import KFold
    from sklearn.metrics import log_loss
    X = np.array(X)
    y = np.array(y).reshape(-1)
    kf = KFold(n_splits=n_splits, shuffle=True)
    kf.get_n_splits(X)
    log_l = []
    for train_index, test_index in kf.split(X):
        model.fit(X[train_index], y[train_index])
        log_l.append(
            log_loss(y[test_index], model.predict_proba(X[test_index]))
        )
    return np.mean(log_l)


def kfold_accuracy(model, X=None, y=None, n_splits=5):
    '''
    Returns the mean of a number (five by default) accuracy for the method.
        model: must have the methods fit(X,y) and predict_proba(X,y) to work
        X: endogenous variables or predictors
        y: exogenous variable or outcome
        n_splits: the number of times to find the log-loss. Default is five.
    '''
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score
    X = np.array(X)
    y = np.array(y).reshape(-1)
    kf = KFold(n_splits=n_splits, shuffle=True)
    kf.get_n_splits(X)
    acc = []
    for train_index, test_index in kf.split(X):
        model.fit(X[train_index], y[train_index])
        acc.append(
            accuracy_score(y[test_index], model.predict(X[test_index]))
        )
    return np.mean(acc)
