import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from evaluation import kfold_log_loss, kfold_accuracy

data_path = ['data']
# %%
X = pd.read_table(os.path.join(*data_path, 'SyntheticData_FirstSet.txt'),
    delimiter='   ',
    header=None,
    dtype=float,
    engine='python')

y = pd.read_table(os.path.join(*data_path, 'membership.txt'),
    delimiter='   ',
    names=['motif'],
    dtype=int,
    engine='python')

train_index = y.groupby('motif').sample(frac=.8).index
test_index = ~y.index.isin(train_index)

y = y['motif'].values.reshape(-1)
# %%
