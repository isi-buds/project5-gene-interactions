# %%
import numpy as np
import pandas as pd
import os

data_path = ['data']
# %%
data_1 = pd.read_table(os.path.join(*data_path, 'SyntheticData_FirstSet.txt'),
                       delimiter='   ',
                       header=None,
                       engine='python')

data_array = data_1.loc[:, 0:].to_numpy().reshape((-1, 51, 51))
# %%


def get_mean(matrix):
    x_prob = np.apply_along_axis(np.sum, 0, matrix)
    y_prob = np.apply_along_axis(np.sum, 1, matrix)

    x_i = np.array(range(matrix.shape[0]))
    y_i = np.array(range(matrix.shape[1]))
    
    x_mean = np.dot(x_prob, x_i)
    y_mean = np.dot(y_prob, y_i)
    return (x_mean, y_mean)


def get_sd(matrix):
    x_mean, y_mean = get_mean(matrix)

    x_prob = np.apply_along_axis(np.sum, 0, matrix)
    y_prob = np.apply_along_axis(np.sum, 1, matrix)

    x_i = np.array(range(matrix.shape[0]))
    y_i = np.array(range(matrix.shape[1]))

    x_diff_sq = (x_i - x_mean) ** 2
    y_diff_sq = (y_i - y_mean) ** 2

    x_var = np.dot(x_prob, x_diff_sq)
    y_var = np.dot(y_prob, y_diff_sq)
    return (x_var ** .5, y_var ** .5)


def get_corr(matrix):
    x_mean, y_mean = get_mean(matrix)
    x_sd, y_sd = get_sd(matrix)
    x_sd += 0.0000001
    y_sd += 0.0000001

    x_diff = np.array(range(matrix.shape[0])) - x_mean
    y_diff = np.array(range(matrix.shape[1])) - y_mean 

    cov = 0
    for i in range(len(x_diff)):
        for j in range(len(y_diff)):
            cov += x_diff[i] * y_diff[j] * matrix[j, i]
    return cov / (x_sd * y_sd)


def coexpression_index(matrix):
    interset = matrix[1:, 1:].sum()
    union = matrix.sum() - matrix[0, 0]
    if union==0:
        return 0
    return interset/union


def entropy_mut_info(probdist):
    probdist += 0.0000001
    probdist_scaled=probdist/np.sum(probdist)
    probdist_log=np.log(probdist_scaled)
    entropy=-np.sum(np.multiply(probdist_scaled,probdist_log))
    x=np.sum(probdist_scaled,axis=0)
    y=np.sum(probdist_scaled,axis=1)
    px=-np.dot(x,np.log(x))
    py=-np.dot(y,np.log(y))
    mut_info=px+py-entropy
    return entropy, mut_info


def fano(matrix):
    x_mean, y_mean = get_mean(matrix)
    x_mean += 0.0000001
    y_mean += 0.0000001
    x_sd, y_sd = get_sd(matrix)
    return (x_sd ** 2 / x_mean, y_sd ** 2 / y_mean)


# %%
data_measures = pd.DataFrame(columns=['x_mean', 'y_mean', 'x_sd', 'y_sd', 'corr', 'coexpress_index', 'entropy', 'mutual_info', 'x_fano', 'y_fano'])
for i in range(data_array.shape[0]):
    data_measures.loc[i, ['x_mean', 'y_mean']] = get_mean(data_array[i])
    data_measures.loc[i, ['x_sd', 'y_sd']] = get_sd(data_array[i])
    data_measures.loc[i, 'corr'] = get_corr(data_array[i])
    data_measures.loc[i, 'coexpress_index'] = coexpression_index(data_array[i])
    data_measures.loc[i, ['entropy', 'mutual_info']] = entropy_mut_info(data_array[i])
    data_measures.loc[i, ['x_fano', 'y_fano']] = fano(data_array[i])

data_measures.to_csv(os.path.join(*data_path, 'firstset_measures.csv'))
# %%
