import numpy as np
import pandas as pd
import os
import h5py
from scipy.sparse import csr_matrix

data_path = ['data']


def get_mean(matrix):
    x_prob = np.apply_along_axis(np.sum, 0, matrix)
    y_prob = np.apply_along_axis(np.sum, 1, matrix)

    x_i = np.array(range(matrix.shape[1]))
    y_i = np.array(range(matrix.shape[0]))
    
    x_mean = np.dot(x_prob, x_i)
    y_mean = np.dot(y_prob, y_i)
    return (x_mean, y_mean)


def get_sd(matrix):
    x_mean, y_mean = get_mean(matrix)

    x_prob = np.apply_along_axis(np.sum, 0, matrix)
    y_prob = np.apply_along_axis(np.sum, 1, matrix)

    x_i = np.array(range(matrix.shape[1]))
    y_i = np.array(range(matrix.shape[0]))

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

    x_diff = np.array(range(matrix.shape[1])) - x_mean
    y_diff = np.array(range(matrix.shape[0])) - y_mean 

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
    Hx=-np.dot(x,np.log(x))
    Hy=-np.dot(y,np.log(y))
    mut_info=Hx+Hy-entropy
    return entropy, mut_info


def fano(matrix):
    x_mean, y_mean = get_mean(matrix)
    x_mean += 0.0000001
    y_mean += 0.0000001
    x_sd, y_sd = get_sd(matrix)
    return (x_sd ** 2 / x_mean, y_sd ** 2 / y_mean)


firstset_input = input("Create measures for first set?(Y/n) ")

if firstset_input == 'Y':
    data_1 = pd.read_table(os.path.join(*data_path, 'SyntheticData_FirstSet.txt'),
                        delimiter='   ',
                        header=None,
                        engine='python')

    data_1_m = pd.DataFrame(columns=['x_mean', 'y_mean', 'x_sd', 'y_sd', 'corr', 'coexpress_index', 'entropy', 'mutual_info', 'x_fano', 'y_fano'])
    for i in range(data_1.shape[0]):
        A = data_1.loc[i].to_numpy().reshape(51,51, order='F')
        data_1_m.loc[i, ['x_mean', 'y_mean']] = get_mean(A)
        data_1_m.loc[i, ['x_sd', 'y_sd']] = get_sd(A)
        data_1_m.loc[i, 'corr'] = get_corr(A)
        data_1_m.loc[i, 'coexpress_index'] = coexpression_index(A)
        data_1_m.loc[i, ['entropy', 'mutual_info']] = entropy_mut_info(A)
        data_1_m.loc[i, ['x_fano', 'y_fano']] = fano(A)

    data_1_m.to_csv(os.path.join(*data_path, 'firstset_measures.csv'), index=False)
    print('firstset_measures.csv created')

secondset_input = input("Create measures for second set?(Y/n) ")

if secondset_input == 'Y':
    file_names_and_membership = {'1-1-11_probdist' : 57, '0010_probdist' : 50,
                             '00-10_probdist' : 32, '0-1-10_probdist' : 29,
                             '0110_probdist' : 53, '-11-10_probdist' : 8,
                             '0100_probdist' : 44}

    data_2_m = pd.DataFrame(columns=['motif', 'x_mean', 'y_mean', 'x_sd', 'y_sd', 'corr', 'coexpress_index', 'entropy', 'mutual_info', 'x_fano', 'y_fano'])

    i = 0
    for file, motif in file_names_and_membership.items():

        hdf = h5py.File('data/'+file+'.h5', mode = 'r')
        
        for key in hdf.keys():
            
            if key != 'parameterset':

                df = hdf[key]

                col = df['col']
                probdist = np.array(df['probdist'])
                row = df['row']

                if probdist.shape != (0,):
                    A = csr_matrix((probdist, (row, col))).toarray()
                    data_2_m.loc[i, 'motif'] = int(motif)
                    data_2_m.loc[i, ['x_mean', 'y_mean']] = get_mean(A)
                    data_2_m.loc[i, ['x_sd', 'y_sd']] = get_sd(A)
                    data_2_m.loc[i, 'corr'] = get_corr(A)
                    data_2_m.loc[i, 'coexpress_index'] = coexpression_index(A)
                    data_2_m.loc[i, ['entropy', 'mutual_info']] = entropy_mut_info(A)
                    data_2_m.loc[i, ['x_fano', 'y_fano']] = fano(A)
                    i += 1

    data_2_m.to_csv(os.path.join(*data_path, 'secondset_measures.csv'), index=False)
    print('secondset_measures.csv created')