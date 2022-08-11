import h5py
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# missing motif 32 and 8
file_names_and_membership = {'1-1-11_probdist' : 57, '0100_probdist' : 50,
                      '0-1-10_probdist' : 29, '0110_probdist' : 53,
                      '00-10_probdist' : 44}

def convert_matrix_to_1D(csr) -> list[float]:
    arr = []

    for x in csr:
        for y in x:
            arr.append(y)

    return arr

def create_dist_df(filenames) -> None:

    dist_df = pd.DataFrame()
    
    for file, motif in filenames.items():

        hdf = h5py.File('data/'+file+'.h5', mode = 'r')
        
        for key in hdf.keys():
            
            if key != 'parameterset':

                df = hdf[key]

                col = df['col']
                probdist = df['probdist']
                row = df['row']

                if probdist.shape != (0,):
                    prob_matrix = csr_matrix((probdist, (row, col)))
                    prob_arr = convert_matrix_to_1D(prob_matrix)
                    new_row = pd.concat([pd.Series(motif), pd.Series(prob_arr)],
                                ignore_index = True)

                    dist_df = dist_df.append(new_row, ignore_index=True)

    dist_df.to_csv('data/replicate-dists-df.csv')
        
create_dist_df(file_names_and_membership)

#dist_df = pd.read_csv('data/replicate-dists-df.csv')

'''
filename = 'data/-11-10_probdist.h5'

hdf = h5py.File(filename, mode = 'r')
#print(hdf.keys())

d1 = hdf['para0']
# <KeysViewHDF5 ['col', 'probdist', 'row']>
#print(d1.keys())

col = d1['col']
probdist = d1['probdist']
row = d1['row']

#print(np.array(col))
#print(np.array(probdist))
print('\n')
#print(np.array(row))

prob_matrix = csr_matrix((probdist, (row, col))).toarray()

print(prob_matrix.shape)



hm = sns.heatmap(prob_matrix)
hm.invert_yaxis()
plt.show()

'''

        
        