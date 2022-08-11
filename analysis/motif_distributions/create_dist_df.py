import h5py
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

# missing motif 32 and 8
motif_file_paths = ['data/1-1-11_probdist.h5', 'data/0100_probdist.h5',
                    'data/0-1-10_probdist.h5', 'data/0110_probdist.h5',
                    'data/00-10_probdist.h5']

membership_indices = {'1-1-11_probdist' : 57, '0100_probdist' : 50,
                      '0-1-10_probdist' : 29, '0110_probdist' : 53,
                      '00-10_probdist' : 44}

def create_dist_df(filenames, row_indices) -> None:

    dist_df = pd.DataFrame()
    
    for file in filenames:

        hdf = h5py.File(file, mode = 'r')
        
        for key in hdf.keys():
            
            if key != 'parameterset':

                df = hdf[key]

                col = df['col']
                probdist = df['probdist']
                row = df['row']

                if probdist.shape != (0, 0):
                    prob_matrix = csr_matrix((probdist, (row, col))).toarray()
                    new_row = pd.Series(prob_matrix.tolist())

                    dist_df = dist_df.append(new_row, ignore_index=True)

    dist_df.to_csv('data/replicate-dists-df.csv')
        
create_dist_df(motif_file_paths, membership_indices)
        
        