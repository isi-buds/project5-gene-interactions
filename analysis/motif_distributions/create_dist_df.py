import h5py
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import seaborn as sns
import matplotlib.pyplot as plt

file_names_and_membership = {'1-1-11_probdist' : 57, '0010_probdist' : 50,
                             '00-10_probdist' : 32, '0-1-10_probdist' : 29,
                             '0110_probdist' : 53, '-11-10_probdist' : 8,
                             '0100_probdist' : 44}


def create_dist_df(filenames) -> None:

    dist_df = pd.DataFrame()

    i =0
    for file, motif in filenames.items():

        hdf = h5py.File('data/'+file+'.h5', mode = 'r')
        
        for key in hdf.keys():
            
            if key != 'parameterset':

                df = hdf[key]

                col = df['col']
                probdist = np.array(df['probdist'])
                row = df['row']
                column_names = ['p(%s,%s)' % (i, j) for i, j in zip(row, col)]

                if probdist.shape != (0,):
                    dist_df.loc[i, 'motif'] = int(motif)
                    dist_df.loc[i, column_names] = probdist
                    i += 1
                    

    dist_df = dist_df.fillna(0)
    dist_df.to_csv('data/secondset-df.csv', index = False)
        
create_dist_df(file_names_and_membership)

dist_df = pd.read_csv('data/secondset-df.csv')
print(dist_df)

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

        
        