import h5py
import os
import pandas as pd
import numpy as np

network_motifs = pd.read_table('data/NetworkMotifs.txt',
                             delimiter='  ',
                             names=['i0', 'i1', 'i2', 'i3'],
                             dtype=int,
                             engine='python')

network_motifs = network_motifs[['i0', 'i2', 'i1', 'i3']].apply(lambda x: '{}{}{}{}'.format(*x), axis=1)
# i2 and i1 where switched for the h5 file names

network_motifs.index += 1

motif_dictionary = {}
for f in os.listdir('data'):
    if f[-12:] == '_probdist.h5':
        key = f[:-12]
        motif_dictionary[f[:-12]] = network_motifs[network_motifs==key].index[0]


def create_downsampled(motif_dictionary) -> None:

    down_sampled = pd.DataFrame()

    i =0
    for file, motif in motif_dictionary.items():

        hdf = h5py.File('data/%s_probdist.h5' % file, mode = 'r')
        
        for key in hdf.keys():
            
            if key != 'parameterset':
                if 'downsample 0.3' in hdf[key].keys():
                    df = hdf[key]['downsample 0.3']

                    col = df['col']
                    probdist = np.array(df['probdist'])
                    row = df['row']
                    column_names = ['p(%s,%s)' % (i, j) for i, j in zip(row, col)]

                    if probdist.shape != (0,):
                        down_sampled.loc[i, 'motif'] = int(motif)
                        down_sampled.loc[i, column_names] = probdist
                        i += 1
                    

    down_sampled = down_sampled.fillna(0)
    down_sampled.to_csv('data/second-downsampled.csv', index = False)
        
create_downsampled(motif_dictionary)

down_sampled = pd.read_csv('data/second-downsampled.csv')
print(down_sampled)

