import h5py
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

filename = 'data/hepatocyte_data.h5'

hdf = h5py.File(filename, mode = 'r')
#print(hdf.keys())

d1 = hdf['Hepatocyte trajectory']

# ['assay_data', 'col_labels', 'row_labels']
#print(d1.keys())

assay_data = d1['assay_data']
#print(assay_data.keys())

data = assay_data['data']
indices = assay_data['indices']
indptr = assay_data['indptr']

csr_matrix = csr_matrix((data, indices, indptr))

# ---------------------------------------------

row_labels = d1['row_labels']

rows = np.array(row_labels)

df = pd.DataFrame.sparse.from_spmatrix(csr_matrix)
num_cols = df.shape[1]

# series containing the number of cells that express the 
# corresponding gene
num_genes = pd.Series()

for gene, data in df.iteritems():
    cell_count = pd.Series((data != 0).sum())
    num_genes = num_genes.append(cell_count)

num_genes.index = [i for i in range(0, num_cols)]
    
#print(num_genes)

# get top 26 most expressed gene
most_expressed = num_genes.nlargest(n = 26)
#print(most_expressed)

top_26_matrix = []

for gene, count in most_expressed.iteritems():
    top_26_matrix.append(df.loc[:, gene])

# rows: genes
# cols: cells
top_26_df = pd.DataFrame(top_26_matrix)
#print(top_26_df.shape)
#print(top_26_df.head())
# make pairwise heatmaps in other file