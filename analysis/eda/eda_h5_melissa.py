import h5py
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np

filename = 'data/hepatocyte_data.h5'

hdf = h5py.File(filename, mode = 'r')
#print(hdf.keys())

d1 = hdf['Hepatocyte trajectory']

#print(d1.keys())

assay_data = d1['assay_data']
#print(assay_data.keys())

data = assay_data['data']
indices = assay_data['indices']
indptr = assay_data['indptr']

csr_matrix = csr_matrix((data, indices, indptr))

#print(csr_matrix.toarray())

plt.spy(csr_matrix.toarray())
plt.show()

plt.hist