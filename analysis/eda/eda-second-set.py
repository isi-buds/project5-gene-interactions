import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy
import matplotlib
from matplotlib.colors import LogNorm
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA, KernelPCA

my_cmap = copy.copy(matplotlib.cm.get_cmap('viridis'))
my_cmap.set_bad(my_cmap.colors[0])
sns.set(font_scale=1.5, style='white')
plt.rc('font', size=12)

data_2 = pd.read_csv('data/secondset-df.csv')

ids = pd.read_table('data/NetworkMotifs.txt',
                             delimiter='  ',
                             names=['i0', 'i1', 'i2', 'i3'],
                             dtype=int,
                             engine='python')

ids = pd.DataFrame(ids.apply(lambda x: '{:2d}{:2d}{:2d}{:2d}'.format(*x), axis=1),
    columns=['id'])

ids.index += 1

data_2 = pd.merge(ids, data_2, how='right', left_index=True, right_on='motif')

# %%
x_i = []
y_j = []

for c in data_2.drop(['motif', 'id'], axis=1).columns:
    x_i.append(int(c.split(',')[0].split('(')[1]))
    y_j.append(int(c.split(',')[1].split(')')[0]))

# %% 

ss_hm_path = os.path.join('analysis','eda', 'second-set-heatmaps')
for f in os.listdir(ss_hm_path):
    path_1 = os.path.join(ss_hm_path, f)
    if os.path.isdir(path_1):
        for f_f in os.listdir(path_1):
            os.remove(os.path.join(path_1, f_f))
    else:
        os.remove(path_1)

# %%
# Histogram
fig, ax = plt.subplots(figsize=(10, 7))
sns.histplot(data_2.drop(['motif','id'], axis=1).to_numpy().reshape(-1), bins=50)
ax.set_yscale('log')
plt.xlabel('probabilty')
plt.title('Second Set')
plt.savefig(os.path.join(ss_hm_path, 'histogram.png'))

# %%

fig, ax = plt.subplots(figsize=(7, 7))
i_2nd = {}

for i in data_2['motif'].unique():
    i_2nd[i] = 0

for index in data_2.groupby('motif').sample(n=10, random_state=1).index:
    values = data_2.drop(['motif','id'], axis=1).iloc[index]
    A = csr_matrix((values, (y_j, x_i))).toarray()[:101, :101]
    hm = sns.heatmap(
        A,
        cbar_kws={'label': 'probability'},
        xticklabels=10,
        yticklabels=10,
        square=True,
        norm=LogNorm(),
        cmap=my_cmap
    )
    hm.invert_yaxis()
    motif = int(data_2.at[index, 'motif'])
    file_name = 'motif-{motif:02}-i-{i_2nd:02}-row-{index}.png'.format(motif=motif,
                                                                 i_2nd=i_2nd[motif],
                                                                 index=index)
    plt.xlabel('gene A')
    plt.ylabel('gene B')
    plt.title('id: %s \n#%s' % (data_2.at[index, 'id'], i_2nd[motif]))
    plt.savefig(os.path.join(ss_hm_path, 'sample', file_name))
    plt.clf()
    i_2nd[motif] += 1

# %% 

# size = (101, 101)
# data_2_array = np.zeros((data_2.shape[0], size[0], size[1]))
# for index in range(data_2.shape[0]):
#     values = data_2.drop(['motif','id'], axis=1).iloc[index]
#     A = csr_matrix((values, (y_j, x_i))).toarray()
#     A[size[0]]  += np.apply_along_axis(np.sum, 0, A[(size[0] + 1):, :])
#     A[:, size[1]] += np.apply_along_axis(np.sum, 1, A[:, (size[1] + 1):])
#     A = A[:size[0], :size[1]]
#     data_2_array[i] = A

# %%

fig, ax = plt.subplots(figsize=(7, 7))
func_list = [np.mean, np.median, np.std, max]
for motif in data_2['motif'].unique():
    for func in func_list:
        values = data_2[data_2['motif'] == motif].drop(['motif', 'id'], axis=1).apply(func)
        A = csr_matrix((values, (y_j, x_i))).toarray()[:101, :101]
        if sum(A.reshape(-1)) > 0:
            
            hm = sns.heatmap(
                A,
                cbar_kws={'label': 'probability'},
                xticklabels=10,
                yticklabels=10,
                square=True,
                norm=LogNorm(),
                cmap=my_cmap
            )
            hm.invert_yaxis()
            file_name = '{motif:02}-{func}.png'.format(motif=motif,
                                                func=func.__name__)
            plt.xlabel('gene A')
            plt.ylabel('gene B')
            title = '%s of motif %s' % (func.__name__, ids.at[motif, 'id'])
            plt.title(title)
            plt.savefig(os.path.join(ss_hm_path, 'summaries', file_name))
            plt.clf()
        else:
            continue

# %%

# %%

def convolution (matrix, kernel):
    k_0, k_1 = kernel.shape
    m_0, m_1 = matrix.shape

    if k_0 % 2 == 0:
        r = int(k_0/2)
    else:
        r = inr((k_0 - 1)/2)

    if k_1 % 2 == 0:
        c = int(k_1/2)
    else:
        c = int((k_1 - 1)/2)

    Z = np.zeros((matrix.shape[0] + 2 * r, matrix.shape[1] + 2 * c))

    Z[r:-r, c:-c] = matrix.copy()
    
    kernel = kernel / sum(kernel.reshape(-1))
    OUT = np.zeros(matrix.shape)
    for i in range(m_0):
        for j in range(m_1):
            A = np.zeros(Z.shape)
            A[i:(i+k_0), j:(j+k_1)] += kernel
            OUT[i, j] = np.dot(A.reshape(-1), Z.reshape(-1))
    return OUT

# # %%
# kernel_pca = KernelPCA(
#     n_components=1000, kernel="rbf", fit_inverse_transform=True
# )

# kernel_pca.fit(data_2_array.reshape(data_2_array.shape[0], -1))

# values = kernel_pca.inverse_transform(
#     kernel_pca.transform(data_2_array[0].reshape(1, -1))
# )
# A = values.reshape(101, 101)
# hm = sns.heatmap(
#     A,
#     cbar_kws={'label': 'probability'},
#     xticklabels=10,
#     yticklabels=10,
#     square=True,
#     norm=LogNorm(),
#     cmap=my_cmap
# )
# hm.invert_yaxis()
# # %%
