# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import copy
import matplotlib
from matplotlib.colors import LogNorm
from scipy.stats import skew

data_path = ['data']

my_cmap = copy.copy(matplotlib.cm.get_cmap('viridis'))
my_cmap.set_bad(my_cmap.colors[0])
sns.set(font_scale=1.5)
plt.rc('font', size=12)

# # Synth Data First Set
# %%

data_1 = pd.read_table(os.path.join(*data_path, 'SyntheticData_FirstSet.txt'),
                       delimiter='   ',
                       header=None,
                       dtype=float,
                       engine='python')

motif_labels = pd.read_table(os.path.join(*data_path, 'membership.txt'),
                             delimiter='   ',
                             names=['motif'],
                             dtype=int,
                             engine='python')


network_motifs = pd.read_table(os.path.join(*data_path, 'NetworkMotifs.txt'),
                             delimiter='  ',
                             names=['i0', 'i1', 'i2', 'i3'],
                             dtype=int,
                             engine='python')

ids = pd.DataFrame(network_motifs.apply(lambda x: '{:2d}{:2d}{:2d}{:2d}'.format(*x), axis=1),
    columns=['id'])

ids.index += 1

motif_labels = pd.merge(motif_labels.loc[:, 'motif'], ids, how='left', left_on='motif', right_index=True)
data_1 = pd.merge(motif_labels, data_1, how='right', left_index=True, right_index=True)

# %% Delete all files in the first-set-heatmaps folder
fs_hm_path = os.path.join('analysis','eda', 'first-set-heatmaps')
for f in os.listdir(fs_hm_path):
    path_1 = os.path.join(fs_hm_path, f)
    if os.path.isdir(path_1):
        for f_f in os.listdir(path_1):
            os.remove(os.path.join(path_1, f_f))
    else:
        os.remove(path_1)

# %% A sample of heatmaps
fig, ax = plt.subplots(figsize=(7, 7))
i_2nd = {}
for i in data_1['motif'].unique():
    i_2nd[i] = 0

for index in data_1.groupby('motif').sample(n=10, random_state=1).index:
    temp_matrix = data_1.loc[index, 0:].astype(float).to_numpy().reshape(51,51, order='F')

    hm = sns.heatmap(
        temp_matrix,
        cbar_kws={'label': 'probability'},
        xticklabels=10,
        yticklabels=10,
        square=True,
        norm=LogNorm(),
        cmap=my_cmap
    )
    hm.invert_yaxis()
    motif = data_1.at[index, 'motif']
    file_name = 'motif-{motif:02}-i-{i_2nd:02}-row-{index}'.format(motif=motif,
                                                                 i_2nd=i_2nd[motif],
                                                                 index=index)
    plt.xlabel('gene A')
    plt.ylabel('gene B')
    plt.title('network motif: %s \n#%s' % (data_1.at[index, 'network motif'], i_2nd[motif]))
    plt.savefig(os.path.join(fs_hm_path, 'sample', file_name))
    plt.clf()
    i_2nd[motif] += 1

# %% Convert images into a pdf
images = []
for f in os.listdir(os.path.join(fs_hm_path, 'sample')):
    images.append(Image.open(os.path.join(fs_hm_path, 'sample', f)).convert('RGB'))

images[0].save(os.path.join(fs_hm_path, 'first-set-heatmaps-samples.pdf'), save_all=True, append_images=images[1:])

# %%
fig, ax = plt.subplots(figsize=(7, 7))
func_list = [np.mean, np.median, np.std, max, skew]
for motif in data_1['motif'].unique():
    for func in func_list:
        if sum(data_1.loc[data_1['motif'] == motif, 0:].apply(func, axis=0)) > 0:
            hm = sns.heatmap(
                data_1.loc[data_1['motif'] == motif, 0:].apply(func, axis=0).to_numpy().reshape(51, 51, order='F'),
                cbar_kws={'label': 'probability'},
                xticklabels=10,
                yticklabels=10,
                square=True,
                norm=LogNorm(),
                cmap=my_cmap
            )
            hm.invert_yaxis()
            file_name = '{motif:02}-{func}'.format(motif=motif,
                                                func=func.__name__)
            plt.xlabel('gene A')
            plt.ylabel('gene B')
            title = '%s of motif %s' % (func.__name__, network_motifs.at[motif, 'network motif'])
            plt.title(title)
            plt.savefig(os.path.join(fs_hm_path, 'summaries', file_name))
            plt.clf()
        else:
            continue


# %%
data_array_bad  = data_1.loc[:, 0:].to_numpy().reshape((5000, 51, 51), order='C') 
# not the same as making a matrix from each row of data_1
'''
hm = sns.heatmap(
    data_array[2],
    cbar_kws={'label': 'probability'},
    xticklabels=10,
    yticklabels=10,
    square=True,
    norm=LogNorm(),
    cmap=my_cmap
)
hm.invert_yaxis()
plt.show()
hm = sns.heatmap(
    data_1.loc[2, 0:].astype(float).to_numpy().reshape(51, 51, order='F'),
    cbar_kws={'label': 'probability'},
    xticklabels=10,
    yticklabels=10,
    square=True,
    norm=LogNorm(),
    cmap=my_cmap
)
hm.invert_yaxis()
plt.show()
'''
# %%
data_array = np.zeros((5000, 51, 51))
for i in range(data_1.shape[0]):
    data_array[i] = data_1.loc[i, 0:].astype(float).to_numpy().reshape(51,51, order='F')

# %%

# for start_row in range(0,46, 5):
#     for start_col in range(0,46, 5):
#         temp = data_array[:, start_row:(start_row + 6), start_col:(start_col + 6)].reshape(-1)
#         plt.hist(temp, bins=50)
#         plt.show()

# %%
fig, ax = plt.subplots(figsize=(10, 7))
sns.histplot(data_array.reshape(-1), bins=50)
ax.set_yscale('log')
plt.xlabel('probabilty')
plt.show()

# %%
# # %% For 3d plot
# x = [i for i in range(51)]
# y = [i for i in range(51)]
# xyz = []
# for x_i in x:
#     for y_i in y:
#         xyz.append([x_i, y_i, data_array[1044, y_i, x_i]])

# import plotly.express as px
# fig = px.scatter_3d(pd.DataFrame(xyz), x=0, y=1, z=2, color=2)
# fig.show()

# %%
