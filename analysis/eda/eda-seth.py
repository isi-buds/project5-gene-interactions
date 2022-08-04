# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Pil import Image
import statsmodels.api as sm
import os
import h5py

data_path = ['..', '..', 'data']

# %%
# # Synth Data First Set

data_1 = pd.read_table(os.path.join(*data_path, 'SyntheticData_FirstSet.txt'),
                       delimiter='   ',
                       header=None,
                       engine='python')

motif_labels = pd.read_table(os.path.join(*data_path, 'membership.txt'),
                             delimiter='   ',
                             header=None,
                             dtype=int,
                             engine='python')

data_1.insert(0, 'motif', motif_labels[0])
# %%

# Empty first-set-heatmaps folder
fs_hm_path = 'first-set-heatmaps'
file_list = [ f for f in os.listdir(fs_hm_path)]
for f in file_list:
    os.remove(os.path.join(fs_hm_path, f))

# %%
i_2nd = {}
for i in data_1['motif'].unique():
    i_2nd[i] = 0
for index in data_1.groupby('motif').sample(n=10, random_state=1).index:
    temp_matrix = data_1.loc[index, 0:].to_numpy().reshape(51,51)
    hm = sns.heatmap(
        temp_matrix,
    )
    hm.invert_yaxis()
    motif = data_1.at[index, 'motif']
    file_name = '{motif:02}-{i_2nd:02}-index{index}'.format(motif=motif,
                                                            i_2nd=i_2nd[motif],
                                                            index=index)
    plt.title(file_name)
    plt.savefig(os.path.join(fs_hm_path, file_name))
    plt.clf()
    i_2nd[motif] += 1

# %%
images = []
file_list = [ f for f in os.listdir(fs_hm_path)]
for f in file_list:
    images.append(Image.open(os.path.join(fs_hm_path, f)).convert('RGB'))

images[0].save(os.path.join(fs_hm_path, 'first-set-heatmaps.pdf'), save_all=True, append_images=images[1:])

# %%
gaus_conv_matrix

