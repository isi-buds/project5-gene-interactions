# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Pil import Image
import os
import h5py

data_path = ['..', '..', 'data']

# %%
# # Synth Data First Set

data_1 = pd.read_table(os.path.join(*data_path, 'SyntheticData_FirstSet.txt'), delimiter='   ', header=None, engine='python')

# %%
# Empty first-set-heatmaps folder
fs_hm_path = 'first-set-heatmaps'
file_list = [ f for f in os.listdir(fs_hm_path)]
for f in file_list:
    os.remove(os.path.join(fs_hm_path, f))
# %%
secondary_i = 0
for index in range(data_1.shape[0]):
    temp_matrix = data_1.iloc[index, :].to_numpy().reshape(51,51)
    condition = temp_matrix[0:3, 0:3].sum() < .01
    if condition:
        hm = sns.heatmap(
            temp_matrix,
        )
        hm.invert_yaxis()
        file_name = '{sec_i:0{width_2}}-index-{index:0{width_1}}'.format(index=index,
                                                                width_1=4, 
                                                                sec_i=secondary_i, 
                                                                width_2=4)
        plt.savefig(os.path.join(fs_hm_path, file_name))
        plt.clf()
        secondary_i += 1

# %%
images = []
file_list = [ f for f in os.listdir(fs_hm_path)]
for f in file_list:
    images.append(Image.open(os.path.join(fs_hm_path, f)).convert('RGB'))

images[0].save(os.path.join(fs_hm_path, 'first-set-heatmaps.pdf'), save_all=True, append_images=images[1:])

# %%

