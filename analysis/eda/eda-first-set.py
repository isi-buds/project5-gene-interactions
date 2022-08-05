# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

data_path = ['..', '..', 'data']

# # Synth Data First Set
# %%

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
# %% Delete all files in the first-set-heatmaps folder
fs_hm_path = 'first-set-heatmaps'
for f in os.listdir(fs_hm_path):
    path_1 = os.path.join(fs_hm_path, f)
    if os.path.isdir(path_1):
        for f_f in os.listdir(path_1):
            os.remove(os.path.join(path_1, f_f))
    else:
        os.remove(path_1)

# %% A sample of heatmaps

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
    title = 'motif-{motif:02}-#{i_2nd:02}-index-{index}'.format(motif=motif,
                                                                 i_2nd=i_2nd[motif],
                                                                 index=index)
    plt.title(title)
    plt.savefig(os.path.join(fs_hm_path, 'sample', title))
    plt.clf()
    i_2nd[motif] += 1

# %% Convert images into a pdf
images = []
for f in os.listdir(os.path.join(fs_hm_path, 'sample')):
    images.append(Image.open(os.path.join(fs_hm_path, 'sample', f)).convert('RGB'))

images[0].save(os.path.join(fs_hm_path, 'first-set-heatmaps-samples.pdf'), save_all=True, append_images=images[1:])

# %%
func_list = [np.mean, np.median, np.std]
for motif in data_1['motif'].unique():
    for func in func_list:
        hm = sns.heatmap(
            data_1.loc[data_1['motif'] == motif, 0:].apply(func, axis=0).to_numpy().reshape(51, 51)
        )
        hm.invert_yaxis()
        title = '{motif:02}-{func}'.format(motif=motif,
                                               func=func.__name__)
        plt.title(title)
        plt.savefig(os.path.join(fs_hm_path, 'summaries', title))
        plt.clf()


# %%
array_data = data_1.loc[:, 0:].to_numpy().reshape((5000, 51, 51))
# %%
for start_row in range(0,46, 5):
    for start_col in range(0,46, 5):
        temp = array_data[:, start_row:(start_row + 6), start_col:(start_col + 6)].reshape(-1)
        plt.hist(temp[temp>0])
        plt.show()
# %% For 3d plot
# x = [i for i in range(51)]
# y = [i for i in range(51)]
# xyz = []
# for x_i in x:
#     for y_i in y:
#         xyz.append([x_i, y_i, array_data[3309, y_i, x_i]])

# import plotly.express as px
# fig = px.scatter_3d(pd.DataFrame(xyz), x=0, y=1, z=2, color=2)
# fig.show()

# %%
