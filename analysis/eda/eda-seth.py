# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

data_path = ['..', '..', 'data']

# %%
# # Synth Data First Set

data_raw = pd.read_table(os.path.join(*data_path, 'SyntheticData_FirstSet.txt'), delimiter='   ', header=None, engine='python')

# %%
plt.figure(figsize=(5*2,5*10))
fig, axes = plt.subplots(ncols=10, nrows=81)
col_index = 0
prev_index = 0
for index in range(2*81):
    temp_matrix = data_raw.iloc[index, :].to_numpy().reshape(51,51)
    condition = temp_matrix[0:3, 0:3].sum() < .01
    if condition & col_index <= 10:
        hm = sns.heatmap(
            temp_matrix,
            ax=axes[(index + 1) // 81, col_index]
        )
        hm.invert_yaxis()
        if prev_index == index:
            col_index += 1
        else:
            col_index = 0
fig.show()
# %%
plt.save('heatmaps-01.png')