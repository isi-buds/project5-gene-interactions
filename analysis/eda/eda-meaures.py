import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(font_scale=1.5)
sns.set_style("whitegrid")
data_m = pd.read_csv('data/firstset_measures.csv')

motif_labels = pd.read_table('data/membership.txt',
                             delimiter='   ',
                             names=['motif'],
                             dtype=int,
                             engine='python')

network_motifs = pd.read_table('data/NetworkMotifs.txt',
                             delimiter='  ',
                             names=['i0', 'i1', 'i2', 'i3'],
                             dtype=int,
                             engine='python')

network_motifs = pd.DataFrame(network_motifs.apply(lambda x: '{} {} {} {}'.format(*x), axis=1),
    columns=['id'])

motif_map = {}
for i, id in enumerate(network_motifs['id']):
    motif_map[i+1] = id

data_m['motif'] = motif_labels['motif'].astype(str).values
data_m['id'] = motif_labels['motif'].map(motif_map).values

# sns.scatterplot(x='x_mean', y='y_mean', data=data_m, hue='id')
# plt.show()

data_2m = pd.read_csv('data/secondset_measures.csv')
data_2m['id'] = data_2m['motif'].map(motif_map).values
data_2m['motif'] = data_2m['motif'].astype(str)


# sns.scatterplot(x='x_mean', y='y_mean', data=data_2m, hue='id')
# plt.show()


data_small = data_2m.groupby('id').sample(n=50)
g = sns.FacetGrid(data_2m, col="id", col_wrap=4)
g.map(sns.scatterplot, "x_mean", "y_mean")
g.add_legend()

plt.show()
