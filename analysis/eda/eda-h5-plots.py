# file has a different naming convention because python cannot 
# import modules with "-" in the file name
import eda_h5
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import copy
import matplotlib
import os
from matplotlib.colors import LogNorm

h5_plots_path = 'analysis/eda/h5-plots'

my_cmap = copy.copy(matplotlib.cm.get_cmap('viridis'))
my_cmap.set_bad(my_cmap.colors[0])
sns.set(font_scale=1.5)
plt.rc('font', size=12)

# delete old plots
for f in os.listdir(h5_plots_path):
    path_1 = os.path.join(h5_plots_path, f)
    if os.path.isdir(path_1):
        for f_f in os.listdir(path_1):
            os.remove(os.path.join(path_1, f_f))
    else:
        os.remove(path_1)

sparse = eda_h5.csr_matrix

fig, ax = plt.subplots()

plt.spy(sparse.toarray())
plt.xlabel('Genes')
plt.ylabel('Cells')

#plt.show()
plt.title('2d Plot of Genes vs. Cells')
plt.savefig('analysis/eda/h5-plots/2d-plot.png')

plt.clf()

#print(sparse)



#--------------------------------------------------
top_26 = eda_h5.top_26_df
genes = eda_h5.rows
shortened_genes = {23560:'Yam1', 7942:'Afp', 19548:'Meg3', 24692:'Malat1',
                   26165:'mt-Rnr2', 25497:'Gpc3', 26163:'mt-Rnr1', 
                   19556:'Rian', 7941:'Alb', 1966:'Camk1d', 8652:'Auts2',
                   7208:'Reln', 22987:'Airn', 17137:'Grb10',
                   13267:'Tenm3', 12743:'lgf2', 15474:'Trf', 10529:'Sox5',
                   22754:'Robo1', 20641:'Adk', 12757:'Kcnq1ot1',
                   16870:'Grip1', 14261:'Pard3', 20768:'Glud1', 
                   10011:'Foxp1', 9986:'Magi1'}


# truncates all values above given max val in series to that value
# returns df w/ first row being gene1 and second row being gene2
def truncate_values(gene1: pd.Series, gene2: pd.Series, max_val: int) -> pd.DataFrame:
    # rows: genes, cols: counts
    combined_df = gene1.to_frame().join(gene2.to_frame()).transpose()
    truncated = combined_df

    for _, col in truncated.iteritems():
        if col.iloc[0] > max_val:
            col.iloc[0] = max_val
        if col.iloc[1] > max_val:
            col.iloc[1] = max_val

    return truncated

# df_2_genes must be a dataframe with 2 rows
def get_hm_matrix(df_2_genes: pd.DataFrame) -> np.array([[int]]):

    unique_pairs = {} # pair : count

    for _, col in df_2_genes.iteritems():
        pair = (col.iloc[0], col.iloc[1])

        if pair not in unique_pairs.keys():
            unique_pairs[pair] = 1
        else:
            unique_pairs[pair] += 1

    temp_x, temp_y = map(max, zip(*unique_pairs))

    res = [[unique_pairs.get((j, i), 0) for i in range(temp_y + 1)] 
                                  for j in range(temp_x + 1)]

    return res


seen = []

fig, ax = plt.subplots(figsize=(7, 7))

# create pairwise gene plots
for index1, gene1 in top_26.iterrows():

    for index2, gene2 in top_26.iterrows():
        
        if gene1.iloc[0] != gene2.iloc[0] and (gene1.iloc[0], gene2.iloc[0]) not in seen:

            truncated = truncate_values(gene1, gene2, 100)
            hm_matrix = get_hm_matrix(truncated)

            hm = sns.heatmap(hm_matrix,
                             cbar_kws={'label': 'probability'},
                             xticklabels=10,
                             yticklabels=10,
                             square=True,
                             norm=LogNorm(),
                             cmap = my_cmap)
            
            hm.invert_yaxis()

            title = 'Gene1-{gene1}-vs-Gene2-{gene2}-hm.png'.format(
                    gene1 = shortened_genes[index1], gene2 = shortened_genes[index2])

            plt.xlabel('gene 1 counts')
            plt.ylabel('gene 2 counts')
            plt.title('{gene1} vs {gene2} Interaction Plot'.format(
                       gene1 = shortened_genes[index1], gene2 = shortened_genes[index2]
            ))

            plt.savefig(os.path.join(h5_plots_path, title))
            plt.clf()

            seen.append((gene1.iloc[0], gene2.iloc[0]))






    
    






