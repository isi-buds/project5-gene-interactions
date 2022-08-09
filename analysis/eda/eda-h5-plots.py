# file has a different naming convention because python cannot 
# import modules with "-" in the file name
import eda_h5
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

h5_plots_path = 'analysis/eda/h5-plots'

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

seen = []

g1_index = 0
# create pairwise gene plots
for gene1 in top_26[0:25]:

    g2_index = 0

    for gene2 in top_26[0:25]:
        
        if gene1 != gene2 and (gene1, gene2) not in seen:

            temp = top_26.iloc[[gene1, gene2], :].to_numpy().reshape(144, 158)
            hm = sns.heatmap(temp)
            title = 'Gene1-{gene1}-vs-Gene2-{gene2}-hm.png'.format(
                    gene1 = genes[g1_index], gene2 = genes[g2_index])

            plt.savefig(os.path.join(h5_plots_path, title))
            plt.clf()

            seen.append((gene1, gene2))

        g2_index += 1
    
    g1_index += 1




