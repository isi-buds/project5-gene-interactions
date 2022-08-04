import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

synthetic_data = pd.read_table('data/SyntheticData_FirstSet.txt', delimiter = '   ', header = None)

#print(synthetic_data.head())
#hm = sns.heatmap(data = synthetic_data)

hm1 = sns.heatmap(synthetic_data.iloc[5, :].to_numpy().reshape(51,51))

hm1.invert_yaxis()

plt.show()


