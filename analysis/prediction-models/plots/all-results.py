import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.5)

with open('analysis/prediction-models/Results.md') as f:
    text = f.read()

text_list = [line.replace(' ', '').split('|') for line in text.split('\n')]
df = pd.DataFrame(text_list[2:], columns=text_list[0])

Xy_map = {'firstset': 'X',
          'firstsetmeasures': 'f(X)',
          'secondset': '2nd X',
          'secondsetmeasures': '2nd f(X)',
          'unreduced': '81',
          'reduced' : '7'
}
df['X'] = df['X'].map(Xy_map)
df['y'] = df['y'].map(Xy_map)
method_map = {'RandomForest': 'Random Forest',
              'Multinomiallogistic': 'Multinomial Logistic',
              'KNearestNeighbors': 'K Nearest Neighbors'}
df['Method'] = df['Method'].map(method_map)

df['X, y'] = df[['X','y']].apply(lambda x: "{}, {} outcomes".format(*x), axis=1)
df['Log Loss'] = df['LogLoss'].astype(float)
df['Accuracy'] = df['Accuracy'].astype(float)

df = df[['Method', 'X, y', 'Log Loss', 'Accuracy']]
# print(df)
df.sort_values('Accuracy', ascending=False).to_csv('data/results.csv', index=False)
#
df = pd.read_csv('data/results.csv')

cat_arg = {'palette': 'colorblind',
     'height': 5,
     'aspect': 2,
     'kind': 'bar',
     'y': 'X, y',
     'hue': 'Method',
     'data': df}

g1 = sns.catplot(x="Log Loss", **cat_arg)

plt.savefig('analysis/prediction-models/plots/images/results-logloss.png')

g2 = sns.catplot(x="Accuracy", **cat_arg)

plt.savefig('analysis/prediction-models/plots/images/results-accuracy.png')
