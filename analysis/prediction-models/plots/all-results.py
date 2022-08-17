import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
import seaborn as sns

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
df['Method'] = df['Method'].map({'RandomForest': 'Random Forest', 'Multinomiallogistic': 'Multinomial Logistic'})

df['X, y'] = df[['X','y']].apply(lambda x: "{}, {} outcomes".format(*x), axis=1)
df['Log Loss'] = df['LogLoss'].astype(float)
df['Accuracy'] = df['Accuracy'].astype(float)

df = df[['Method', 'X, y', 'Log Loss', 'Accuracy']]
# print(df)

palette = 'hls'

g1 = sns.catplot(x="Method", y="Log Loss",
                hue="X, y", data=df,
                kind="bar", palette=palette,
                height=5, aspect=.8)

plt.savefig('analysis/prediction-models/plots/images/results-logloss.png')

g2 = sns.catplot(x="Method", y="Accuracy",
                hue="X, y", data=df,
                kind="bar", palette=palette,
                height=5, aspect=.8)

plt.savefig('analysis/prediction-models/plots/images/results-accuracy.png')
