import copy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

my_cmap = copy.copy(matplotlib.cm.get_cmap('viridis'))
my_cmap.set_bad(my_cmap.colors[0])
sns.set(font_scale=1.5, style='white')
plt.rc('font', size=12)

ids = pd.read_table('data/NetworkMotifs.txt',
                             delimiter='  ',
                             names=['i0', 'i1', 'i2', 'i3'],
                             dtype=int,
                             engine='python')
ids = pd.DataFrame(ids.apply(lambda x: '{:2d}{:2d}{:2d}{:2d}'.format(*x), axis=1),
    columns=['id'])
ids.index += 1

data_2m = pd.read_csv('data/secondset_measures.csv')
data_2m = pd.merge(ids, data_2m, how='right', left_index=True, right_on='motif')
X = data_2m.drop(['motif', 'id'], axis=1)
y = data_2m['id']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, 
                                                        random_state=42)

# top models
mlr = LogisticRegression(solver = 'saga', max_iter = 200,
                                           multi_class = 'multinomial')
rfc_params = {'max_depth': 10, 'max_features': 0.5, 'n_estimators': 500}
rfc = RandomForestClassifier(**rfc_params, n_jobs=-1)
# make plots
fig, ax = plt.subplots(figsize=(10, 10))
p = []
for m, name in [(mlr, 'Multinomial Logistic Regression'), (rfc, 'Random Forest')]:
    m.fit(X_train, y_train)
    p_data = pd.DataFrame(m.predict_proba(X_test), columns=m.classes_)
    p_data['id'] = y_test.values
    mean_p = p_data.groupby('id').mean()
    sd_p = p_data.groupby('id').std()
    hm = sns.heatmap(
        mean_p,
        square=True,
        cbar_kws={'label': 'probability'},
        annot=True,
        annot_kws={"size": 18},
        cmap=my_cmap
    )
    plt.title(f'{name}')
    plt.xlabel('predicted')
    plt.xticks(rotation=45)
    plt.ylabel('true')
    plt.yticks(rotation=45)
    plt.savefig(f'analysis/prediction-models/plots/images/{name.lower().replace(" ", "-")}-heatmap.png')
    plt.clf()
    p.append([name, mean_p, sd_p])
