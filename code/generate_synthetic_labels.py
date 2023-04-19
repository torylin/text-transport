import pandas as pd
from sklearn.linear_model import LinearRegression
import pdb
import numpy as np
import os

data_dir = '/home/victorialin/Documents/2022-2023/causal_text/data/amazon_synthetic/'

df0 = pd.read_csv(os.path.join(data_dir, 'combined_liwc_categorymusic.csv'))
df1 = pd.read_csv(os.path.join(data_dir, 'combined_liwc_categoryoffice.csv'))
prob_df0 = pd.read_csv(os.path.join(data_dir, 'music_reviews_pred_numerical.csv'))
prob_df1 = pd.read_csv(os.path.join(data_dir, 'office_reviews_pred_numerical.csv'))

df = pd.concat([df0, df1], axis=0)
prob_df = pd.concat([prob_df0, prob_df1], axis=0)

X = df.drop(['reviewText', 'helpful'], axis=1)
y = df['helpful'].values

model = LinearRegression()
model.fit(X, y)

features = pd.DataFrame({'feature': X.columns.tolist(), 'coef': model.coef_})
features = features.reindex(features.coef.abs().sort_values(ascending=False).index)
print(features['feature'][0:10].values)

np.random.seed(230418)
mu, sigma = 0, 1
noise0 = np.random.normal(mu, sigma, df0.shape[0])
noise1 = np.random.normal(mu, sigma, df1.shape[0])

# label_synth0 = df0[features['feature'][0:10].values].sum(axis=1) + prob_df0['pred']*0.5 + noise0
# label_synth1 = df1[features['feature'][0:10].values].sum(axis=1) + prob_df1['pred']*0.5 + noise1
# label_synth0 = df0[features['feature'][0:10].values].sum(axis=1)*2 + prob_df0['pred']*0.5 + noise0
# label_synth1 = df1[features['feature'][0:10].values].sum(axis=1)*2 + prob_df1['pred']*0.5 + noise1
label_synth0 = prob_df0['pred']*0.5 + noise0
label_synth1 = prob_df1['pred']*0.5 + noise1

df0.drop(['helpful'], axis=1, inplace=True)
df1.drop(['helpful'], axis=1, inplace=True)

df0['label_synthetic'] = label_synth0
df1['label_synthetic'] = label_synth1

df0.to_csv(os.path.join(data_dir, 'music_reviews_label_synthetic_numerical_nolex.csv'), index=False)
df1.to_csv(os.path.join(data_dir, 'office_reviews_label_synthetic_numerical_nolex.csv'), index=False)
