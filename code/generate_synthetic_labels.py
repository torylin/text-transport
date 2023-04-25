import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pdb
import numpy as np
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/home/victorialin/Documents/2022-2023/causal_text/data/amazon_synthetic/')
    parser.add_argument('--combination-type', type=str, default='predreg')
    parser.add_argument('--lexicon-weight', type=float, default=1.0)
    parser.add_argument('--embedding-weight', type=float, default=0.5)
    args = parser.parse_args()

    return args

args = get_args()

df0 = pd.read_csv(os.path.join(args.data_dir, 'combined_liwc_categorymusic.csv'))
df1 = pd.read_csv(os.path.join(args.data_dir, 'combined_liwc_categoryoffice.csv'))
prob_df0 = pd.read_csv(os.path.join(args.data_dir, 'music_reviews_pred_numerical.csv'))
prob_df1 = pd.read_csv(os.path.join(args.data_dir, 'office_reviews_pred_numerical.csv'))

df = pd.concat([df0, df1], axis=0)
prob_df = pd.concat([prob_df0, prob_df1], axis=0)

X = df.drop(['reviewText', 'helpful'], axis=1)
y = df['helpful'].values

model = LinearRegression()
model.fit(X, y)

features = pd.DataFrame({'feature': X.columns.tolist(), 'coef': model.coef_})
features = features.reindex(features.coef.abs().sort_values(ascending=False).index)
print(features['feature'][0:10].values)
lex_pred = (df[features['feature'][0:10].values]*features.coef[0:10].values).sum(axis=1).values
print('Lexical model RMSE: {:.3f}'.format(np.sqrt(np.sum((model.predict(X)-y)**2))))
print('Lexical model RMSE (10 features): {:.3f}'.format(np.sqrt(np.sum((lex_pred-y)**2))))
print('Embedding model RMSE: {:.3f}'.format(np.sqrt(np.sum((prob_df['pred'].values-y)**2))))

np.random.seed(230418)
mu, sigma = 0, 1
noise0 = np.random.normal(mu, sigma, df0.shape[0])
noise1 = np.random.normal(mu, sigma, df1.shape[0])

if args.combination_type == 'interaction':
    X_new = df[features['feature'][0:10].values]
    X_new['pred'] = prob_df['pred']
    X_new['helpful'] = y
    interaction_model = smf.ols(
        formula='helpful ~ sexual*pred + female*pred + filler*pred + netspeak*pred + home*pred + informal*pred + shehe*pred + nonflu*pred + assent*pred + death*pred', data=X_new)
    res = interaction_model.fit()
    preds = res.predict()
    print('Interaction model RMSE: {:.3f}'.format(np.sqrt(np.sum((preds-y)**2))))
    label_synth0 = preds[:df0.shape[0]] + noise0
    label_synth1 = preds[df0.shape[0]:] + noise1
    pdb.set_trace()
elif args.combination_type == 'reg':
    lex_pred = (df[features['feature'][0:10].values]*features.coef[0:10].values).sum(axis=1).values
    embed_pred = prob_df['pred'].values
    X_new = pd.DataFrame({'lex_pred': lex_pred, 'embed_pred': embed_pred})
    reg_model = LinearRegression()
    reg_model.fit(X_new, y)
    print(reg_model.coef_)
    preds = reg_model.predict(X_new)
    print('Combined model RMSE: {:.3f}'.format(np.sqrt(np.sum((preds-y)**2))))
    label_synth0 = preds[:df0.shape[0]] + noise0
    label_synth1 = preds[df0.shape[0]:] + noise1
    pdb.set_trace()

elif args.combination_type == 'direct':
    label_synth0 = (df0[features['feature'][0:10].values]*features.coef[0:10].values).sum(axis=1)*args.lexicon_weight + prob_df0['pred']*args.embedding_weight + noise0
    label_synth1 = (df1[features['feature'][0:10].values]*features.coef[0:10].values).sum(axis=1)*args.lexicon_weight + prob_df1['pred']*args.embedding_weight + noise1
# label_synth0 = df0[features['feature'][0:10].values].sum(axis=1)*2 + prob_df0['pred']*0.5 + noise0
# label_synth1 = df1[features['feature'][0:10].values].sum(axis=1)*2 + prob_df1['pred']*0.5 + noise1
# label_synth0 = prob_df0['pred']*0.5 + noise0
# label_synth1 = prob_df1['pred']*0.5 + noise1

df0.drop(['helpful'], axis=1, inplace=True)
df1.drop(['helpful'], axis=1, inplace=True)

df0['label_synthetic'] = label_synth0
df1['label_synthetic'] = label_synth1

if args.combination_type == 'interaction':
    df0.to_csv(os.path.join(args.data_dir, 'music_reviews_label_synthetic_numerical_coef_interaction.csv'), index=False)
    df1.to_csv(os.path.join(args.data_dir, 'office_reviews_label_synthetic_numerical_coef_interaction.csv'), index=False)
elif args.combination_type == 'predreg':
    df0.to_csv(os.path.join(args.data_dir, 'music_reviews_label_synthetic_numerical_coef_predreg.csv'), index=False)
    df1.to_csv(os.path.join(args.data_dir, 'office_reviews_label_synthetic_numerical_coef_predreg.csv'), index=False)
elif args.combination_type == 'direct':
    df0.to_csv(os.path.join(args.data_dir, 'music_reviews_label_synthetic_numerical_coef_direct{}{}.csv'.format(args.lexicon_weight, args.embedding_weight)), index=False)
    df1.to_csv(os.path.join(args.data_dir, 'office_reviews_label_synthetic_numerical_coef_direct{}{}.csv'.format(args.lexicon_weight, args.embedding_weight)), index=False)