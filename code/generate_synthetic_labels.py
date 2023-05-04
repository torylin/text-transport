import pandas as pd
import pdb
import numpy as np
import os
import argparse
from sklearn.linear_model import LogisticRegressionCV

from utils import get_unnoised_labels, noise_labels

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/home/victorialin/Documents/2022-2023/causal_text/data/amazon_synthetic/')
    parser.add_argument('--combination-type', type=str, default='predreg')
    parser.add_argument('--lexicon-weight', type=float, default=1.0)
    parser.add_argument('--embedding-weight', type=float, default=0.5)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--resample-prpt', action='store_true')
    args = parser.parse_args()

    return args

args = get_args()

df0 = pd.read_csv(os.path.join(args.data_dir, 'combined_liwc_categorymusic.csv'))
df1 = pd.read_csv(os.path.join(args.data_dir, 'combined_liwc_categoryoffice.csv'))
prob_df0 = pd.read_csv(os.path.join(args.data_dir, 'music_reviews_pred_numerical.csv'))
prob_df1 = pd.read_csv(os.path.join(args.data_dir, 'office_reviews_pred_numerical.csv'))

label_synth0, label_synth1 = get_unnoised_labels(df0, df1, prob_df0, prob_df1, args.combination_type, args.lexicon_weight, args.embedding_weight)

df0['label_synthetic_unnoised'] = label_synth0
df1['label_synthetic_unnoised'] = label_synth1

noise0, noise1 = noise_labels(label_synth0, label_synth1)
label_synth0 += noise0
label_synth1 += noise1

df0.drop(['helpful'], axis=1, inplace=True)
df1.drop(['helpful'], axis=1, inplace=True)

df0['label_synthetic'] = label_synth0
df1['label_synthetic'] = label_synth1

if args.save and not args.resample_prpt:
    if args.combination_type == 'interaction':
        df0.to_csv(os.path.join(args.data_dir, 'music_reviews_label_synthetic_numerical_coef_interaction.csv'), index=False)
        df1.to_csv(os.path.join(args.data_dir, 'office_reviews_label_synthetic_numerical_coef_interaction.csv'), index=False)
    elif args.combination_type == 'predreg':
        df0.to_csv(os.path.join(args.data_dir, 'music_reviews_label_synthetic_numerical_coef_predreg.csv'), index=False)
        df1.to_csv(os.path.join(args.data_dir, 'office_reviews_label_synthetic_numerical_coef_predreg.csv'), index=False)
    elif args.combination_type == 'direct':
        df0.to_csv(os.path.join(args.data_dir, 'music_reviews_label_synthetic_numerical_coef_direct{}{}.csv'.format(args.lexicon_weight, args.embedding_weight)), index=False)
        df1.to_csv(os.path.join(args.data_dir, 'office_reviews_label_synthetic_numerical_coef_direct{}{}.csv'.format(args.lexicon_weight, args.embedding_weight)), index=False)

if args.resample_prpt:
    top10 = ['sexual', 'female', 'filler', 'netspeak', 'home', 'informal', 'shehe', 'nonflu', 'assent', 'death']
    df0['corpus'] = 0
    df1['corpus'] = 1
    df = pd.concat([df0, df1], axis=0)
    pdb.set_trace()
    clf = LogisticRegressionCV(cv=5, random_state=230503, max_iter=10000)
    clf.fit(df[top10], df['corpus'].values)
    probs = clf.predict_proba(df[top10])
    df['pt'] = probs[:,1]
    df['pr'] = probs[:,0]
    df0_new = df.sample(n=5000, replace=True, weights='pr', random_state=230425)
    df1_new = df.sample(n=5000, replace=True, weights='pt', random_state=102938)
    df0_new.drop(['corpus'], axis=1, inplace=True)
    df1_new.drop(['corpus'], axis=1, inplace=True)

    if args.save and args.combination_type == 'direct':
        df0_new.to_csv(os.path.join(args.data_dir, 'pr_sampled_label_synthetic_numerical_coef_direct{}{}.csv'.format(args.lexicon_weight, args.embedding_weight)), index=False)
        df1_new.to_csv(os.path.join(args.data_dir, 'pt_sampled_label_synthetic_numerical_coef_direct{}{}.csv'.format(args.lexicon_weight, args.embedding_weight)), index=False)

pdb.set_trace()
