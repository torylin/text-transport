import pdb
import os
import argparse
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from nltk import tokenize
from tqdm import tqdm

def get_prob(model, clf, sent_group):
    sents = tokenize.sent_tokenize(sent_group)
    sents_embed = model.encode(sents)
    probs = clf.predict_proba(sents_embed)
    p0 = np.prod(probs[:,0])
    p1 = np.prod(probs[:,1])
    # if p0 > p1:
    #     return np.array([p0, 1-p0])
    # else:
    #     return np.array([1-p1, p1])
    return np.array([p0, p1])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/home/victorialin/Documents/2022-2023/causal_text/data/')
    parser.add_argument('--seed', type=int, default=230224)
    parser.add_argument('--treatment', type=str, default='treatycommit')
    parser.add_argument('--marginal-probs', action='store_true')
    args = parser.parse_args()

    return args

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

args = get_args()

df_random = pd.read_csv(os.path.join(args.data_dir, 'hk_rct', 'HKData.csv'))
df_target = pd.read_csv(os.path.join(args.data_dir, 'hk_speeches', 'target_corpus.csv'))
df_random['text_full'] = df_random.text1.fillna('') + ' ' + df_random.text2.fillna('') + ' ' + df_random.text3.fillna('')

model = SentenceTransformer('all-mpnet-base-v2')
model.to(device)

sentences = np.hstack([df_random['text_full'].values, df_target['text_full'].values])
labels = np.array([0]*df_random.shape[0] + [1]*df_target.shape[0])
sentences, labels = shuffle(sentences, labels, random_state=args.seed)

embeds = model.encode(sentences, show_progress_bar=True)

X_train, X_test, y_train, y_test, sentence_train, sentence_test = train_test_split(embeds, labels, sentences, test_size=0.9, random_state=args.seed)

clf = LogisticRegressionCV(cv=5, random_state=args.seed).fit(X_train, y_train)

test_df = pd.DataFrame({'text_full': sentence_test})
df_random_test = pd.merge(df_random, test_df, on=['text_full'], how='inner').drop_duplicates(subset=['text_full'])
df_target_test = pd.merge(df_target, test_df, on=['text_full'], how='inner').drop_duplicates(subset=['text_full'])

if args.marginal_probs:
    prob_list = []
    for sent_group in tqdm(df_random_test['text_full'].values):
        prob_list.append(get_prob(model, clf, sent_group))
    probs = np.vstack(prob_list)

else:
    test_embeds = model.encode(df_random_test['text_full'].values, show_progress_bar=True)
    probs = clf.predict_proba(test_embeds)

df_random_corp = pd.read_pickle(os.path.join(args.data_dir, 'hk_rct', 'randomization_corpus.pkl'))
if args.treatment == 'treatycommit':
    treat_prob = df_random_corp['treatyobligation'].mean()
else:
    treat_prob = df_random_corp[args.treatment].mean()
corp_prob = np.array([df_random_test.shape[0]/(df_random_test.shape[0]+df_target_test.shape[0]), 
                      df_target_test.shape[0]/(df_random_test.shape[0]+df_target_test.shape[0])])

mu1 = np.sum(probs[:,1]*corp_prob[0]/(corp_prob[1]*probs[:,0])*df_random_test['resp'].values*df_random_test[args.treatment].values)/(df_random_test.shape[0]*treat_prob)
mu0 = np.sum(probs[:,1]*corp_prob[0]/(corp_prob[1]*probs[:,0])*df_random_test['resp'].values*(1-df_random_test[args.treatment].values))/(df_random_test.shape[0]*(1-treat_prob))
est = mu1-mu0
# print(mu1)
# print(mu0)
# print(mu1-mu0)

val_list = []
for i in tqdm(range(df_random_test.shape[0])):
    for j in range(df_random_test.shape[0]):
        d_xi = probs[i, 1]*corp_prob[0]/(corp_prob[1]*probs[i, 0])
        d_xj = probs[j, 1]*corp_prob[0]/(corp_prob[1]*probs[j, 0])
        yi = df_random_test['resp'].values[i]
        yj = df_random_test['resp'].values[j]
        pr_xixj = (1/df_random_test.shape[0])*(1/(df_random_test.shape[0]-1))
        pr_xi = 1/df_random_test.shape[0]
        pr_xj = 1/df_random_test.shape[0]

        val = d_xi*d_xj*yi*yj*(pr_xixj-pr_xi*pr_xj)/pr_xixj
        val_list.append(val)

varhat = np.sum(val_list)/(df_random_test.shape[0]**2)

print('Estimate: {}'.format(est))
print('95% CI: [{}, {}]'.format(est-1.96*np.sqrt(varhat), est+1.96*np.sqrt(varhat)))

pdb.set_trace()