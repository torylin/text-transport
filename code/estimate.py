import pdb
import os
import gc
import argparse
import pandas as pd
import numpy as np
import torch
import itertools
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, AutoModel, pipeline
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegressionCV, LinearRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nltk import tokenize
from tqdm import tqdm
from functools import partial
from transformers.utils import logging
from tabulate import tabulate

from utils import get_embeds

logging.set_verbosity_error()

def get_prob_clf(model, clf, sent_group, args):
    sents = tokenize.sent_tokenize(sent_group)
    if args.lm_library == 'sentence-transformers':
        sents_embed = model.encode(sents)
    elif args.lm_library == 'transformers':
        sents_embed = get_embeds(sents, tokenizer, model, device)
    try:
        probs = clf.predict_proba(sents_embed)
    except:
        pdb.set_trace()
    p0 = np.prod(probs[:,0])
    p1 = np.prod(probs[:,1])
    # if p0 > p1:
    #     return np.array([p0, 1-p0])
    # else:
    #     return np.array([1-p1, p1])
    return np.array([p0, p1])

def get_prob_mlm(dataloader):
    all_probs = []
    for batch in tqdm(dataloader):
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probabilities = np.repeat(1.0, predictions.shape[0])
        for j in range(0, predictions.shape[0]):
            for i in range(1, len(batch['input_ids'][0])):
                if batch['input_ids'][j, i] != 0:
                    probabilities[j] *= predictions[j, i, batch['input_ids'][j, i]].item()
        all_probs += probabilities.tolist()

    all_probs = np.array(all_probs)

    return all_probs

def get_prob_clm(dataloader):
    all_probs = []
    for batch in tqdm(dataloader):
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        outputs = model.generate(**batch, max_new_tokens=0, return_dict_in_generate=True, output_scores=True)
        probabilities = torch.exp(model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True))
        all_probs += probabilities.flatten().tolist()
    
    all_probs = np.array(all_probs)
    
    return all_probs

def get_var_clf(idx, df, weights, y_mean):
    i = idx[0]
    j = idx[1]
    # d_xi = probs[i, 1]*corp_prob[0]/(corp_prob[1]*probs[i, 0])
    # d_xj = probs[j, 1]*corp_prob[0]/(corp_prob[1]*probs[j, 0])
    # d_xi /= weight_norm
    # d_xj /= weight_norm
    d_xi = weights[i]
    d_xj = weights[j]
    # yi = df['resp'].values[i] - y_mean
    # yj = df['resp'].values[j] - y_mean
    yi = df['resp'].values[i]
    yj = df['resp'].values[j]
    if i != j:
        return 0
    val = d_xi*d_xj*yi*yj*(1-1/df.shape[0])
    # pr_xixj = pr_xi = pr_xj = 1/df.shape[0]

    # val = d_xi*d_xj*yi*yj*(pr_xixj-pr_xi*pr_xj)/pr_xixj

    return val

def get_var_lm(idx, df, weights, y_mean):
    i = idx[0]
    j = idx[1]
    # p_xi = all_probs[i]/(df.shape[0]*norm_sum)
    # p_xj = all_probs[j]/(df.shape[0]*norm_sum)
    # p_xi /= weight_norm
    # p_xj /= weight_norm
    p_xi = weights[i]
    p_xj = weights[j]
    # yi = df['resp'].values[i] - y_mean
    # yj = df['resp'].values[j] - y_mean
    yi = df['resp'].values[i]
    yj = df['resp'].values[j]
    if i != j:
        return 0
    val = p_xi*p_xj*yi*yj*(1-1/df.shape[0])
    # pr_xixj = pr_xi = pr_xj = 1/df.shape[0]

    # val = p_xi*p_xj*yi*yj*(pr_xixj-pr_xi*pr_xj)/pr_xixj

    return val

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/home/victorialin/Documents/2022-2023/causal_text/data/')
    parser.add_argument('--method', type=str, default='clf')
    parser.add_argument('--representation', type=str, default='embedding')
    parser.add_argument('--seed', type=int, default=230224)
    parser.add_argument('--treatment', type=str, default='treatycommit')
    parser.add_argument('--clf', type=str, default='logistic')
    parser.add_argument('--estimate', type=str, default='diff')
    parser.add_argument('--lm-name', type=str, default='bert-base-uncased')
    parser.add_argument('--lm-library', type=str, default='transformers')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--marginal-probs', action='store_true')
    parser.add_argument('--ci', action='store_true')
    parser.add_argument('--scale', action='store_true')
    args = parser.parse_args()

    return args

gc.collect()
torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

args = get_args()

# df_random = pd.read_csv(os.path.join(args.data_dir, 'hk_rct', 'HKData.csv'))
df_random = pd.read_csv(os.path.join(args.data_dir, 'hk_rct', 'HKRepData.csv'))
df_target = pd.read_csv(os.path.join(args.data_dir, 'hk_speeches', 'target_corpus_fullprobs.csv'))
df_random['text_full'] = df_random.text1.fillna('') + ' ' + df_random.text2.fillna('') + ' ' + df_random.text3.fillna('')

df_random_corp = pd.read_pickle(os.path.join(args.data_dir, 'hk_rct', 'randomization_corpus_random_sample.pkl'))
if args.treatment == 'treatycommit':
    treat_prob = df_random_corp['treatyobligation'].mean()
else:
    treat_prob = df_random_corp[args.treatment].mean()

if args.method == 'mlm' or args.method == 'clm':
    if args.method == 'mlm':
        get_prob_lm = get_prob_mlm
        tokenizer = AutoTokenizer.from_pretrained(args.lm_name, max_length=512, truncation=True, padding=True)
        model = AutoModelForMaskedLM.from_pretrained(args.lm_name)
    elif args.method == 'clm':
        get_prob_lm = get_prob_clm
        tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.lm_name)

    model.to(device)

    if args.marginal_probs:
        tokens1 = tokenizer(df_random.text1.fillna('').astype(str).values.tolist(), return_tensors='pt', padding=True, truncation=True)
        tokens2 = tokenizer(df_random.text2.fillna('').astype(str).values.tolist(), return_tensors='pt', padding=True, truncation=True)
        tokens3 = tokenizer(df_random.text3.fillna('').astype(str).values.tolist(), return_tensors='pt', padding=True, truncation=True)

        ds1 = Dataset.from_dict(tokens1).with_format('torch')
        dataloader1 = DataLoader(ds1, batch_size=args.batch_size, shuffle=False)

        ds2 = Dataset.from_dict(tokens2).with_format('torch')
        dataloader2 = DataLoader(ds2, batch_size=args.batch_size, shuffle=False)

        ds3 = Dataset.from_dict(tokens3).with_format('torch')
        dataloader3 = DataLoader(ds3, batch_size=args.batch_size, shuffle=False)

        all_probs1 = get_prob_lm(dataloader1)
        all_probs2 = get_prob_lm(dataloader2)
        all_probs3 = get_prob_lm(dataloader3)

        all_probs = all_probs1*all_probs2*all_probs3

    else:
        tokens = tokenizer(df_random['text_full'].astype(str).values.tolist(), return_tensors='pt', padding=True, truncation=True)
        ds = Dataset.from_dict(tokens).with_format('torch')
        dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

        all_probs = get_prob_lm(dataloader)
    
    norm_sum = np.sum(all_probs)
    pr_x = 1/df_random.shape[0]

    if args.estimate == 'lr':
        y_adj = df_random['resp']*all_probs/(pr_x*norm_sum)
        lr = LinearRegression()
        lr.fit(df_random.drop(['resp', 'numtexts', 'text1', 'text2', 'text3', 'text_full', 'resp_id'], axis=1), y_adj)
        print(df_random.drop(['resp', 'numtexts', 'text1', 'text2', 'text3', 'text_full', 'resp_id'], axis=1).columns)
        print(lr.coef_)
        print(lr.intercept_)
        quit()

    n1 = np.sum(df_random[args.treatment].values)
    n0 = np.sum(1-df_random[args.treatment].values)
    weights1 = all_probs*df_random[args.treatment].values/(pr_x*norm_sum)
    weight_norm1 = np.sum(weights1)/n1
    weights1 /= weight_norm1
    mu1 = np.sum(weights1*df_random['resp'].values)/n1

    weights0 = all_probs*(1-df_random[args.treatment].values)/(pr_x*norm_sum)
    weight_norm0 = np.sum(weights0)/n0
    weights0 /= weight_norm0
    mu0 = np.sum(weights0*df_random['resp'].values)/n0

    # mu1 = np.sum(all_probs*df_random['resp'].values*df_random[args.treatment].values/(pr_x*norm_sum))/(df_random.shape[0]*treat_prob)
    # mu0 = np.sum(all_probs*df_random['resp'].values*(1-df_random[args.treatment].values)/(pr_x*norm_sum))/(df_random.shape[0]*(1-treat_prob))
    est = mu1 - mu0

    if args.ci:
    
        # idxs1 = list(itertools.product(df_random[df_random[args.treatment]==1].index, df_random[df_random[args.treatment]==1].index))
        # var_list1 = np.array(list(map(partial(get_var_lm, df=df_random, weights=weights1,
                                            #   y_mean=np.mean(df_random[df_random[args.treatment]==1]['resp'].values)), tqdm(idxs1))))
        y_mean1 = np.mean(df_random[df_random[args.treatment]==1]['resp'].values)
        # var_list1 = (weights1[df_random[df_random[args.treatment]==1].index]**2)*((df_random[df_random[args.treatment]==1]['resp'].values-y_mean1)**2)*(1-1/df_random.shape[0])
        var_list1 = (weights1[df_random[df_random[args.treatment]==1].index]**2)*((df_random[df_random[args.treatment]==1]['resp'].values)**2)*(1-1/df_random.shape[0])
        # var_list1 = (weights1[df_random[df_random[args.treatment]==1].index]**2)*((df_random[df_random[args.treatment]==1]['resp'].values-y_mean1)**2)
        # varhat1 = np.sum(var_list1)/((df_random.shape[0]*treat_prob)**2)
        varhat1 = np.sum(var_list1)/(n1**2)
        # varhat1 = np.sum(var_list1)/(n1*(n1-1))

        # idxs0 = list(itertools.product(df_random[df_random[args.treatment]==0].index, df_random[df_random[args.treatment]==0].index))
        # var_list0 = np.array(list(map(partial(get_var_lm, df=df_random, weights=weights0,
                                            #   y_mean=np.mean(df_random[df_random[args.treatment]==0]['resp'].values)), tqdm(idxs0))))
        # varhat0 = np.sum(var_list0)/((df_random.shape[0]*(1-treat_prob))**2)
        y_mean0 = np.mean(df_random[df_random[args.treatment]==0]['resp'].values)
        # var_list0 = (weights0[df_random[df_random[args.treatment]==0].index]**2)*((df_random[df_random[args.treatment]==0]['resp'].values-y_mean0)**2)*(1-1/df_random.shape[0])
        var_list0 = (weights0[df_random[df_random[args.treatment]==0].index]**2)*((df_random[df_random[args.treatment]==0]['resp'].values)**2)*(1-1/df_random.shape[0])
        # var_list0 = (weights0[df_random[df_random[args.treatment]==0].index]**2)*((df_random[df_random[args.treatment]==0]['resp'].values-y_mean0)**2)
        varhat0 = np.sum(var_list0)/(n0**2)
        # varhat0 = np.sum(var_list1)/(n0*(n0-1))

        varhat = varhat1 + varhat0

elif args.method == 'clf':

    sentences = np.hstack([df_random['text_full'].values, df_target['text_full'].values])
    labels = np.array([0]*df_random.shape[0] + [1]*df_target.shape[0])
    sentences, labels = shuffle(sentences, labels, random_state=args.seed)

    if args.lm_library == 'sentence-transformers':
        model = SentenceTransformer(args.lm_name)
        model.to(device)
        embeds = model.encode(sentences, show_progress_bar=True)

    elif args.lm_library == 'transformers':
        tokenizer = AutoTokenizer.from_pretrained(args.lm_name, max_length=512, truncation=True)
        model = AutoModel.from_pretrained(args.lm_name)
        model.to(device)
        if 'Masked' not in model.config.architectures[0]:
            tokenizer.pad_token = tokenizer.eos_token
        
        embeds = get_embeds(sentences.astype(str).tolist(), tokenizer, model, device)

    X_train, X_test, y_train, y_test, sentence_train, sentence_test = train_test_split(embeds, labels, sentences, test_size=0.9, random_state=args.seed)

    if args.clf == 'logistic':
        clf = LogisticRegressionCV(cv=5, random_state=args.seed, max_iter=1000)
    elif args.clf == 'elasticnet':
        clf = LogisticRegressionCV(cv=5, random_state=args.seed, penalty='elasticnet', solver='saga', l1_ratios=[0.5]*5, max_iter=1000)
    if args.scale:
        scaler = StandardScaler()
        scaler.fit(X_train)
        clf.fit(scaler.transform(X_train), y_train)
    else:
        clf.fit(X_train, y_train)

    train_df = pd.DataFrame({'text_full': sentence_train})
    test_df = pd.DataFrame({'text_full': sentence_test})
    df_random_test = pd.merge(df_random, test_df, on=['text_full'], how='inner').drop_duplicates(subset=['text_full']).reset_index(drop=True)
    df_target_test = pd.merge(df_target, test_df, on=['text_full'], how='inner').drop_duplicates(subset=['text_full']).reset_index(drop=True)
    df_random_train = pd.merge(df_random, train_df, on=['text_full'], how='inner').drop_duplicates(subset=['text_full']).reset_index(drop=True)
    df_target_train = pd.merge(df_target, train_df, on=['text_full'], how='inner').drop_duplicates(subset=['text_full']).reset_index(drop=True)

    df_random_train.to_csv(os.path.join(args.data_dir, 'hk_rct', 'random_train.csv'), index=False)
    df_target_train.to_csv(os.path.join(args.data_dir, 'hk_speeches', 'target_train.csv'), index=False)
    df_random_test.to_csv(os.path.join(args.data_dir, 'hk_rct', 'random_test.csv'), index=False)
    df_target_test.to_csv(os.path.join(args.data_dir, 'hk_speeches', 'target_test.csv'), index=False)

    # test_embeds = model.encode(df_random_test['text_full'].values, show_progress_bar=True)
    # probs = clf.predict_proba(test_embeds)

    if args.marginal_probs:
        prob_list = []
        for sent_group in tqdm(df_random_test['text_full'].values):
            prob_list.append(get_prob_clf(model, clf, sent_group, args))
        probs = np.vstack(prob_list)
        # marginal_probs = np.vstack(prob_list)
        # probs = np.hstack([probs[:,0].reshape(-1, 1), marginal_probs[:,1].reshape(-1, 1)])

    else:
        if args.lm_library == 'sentence-transformers':
            test_embeds = model.encode(df_random_test['text_full'].values, show_progress_bar=True)
        elif args.lm_library == 'transformers':
            test_embeds = get_embeds(df_random_test['text_full'].values.astype(str).tolist(), tokenizer, model, device)

        if args.scale:
            probs = clf.predict_proba(scaler.transform(test_embeds))
        else:
            probs = clf.predict_proba(test_embeds)

    corp_prob = np.array([df_random_test.shape[0]/(df_random_test.shape[0]+df_target_test.shape[0]), 
                        df_target_test.shape[0]/(df_random_test.shape[0]+df_target_test.shape[0])])

    if args.estimate == 'lr':
        y_adj = probs[:,1]*corp_prob[0]/(corp_prob[1]*probs[:,0])*df_random_test['resp'].values
        lr = LinearRegression()
        lr.fit(df_random_test.drop(['resp', 'numtexts', 'text1', 'text2', 'text3', 'text_full', 'resp_id'], axis=1), y_adj)
        print(df_random_test.drop(['resp', 'numtexts', 'text1', 'text2', 'text3', 'text_full', 'resp_id'], axis=1).columns)
        print(lr.coef_)
        print(lr.intercept_)
        quit()

    n1 = np.sum(df_random_test[args.treatment].values)
    n0 = np.sum(1-df_random_test[args.treatment].values)
    weights = np.repeat(1.0, df_random_test.shape[0])

    weights1 = probs[:,1]*corp_prob[0]/(corp_prob[1]*probs[:,0])*df_random_test[args.treatment].values
    weight_norm1 = np.sum(weights1)/n1
    weights1 /= weight_norm1
    mu1 = np.sum(weights1*df_random_test['resp'].values)/n1

    weights0 = probs[:,1]*corp_prob[0]/(corp_prob[1]*probs[:,0])*(1-df_random_test[args.treatment].values)
    weight_norm0 = np.sum(weights0)/n0
    weights0 /= weight_norm0
    mu0 = np.sum(weights0*df_random_test['resp'].values)/n0

    # mu1 = np.sum(probs[:,1]*corp_prob[0]/(corp_prob[1]*probs[:,0])*df_random_test['resp'].values*df_random_test[args.treatment].values)/(df_random_test.shape[0]*treat_prob)
    # mu0 = np.sum(probs[:,1]*corp_prob[0]/(corp_prob[1]*probs[:,0])*df_random_test['resp'].values*(1-df_random_test[args.treatment].values))/(df_random_test.shape[0]*(1-treat_prob))
    est = mu1 - mu0

    if args.ci:

        # idxs = list(itertools.product(range(df_random_test.shape[0]), range(df_random_test.shape[0])))
        # var_list = np.array(list(map(partial(get_var_clf, df=df_random_test), tqdm(idxs))))
        # varhat = np.sum(var_list)/(df_random_test.shape[0]**2)
        y_mean1 = np.mean(df_random_test[df_random_test[args.treatment]==1]['resp'].values)
        # var_list1 = (weights1[df_random_test[df_random_test[args.treatment]==1].index]**2)*((df_random_test[df_random_test[args.treatment]==1]['resp'].values-y_mean1)**2)*(1-1/df_random_test.shape[0])
        var_list1 = (weights1[df_random_test[df_random_test[args.treatment]==1].index]**2)*((df_random_test[df_random_test[args.treatment]==1]['resp'].values)**2)*(1-1/df_random_test.shape[0])
        # var_list1 = (weights1[df_random_test[df_random_test[args.treatment]==1].index]**2)*((df_random_test[df_random_test[args.treatment]==1]['resp'].values-y_mean1)**2)

        # idxs1 = list(itertools.product(df_random_test[df_random_test[args.treatment]==1].index, df_random_test[df_random_test[args.treatment]==1].index))
        # var_list1 = np.array(list(map(partial(get_var_clf, df=df_random_test, weights=weights1, 
        #                                       y_mean=np.mean(df_random_test[df_random_test[args.treatment]==1]['resp'].values)), tqdm(idxs1))))
        # varhat1 = np.sum(var_list1)/((df_random_test.shape[0]*treat_prob)**2)
        varhat1 = np.sum(var_list1)/(n1**2)
        # varhat1 = np.sum(var_list1)/(n1*(n1-1))
        # idxs0 = list(itertools.product(df_random_test[df_random_test[args.treatment]==0].index, df_random_test[df_random_test[args.treatment]==0].index))
        # var_list0 = np.array(list(map(partial(get_var_clf, df=df_random_test, weights=weights0,
        #                                       y_mean=np.mean(df_random_test[df_random_test[args.treatment]==0]['resp'].values)), tqdm(idxs0))))
        # varhat0 = np.sum(var_list0)/((df_random_test.shape[0]*(1-treat_prob))**2)
        y_mean0 = np.mean(df_random_test[df_random_test[args.treatment]==0]['resp'].values)
        # var_list0 = (weights0[df_random_test[df_random_test[args.treatment]==0].index]**2)*((df_random_test[df_random_test[args.treatment]==0]['resp'].values-y_mean0)**2)*(1-1/df_random_test.shape[0])
        var_list0 = (weights0[df_random_test[df_random_test[args.treatment]==0].index]**2)*((df_random_test[df_random_test[args.treatment]==0]['resp'].values)**2)*(1-1/df_random_test.shape[0])
        # var_list0 = (weights0[df_random_test[df_random_test[args.treatment]==0].index]**2)*((df_random_test[df_random_test[args.treatment]==0]['resp'].values-y_mean0)**2)
        varhat0 = np.sum(var_list0)/(n0**2)
        # varhat0 = np.sum(var_list0)/(n0*(n0-1))

        varhat = varhat1 + varhat0
        # varhat_alt = (np.sum(var_list1) + np.sum(var_list0))/(df_random_test.shape[0]**2)

rows = [['{:.3f}'.format(mu1), '{:.3f}'.format(mu0), '{:.3f}'.format(est)]]
print('mu1: {}'.format(mu1))
print('mu0: {}'.format(mu0))
print('Estimate: {}'.format(est))
if args.ci:
    rows[0] += ['[{:.3f}, {:.3f}]'.format(est-1.96*np.sqrt(varhat), est+1.96*np.sqrt(varhat))]
    # print('95% CI: [{}, {}]'.format(est-1.96*np.sqrt(varhat), est+1.96*np.sqrt(varhat)))
    print('95% CI: [{}, {}]'.format(est-1.96*np.sqrt(varhat), est+1.96*np.sqrt(varhat)))
    # print('95% CI alt: [{}, {}]'.format(est-1.96*np.sqrt(varhat_alt), est+1.96*np.sqrt(varhat_alt)))

print(tabulate(rows, headers=['mu1', 'mu0', 'est', 'ci'], tablefmt='latex'))
