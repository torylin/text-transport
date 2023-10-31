import torch
import nltk
import contractions
import numpy as np
from tqdm import tqdm
from collections import Counter
from datasets import Dataset
from torch.utils.data import DataLoader
from empath import Empath
import pdb
import os
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegressionCV, ElasticNet
from sklearn.metrics import r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
import liwc

def get_embeds(sentences, tokenizer, model, device, progress=True):
    tokens = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    ds = Dataset.from_dict(tokens).with_format('torch')
    dataloader = DataLoader(ds, batch_size=16, shuffle=False)
    embeds_list = []
    if progress:
        for batch in tqdm(dataloader):
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            with torch.no_grad():
                outputs = model(**batch)
            out = outputs.last_hidden_state.mean(axis=1)
            embeds_list.append(out.detach().cpu())
    else:
        for batch in dataloader:
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            with torch.no_grad():
                outputs = model(**batch)
            out = outputs.last_hidden_state.mean(axis=1)
            embeds_list.append(out.detach().cpu())
            
    embeds = np.vstack(embeds_list)
    
    return embeds

def get_liwc_labels(utterance, rel_cats, parse, binary=False):
    try:
        tokens = nltk.word_tokenize(contractions.fix(utterance).lower())
    except:
        pdb.set_trace()
    counts = dict(Counter(category for token in tokens for category in parse(token)))

    # print(counts)

    label_vec = np.zeros(len(rel_cats))
    bin_label_vec = np.zeros(len(rel_cats))

    for i in range(len(rel_cats)):
        if not binary:
            label_vec[i] += counts.get(rel_cats[i], 0)
        else:
            if counts.get(rel_cats[i], 0) > 0:
                bin_label_vec[i] = 1

    if not binary:
        return label_vec
    return bin_label_vec

def get_empath_labels(utterance, lexicon):
    return list(lexicon.analyze(utterance.lower()).values())

def get_unnoised_labels(df0, df1, prob_df0, prob_df1, combination_type, lexicon_weight=1.0, embedding_weight=0.0, output=False):

    df = pd.concat([df0, df1], axis=0)
    prob_df = pd.concat([prob_df0, prob_df1], axis=0)

    X = df.drop(['reviewText', 'helpful'], axis=1)
    y = df['helpful'].values

    model = LinearRegression()
    # model = ElasticNet(l1_ratio=0.5)
    model.fit(X, y)

    features = pd.DataFrame({'feature': X.columns.tolist(), 'coef': model.coef_})
    features = features.reindex(features.coef.abs().sort_values(ascending=False).index)
    lex_pred = (df[features['feature'][0:10].values]*features.coef[0:10].values).sum(axis=1).values
    if output:
        print(features['feature'][0:10].values)
        print('Lexical model RMSE: {:.3f}'.format(np.sqrt(np.sum((model.predict(X)-y)**2))))
        print('Lexical model RMSE (10 features): {:.3f}'.format(np.sqrt(np.sum((lex_pred-y)**2))))
        print('Embedding model RMSE: {:.3f}'.format(np.sqrt(np.sum((prob_df['pred'].values-y)**2))))
        print('Lexical model R^2: {:.3f}'.format(model.score(X, y)))
        print('Lexical model R^2 (10 features): {:.3f}'.format(r2_score(y, lex_pred)))
        print('Embedding model R^2: {:.3f}'.format(r2_score(y, prob_df['pred'].values)))

    if combination_type == 'interaction':
        X_new = df[features['feature'][0:10].values]
        X_new['pred'] = prob_df['pred']
        X_new['helpful'] = y
        interaction_model = smf.ols(
            formula='helpful ~ sexual*pred + female*pred + filler*pred + netspeak*pred + home*pred + informal*pred + shehe*pred + nonflu*pred + assent*pred + death*pred', data=X_new)
        res = interaction_model.fit()
        preds = res.predict()
        if output:
            print('Interaction model RMSE: {:.3f}'.format(np.sqrt(np.sum((preds-y)**2))))
            print('Interaction model R^2: {:.3f}'.format(r2_score(y, preds)))
        label_synth0 = preds[:df0.shape[0]]
        label_synth1 = preds[df0.shape[0]:]
    elif combination_type == 'predreg':
        lex_pred = (df[features['feature'][0:10].values]*features.coef[0:10].values).sum(axis=1).values
        embed_pred = prob_df['pred'].values
        X_new = pd.DataFrame({'lex_pred': lex_pred, 'embed_pred': embed_pred})
        reg_model = LinearRegression()
        reg_model.fit(X_new, y)
        print(reg_model.coef_)
        preds = reg_model.predict(X_new)
        if output:
            print('Combined model RMSE: {:.3f}'.format(np.sqrt(np.sum((preds-y)**2))))
            print('Combined model R^2: {:.3f}'.format(r2_score(y, preds)))
        label_synth0 = preds[:df0.shape[0]] 
        label_synth1 = preds[df0.shape[0]:]

    elif combination_type == 'direct':
        label_synth0 = (df0[features['feature'][0:10].values]*features.coef[0:10].values).sum(axis=1)*lexicon_weight + prob_df0['pred']*embedding_weight
        label_synth1 = (df1[features['feature'][0:10].values]*features.coef[0:10].values).sum(axis=1)*lexicon_weight + prob_df1['pred']*embedding_weight

    return (label_synth0, label_synth1)

def noise_labels(label_synth0, label_synth1=None, mu=0, sigma=1, seed=230418):
    np.random.seed(seed)
    noise0 = np.random.normal(mu, sigma, len(label_synth0))
    if label_synth1 is not None:
        noise1 = np.random.normal(mu, sigma, len(label_synth1))

        return (noise0, noise1)
    return noise0

def amazon_sample(df, high_prob_df, seed0=230425, seed1=102938, existing_probs=False):

    df0_new = df.sample(n=5000, replace=True, weights='pr', random_state=seed0)
    df1_new = df.sample(n=5000, replace=True, weights='pt', random_state=seed1)
    df0_new.drop(['corpus'], axis=1, inplace=True)
    df1_new.drop(['corpus'], axis=1, inplace=True)

    if existing_probs:
        df0_new = pd.merge(df0_new, high_prob_df, on=['reviewText'], how='inner', suffixes=('', '_y'))
        df0_new.drop(df0_new.filter(regex='_y$').columns, axis=1, inplace=True)

    return (df0_new, df1_new)


# def emobank_sample(data_dir='/home/victorialin/Documents/2022-2023/causal_text/data/emobank/',
#                    outcome='V_writer', outcome_reader='V_reader', outcome_name='valence',
#                    seedhigh=230425, seedlow=102938, existing_probs=False, existing_clf_probs=False):

def emobank_sample(df_combined, high_prob_df, low_prob_df,
                   seedhigh=230425, seedlow=102938, existing_probs=False, existing_clf_probs=False):
    
    # emobank = pd.read_csv(os.path.join(data_dir, 'emobank.csv'))
    # df_writer = pd.read_csv(os.path.join(data_dir, 'writer.csv'))
    # df_reader = pd.read_csv(os.path.join(data_dir, 'reader.csv'))

    # emobank.columns = ['id', 'split', 'V_combined', 'A_combined', 'D_combined', 'text']
    
    # df_writer.drop(['stdV', 'stdA', 'stdD', 'N'], axis=1, inplace=True)
    # df_writer.columns = ['id', 'V_writer', 'A_writer', 'D_writer']

    # df_reader.drop(['stdV', 'stdA', 'stdD', 'N'], axis=1, inplace=True)
    # df_reader.columns = ['id', 'V_reader', 'A_reader', 'D_reader']

    # df_combined = emobank.merge(df_writer.merge(df_reader, on='id'), on='id')

    # writer_median = df_combined[outcome].median()

    # df_combined = df_combined[df_combined[outcome] != writer_median].reset_index(drop=True)

    df_highwriterintent = df_combined.sample(n=5000, replace=True, weights='P(V>3)', random_state=seedhigh)
    df_lowwriterintent = df_combined.sample(n=5000, replace=True, weights='P(V<3)', random_state=seedlow)

    # if existing_probs or existing_clf_probs:
    #     df_highwriterintent = pd.merge(df_highwriterintent, high_prob_df, on=['text'], how='inner', suffixes=('', '_y'))
    #     df_lowwriterintent = pd.merge(df_lowwriterintent, low_prob_df,  on=['text'], how='inner', suffixes=('', '_y'))
    #     df_highwriterintent.drop(df_highwriterintent.filter(regex='_y$').columns, axis=1, inplace=True)
    #     df_lowwriterintent.drop(df_lowwriterintent.filter(regex='_y$').columns, axis=1, inplace=True)

    return (df_combined, df_highwriterintent, df_lowwriterintent)

def list_func(row):
    row['id'] = row['id'].replace('\t', '').split('\n')
    row['text'] = row['text'].replace('\t', '').split('\n')
    try:
        row['hate_speech_idx'] = int(row['hate_speech_idx'].replace('[', '').replace(']', ''))
    except:
        pass

    if len(row['id']) != len(row['text']):
        # print('Lengths not equal')
        row['text'] = 'Lengths not equal'
        
    return row

def process_data(df):
    df_new = df.apply(list_func, axis=1)
    df_new = df_new[df_new['text'] != 'Lengths not equal']
    df_new = df_new.explode(['id', 'text'])

    df_new['idx'] = df_new.apply(lambda row: row['id'].split('.')[0], axis=1)
    df_new = df_new[df_new['idx'] != '']
    df_new['idx'] = df_new['idx'].astype(int)

    df_new['hatespeech'] = 0
    df_new.loc[df_new['hate_speech_idx'] == df_new['idx'], 'hatespeech'] = 1

    df_new['id'] = df_new['id'].apply(lambda row: row.split('.')[1])
    df_new['text'] = df_new['text'].apply(lambda row: '.'.join(row.split('.')[1:]))

    df_new.drop(['hate_speech_idx', 'response'], axis=1, inplace=True)

    return df_new

def hatespeech_sample(df_r, df_t, data_dir='/home/victorialin/Documents/2022-2023/causal_text/data/hatespeech/',
                      seedreddit=230607, seedgab=230607, existing_probs=False, existing_clf_probs=False):


    reddit_new = df_r.sample(1100, random_state=seedreddit)
    gab_new = df_t.sample(1800, random_state=seedgab)

    # reddit_new = process_data(reddit_downsample)
    # gab_new = process_data(gab_downsample)
    

    # if existing_probs or existing_clf_probs:
    #     if existing_probs:
    #         reddit_df = pd.read_csv(os.path.join(data_dir, 'reddit_liwc_w_probs.csv'))
    #         gab_df = pd.read_csv(os.path.join(data_dir, 'reddit_liwc_w_probs.csv'))
    if existing_clf_probs:
        reddit_df = pd.read_csv(os.path.join(data_dir, 'reddit_liwc_w_clf_probs.csv'))
        gab_df = pd.read_csv(os.path.join(data_dir, 'gab_liwc_w_clf_probs.csv'))
        reddit_new = pd.merge(reddit_new, reddit_df, on=['text'], how='inner', suffixes=('', '_y'))
        gab_new = pd.merge(gab_new, gab_df,  on=['text'], how='inner', suffixes=('', '_y'))
        reddit_new.drop(reddit_new.filter(regex='_y$').columns, axis=1, inplace=True)
        gab_new.drop(gab_new.filter(regex='_y$').columns, axis=1, inplace=True)

    return(reddit_new, gab_new)


def get_lexical_df(lexicon_name, df, text_col, outcome, pr_col=None, pt_col=None):

    if lexicon_name == 'empath':
        lexicon = Empath()
        category_names = list(lexicon.cats.keys())
        df['label_count'] = df[text_col].apply(get_empath_labels, args=(lexicon, ))

    elif lexicon_name == 'liwc':

        parse, category_names = liwc.load_token_parser('/home/victorialin/Documents/liwc_dict/LIWC2015_English_Flat.dic')
        df['label_count'] = df[text_col].apply(get_liwc_labels, args=(category_names, parse, True))

    count_df = pd.DataFrame(np.stack(df['label_count'].values, axis=0), columns=category_names)

    count_df[text_col] = df[text_col].values
    count_df[outcome] = df[outcome].values
    
    if pr_col is not None:
        count_df[pr_col] = df[pr_col].values
    if pt_col is not None:
        count_df[pt_col] = df[pt_col].values

    del(df)

    return count_df