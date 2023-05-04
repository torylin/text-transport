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
from sklearn.linear_model import LinearRegression, ElasticNet
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

def emobank_sample(data_dir='/home/victorialin/Documents/2022-2023/causal_text/data/emobank/',
                   outcome='V_writer', outcome_reader='V_reader', outcome_name='valence',
                   seedhigh=230425, seedlow=102938):
    
    emobank = pd.read_csv(os.path.join(data_dir, 'emobank.csv'))
    df_writer = pd.read_csv(os.path.join(data_dir, 'writer.csv'))
    df_reader = pd.read_csv(os.path.join(data_dir, 'reader.csv'))

    emobank.columns = ['id', 'split', 'V_combined', 'A_combined', 'D_combined', 'text']
    
    df_writer.drop(['stdV', 'stdA', 'stdD', 'N'], axis=1, inplace=True)
    df_writer.columns = ['id', 'V_writer', 'A_writer', 'D_writer']

    df_reader.drop(['stdV', 'stdA', 'stdD', 'N'], axis=1, inplace=True)
    df_reader.columns = ['id', 'V_reader', 'A_reader', 'D_reader']

    df_combined = emobank.merge(df_writer.merge(df_reader, on='id'), on='id')

    writer_median = df_combined[outcome].median()

    df_combined = df_combined[df_combined[outcome] != writer_median].reset_index(drop=True)

    df_combined['P(V>3)'] = 0.2
    df_combined['P(V<3)'] = 0.2
    df_combined.loc[df_combined[outcome] > writer_median, 'P(V>3)'] = 0.8
    df_combined.loc[df_combined[outcome] < writer_median, 'P(V<3)'] = 0.8

    df_highwriterintent = df_combined.sample(n=5000, replace=True, weights='P(V>3)', random_state=seedhigh)
    df_lowwriterintent = df_combined.sample(n=5000, replace=True, weights='P(V<3)', random_state=seedlow)

    return (df_combined, df_highwriterintent, df_lowwriterintent)

def get_liwc_df(df, text_col, outcome):
    parse, category_names = liwc.load_token_parser('/home/victorialin/Documents/liwc_dict/LIWC2015_English_Flat.dic')
    df['label_count'] = df[text_col].apply(get_liwc_labels, args=(category_names, parse, True))

    count_df = pd.DataFrame(np.stack(df['label_count'].values, axis=0), columns=category_names)

    count_df[text_col] = df[text_col].values
    count_df[outcome] = df[outcome].values

    return count_df