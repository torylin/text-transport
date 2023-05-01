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
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

def get_embeds(sentences, tokenizer, model, device):
    tokens = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    ds = Dataset.from_dict(tokens).with_format('torch')
    dataloader = DataLoader(ds, batch_size=16, shuffle=False)
    embeds_list = []
    for batch in tqdm(dataloader):
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

    # model = LinearRegression()
    model = ElasticNet(l1_ratio=0.5)
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

def noise_labels(label_synth0, label_synth1, mu=0, sigma=1, seed=230418):
    np.random.seed(seed)
    noise0 = np.random.normal(mu, sigma, len(label_synth0))
    noise1 = np.random.normal(mu, sigma, len(label_synth1))

    return (noise0, noise1)