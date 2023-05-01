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
from sklearn.linear_model import LogisticRegressionCV, LinearRegression, ElasticNet
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from nltk import tokenize
from tqdm import tqdm
from functools import partial
from transformers.utils import logging
from tabulate import tabulate
from statsmodels.regression.linear_model import OLS, WLS
import liwc
from empath import Empath
import functools

from utils import get_embeds, get_liwc_labels, get_empath_labels, get_unnoised_labels, noise_labels

logging.set_verbosity_error()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/home/victorialin/Documents/2022-2023/causal_text/data/hk/')
    parser.add_argument('--random-csv', type=str, default='HKRepData_textfull.csv')
    parser.add_argument('--target-csv', type=str, default='target_corpus_fullprobs.csv')
    parser.add_argument('--method', type=str, default='clm')
    parser.add_argument('--representation', type=str, default='embedding')
    parser.add_argument('--seed', type=int, default=230224)
    parser.add_argument('--treatment', type=str, nargs='+', default=['treatycommit'])
    parser.add_argument('--outcome', type=str, default='resp')
    parser.add_argument('--outcome-model', type=str, default='elasticnet')
    parser.add_argument('--outcome-lm-name', type=str, default='all-mpnet-base-v2')
    parser.add_argument('--outcome-lm-library', type=str, default='sentence-transformers')
    parser.add_argument('--text-col', type=str, default='text_full')
    parser.add_argument('--clf', type=str, default='lr')
    parser.add_argument('--estimate', type=str, default='diff')
    parser.add_argument('--lm-name', type=str, default='gpt2')
    parser.add_argument('--lm-library', type=str, default='transformers')
    parser.add_argument('--pr-lm-library', type=str, default='transformers')
    parser.add_argument('--pr-lm-name', type=str)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--train-size', type=float, default=0.1)
    parser.add_argument('--bootstrap-iters', type=int, default=0)
    parser.add_argument('--combination-type', type=str)
    parser.add_argument('--lexicon-weight', type=float)
    parser.add_argument('--embedding-weight', type=float)
    parser.add_argument('--marginal-probs', action='store_true')
    parser.add_argument('--ci', action='store_true')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--output-outcomes', action='store_true')
    parser.add_argument('--save-csvs', action='store_true')
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()

    return args

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

def get_prob_mlm(dataloader, model):
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

def get_prob_clm(dataloader, model):
    all_probs = []
    for batch in tqdm(dataloader):
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        outputs = model.generate(**batch, max_new_tokens=0, return_dict_in_generate=True, output_scores=True)
        probabilities = torch.exp(model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True))
        all_probs += probabilities.flatten().tolist()
    
    all_probs = np.array(all_probs)
    
    return all_probs

def get_var_clf(idx, df, weights, y_mean, args):
    i = idx[0]
    j = idx[1]
    d_xi = weights[i]
    d_xj = weights[j]
    yi = df[args.outcome].values[i]
    yj = df[args.outcome].values[j]
    if i != j:
        return 0
    val = d_xi*d_xj*yi*yj*(1-1/df.shape[0])

    return val

def get_var_lm(idx, df, weights, y_mean, args):
    i = idx[0]
    j = idx[1]
    p_xi = weights[i]
    p_xj = weights[j]
    yi = df[args.outcome].values[i]
    yj = df[args.outcome].values[j]
    if i != j:
        return 0
    val = p_xi*p_xj*yi*yj*(1-1/df.shape[0])

    return val

def get_estimate(df, df_random_train, treatment='a'):

    y = df[args.outcome].values.flatten()
    n = df.shape[0]

    if not treatment == 'none':
        a = df[treatment].values.flatten()
        n1 = np.sum(a)
        n0 = np.sum(1-a)

    if args.method == 'clm' or args.method == 'mlm':
        if args.pr_lm_name is None:
            weights_noadj = all_probs/(pr_x*norm_sum)
        else:
            weights_noadj = all_probs*norm_sum_pr/(all_probs_pr*norm_sum)
    elif args.method == 'clf':
        weights_noadj = probs[:,1]*corp_prob[0]/(corp_prob[1]*probs[:,0])
    
    weight_norm = np.sum(weights_noadj)/n
    weights = weights_noadj/weight_norm

    if not treatment == 'none':

        weights1_noadj = weights_noadj*a
        weights0_noadj = weights_noadj*(1-a)

        weight_norm1 = np.sum(weights1_noadj)/n1
        weights1 = weights1_noadj/weight_norm1

        weight_norm0 = np.sum(weights0_noadj)/n0
        weights0 = weights0_noadj/weight_norm0

    if args.estimate == 'diff':
        if not treatment == 'none':
            mu1 = np.sum(weights1*y)/n1
            mu0 = np.sum(weights0*y)/n0

        mu = np.sum(weights*y)/n
    
    elif args.estimate == 'dr':
        if args.outcome_model == 'lr':
            mu1_model = LinearRegression()
            mu0_model = LinearRegression()
            mu_model = LinearRegression()
        elif args.outcome_model == 'elasticnet':
            mu1_model = ElasticNet(l1_ratio=0.5, max_iter=10000)
            mu0_model = ElasticNet(l1_ratio=0.5, max_iter=10000)
            mu_model = ElasticNet(l1_ratio=0.5, max_iter=10000)
        elif args.outcome_model == 'svm':
            mu1_model = SVR()
            mu0_model = SVR()
            mu_model = SVR()
        
        if not treatment == 'none':
            a_train = df_random_train[treatment].values.flatten()
            mu1_model.fit(train_embeds[a_train==1], df_random_train[a_train==1][args.outcome].values.flatten())
            mu0_model.fit(train_embeds[a_train==0], df_random_train[a_train==0][args.outcome].values.flatten())
            mu1 = np.sum((weights1*(y-mu1_model.predict(embeds_r)))[a==1])/n1 + np.mean(mu1_model.predict(embeds_t))
            mu0 = np.sum((weights0*(y-mu0_model.predict(embeds_r)))[a==0])/n0 + np.mean(mu0_model.predict(embeds_t))
            print('mu1 model R^2: {:.3f}'.format(mu1_model.score(embeds_r[a==1], y[a==1])))
            print('mu0 model R^2: {:.3f}'.format(mu0_model.score(embeds_r[a==0], y[a==0])))

        mu_model.fit(train_embeds, df_random_train[args.outcome].values.flatten())
        mu = np.sum(weights*(y-mu_model.predict(embeds_r)))/n + np.mean(mu_model.predict(embeds_t))
        print('mu model R^2: {:.3f}'.format(mu_model.score(embeds_r, y)))

    if not treatment == 'none':
        est = mu1 - mu0

    if args.ci:
        if not treatment == 'none':
            if args.estimate == 'diff':
                var_list1 = ((weights1*y)[a==1]-mu1)**2
            elif args.estimate == 'dr':
                var_list1 = (((weights1*y)-mu1_model.predict(embeds_r))[a==1])**2
            # idxs1 = list(itertools.product(df_random_test[df_random_test[args.treatment]==1].index, df_random_test[df_random_test[args.treatment]==1].index))
            # var_list1 = np.array(list(map(partial(get_var_clf, df=df_random_test, weights=weights1, 
            #                                       y_mean=np.mean(df_random_test[df_random_test[args.treatment]==1]['resp'].values)), tqdm(idxs1))))
            varhat1 = np.sum(var_list1)/(n1**2)
            # idxs0 = list(itertools.product(df_random_test[df_random_test[args.treatment]==0].index, df_random_test[df_random_test[args.treatment]==0].index))
            # var_list0 = np.array(list(map(partial(get_var_clf, df=df_random_test, weights=weights0,
            #                                       y_mean=np.mean(df_random_test[df_random_test[args.treatment]==0]['resp'].values)), tqdm(idxs0))))
            if args.estimate == 'diff':
                var_list0 = ((weights0*y)[a==0]-mu0)**2
            elif args.estimate == 'dr':
                var_list0 = (((weights0*y)-mu0_model.predict(embeds_r))[a==0])**2
            varhat0 = np.sum(var_list0)/(n0**2)
            varhat = varhat1 + varhat0

        if args.estimate == 'diff':
            var_list_overall = ((weights*y)-mu)**2
        elif args.estimate == 'dr':
            var_list_overall = ((weights*y)-mu_model.predict(embeds_r))**2
        varhat_overall = np.sum(var_list_overall)/(n**2)

        if not treatment == 'none':
            return (mu1, mu0, est, varhat, mu, varhat_overall)

        return (mu, varhat_overall)
    
    if not treatment == 'none':
        return (mu1, mu0, est, mu)
    
    return mu

gc.collect()
torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

args = get_args()

# df_random = pd.read_csv(os.path.join(args.data_dir, 'hk_rct', 'HKData.csv'))
df_random = pd.read_csv(os.path.join(args.data_dir, args.random_csv))
df_target = pd.read_csv(os.path.join(args.data_dir, args.target_csv))
if args.bootstrap_iters > 0:
    df0 = pd.read_csv(os.path.join(args.data_dir, 'combined_liwc_categorymusic.csv'))
    df1 = pd.read_csv(os.path.join(args.data_dir, 'combined_liwc_categoryoffice.csv'))
    prob_df0 = pd.read_csv(os.path.join(args.data_dir, 'music_reviews_pred_numerical.csv'))
    prob_df1 = pd.read_csv(os.path.join(args.data_dir, 'office_reviews_pred_numerical.csv'))
    label_synth0_unnoised, label_synth1_unnoised = get_unnoised_labels(df0, df1, prob_df0, prob_df1, 
                                                                       args.combination_type, args.lexicon_weight, args.embedding_weight)

# df_random_corp = pd.read_pickle(os.path.join(args.data_dir, 'hk_rct', 'randomization_corpus_random_sample.pkl'))
# if args.treatment == 'treatycommit':
#     treat_prob = df_random_corp['treatyobligation'].mean()
# else:
#     treat_prob = df_random_corp[args.treatment].mean()

sentences = np.hstack([df_random[args.text_col].values, df_target[args.text_col].values])
labels = np.array([0]*df_random.shape[0] + [1]*df_target.shape[0])
sentences, labels = shuffle(sentences, labels, random_state=args.seed)
y_train, y_test, sentence_train, sentence_test = train_test_split(labels, sentences, train_size=args.train_size, random_state=args.seed)
if args.method == 'clf':
    if args.lm_library == 'sentence-transformers':
        model = SentenceTransformer(args.lm_name)
        model.to(device)
        # embeds = model.encode(sentences, show_progress_bar=True)
        X_train = model.encode(sentence_train, show_progress_bar=True)

    elif args.lm_library == 'transformers':
        tokenizer = AutoTokenizer.from_pretrained(args.lm_name, max_length=512, truncation=True)
        model = AutoModel.from_pretrained(args.lm_name)
        model.to(device)
        if 'Masked' not in model.config.architectures[0]:
            tokenizer.pad_token = tokenizer.eos_token
        
        # embeds = get_embeds(sentences.astype(str).tolist(), tokenizer, model, device)
        X_train = get_embeds(sentence_train.astype(str).tolist(), tokenizer, model, device)

    # X_train, X_test, y_train, y_test, sentence_train, sentence_test = train_test_split(embeds, labels, sentences, train_size=args.train_size, random_state=args.seed)
    # pdb.set_trace
    elif args.lm_library == 'lexicon':
        if args.lm_name == 'liwc':
            parse, category_names = liwc.load_token_parser('/home/victorialin/Documents/liwc_dict/LIWC2015_English_Flat.dic')
            rel_cats = ['sexual', 'female', 'filler', 'netspeak', 'home', 'informal', 'shehe', 'nonflu', 'assent', 'death']
            X_train = np.array(list(map(functools.partial(get_liwc_labels, rel_cats=rel_cats, parse=parse, binary=True), sentence_train)))
            # X_train = X_train[:,np.in1d(category_names, ['sexual', 'female', 'filler', 'netspeak', 'home', 'informal', 'shehe', 'nonflu', 'assent', 'death'])]
        elif args.lm_name == 'empath':
            lexicon = Empath()
            category_names = list(lexicon.cats.keys())
            X_train = np.array(list(map(functools.partial(get_empath_labels, lexicon=lexicon), sentence_train)))
            X_train[X_train > 1] = 1
        
train_df = pd.DataFrame({args.text_col: sentence_train})
test_df = pd.DataFrame({args.text_col: sentence_test})
df_random_test = pd.merge(df_random, test_df, on=[args.text_col], how='inner').drop_duplicates(subset=[args.text_col]).reset_index(drop=True)
df_target_test = pd.merge(df_target, test_df, on=[args.text_col], how='inner').drop_duplicates(subset=[args.text_col]).reset_index(drop=True)
df_random_train = pd.merge(df_random, train_df, on=[args.text_col], how='inner').drop_duplicates(subset=[args.text_col]).reset_index(drop=True)
df_target_train = pd.merge(df_target, train_df, on=[args.text_col], how='inner').drop_duplicates(subset=[args.text_col]).reset_index(drop=True)

df = df_random_test
df_t = df_target_test
pr_x = 1/df.shape[0]

if args.estimate == 'dr':
    if args.outcome_lm_library == 'sentence-transformers':
        outcome_lm = SentenceTransformer(args.outcome_lm_name)
        outcome_lm.to(device)
        train_embeds = outcome_lm.encode(df_random_train[args.text_col].values, show_progress_bar=True)
        embeds_r = outcome_lm.encode(df[args.text_col].values, show_progress_bar=True)
        embeds_t = outcome_lm.encode(df_t[args.text_col].values, show_progress_bar=True)

    elif args.outcome_lm_library == 'transformers':
        outcome_tokenizer = AutoTokenizer.from_pretrained(args.outcome_lm_name, max_length=512, truncation=True)
        outcome_lm = AutoModel.from_pretrained(args.outcome_lm_name)
        outcome_lm.to(device)
        if 'Masked' not in model.config.architectures[0]:
            outcome_tokenizer.pad_token = outcome_tokenizer.eos_token
        
        train_embeds = get_embeds(df_random_train[args.text_col].values.tolist(), outcome_tokenizer, outcome_lm, device)
        embeds_r = get_embeds(df[args.text_col].values.tolist(), outcome_tokenizer, outcome_lm, device)
        embeds_t = get_embeds(df_t[args.text_col].values.tolist(), outcome_tokenizer, outcome_lm, device)
    
    elif args.outcome_lm_library == 'lexicon':
        if args.outcome_lm_name == 'liwc':
            parse, category_names = liwc.load_token_parser('/home/victorialin/Documents/liwc_dict/LIWC2015_English_Flat.dic')
            # rel_cats = ['sexual', 'female', 'filler', 'netspeak', 'home', 'informal', 'shehe', 'nonflu', 'assent', 'death']
            rel_cats = ['posemo', 'negemo', 'anx', 'nonflu', 'filler', 'home', 'assent']
            train_embeds = np.array(list(map(functools.partial(get_liwc_labels, rel_cats=rel_cats, parse=parse, binary=True), 
                                             df_random_train[args.text_col].values.tolist())))
            embeds_r = np.array(list(map(functools.partial(get_liwc_labels, rel_cats=rel_cats, parse=parse, binary=True), df[args.text_col].values.tolist())))
            embeds_t = np.array(list(map(functools.partial(get_liwc_labels, rel_cats=rel_cats, parse=parse, binary=True), df_t[args.text_col].values.tolist())))
        elif args.outcome_lm_name == 'empath':
            lexicon = Empath()
            category_names = list(lexicon.cats.keys())
            train_embeds = np.array(list(map(functools.partial(get_empath_labels, lexicon=lexicon), df_random_train[args.text_col].values.tolist())))
            embeds_r = np.array(list(map(functools.partial(get_empath_labels, lexicon=lexicon), df[args.text_col].values.tolist())))
            embeds_t = np.array(list(map(functools.partial(get_empath_labels, lexicon=lexicon), df_t[args.text_col].values.tolist())))
            train_embeds[train_embeds > 1] = 1
            embeds_r[embeds_r > 1] = 1
            embeds_t[embeds_t > 1] = 1
            
if args.save_csvs:
    df_random_train.to_csv(os.path.join(args.data_dir, 'random_train.csv'), index=False)
    df_target_train.to_csv(os.path.join(args.data_dir, 'target_train.csv'), index=False)
    df_random_test.to_csv(os.path.join(args.data_dir, 'random_test.csv'), index=False)
    df_target_test.to_csv(os.path.join(args.data_dir, 'target_test.csv'), index=False)

if args.method == 'mlm' or args.method == 'clm':
    
    if args.method == 'mlm':
        get_prob_lm = get_prob_mlm
        tokenizer = AutoTokenizer.from_pretrained(args.lm_name, max_length=512, truncation=True, padding=True)
        model = AutoModelForMaskedLM.from_pretrained(args.lm_name)

        if args.pr_lm_name is not None:
            tokenizer_pr = AutoTokenizer.from_pretrained(args.pr_lm_name, max_length=512, truncation=True, padding=True)
            model_pr = AutoModelForMaskedLM.from_pretrained(args.pr_lm_name)
    
    elif args.method == 'clm':
        get_prob_lm = get_prob_clm
        tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.lm_name)

        if args.pr_lm_name is not None:
            tokenizer_pr = AutoTokenizer.from_pretrained(args.lm_name)
            tokenizer_pr.padding_side = 'left'
            tokenizer_pr.pad_token = tokenizer_pr.eos_token
            model_pr = AutoModelForCausalLM.from_pretrained(args.pr_lm_name)

    model.to(device)
    if args.pr_lm_name is not None:
        model_pr.to(device)

    if args.marginal_probs:
        tokens1 = tokenizer(df.text1.fillna('').astype(str).values.tolist(), return_tensors='pt', padding=True, truncation=True)
        tokens2 = tokenizer(df.text2.fillna('').astype(str).values.tolist(), return_tensors='pt', padding=True, truncation=True)
        tokens3 = tokenizer(df.text3.fillna('').astype(str).values.tolist(), return_tensors='pt', padding=True, truncation=True)

        ds1 = Dataset.from_dict(tokens1).with_format('torch')
        dataloader1 = DataLoader(ds1, batch_size=args.batch_size, shuffle=False)

        ds2 = Dataset.from_dict(tokens2).with_format('torch')
        dataloader2 = DataLoader(ds2, batch_size=args.batch_size, shuffle=False)

        ds3 = Dataset.from_dict(tokens3).with_format('torch')
        dataloader3 = DataLoader(ds3, batch_size=args.batch_size, shuffle=False)

        all_probs1 = get_prob_lm(dataloader1, model)
        all_probs2 = get_prob_lm(dataloader2, model)
        all_probs3 = get_prob_lm(dataloader3, model)

        all_probs = all_probs1*all_probs2*all_probs3

    else:
        tokens = tokenizer(df[args.text_col].astype(str).values.tolist(), return_tensors='pt', padding=True, truncation=True)
        ds = Dataset.from_dict(tokens).with_format('torch')
        dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

        all_probs = get_prob_lm(dataloader, model)

        if args.pr_lm_name is not None:
            if args.pr_lm_name == args.lm_name:
                all_probs_pr = get_prob_lm(dataloader, model_pr)
            else:
                tokens_pr = tokenizer_pr(df[args.text_col].astype(str).values.tolist(), return_tensors='pt', padding=True, truncation=True)
                ds_pr = Dataset.from_dict(tokens).with_format('torch')
                dataloader_pr = DataLoader(ds_pr, batch_size=args.batch_size, shuffle=False)

                all_probs_pr = get_prob_lm(dataloader_pr, model_pr)

    norm_sum = np.sum(all_probs)
    if args.pr_lm_name is not None:
        norm_sum_pr = np.sum(all_probs_pr)

    pdb.set_trace()

elif args.method == 'clf':

    if args.clf == 'lr':
        clf = LogisticRegressionCV(cv=5, random_state=args.seed, max_iter=10000)
        # reg = LinearRegression()
        # train_embeds = get_embeds(pd.concat([df_random_train, df_target_train], axis=0)[args.text_col].values.astype(str).tolist(), tokenizer, model, device)
        # y_train = pd.concat([df_random_train, df_target_train], axis=0).V_reader
        # reg.fit(train_embeds, y_train)
        # test_embeds = get_embeds(df[args.text_col].values.astype(str).tolist(), tokenizer, model, device)
        # print(reg.score(test_embeds, df.V_reader))
        # pdb.set_trace()
    elif args.clf == 'elasticnet':
        clf = LogisticRegressionCV(cv=5, random_state=args.seed, penalty='elasticnet', solver='saga', l1_ratios=[0.5]*5, max_iter=10000)
    if args.scale:
        scaler = StandardScaler()
        scaler.fit(X_train)
        clf.fit(scaler.transform(X_train), y_train)
    else:
        clf.fit(X_train, y_train)
    if args.marginal_probs:
        prob_list = []
        for sent_group in tqdm(df[args.text_col].values):
            prob_list.append(get_prob_clf(model, clf, sent_group, args))
        probs = np.vstack(prob_list)

    else:
        if args.lm_library == 'sentence-transformers':
            test_embeds = model.encode(df[args.text_col].values, show_progress_bar=True)
        elif args.lm_library == 'transformers':
            test_embeds = get_embeds(df[args.text_col].values.astype(str).tolist(), tokenizer, model, device)
        elif args.lm_library == 'lexicon':
            if args.lm_name == 'liwc':
                test_embeds = np.array(list(map(functools.partial(get_liwc_labels, rel_cats=rel_cats, parse=parse, binary=True), df[args.text_col].values)))
            elif args.lm_name == 'empath':
                test_embeds = np.array(list(map(functools.partial(get_empath_labels, lexicon=lexicon), df[args.text_col].values)))
                test_embeds[test_embeds > 1] = 1
        if args.scale:
            probs = clf.predict_proba(scaler.transform(test_embeds))
        else:
            probs = clf.predict_proba(test_embeds)
            print('clf accuracy: {:.3f}'.format(clf.score(test_embeds, [0]*test_embeds.shape[0])))
    corp_prob = np.array([df.shape[0]/(df.shape[0]+df_t.shape[0]), 
                        df_t.shape[0]/(df.shape[0]+df_t.shape[0])])

headers = []
headers2 = []

if args.treatment != ['none']:
    headers += ['$\hat{\\tau}_{R \\rightarrow T}$']
if args.validate:
    if args.treatment != ['none']:
        headers += ['$\hat{\\tau}_R$', '$\hat{\\tau}_T$']
    headers2 += ['$\hat{\mu}(P_{R \\rightarrow T})$', '$\hat{\mu}(P_R)$', '$\hat{\mu}(P_T)$']
if args.output_outcomes:
    headers = ['$\hat{\mu}(P_1)$', '$\hat{\mu}(P_0)$'] + headers

headers = ['Treatment'] + headers


if args.validate:
    mu_r = np.mean(df[args.outcome].values)
    mu_t = np.mean(df_t[args.outcome].values)
    if args.ci:
        varhat_overall_r = np.sum((df[args.outcome].values-np.mean(df[args.outcome].values))**2)/(df.shape[0]**2)
        varhat_overall_t = np.sum((df_t[args.outcome].values-np.mean(df_t[args.outcome].values))**2)/(df_t.shape[0]**2)

rows = []

if args.treatment != ['none']:
    if args.treatment == ['all']:
        treatments = ['treatycommit', 'brave', 'evil', 'flag', 'threat', 'economy', 'treatyviolation']
        treatment_names = ['Commitment', 'Bravery', 'Mistreatment', 'Flags', 'Threat', 'Economy', 'Violation']
    else:
        treatments = args.treatment
        treatment_names = args.treatment
        
if args.bootstrap_iters > 0:
    if args.validate:
        mu_r_list = np.zeros(args.bootstrap_iters)
        mu_t_list = np.zeros(args.bootstrap_iters)
    if args.treatment != ['none']:
        bootmu1_dict = {treatment: np.zeros(args.bootstrap_iters) for treatment in treatments}
        bootmu0_dict = {treatment: np.zeros(args.bootstrap_iters) for treatment in treatments}
        bootest_dict = {treatment: np.zeros(args.bootstrap_iters) for treatment in treatments}
        if args.validate:
            tau_r_dict = {treatment: np.zeros(args.bootstrap_iters) for treatment in treatments}
            tau_t_dict = {treatment: np.zeros(args.bootstrap_iters) for treatment in treatments}

        mu_list = np.zeros(args.bootstrap_iters)
        
    for iter in tqdm(range(args.bootstrap_iters)):
        noise0, noise1 = noise_labels(label_synth0_unnoised, label_synth1_unnoised, seed=iter, sigma=1.0)

        df_random_boot = df_random
        df_target_boot = df_target
        df_random_boot[args.outcome] = label_synth0_unnoised + noise0
        df_target_boot[args.outcome] = label_synth1_unnoised + noise1
        df_boot = pd.merge(df_random_boot, test_df, on=[args.text_col], how='inner').drop_duplicates(subset=[args.text_col]).reset_index(drop=True)
        df_t_boot = pd.merge(df_target_boot, test_df, on=[args.text_col], how='inner').drop_duplicates(subset=[args.text_col]).reset_index(drop=True)
        df_random_train_boot = pd.merge(df_random_boot, train_df, on=[args.text_col], how='inner').drop_duplicates(subset=[args.text_col]).reset_index(drop=True)
        df_target_train_boot = pd.merge(df_target_boot, train_df, on=[args.text_col], how='inner').drop_duplicates(subset=[args.text_col]).reset_index(drop=True)
    
        if args.validate:
            mu_r_list[iter] = np.mean(df_boot[args.outcome].values)
            mu_t_list[iter] = np.mean(df_t_boot[args.outcome].values)

        if args.treatment != ['none']:
            for i in range(len(treatments)):
                treatment = treatments[i]
                a = df_boot[treatment].values.flatten()
                n1 = np.sum(a)
                n0 = np.sum(1-a)
                bootmu1_dict[treatment][iter], bootmu0_dict[treatment][iter], bootest_dict[treatment][iter], mu_list[iter] = get_estimate(df_boot, df_random_train_boot, treatment)

                if args.validate:
                    a_t = df_t_boot[treatment].values.flatten()
                    n1_t = np.sum(a_t)
                    n0_t = np.sum(1-a_t)
                    tau_r_dict[treatment][iter] = np.mean(df_boot[args.outcome].values[a==1]) - np.mean(df_boot[args.outcome].values[a==0])
                    tau_t_dict[treatment][iter] = np.mean(df_t_boot[args.outcome].values[a_t==1]) - np.mean(df_t_boot[args.outcome].values[a_t==0])
        else:
            mu_list[iter] = get_estimate(df_boot, df_random_train_boot, args.treatment[0])

    if args.treatment != ['none']: 
        for i in range(len(treatments)):
            treatment = treatments[i]
            est_boot = np.mean(bootest_dict[treatment])
            ci_025 = np.quantile(bootest_dict[treatment], 0.025)
            ci_975 = np.quantile(bootest_dict[treatment], 0.975)
            row = [treatment_names[i], '{:.3f} [{:.3f}, {:.3f}]'.format(est_boot, ci_025, ci_975)]

            if args.validate:
                tau_r_025 = np.quantile(tau_r_dict[treatment], 0.025)
                tau_r_975 = np.quantile(tau_r_dict[treatment], 0.975)
                tau_t_025 = np.quantile(tau_t_dict[treatment], 0.025)
                tau_t_975 = np.quantile(tau_t_dict[treatment], 0.975)
                row += ['{:.3f} [{:.3f}, {:.3f}]'.format(np.mean(tau_r_dict[treatment]), tau_r_025, tau_r_975), 
                        '{:.3f} [{:.3f}, {:.3f}]'.format(np.mean(tau_t_dict[treatment]), tau_t_025, tau_t_975)]
            
            rows.append(row)

    mu_boot = np.mean(mu_list)
    mu_025 = np.quantile(mu_list, 0.025)
    mu_975 = np.quantile(mu_list, 0.975)

    if args.validate:
        mu_r_boot = np.mean(mu_r_list)
        mu_r_025 = np.quantile(mu_r_list, 0.025)
        mu_r_975 = np.quantile(mu_r_list, 0.975)
        mu_t_boot = np.mean(mu_t_list)
        mu_t_025 = np.quantile(mu_t_list, 0.025)
        mu_t_975 = np.quantile(mu_t_list, 0.975)

else:
    if args.treatment != ['none']:
        for i in range(len(treatments)):
            treatment = treatments[i]
            a = df[treatment].values.flatten()
            n1 = np.sum(a)
            n0 = np.sum(1-a)
            if args.ci:
                mu1, mu0, est, varhat, mu, varhat_overall = get_estimate(df, df_random_train, treatment)
            else:
                mu1, mu0, est, mu = get_estimate(df, df_random_train, treatment)

            if args.validate:
                a_t = df_t[treatment].values.flatten()
                n1_t = np.sum(a_t)
                n0_t = np.sum(1-a_t)
                tau_r = np.mean(df[args.outcome].values[a==1]) - np.mean(df[args.outcome].values[a==0])
                tau_t = np.mean(df_t[args.outcome].values[a_t==1]) - np.mean(df_t[args.outcome].values[a_t==0])
                if args.ci:
                    varhat_r = np.sum(((df[args.outcome].values-np.mean(df[args.outcome].values))[a==1])**2)/(n1**2) + np.sum(((df[args.outcome].values-np.mean(df[args.outcome].values))[a==0])**2)/(n0**2)
                    varhat_t = np.sum(((df_t[args.outcome].values-np.mean(df_t[args.outcome].values))[a_t==1])**2)/(n1_t**2) + np.sum(((df_t[args.outcome].values-np.mean(df_t[args.outcome].values))[a_t==0])**2)/(n0_t**2)
            
            if args.ci:
                row = ['{:.3f} [{:.3f}, {:.3f}]'.format(est, est-1.96*np.sqrt(varhat), est+1.96*np.sqrt(varhat))]
            else:
                row = ['{:.3f}'.format(est)]
            if args.output_outcomes:
                row = ['{:.3f} '.format(mu1), '{:.3f}'.format(mu0)] + row
            
            row = [treatment_names[i]] + row
            if args.validate:
                if args.ci:
                    row += ['{:.3f} [{:.3f}, {:.3f}]'.format(tau_r, tau_r-1.96*np.sqrt(varhat_r), tau_r+1.96*np.sqrt(varhat_r)), 
                            '{:.3f} [{:.3f}, {:.3f}]'.format(tau_t, tau_t-1.96*np.sqrt(varhat_t), tau_t+1.96*np.sqrt(varhat_t))]   
                else:
                    row += ['{:.3f}'.format(tau_r), '{:.3f}'.format(tau_t)]
        
            rows.append(row)

    else:
        if args.ci:
            mu, varhat_overall = get_estimate(df, df_random_train, args.treatment[0])
        else:
            mu = get_estimate(df, df_random_train, args.treatment[0])

print(tabulate(rows, headers=headers, tablefmt='latex_raw'))

if args.validate:
    if args.ci:
        rows2 = [['Outcome', 
                '{:.3f} [{:.3f}, {:.3f}]'.format(mu, mu-1.96*np.sqrt(varhat_overall), mu+1.96*np.sqrt(varhat_overall)), 
                '{:.3f} [{:.3f}, {:.3f}]'.format(mu_r, mu_r-1.96*np.sqrt(varhat_overall_r), mu_r+1.96*np.sqrt(varhat_overall_r)), 
                '{:.3f} [{:.3f}, {:.3f}]'.format(mu_t, mu_t-1.96*np.sqrt(varhat_overall_t), mu_t+1.96*np.sqrt(varhat_overall_t))]]
    elif args.bootstrap_iters > 0:
        rows2 = [['Outcome', 
                '{:.3f} [{:.3f}, {:.3f}]'.format(mu_boot, mu_025, mu_975), 
                '{:.3f} [{:.3f}, {:.3f}]'.format(mu_r_boot, mu_r_025, mu_r_975), 
                '{:.3f} [{:.3f}, {:.3f}]'.format(mu_t_boot, mu_t_025, mu_t_975)]]
    else:
        rows2 = [['Outcome', '{:.3f}'.format(mu), '{:.3f}'.format(mu_r), '{:.3f}'.format(mu_t)]]

    print(tabulate(rows2, headers=headers2, tablefmt='latex_raw'))