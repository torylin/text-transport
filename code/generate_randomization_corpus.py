import pdb
import argparse
import os
from nltk import tokenize
import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
import random
from random import choices, sample

def eliminate_dups(sentences, df_dict, stems):
    info_dict = {}
    info_dict['numtexts'] = []
    for stem in stems:
        info_dict[stem] = []
    info_dict['text1'] = []
    info_dict['text2'] = []
    info_dict['text3'] = []
    info_dict['text_full'] = []
    # keep_sentences = []
    for sent_group in tqdm(sentences):
        sent_type = []
        for sent in sent_group:
            for stem in stems:
                if sent in df_dict[stem][0].values.tolist():
                    sent_type.append(stem)
                    break
        if len(set(sent_type)) == len(sent_type):
            info_dict['numtexts'].append(len(sent_group))
            for stem in stems:
                if stem in sent_type:
                    info_dict[stem].append(1)
                else:
                    info_dict[stem].append(0)
            for i in range(len(sent_group)):
                info_dict['text{}'.format(i+1)].append(sent_group[i])
            info_dict['text_full'].append(' '.join(sent_group))
    return info_dict

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/home/victorialin/Documents/2022-2023/causal_text/data/hk_rct/')
    parser.add_argument('--k', type=int, default=17)
    parser.add_argument('--method', type=str, default='sample')
    parser.add_argument('--seed', type=int, default=230313)
    args = parser.parse_args()

    return args

args = get_args()

random.seed(args.seed)

stems = ['brave', 'economy', 'evil', 'flag', 'threat', 'treatyobligation', 'treatyviolation']
other_stems = ['brave', 'economy', 'evil', 'flag', 'threat', 'treatyviolation']
df_dict = {}
df_allcomb = []
for stem in stems:
    df_dict[stem] = pd.read_csv(os.path.join(args.data_dir, 'HKarms{}.csv'.format(stem)), header=None)
    df_allcomb += df_dict[stem][0].values.tolist()

if args.method == 'sample':

    text_dict = {}
    text_dict['numtexts'] = []
    text_dict['text1'] = []
    text_dict['text2'] = []
    text_dict['text3'] = []
    for stem in stems:
        text_dict[stem] = []

    for i in tqdm(range(200000)):
        # treat = sample([True, False], counts=[5, 1], k=1)[0]
        treat = sample([True, False], 1)[0]
        num_texts = sample([2, 3], 1)[0]
        sample_stems = choices(stems, k=num_texts)
        if treat:
            while 'treatyobligation' not in sample_stems:
                sample_stems = choices(stems, k=num_texts)
        else:
            while 'treatyobligation' in sample_stems:
                sample_stems = choices(stems, k=num_texts)
        # if treat:
        #     sample_stems = ['treatyobligation'] + choices(other_stems, k=num_texts-1)
        # else:
        #     sample_stems = choices(other_stems, k=num_texts)
        
        sample_stems = set(sample_stems)

        text_dict['numtexts'].append(len(sample_stems))
        for i in range(num_texts):
            text_dict['text{}'.format(i+1)].append(sample(df_dict[stem][0].values.tolist(), k=1)[0])
        if num_texts == 2:
            text_dict['text3'].append('')
        for stem in stems:
            if stem in sample_stems:
                text_dict[stem].append(1)
            else:
                text_dict[stem].append(0)

    df_sent = pd.DataFrame(text_dict)
    df_sent['text_full'] = df_sent.text1.fillna('') + ' ' + df_sent.text2.fillna('') + ' ' + df_sent.text3.fillna('')
    df_sent.to_pickle(os.path.join(args.data_dir, 'randomization_corpus_random_sample.pkl'))
    # STOP HERE`

elif args.method == 'hybrid':

    df_allcomb = []
    for stem in stems:
        if stem != 'treatyobligation':
            df_allcomb += choices(df_dict[stem][0].values.tolist(), k=args.k)

    treat_sent2 = list(itertools.product(choices(df_dict['treatyobligation'][0].values.tolist(), k=args.k*12), df_allcomb))
    treat_sent2 = eliminate_dups(treat_sent2, df_dict, stems)
    treat_sent2['text3'] = ['']*len(treat_sent2['numtexts'])

    notreat_sent2 = []
    for i in range(len(stems)-1):
        for j in range(i+1, len(stems)):
            if stems[i] != 'treatyobligation' and stems[j] != 'treatyobligation':
                notreat_sent2 += list(itertools.product(choices(df_dict[stems[i]][0].values.tolist(), k=args.k),
                                                        choices(df_dict[stems[j]][0].values.tolist(), k=args.k)))
    notreat_sent2 = eliminate_dups(notreat_sent2, df_dict, stems)
    notreat_sent2['text3'] = ['']*len(notreat_sent2['numtexts'])

    treat_sent3 = []
    for i in range(len(stems)-1):
        for j in range(i+1, len(stems)):
            if stems[i] != 'treatyobligation' and stems[j] != 'treatyobligation':
                treat_sent3 += list(itertools.product(choices(df_dict['treatyobligation'][0].values.tolist(), k=args.k*10),
                                                    choices(df_dict[stems[i]][0].values.tolist(), k=args.k),
                                                    choices(df_dict[stems[j]][0].values.tolist(), k=args.k)))
    treat_sent3 = choices(treat_sent3, k=len(treat_sent2['numtexts']))
    treat_sent3 = eliminate_dups(treat_sent3, df_dict, stems)

    notreat_sent3 = []
    for i in range(len(stems)-2):
        for j in range(i+1, len(stems)-1):
            for k in range(j+1, len(stems)):
                if stems[i] != 'treatyobligation' and stems[j] != 'treatyobligation' and stems[k] != 'treatyobligation':
                    notreat_sent3 += list(itertools.product(choices(df_dict[stems[i]][0].values.tolist(), k=args.k),
                                                            choices(df_dict[stems[j]][0].values.tolist(), k=args.k),
                                                            choices(df_dict[stems[k]][0].values.tolist(), k=args.k)))
    notreat_sent3 = choices(notreat_sent3, k=len(notreat_sent2['numtexts']))
    notreat_sent3 = eliminate_dups(notreat_sent3, df_dict, stems)

    dicts = [treat_sent2, notreat_sent2, treat_sent3, notreat_sent3]
    df_list = []
    for dict in dicts:
        df_list.append(pd.DataFrame(dict))

    df_sent = pd.concat(df_list)
    df_sent.to_pickle(os.path.join(args.data_dir, 'randomization_corpus_sample.pkl'))

elif args.method == 'original':

    sent3 = list(itertools.permutations(df_allcomb, 3))
    sent2 = list(itertools.permutations(df_allcomb, 2))

    sent3 = eliminate_dups(sent3, df_dict, stems)
    df_sent3 = pd.DataFrame(sent3)
    df_sent3.to_pickle(os.path.join(args.data_dir, 'randomization_corpus_sent3.pkl'))

    sent2 = eliminate_dups(sent2, df_dict, stems)
    sent2['text3'] = [np.nan]*len(sent2['text2'])
    df_sent2 = pd.DataFrame(sent2)
    df_sent2.to_pickle(os.path.join(args.data_dcir, 'randomization_corpus_sent2.pkl'))

    df_sent2 = pd.read_pickle(os.path.join(args.data_dir, 'randomization_corpus_sent2.pkl'))
    df_sent3 = pd.read_pickle(os.path.join(args.data_dir, 'randomization_corpus_sent3.pkl'))
    df_sent = pd.concat([df_sent2, df_sent3])
    df_sent.to_pickle(os.path.join(args.data_dir, 'randomization_corpus.pkl'))
