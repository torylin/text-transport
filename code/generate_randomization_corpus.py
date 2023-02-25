import pdb
import argparse
import os
from nltk import tokenize
import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm

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
    args = parser.parse_args()

    return args

args = get_args()

stems = ['brave', 'economy', 'evil', 'flag', 'threat', 'treatyobligation', 'treatyviolation']
df_dict = {}
df_allcomb = []
for stem in stems:
    df_dict[stem] = pd.read_csv(os.path.join(args.data_dir, 'HKarms{}.csv'.format(stem)), header=None)
    df_allcomb += df_dict[stem][0].values.tolist()

sent3 = list(itertools.permutations(df_allcomb, 3))
sent2 = list(itertools.permutations(df_allcomb, 2))

sent3 = eliminate_dups(sent3, df_dict, stems)
df_sent3 = pd.DataFrame(sent3)
df_sent3.to_pickle(os.path.join(args.data_dir, 'randomization_corpus_sent3.pkl'))

sent2 = eliminate_dups(sent2, df_dict, stems)
sent2['text3'] = [np.nan]*len(sent2['text2'])
df_sent2 = pd.DataFrame(sent2)
df_sent2.to_pickle(os.path.join(args.data_dir, 'randomization_corpus_sent2.pkl'))

df_sent2 = pd.read_pickle(os.path.join(args.data_dir, 'randomization_corpus_sent2.pkl'))
df_sent3 = pd.read_pickle(os.path.join(args.data_dir, 'randomization_corpus_sent3.pkl'))
df_sent = pd.concat([df_sent2, df_sent3])
df_sent.to_pickle(os.path.join(args.data_dir, 'randomization_corpus.pkl'))
