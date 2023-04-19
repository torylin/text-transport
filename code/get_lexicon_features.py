import pandas as pd
import numpy as np
from empath import Empath
import liwc
import argparse
import pdb
import os
from utils import get_liwc_labels, get_empath_labels
from tqdm import tqdm
tqdm.pandas()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/home/victorialin/Documents/2022-2023/causal_text/data/amazon_synthetic_pryzant/')
    parser.add_argument('--csv', type=str, default='music_preprocessed.tsv')
    parser.add_argument('--lexicon', type=str, default='liwc')
    parser.add_argument('--text-col', type=str, default='text')
    parser.add_argument('--outcome', type=str, default='Y')
    parser.add_argument('--split-var', type=str, default='C')
    parser.add_argument('--split-values', nargs='+', default=[0, 1])
    parser.add_argument('--out-path', type=str, default='music')
    args = parser.parse_args()

    return args

args = get_args()

if args.split_values == ['True', 'False']:
    args.split_values = [True, False]
elif args.split_values == ['0', '1']:
    args.split_values = [0, 1]

if args.csv.split('.')[-1] == 'tsv':
    df = pd.read_csv(os.path.join(args.data_dir, args.csv), sep='\t', index_col=0)
else:
    df = pd.read_csv(os.path.join(args.data_dir, args.csv))

if args.lexicon == 'liwc':
    parse, category_names = liwc.load_token_parser('/home/victorialin/Documents/liwc_dict/LIWC2015_English_Flat.dic')
    df['label_count'] = df[args.text_col].progress_apply(get_liwc_labels, args=(category_names, parse, True))

elif args.lexicon == 'empath':
    lexicon = Empath()
    category_names = list(lexicon.cats.keys())
    df['label_count'] = df[args.text_col].progress_apply(get_empath_labels, args=(lexicon, ))

count_df = pd.DataFrame(np.stack(df['label_count'].values, axis=0), columns=category_names)
if args.lexicon == 'empath':
    count_df[count_df[category_names] > 1] = 1

count_df[args.text_col] = df[args.text_col]
count_df[args.outcome] = df[args.outcome]

count_df0 = count_df[df[args.split_var]==args.split_values[0]]
count_df1 = count_df[df[args.split_var]==args.split_values[1]]

count_df0.to_csv(os.path.join(args.data_dir, '{}_{}_{}{}.csv'.format(args.out_path,args.lexicon, args.split_var, args.split_values[0])), index=False)
count_df1.to_csv(os.path.join(args.data_dir, '{}_{}_{}{}.csv'.format(args.out_path,args.lexicon, args.split_var, args.split_values[1])), index=False)