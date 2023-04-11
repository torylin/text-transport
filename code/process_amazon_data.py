import pandas as pd
import numpy as np
from empath import Empath
import liwc
import argparse
import pdb
from utils import get_liwc_labels, get_empath_labels
from tqdm import tqdm
tqdm.pandas()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lexicon', type=str, default='liwc')
    args = parser.parse_args()

    return args

args = get_args()

df = pd.read_csv('../data/amazon_synthetic/music_preprocessed.tsv', sep='\t', index_col=0)

if args.lexicon == 'liwc':
    parse, category_names = liwc.load_token_parser('/home/victorialin/Documents/liwc_dict/LIWC2015_English_Flat.dic')
    df['label_count'] = df['text'].progress_apply(get_liwc_labels, args=(category_names, parse, True))

elif args.lexicon == 'empath':
    lexicon = Empath()
    category_names = list(lexicon.cats.keys())
    df['label_count'] = df['text'].progress_apply(get_empath_labels, args=(lexicon, ))

count_df = pd.DataFrame(np.stack(df['label_count'].values, axis=0), columns=category_names)
if args.lexicon == 'empath':
    count_df[count_df[category_names] > 1] = 1

count_df['text'] = df['text']
count_df['Y'] = df['Y']

count_df0 = count_df[df['C']==0]
count_df1 = count_df[df['C']==1]

count_df0.to_csv('../data/amazon_synthetic/music_{}_c0.csv'.format(args.lexicon), index=False)
count_df1.to_csv('../data/amazon_synthetic/music_{}_c1.csv'.format(args.lexicon), index=False)