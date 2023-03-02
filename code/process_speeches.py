import pdb
import argparse
import os
import pickle
from nltk import tokenize
import numpy as np
import re
import pandas as pd

def get_sent_groups(text, num_sentences, sent_probs, seed):
    text = re.sub("\[\[.*?\]\]","", text)
    sentences = tokenize.sent_tokenize(text)
    sentences = [sent for sent in sentences if len(tokenize.word_tokenize(sent)) > 4]

    np.random.seed(seed)
    sent_lengths = []
    while np.sum(sent_lengths) < len(sentences):
        sent_lengths.append(np.random.choice(num_sentences, p=sent_probs))

    sent_groups = []
    idx = 0
    for length in sent_lengths:
        sent_groups.append(' '.join(sentences[idx:idx+length]))
        idx += length
    if np.sum(sent_lengths) < len(sentences):
        sent_groups.append(' '.join(sentences[idx:]))
    
    return sent_groups

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/home/victorialin/Documents/2022-2023/causal_text/data/')
    parser.add_argument('--seed', type=int', default=230224)
    args = parser.parse_args()

    return args

args = get_args()

sample_sent_probs = np.load(os.path.join(args.data_dir, 'hk_rct/sample_sent_prop.npy'))
full_sent_probs = np.load(os.path.join(args.data_dir, 'hk_rct/full_sent_prop.npy'))

sent_groups = []

for filename in os.listdir(os.path.join(args.data_dir, 'hk_speeches', 'text')):
    with open(os.path.join(args.data_dir, 'hk_speeches', 'text', filename)) as file:
        text = file.read().replace('\n', '')

    sent_groups += get_sent_groups(text, range(2, 8), sample_sent_probs, args.seed)

df = pd.DataFrame({'text_full': sent_groups})
df.to_csv(os.path.join(args.data_dir, 'hk_speeches', 'target_corpus.csv'), index=False)