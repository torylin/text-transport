import os
import torch
import argparse
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import tqdm
import pdb

from utils import get_embeds

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/home/victorialin/Documents/2022-2023/causal_text/data/hk_rct/')
    parser.add_argument('--output-dir', type=str, default='/home/victorialin/Documents/2022-2023/causal_text/plots/sanity_check/')
    parser.add_argument('--lm-library', type=str, default='transformers')
    parser.add_argument('--lm-name', type=str, default='bert-base-uncased')
    parser.add_argument('--title', type=str)
    parser.add_argument('--seed', type=int, default=230314)
    args = parser.parse_args()

    return args

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

args = get_args()

if args.lm_library == 'transformers':
    tokenizer = AutoTokenizer.from_pretrained(args.lm_name, max_length=512, truncation=True)
    model = AutoModel.from_pretrained(args.lm_name)
    model.to(device)
    if 'Masked' not in model.config.architectures[0]:
        tokenizer.pad_token = tokenizer.eos_token
   
elif args.lm_library == 'sentence-transformers':
    model = SentenceTransformer(args.lm_name)
    model.to(device)

stems = ['brave', 'economy', 'evil', 'flag', 'threat', 'treatyobligation', 'treatyviolation']
df_dict = {}
embeds_dict = {}
for stem in tqdm(stems):
    df_dict[stem] = pd.read_csv(os.path.join(args.data_dir, 'HKarms{}.csv'.format(stem)), header=None)
    if args.lm_library == 'transformers':
        embeds_dict[stem] = get_embeds(df_dict[stem][0].values.tolist(), tokenizer, model, device)
    elif args.lm_library == 'sentence-transformers':
        embeds_dict[stem] = model.encode(df_dict[stem][0].values.tolist(), show_progress_bar=True)

colors = []
embeds_list = []
for k, v in embeds_dict.items():
    colors += [k]*v.shape[0]
    embeds_list.append(v)

embeds = np.vstack(embeds_list)

tsne = TSNE(n_components=2, learning_rate='auto', init='pca', random_state=args.seed)
embeds_t = tsne.fit_transform(embeds)

tsne_df = pd.DataFrame()
tsne_df['c1'] = embeds_t[:,0]
tsne_df['c2'] = embeds_t[:,1]

ax = sns.scatterplot(x='c1', y='c2', hue=colors, data=tsne_df)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
if args.title == None:
    ax.set(title='{} embeddings'.format(args.lm_name))
    ax.get_figure().savefig(os.path.join(args.output_dir, '{}.png'.format(args.lm_name)), bbox_inches='tight')
else:
    ax.set(title='{} embeddings'.format(args.title))
    ax.get_figure().savefig(os.path.join(args.output_dir, '{}.png'.format(args.title)), bbox_inches='tight')