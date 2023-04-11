import torch
import nltk
import contractions
import numpy as np
from tqdm import tqdm
from collections import Counter
from datasets import Dataset
from torch.utils.data import DataLoader
from empath import Empath

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