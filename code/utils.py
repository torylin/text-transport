import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader

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