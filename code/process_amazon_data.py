import pandas as pd
import json
import gzip
import pdb
from tqdm import tqdm

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

music_df = getDF('/home/victorialin/Documents/2022-2023/causal_text/data/amazon_synthetic/Musical_Instruments_5.json.gz')
music_df = music_df[~music_df['vote'].isna()][['reviewText', 'vote']]
music_df['vote'] = music_df['vote'].str.replace(',', '').astype(int)
music_df.dropna(inplace=True)
music_df.rename(columns={'vote': 'helpful'}, inplace=True)
music_df.to_csv('/home/victorialin/Documents/2022-2023/causal_text/data/amazon_synthetic/music_reviews_raw.csv', index=False)

office_df = getDF('/home/victorialin/Documents/2022-2023/causal_text/data/amazon_synthetic/Office_Products_5.json.gz')
office_df = office_df[~office_df['vote'].isna()][['reviewText', 'vote']]
office_df['vote'] = office_df['vote'].str.replace(',', '').astype(int)
office_df.dropna(inplace=True)
office_df.rename(columns={'vote': 'helpful'}, inplace=True)
office_df.to_csv('/home/victorialin/Documents/2022-2023/causal_text/data/amazon_synthetic/office_reviews_raw.csv', index=False)