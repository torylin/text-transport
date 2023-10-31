import os
import openai
import tiktoken
import numpy as np
import pdb
import pandas as pd
import time
from tqdm import tqdm

openai.organization = 'org-uCzLn1lPkfNNSs8oq6jIOWcG'
openai.api_key = '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'

responses = openai.Image.create(
    prompt='A nice background for a Powerpoint slide in neutral colors',
    n=5,
    size='1024x1024'
)
image_url = responses['data'][0]['url']
print(image_url)
pdb.set_trace()