import os
import openai
import tiktoken
import numpy as np
import pdb
import pandas as pd

openai.organization = 'org-uCzLn1lPkfNNSs8oq6jIOWcG'
openai.api_key = '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'

def get_num_tokens(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

data_dir = '/home/victorialin/Documents/2022-2023/causal_text/data/amazon_synthetic/'

music_df = pd.read_csv(os.path.join(data_dir, 'music_reviews_label_synthetic_numerical_coef_direct1.00.0.csv'))
office_df = pd.read_csv(os.path.join(data_dir, 'office_reviews_label_synthetic_numerical_coef_direct1.00.0.csv'))

music_instructions = "You are writing a review for your purchase of a musical instrument on Amazon. Consider the following sentence: "
office_instructions = 'You are writing a review for your purchase of an office product on Amazon. Consider the following sentence: '
neutral_instructions='You are writing a review for your purchase on Amazon. Consider the following sentence: '
# review = "I'm a woodwind girl, but wanted something to take camping for around the campfire. Ordered this because it was a kit and on price point alone.  For the price paid, this has given me a lifelong hobby and a way to connect with my kids. Huge value."
model = 'text-davinci-003'

for review in office_df['reviewText'].values[0:10]:
    for instructions in [music_instructions, office_instructions, neutral_instructions]:

        instr_len = get_num_tokens(instructions, model)
        review_len = get_num_tokens(review, model)

        # pdb.set_trace()

        prompt = instructions+review

        response = openai.Completion.create(
            model=model,
            max_tokens=0,
            temperature=0,
            logprobs=0,
            prompt=prompt,
            echo=True
        )

        sentence_logprobs = response.choices[0].logprobs.token_logprobs[instr_len-1:(instr_len-1+review_len)]

        # print(sentence_logprobs)
        # print(np.exp(sentence_logprobs))
        print(np.product(np.exp(sentence_logprobs)))
    
    print()

# pdb.set_trace()