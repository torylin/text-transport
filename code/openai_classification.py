import os
import openai
import tiktoken
import numpy as np
import pdb
import pandas as pd
import time
from tqdm import tqdm

import os
import openai
import tiktoken
import numpy as np
import pdb
import pandas as pd
import time
from tqdm import tqdm

openai.organization = ''
openai.api_key = ''

def get_num_tokens(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    string = string.strip()
    encoding = tiktoken.encoding_for_model(encoding_name)
    total_num_tokens = len(encoding.encode(string))
    
    return total_num_tokens

def add_prefix(lst, prefix, suffix=''):
    pre_res = list(map(lambda x: prefix + x + suffix, lst))
    return pre_res

data_dir = '/home/victorialin/Documents/2022-2023/causal_text/data/amazon_synthetic/'
# data_dir = '/home/victorialin/Documents/2022-2023/causal_text/data/emobank/'
# data_dir = '/home/victorialin/Documents/2022-2023/causal_text/data/hk/'

# df_r = pd.read_csv(os.path.join(data_dir, 'music_reviews_label_synthetic_numerical_coef_direct1.00.0.csv'))
# df_t = pd.read_csv(os.path.join(data_dir, 'office_reviews_label_synthetic_numerical_coef_direct1.00.0.csv'))
df_r = pd.read_csv(os.path.join(data_dir, 'music_reviews_label_synthetic_numerical_coef_predreg.csv'))
df_t = pd.read_csv(os.path.join(data_dir, 'office_reviews_label_synthetic_numerical_coef_predreg.csv'))
text_col = 'reviewText'
df_r = df_r.sample(1000, random_state=230510)
df_t = df_t.sample(1000, random_state=230510)

# pdb.set_trace()

# df_r = pd.read_csv(os.path.join(data_dir, 'HKRepData_textfull.csv'))
# text_col = 'text_full'

df = df_r
# df = df_t
category = 'music'
category1 = 'music'
category2 = 'office'

# batches = list(range(0, df.shape[0]+1, 100))

# df_r = pd.read_csv(os.path.join(data_dir, 'highwriterintent_valence_liwc.csv'))
# df_t = pd.read_csv(os.path.join(data_dir, 'lowwriterintent_valence_liwc.csv'))
# df = df_r
# # df = df_t
# category = 'highwriterintent'
# category1 = 'high'
# category2 = 'low'
# text_col = 'text'

batches = list(range(0, df.shape[0]+1, 5))

model = 'text-davinci-003'

# r_prompts = add_prefix(df[text_col].values, '', '\nPositive or negative?: Positive')
# t_prompts = add_prefix(df[text_col].values, '', '\nPositive or negative?: Negative')

r_prompts = add_prefix(df[text_col].values, '', '\nMusic or office?: Music')
t_prompts = add_prefix(df[text_col].values, '', '\nMusic or office?: Office')

# instr_len = get_num_tokens('\nPositive or negative?: Positive', encoding_name=model)

for b in tqdm(range(len(batches)-1)):
    r_probs = []
    t_probs = []
    r_prompts_batch = r_prompts[batches[b]:batches[b+1]]
    t_prompts_batch = t_prompts[batches[b]:batches[b+1]]

    r_responses = openai.Completion.create(
        model=model,
        max_tokens=0,
        temperature=0,
        logprobs=0,
        prompt=r_prompts_batch,
        echo=True
    )

    t_responses = openai.Completion.create(
        model=model,
        max_tokens=0,
        temperature=0,
        logprobs=0,
        prompt=t_prompts_batch,
        echo=True
    )

    time.sleep(5)

    for i in range(len(t_prompts_batch)):
        # review_len, review_sentence_len = get_num_tokens(df[text_col].values[batches[b]+i], model)
        # review_sentence_len = [0] + review_sentence_len

        # r_mean_sentence_probs = []
        # t_mean_sentence_probs = []
        # for j in range(len(review_sentence_len)-2):

        #     r_sentence_logprobs = r_responses.choices[i].logprobs.token_logprobs[r_instr_len+np.sum(review_sentence_len[:1+j]):(r_instr_len+np.sum(review_sentence_len[:2+j]))]
        #     r_mean_sentence_probs.append(np.product(np.exp(r_sentence_logprobs)))

        #     t_sentence_logprobs = t_responses.choices[i].logprobs.token_logprobs[t_instr_len+np.sum(review_sentence_len[:1+j]):(t_instr_len+np.sum(review_sentence_len[:2+j]))]
        #     t_mean_sentence_probs.append(np.product(np.exp(t_sentence_logprobs)))

        # r_sentence_probs.append(np.mean(r_mean_sentence_probs))
        # t_sentence_probs.append(np.mean(t_mean_sentence_probs))

        r_probs.append(np.exp(r_responses.choices[i].logprobs.token_logprobs[-1]))
        t_probs.append(np.exp(t_responses.choices[i].logprobs.token_logprobs[-1]))

    df_batch = df.iloc[batches[b]:(batches[b+1])].copy()

    df_batch['{}_prompt_probs'.format(category1)] = r_probs
    df_batch['{}_prompt_probs'.format(category2)] = t_probs


    # if not os.path.exists(os.path.join(data_dir, 'HKRepData_textfull_w_probs.csv')):
    #     df_batch.to_csv(os.path.join(data_dir, 'HKRepData_textfull_w_probs.csv'), index=False)
    # else:
    #     df_batch.to_csv(os.path.join(data_dir, 'HKRepData_textfull_w_probs.csv'), mode='a', index=False, header=False)
   
    # if not os.path.exists(os.path.join(data_dir, '{}_reviews_label_synthetic_numerical_coef_direct1.00.0_w_probs.csv'.format(category))):
    #     df_batch.to_csv(os.path.join(data_dir, '{}_reviews_label_synthetic_numerical_coef_direct1.00.0_w_probs.csv'.format(category)), index=False)
    # else:
    #     df_batch.to_csv(os.path.join(data_dir, '{}_reviews_label_synthetic_numerical_coef_direct1.00.0_w_probs.csv'.format(category)), mode='a', index=False, header=False)

    if not os.path.exists(os.path.join(data_dir, '{}_reviews_label_synthetic_numerical_coef_predreg_w_clf_probs.csv'.format(category))):
        df_batch.to_csv(os.path.join(data_dir, '{}_reviews_label_synthetic_numerical_coef_predreg_w_clf_probs.csv'.format(category)), index=False)
    else:
        df_batch.to_csv(os.path.join(data_dir, '{}_reviews_label_synthetic_numerical_coef_predreg_w_clf_probs.csv'.format(category)), mode='a', index=False, header=False)


    # if not os.path.exists(os.path.join(data_dir, '{}_valence_liwc_w_clf_probs.csv'.format(category))):
    #     df_batch.to_csv(os.path.join(data_dir, '{}_valence_liwc_w_clf_probs.csv'.format(category)), index=False)
    # else:
    #     df_batch.to_csv(os.path.join(data_dir, '{}_valence_liwc_w_clf_probs.csv'.format(category)), mode='a', index=False, header=False)