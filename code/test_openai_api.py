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

batch_size = 50
max_tokens = 249999
max_queries = 2999

def get_num_tokens(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    string = string.strip()
    encoding = tiktoken.encoding_for_model(encoding_name)
    total_num_tokens = len(encoding.encode(string))
    sentences = string.split('.')
    sentence_num_tokens = []
    for sentence in sentences:
        sentence_num_tokens.append(len(encoding.encode(sentence)))
    return total_num_tokens, sentence_num_tokens
    # return total_num_tokens

def add_prefix(lst, prefix):
    pre_res = list(map(lambda x: prefix + x, lst))
    return pre_res

# data_dir = '/home/victorialin/Documents/2022-2023/causal_text/data/amazon_synthetic/'
# data_dir = '/home/victorialin/Documents/2022-2023/causal_text/data/hatespeech/'
data_dir = '/home/victorialin/Documents/2022-2023/causal_text/data/emobank/'
# data_dir = '/home/victorialin/Documents/2022-2023/causal_text/data/hk/'


# df_r = pd.read_csv(os.path.join(data_dir, 'music_reviews_label_synthetic_numerical_coef_direct1.00.0.csv'))
# df_t = pd.read_csv(os.path.join(data_dir, 'office_reviews_label_synthetic_numerical_coef_direct1.00.0.csv'))
# df_r = pd.read_csv(os.path.join(data_dir, 'music_reviews_label_synthetic_numerical_coef_predreg.csv'))
# df_t = pd.read_csv(os.path.join(data_dir, 'office_reviews_label_synthetic_numerical_coef_predreg.csv'))
# text_col = 'reviewText'
# df_r = df_r.sample(1000, random_state=230510)
# df_t = df_t.sample(1000, random_state=230510)
# df_r = pd.read_csv(os.path.join(data_dir, 'reddit_liwc_full.csv'))
# df_t = pd.read_csv(os.path.join(data_dir, 'gab_liwc_full.csv'))

# pdb.set_trace()

# df_r = pd.read_csv(os.path.join(data_dir, 'HKRepData_textfull.csv'))
# text_col = 'text_full'

df_r = pd.read_csv(os.path.join(data_dir, 'emobank_valence_liwc.csv'))
# df_t = pd.read_csv(os.path.join(data_dir, 'lowwriterintent_valence_liwc.csv'))
df = df_r
# df = df_t
category = 'emobank'
category1 = 'high'
category2 = 'low'
text_col = 'text'

df = df_r
# df = df_t
# category = 'music'
# category1 = 'music'
# category2 = 'office'
# category = 'reddit'
# category1 = 'reddit'
# category2 = 'gab'
# text_col = 'text'



batches = list(range(0, df.shape[0]+1, batch_size))
# pdb.set_trace()

# r_instructions = "You are writing a comment on a conservative subreddit of the social media site Reddit. Consider the following sentence: "
# t_instructions = 'You are writing a comment on the alt-right social media site Gab. Consider the following sentence: '
# r_instructions = "You are writing a review for your purchase of a musical instrument on Amazon. Consider the following sentence: "
# t_instructions = 'You are writing a review for your purchase of an office product on Amazon. Consider the following sentence: '
# neutral_instructions = 'You are writing a review for your purchase on Amazon. Consider the following sentence: '
# t_instructions = 'You are writing a United States Congressional speech on the Hong Kong democracy protests of 2019. Consider the following sentence: '

r_instructions = "You are writing a positive statement. Consider the following sentence: "
t_instructions = 'You are writing a negative statement. Consider the following sentence: '
# neutral_instructions = 'You are writing a review for your purchase on Amazon. Consider the following sentence: '

model = 'text-davinci-003'

r_prompts = add_prefix(df[text_col].values, r_instructions)
t_prompts = add_prefix(df[text_col].values, t_instructions)

r_instr_len, r_sentences_len = get_num_tokens(r_instructions, model)
t_instr_len, t_sentences_len = get_num_tokens(t_instructions, model)

# print('Waiting 60 seconds, just in case')
# time.sleep(60)

num_requests = 0
num_tokens = 0
start_time = time.time()
for b in tqdm(range(len(batches)-1)):
# for p in tqdm(range(len(r_prompts))):
    r_probs = []
    r_token_probs = []
    r_sentence_probs = []
    t_probs = []
    t_token_probs = []
    t_sentence_probs = []
    r_prompts_batch = r_prompts[batches[b]:batches[b+1]]
    t_prompts_batch = t_prompts[batches[b]:batches[b+1]]

    # r_responses = openai.Completion.create(
    #     model=model,
    #     max_tokens=0,
    #     temperature=0,
    #     logprobs=0,
    #     prompt=r_prompts[p],
    #     echo=True
    # )

    # t_responses = openai.Completion.create(
    #     model=model,
    #     max_tokens=0,
    #     temperature=0,
    #     logprobs=0,
    #     prompt=t_prompts[p],
    #     echo=True
    # )

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

    num_requests += 2*batch_size
    num_tokens += r_responses.usage.total_tokens
    num_tokens += t_responses.usage.total_tokens


    # pdb.set_trace()

    # time.sleep(5)

    for i in range(len(t_prompts_batch)):
        review_len, review_sentence_len = get_num_tokens(df[text_col].values[batches[b]+i], model)
        review_sentence_len = [0] + review_sentence_len

        r_mean_sentence_probs = []
        t_mean_sentence_probs = []
        for j in range(len(review_sentence_len)-2):

            r_sentence_logprobs = r_responses.choices[i].logprobs.token_logprobs[r_instr_len+np.sum(review_sentence_len[:1+j]):(r_instr_len+np.sum(review_sentence_len[:2+j]))]
            r_mean_sentence_probs.append(np.product(np.exp(r_sentence_logprobs)))

            t_sentence_logprobs = t_responses.choices[i].logprobs.token_logprobs[t_instr_len+np.sum(review_sentence_len[:1+j]):(t_instr_len+np.sum(review_sentence_len[:2+j]))]
            t_mean_sentence_probs.append(np.product(np.exp(t_sentence_logprobs)))

        r_sentence_probs.append(np.mean(r_mean_sentence_probs))
        t_sentence_probs.append(np.mean(t_mean_sentence_probs))

        r_sentence_logprobs = r_responses.choices[i].logprobs.token_logprobs[r_instr_len:(r_instr_len+review_len)]
        t_sentence_logprobs = t_responses.choices[i].logprobs.token_logprobs[t_instr_len:(t_instr_len+review_len)]

        r_probs.append(np.product(np.exp(r_sentence_logprobs)))
        t_probs.append(np.product(np.exp(t_sentence_logprobs)))

        r_token_probs.append(np.nanmean(np.exp(r_sentence_logprobs)))
        t_token_probs.append(np.nanmean(np.exp(t_sentence_logprobs)))

    # review_len, review_sentence_len = get_num_tokens(df[text_col].values[p], model)
    # review_sentence_len = [0] + review_sentence_len

    # r_mean_sentence_probs = []
    # t_mean_sentence_probs = []
    # for j in range(len(review_sentence_len)-2):

    #     r_sentence_logprobs = r_responses.choices[0].logprobs.token_logprobs[r_instr_len+np.sum(review_sentence_len[:1+j]):(r_instr_len+np.sum(review_sentence_len[:2+j]))]
    #     r_mean_sentence_probs.append(np.product(np.exp(r_sentence_logprobs)))

    #     t_sentence_logprobs = t_responses.choices[0].logprobs.token_logprobs[t_instr_len+np.sum(review_sentence_len[:1+j]):(t_instr_len+np.sum(review_sentence_len[:2+j]))]
    #     t_mean_sentence_probs.append(np.product(np.exp(t_sentence_logprobs)))

    # r_sentence_probs.append(np.mean(r_mean_sentence_probs))
    # t_sentence_probs.append(np.mean(t_mean_sentence_probs))

    # r_sentence_logprobs = r_responses.choices[0].logprobs.token_logprobs[r_instr_len:(r_instr_len+review_len)]
    # t_sentence_logprobs = t_responses.choices[0].logprobs.token_logprobs[t_instr_len:(t_instr_len+review_len)]

    # r_probs.append(np.product(np.exp(r_sentence_logprobs)))
    # t_probs.append(np.product(np.exp(t_sentence_logprobs)))

    # r_token_probs.append(np.nanmean(np.exp(r_sentence_logprobs)))
    # t_token_probs.append(np.nanmean(np.exp(t_sentence_logprobs)))

    df_batch = df.iloc[batches[b]:(batches[b+1])].copy()
    # df_batch = df.iloc[p].copy()

    df_batch['{}_prompt_probs'.format(category1)] = r_probs
    df_batch['{}_prompt_sentence_probs'.format(category1)] = r_sentence_probs
    df_batch['{}_prompt_token_probs'.format(category1)] = r_token_probs

    df_batch['{}_prompt_probs'.format(category2)] = t_probs
    df_batch['{}_prompt_sentence_probs'.format(category2)] = t_sentence_probs
    df_batch['{}_prompt_token_probs'.format(category2)] = t_token_probs

    # if not os.path.exists(os.path.join(data_dir, 'HKRepData_textfull_w_probs.csv')):
    #     df_batch.to_csv(os.path.join(data_dir, 'HKRepData_textfull_w_probs.csv'), index=False)
    # else:
    #     df_batch.to_csv(os.path.join(data_dir, 'HKRepData_textfull_w_probs.csv'), mode='a', index=False, header=False)
   
    # if not os.path.exists(os.path.join(data_dir, '{}_reviews_label_synthetic_numerical_coef_direct1.00.0_w_probs_full.csv'.format(category))):
    #     df_batch.to_csv(os.path.join(data_dir, '{}_reviews_label_synthetic_numerical_coef_direct1.00.0_w_probs_full.csv'.format(category)), index=False)
    # else:
    #     df_batch.to_csv(os.path.join(data_dir, '{}_reviews_label_synthetic_numerical_coef_direct1.00.0_w_probs_full.csv'.format(category)), mode='a', index=False, header=False)

    # if not os.path.exists(os.path.join(data_dir, '{}_reviews_label_synthetic_numerical_coef_predreg_w_probs.csv'.format(category))):
    #     df_batch.to_csv(os.path.join(data_dir, '{}_reviews_label_synthetic_numerical_coef_predreg_w_probs.csv'.format(category)), index=False)
    # else:
    #     df_batch.to_csv(os.path.join(data_dir, '{}_reviews_label_synthetic_numerical_coef_predreg_w_probs.csv'.format(category)), mode='a', index=False, header=False)


    if not os.path.exists(os.path.join(data_dir, '{}_liwc_w_probs_full.csv'.format(category))):
        df_batch.to_csv(os.path.join(data_dir, '{}_liwc_w_probs_full.csv'.format(category)), index=False)
    else:
        df_batch.to_csv(os.path.join(data_dir, '{}_liwc_w_probs_full.csv'.format(category)), mode='a', index=False, header=False)

    # if not os.path.exists(os.path.join(data_dir, '{}_valence_liwc_w_probs.csv'.format(category))):
    #     df_batch.to_csv(os.path.join(data_dir, '{}_valence_liwc_w_probs.csv'.format(category)), index=False)
    # else:
    #     df_batch.to_csv(os.path.join(data_dir, '{}_valence_liwc_w_probs.csv'.format(category)), mode='a', index=False, header=False)

    if b < len(batches)-2:
        next_batch_df = df.iloc[batches[b+1]:(batches[b+2])].copy()
        combined_text = ''
        for r in range(next_batch_df.shape[0]):
            combined_text = combined_text + r_instructions + next_batch_df[text_col].values[r] + t_instructions + next_batch_df[text_col].values[r]

        next_tokens, _ = get_num_tokens(combined_text, model)
    else:
        next_tokens = 0
    
    print(num_tokens)
    print(num_tokens + next_tokens)
    print(num_requests)
    print(num_requests + 2*batch_size)

    if (num_tokens + next_tokens > max_tokens) or (num_requests + 2*batch_size > max_queries):
        num_tokens = 0
        num_requests = 0
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('Waiting {} seconds'.format(70 - elapsed_time))
        if 70-elapsed_time > 0:
            time.sleep(70 - elapsed_time)
        