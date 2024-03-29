import pdb
import os
import gc
import torch
import argparse
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer, AutoModelForMaskedLM, AutoModelForCausalLM
from sklearn.model_selection import train_test_split

def preprocess(df, tokenizer, args):
    return tokenizer(df[args.text_col], truncation=True, padding="max_length")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/home/victorialin/Documents/2022-2023/causal_text/data/hk/')
    parser.add_argument('--csv', type=str, default='target_train.csv')
    parser.add_argument('--text-col', type=str, default='text_full')
    parser.add_argument('--lm-type', type=str, default='clm')
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--seed', type=int, default=230301)
    parser.add_argument('--mlm-prob', type=float, default=0.15)
    parser.add_argument('--output-dir', type=str, default='/home/victorialin/Documents/2022-2023/causal_text/models/hk/')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--decay', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--eval', action='store_true')

    args = parser.parse_args()

    return args

gc.collect()
torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

args = get_args()

df_target = pd.read_csv(os.path.join(args.data_dir, args.csv))

if args.lm_type == 'mlm':
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, max_length=512, padding=True, truncation=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_prob)
elif args.lm_type == 'clm':
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, max_length=512, padding=True, truncation=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
model.to(device)

df_target_train, df_target_val = train_test_split(df_target, test_size=0.33)

train_dat = Dataset.from_pandas(df_target_train)
val_dat = Dataset.from_pandas(df_target_val)

train_dat = train_dat.map(preprocess, fn_kwargs={'tokenizer': tokenizer, 'args': args}, batched=True)
train_dat = train_dat.remove_columns([args.text_col, '__index_level_0__'])
val_dat = val_dat.map(preprocess, fn_kwargs={'tokenizer': tokenizer, 'args': args}, batched=True)
val_dat = val_dat.remove_columns([args.text_col, '__index_level_0__'])

training_args = TrainingArguments(
    output_dir=os.path.join(args.output_dir, args.model),
    learning_rate=args.lr,
    num_train_epochs=args.num_epochs,
    weight_decay=args.decay,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    seed=args.seed
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dat,
    eval_dataset=val_dat,
    tokenizer=tokenizer,
    data_collator=data_collator
)

if not args.eval:
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, args.model, 'best_model'))
else:
    trainer.evaluate()