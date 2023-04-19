import gc
import os
import pdb
import torch
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, Features, Value
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer

def preprocess(df, tokenizer, args):
    return tokenizer(df[args.text_col], truncation=True, padding="max_length")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text-col', type=str, default='reviewText')
    parser.add_argument('--label', type=str, default='helpful')
    parser.add_argument('--label-type', type=str, default='binary')
    parser.add_argument('--model', type=str, default='distilbert-base-uncased')
    parser.add_argument('--seed', type=int, default=230418)
    parser.add_argument('--output-dir', type=str, default='/home/victorialin/Documents/2022-2023/causal_text/models/amazon_synthetic/')
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

music_df = pd.read_csv('/home/victorialin/Documents/2022-2023/causal_text/data/amazon_synthetic/music_reviews_raw.csv')
music_df = music_df.sample(15000, random_state=args.seed).reset_index(drop=True)
office_df = pd.read_csv('/home/victorialin/Documents/2022-2023/causal_text/data/amazon_synthetic/office_reviews_raw.csv')
office_df = office_df.sample(15000, random_state=args.seed).reset_index(drop=True)
if args.label_type == 'binary':
    cutoff = music_df[args.label].median()
    music_df[args.label] = music_df[args.label] > cutoff
    office_df[args.label] = office_df[args.label] > cutoff
    num_labels = 2
else:
    num_labels = 1
music_df['category'] = 'music'
office_df['category'] = 'office'

df = pd.concat([music_df, office_df]).reset_index(drop=True)

train_df, val_df = train_test_split(df, test_size=0.33, random_state=args.seed)

train_dat = Dataset.from_pandas(train_df[[args.text_col, args.label]])
val_dat = Dataset.from_pandas(val_df[[args.text_col, args.label]])

model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained(args.model, padding=True, truncation=True, max_length=512)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model.to(device)

train_dat = train_dat.map(preprocess, fn_kwargs={'tokenizer': tokenizer, 'args': args}, batched=True)
train_dat = train_dat.remove_columns([args.text_col, '__index_level_0__'])
train_dat = train_dat.rename_column(args.label, 'label')

val_dat = val_dat.map(preprocess, fn_kwargs={'tokenizer': tokenizer, 'args': args}, batched=True)
val_dat = val_dat.remove_columns([args.text_col, '__index_level_0__'])
val_dat = val_dat.rename_column(args.label, 'label')

if args.label_type == 'binary':
    train_dat = train_dat.cast_column('label', Value('int64'))
    val_dat = val_dat.cast_column('label', Value('int64'))
else:
    train_dat = train_dat.cast_column('label', Value('float32'))
    val_dat = val_dat.cast_column('label', Value('float32'))

training_args = TrainingArguments(
    output_dir=os.path.join(args.output_dir, args.model, args.label, args.label_type),
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
    trainer.save_model(os.path.join(args.output_dir, args.model, args.label, args.label_type, 'best_model'))
else:
    trainer.evaluate()

train_preds = trainer.predict(train_dat)
val_preds = trainer.predict(val_dat)

if args.label_type == 'binary':
    train_preds = torch.nn.functional.softmax(torch.tensor(train_preds[0]))
    val_preds = torch.nn.functional.softmax(torch.tensor(val_preds[0]))
    train_df['pred_prob'] = train_preds[:,1]
    val_df['pred_prob'] = val_preds[:,1]
else:
    train_df['pred'] = train_preds[0]
    val_df['pred'] = val_preds[0]

# prob_df = pd.concat([train_df, val_df], axis=0)
prob_df = val_df
prob_df = prob_df.sort_index().reset_index(drop=True)

music_df = prob_df[prob_df['category']=='music'].drop('category', axis=1).reset_index(drop=True)
office_df = prob_df[prob_df['category']=='office'].drop('category', axis=1).reset_index(drop=True)

music_df.to_csv('/home/victorialin/Documents/2022-2023/causal_text/data/amazon_synthetic/music_reviews_pred_{}.csv'.format(args.label_type))
office_df.to_csv('/home/victorialin/Documents/2022-2023/causal_text/data/amazon_synthetic/office_reviews_pred_{}.csv'.format(args.label_type))
prob_df.to_csv('/home/victorialin/Documents/2022-2023/causal_text/data/amazon_synthetic/combined_reviews_pred_{}.csv'.format(args.label_type))

