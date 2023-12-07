import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import numpy as np

tokenizer = DistilBertTokenizer.from_pretrained('fine_tuned_distilbert_sentiment')
model = DistilBertForSequenceClassification.from_pretrained('fine_tuned_distilbert_sentiment')

output_csv = 'analysis.csv'
df = pd.read_csv(output_csv)

label_mapping = {1: 0, -1: 1, 0: 2, 2: 3}
df['sentiment'] = df['sentiment'].map(label_mapping)
print(df)

batch_size = 8

all_predicted_labels = np.empty(0, dtype=int)

for i in range(0, len(df), batch_size):
    batch_df = df.iloc[i:i+batch_size]
    tokenized_batch = tokenizer(batch_df['keywords'].tolist(), padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(tokenized_batch['input_ids'], attention_mask=tokenized_batch['attention_mask'])

    predicted_labels = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    all_predicted_labels = np.concatenate([all_predicted_labels, predicted_labels])

df['predicted_sentiment'] = all_predicted_labels
df['prediction_accuracy'] = np.where(df['predicted_sentiment'] == df['sentiment'], '1', '0')

print(df['predicted_sentiment'].unique())

print(df)
print(df['predicted_sentiment'])
print(df['sentiment'])

affirm_words = set(open('lexicons/AFFIRM_WORDS.txt').read().splitlines())
doubt_words = set(open('lexicons/DOUBT_WORDS.txt').read().splitlines())
print(affirm_words)
print(doubt_words)

def make_list(s):
    return s.strip('[]').replace("'", "").split(', ') if isinstance(s, str) else s

df['keywords'] = df['keywords'].apply(make_list)
df['affirm_word_count'] = df['keywords'].apply(lambda x: sum(1 for word in x if word in affirm_words))
df['doubt_word_count'] = df['keywords'].apply(lambda x: sum(1 for word in x if word in doubt_words))

print(df)
print(df[['keywords', 'affirm_word_count', 'doubt_word_count']])

false_df = df[df['prediction_accuracy'] == '0'].copy()

print(false_df)

false_affirm_pos = false_df[(false_df['sentiment'] == 0)]['affirm_word_count'].sum()
false_doubt_pos = false_df[(false_df['sentiment'] == 0)]['doubt_word_count'].sum()
false_affirm_neg = false_df[(false_df['sentiment'] == 1)]['affirm_word_count'].sum()
false_doubt_neg = false_df[(false_df['sentiment'] == 1)]['doubt_word_count'].sum()

false_pos_total = ((false_df['sentiment'] == 0)).sum()
false_neg_total = ((false_df['sentiment'] == 1)).sum()

false_affirm_pos_ratio = false_affirm_pos/false_pos_total
false_doubt_pos_ratio = false_doubt_pos/false_pos_total
false_affirm_neg_ratio = false_affirm_neg/false_neg_total
false_doubt_neg_ratio = false_doubt_neg/false_neg_total

print(false_affirm_pos, false_doubt_pos, false_affirm_neg, false_doubt_neg)
print(false_affirm_pos_ratio, false_doubt_pos_ratio, false_affirm_neg_ratio, false_doubt_neg_ratio)

affirm_pos = df[df['sentiment'] == 0]['affirm_word_count'].sum()
doubt_pos = df[df['sentiment'] == 0]['doubt_word_count'].sum()
affirm_neg = df[df['sentiment'] == 1]['affirm_word_count'].sum()
doubt_neg = df[df['sentiment'] == 1]['doubt_word_count'].sum()

pos_total = (df['sentiment'] == 0).sum()
neg_total = (df['sentiment'] == 1).sum()

affirm_pos_ratio = affirm_pos/pos_total
doubt_pos_ratio = doubt_pos/pos_total
affirm_neg_ratio = affirm_neg/neg_total
doubt_neg_ratio = doubt_neg/neg_total

print(affirm_pos, doubt_pos, affirm_neg, doubt_neg)
print(affirm_pos_ratio, doubt_pos_ratio, affirm_neg_ratio, doubt_neg_ratio)