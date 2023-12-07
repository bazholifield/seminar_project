import torch
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW

df = pd.read_csv("twitter_sentiment_data.csv")

label_mapping = {1: 0, -1: 1, 0: 2, 2: 3}
df['label'] = df['sentiment'].map(label_mapping)

print(df['label'].unique())

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)

tokenized = tokenizer(df['message'].tolist(), padding=True, truncation=True, return_tensors='pt')

dataset = TensorDataset(tokenized['input_ids'], tokenized['attention_mask'], torch.tensor(df['label'].tolist()))
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 3
total_steps = len(dataloader) * epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

model.train()
for epoch in range(epochs):
    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        labels = labels.to(torch.long)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

model.save_pretrained('fine_tuned_distilbert_sentiment')
tokenizer.save_pretrained('fine_tuned_distilbert_sentiment')