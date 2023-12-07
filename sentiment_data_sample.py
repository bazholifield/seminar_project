import pandas as pd

df = pd.read_csv('twitter_sentiment_data.csv')
df.head(1000).to_csv('twitter_sentiment_sample.csv', index=False)
