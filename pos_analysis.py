import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import MWETokenizer

def configure_mwe_tokenizer():
    tk = MWETokenizer()
    tk.add_mwe(('peer', 'review'))
    tk.add_mwe(('non', 'peer', 'review'))
    tk.add_mwe(('climate', 'scientist'))
    tk.add_mwe(('climate', 'scientist'))
    tk.add_mwe(('medium', 'outlet'))
    tk.add_mwe(('nobel', 'prize', 'win'))
    tk.add_mwe(('nobel', 'win'))
    tk.add_mwe(('nobel', 'peace', 'prize'))
    tk.add_mwe(('nobel', 'laureate'))
    tk.add_mwe(('nobel', 'laureates'))
    tk.add_mwe(('prize', 'win'))
    tk.add_mwe(('ocasio', 'cortez'))
    return tk

def process_row(row):
    sentiment = row['sentiment']
    message = row['message']
    message_id = row['tweetid']

    tokens = nltk.word_tokenize(message)

    tk = configure_mwe_tokenizer()
    tokens = tk.tokenize(tokens)

    verbs = [word for word, pos in nltk.pos_tag(tokens) if pos.startswith('VB')]
    adjectives = [word for word, pos in nltk.pos_tag(tokens) if pos.startswith('JJ')]
    nouns = [word for word, pos in nltk.pos_tag(tokens) if pos.startswith('NN')]

    analysis_result = {
        'id': message_id,
        'sentiment': sentiment,
        'verbs': verbs,
        'adjectives': adjectives,
        'nouns' : nouns,
        'keywords' : verbs + adjectives + nouns
    }

    return analysis_result

def analyze_sentiment_data(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    analysis_results = []
    for _, row in df.iterrows():
        result = process_row(row)
        analysis_results.append(result)

    result_df = pd.DataFrame(analysis_results)
    result_df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    input_csv = 'twitter_sentiment_data.csv'
    output_csv = 'analysis.csv'

    analyze_sentiment_data(input_csv, output_csv)