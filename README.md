# Climate Change Discourse Analysis

This project analyzed rhetorical and opinion-framing differences between climate change-affirming and climate change-denying tweets. Using a labeled corpus of tweets, it fine-tunes a DistilBERT classifier for stance detection and combines it with POS analysis and sentiment lexicon comparisons to quantify how each side frames its arguments. Results indicate measurable differences in inflammatory language usage across stances.

# Setup

To get started:
1. Create a Python 3.10 environment
2. Install the following libraries: 
	pandas nltk transformers torch numpy

pos_analysis.py - Used to tokenize and categorize parts of speech in the message of each tweet.

pos_finetune.py - Used to fine-tune DistilBertForSequenceClassification.

pos_sentiment_vocab - Used to compare the text of each tweet to the provided lexicons, and run the fine-tuned DistilBertForSequenceClassification model. Outputs ratios of tweets using opinion-framing devices. 

# Data
Dataset: [Twitter Climate Change Sentiment Dataset](https://www.kaggle.com/datasets/edqian/twitter-climate-change-sentiment-dataset)  
Source: Kaggle (edqian)
