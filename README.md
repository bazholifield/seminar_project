# Setup

To get started:
1. Create a Python 3.10 environment
2. Install the follwing libraries: 
	pandas nltk transformers torch numpy

pos_analysis.py - Used to tokenize and categorize parts of speech in the message of each tweet.

pos_finetune.py - Used to fine-tune DistilBertForSequenceClassification.

pos_sentiment_vocab - Used to compare the text of each tweet to the provided lexicons, and run the fine-tuned DistilBertForSequenceClassification model. Outputs ratios of tweets using opinion-framing devices. 