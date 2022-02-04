# Recurrent neural network (LSTM model) for hatred speech detection from tweets

#### The goal of this project is to detect hate speech in tweets using RNN, classifying if a hate speech in tweet contains a racist/sexist message. The [dataset](https://www.kaggle.com/arkhoshghalb/twitter-sentiment-analysis-hatred-speech?select=test.csv) used in this task consist of tweets message and its labels, where the tweet is considered racista/sexist if its label is '1', otherwise '0', and the main objective is to predict this labels.

Run the train.py:
* example usage: python3 train.py --file ./data/train.csv

* If run with default settings, generated files:
  1. counter.pickle -- with token frequencies of initial data for frequency filtering
  2. vocab.json -- with mapping from tokens to ids
  3. checkpoint.pth -- the trained model


Run the test.py:
* example usage: python3 test.py --file ./data/test.csv

* If run with default settings, generated files:
  1. net_output.txt -- file with labels, one per line


