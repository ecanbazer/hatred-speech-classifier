import pandas as pd
import params
from params import awesome_preprocessor
from params import index_and_pad
import json
from collections import Counter

"""# Run on new data

A few things that we will need to do, to be able to process new data:

- again import all needed libs

- save the vocabulary that is created for frequency filtering and used in awesome_preprocessor and load it before the preprocessing

- load mapping of tokens to indexes from the saved file vocab

- store max_len from Dataset in a file. Some other values also need to be stored (like the ones used for defining the net: vocab_size, output_size etc.), so it will be useful to create a file like params.py where we will define all of them.

- perform preprocessing - awesome_preprocessor, index_and_pad, and creating a data loader

- running the net and saving its results
"""

test = pd.read_csv("data/test.csv") # we don't need test for now


train_word_list = params.train_word_list
stops = params.stops
lemmatizer = params.lemmatizer
counter = params.counter

test["cleaned"] = awesome_preprocessor(test["tweet"], stops, lemmatizer, counter)

vocab_size = params.vocab_size
output_size = params.output_size
embedding_dim = params.embedding_dim
hidden_dim = params.hidden_dim
n_layers = params.n_layers

with open('vocab.json') as json_file:
    vocab = json.load(json_file)
max_len = params.max_len

test_padded = index_and_pad(list(test['cleaned']), vocab, max_len)

print(test_padded[:10])

#net = HatredSpeechLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

#state_dict = torch.load('checkpoint.pth')
#print(state_dict.keys())
#net.load_state_dict(state_dict)




