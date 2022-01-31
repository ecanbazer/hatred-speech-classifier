# example usage: python3 test.py --file ./data/test.csv

import pickle
import json
import argparse
import torch
import pandas as pd
import params
from functions import awesome_preprocessor, index_and_pad, HatredSpeechLSTM, lemmatizer, stops

parser = argparse.ArgumentParser(description='Script to run LSTM system for hatred speech detection.')
parser.add_argument('--file', type=str, help='The csv file with tweets we want to process.')

if __name__ == '__main__':
    args = parser.parse_args()
    test = pd.read_csv(args.file)

    ##### prepare data #####
    # load counter on train data for frequency filtering
    with open('counter.pickle', 'rb') as inputfile:
        counter = pickle.load(inputfile)

    # clean the data
    test["cleaned"] = awesome_preprocessor(test["tweet"], stops, lemmatizer, counter)

    # open mapping from tokens to indexes built on training data
    with open('vocab.json') as json_file:
        vocab = json.load(json_file)

    max_len = params.max_len

    # map tokens to indexes and pad
    test_padded = index_and_pad(list(test['cleaned']), vocab, max_len)

    # create dataloader


    #### prepare the net ####
    # net params
    vocab_size = params.vocab_size
    output_size = params.output_size
    embedding_dim = params.embedding_dim
    hidden_dim = params.hidden_dim
    n_layers = params.n_layers
    drop_prob = params.drop_prob

    if torch.cuda.is_available():
        print(torch.cuda.is_available())
        train_on_gpu = True
    else:
        train_on_gpu = False

    net = HatredSpeechLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob, train_on_gpu)
    state_dict = torch.load('checkpoint.pth')
    net.load_state_dict(state_dict)

    if train_on_gpu:
        net.cuda()
        
    # run net

    # save results

    # freeze the dependencies




