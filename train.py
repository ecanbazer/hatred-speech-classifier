# example usage: python3 train.py --file ./data/train.csv

import pickle
import json
import argparse
import nltk
import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from functions import awesome_preprocessor, evaluate, Dataset, HatredSpeechLSTM, stops, lemmatizer

parser = argparse.ArgumentParser(description='Script to train LSTM system for hatred speech detection.')
parser.add_argument('--file', type=str, help='The csv file with tweets we want to train our system on.')

if __name__ == '__main__':
    args = parser.parse_args()
    train = pd.read_csv(args.file)

    # create a vocabulary of the dataset and count the occurences of each word (for frequency filtering)
    tokens = nltk.word_tokenize(" ".join(list(train.tweet)))
    vocab = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    counter = Counter(vocab)

    # save counter to use it on new data too
    with open('counter.pickle', 'wb') as outputfile:
        pickle.dump(counter, outputfile)

    train["cleaned"] = awesome_preprocessor(train["tweet"], stops, lemmatizer, counter)

    dataset = Dataset(train)
    dataset.vectorizer()

    # write down the max len of tweets in training data
    with open("params.py","a") as params:
        params.write("\n")
        params.write(f'max_len = {dataset.max_len}\n')

    # save the token-int mapping built on training data
    with open("vocab.json", "w") as outfile:
        json.dump(dataset.vocab, outfile)

    batch_size = params.batch_size
    train_loader, valid_loader, test_loader, vocab = dataset.loaders(batch_size)

    # obtain one batch of training data
    dataiter = iter(train_loader)
    sample_x, sample_y = dataiter.next()
    print('Sample input size: ', sample_x.size())  # batch_size, seq_length
    #print('Sample input: \n', sample_x)
    print()
    print('Sample label size: ', sample_y.size())  # batch_size
    #print('Sample label: \n', sample_y)

    print(len(train_loader))

    # Instantiate the model with hyperparams
    vocab_size = len(dataset.vocab) + 1  # +1 is for zeros in padding
    with open("params.py","a") as params:
        params.write("\n")
        params.write(f'vocab_size = {vocab_size}\n')

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
    print(net)

    if train_on_gpu:
        net.cuda()

    # training params
    lr = params.lr
    if params.criterion == 'BCE':
        criterion = nn.BCELoss()
    else:
        raise ValueError('Unknown loss is defined in params.py')
    if params.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    else:
        raise ValueError('Unknown optimizer is defined in params.py')

    epochs = params.epochs
    clip = params.clip  # gradient clipping

    # training
    model, loss_train_all_epochs, loss_val_all_epochs, fmeasure_val_all_epochs = train(net, epochs, train_loader, clip, batch_size,
                                                                                       criterion, optimizer, train_on_gpu, valid_loader)
    # returned 'model' is the best among epochs according to fmeasure on validation data

    # Display the training and validation losses and (validation) f-measure over epochs
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(loss_train_all_epochs)
    plt.title('Training loss')
    plt.subplot(1, 3, 2)
    plt.plot(loss_val_all_epochs)
    plt.title('Validation loss')
    plt.subplot(1, 3, 3)
    plt.plot(fmeasure_val_all_epochs)
    plt.title('Validation fmeasure')
    plt.show()
    plt.savefig('output.png')  # save picture with losses and fmeasure

    # save the model to a checkpoint
    torch.save(model.state_dict(), 'checkpoint.pth')

    # evaluate on test
    loss_last_epoch, f_measure = evaluate(net, test_loader, batch_size, criterion, train_on_gpu)
    print(loss_last_epoch, f_measure)