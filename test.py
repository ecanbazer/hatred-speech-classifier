# example usage: python3 test.py --file ./data/test.csv

# If run with default settings, generated files:
# net_output.txt -- file with labels, one per line

import pickle
import json
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import params
from functions import awesome_preprocessor, index_and_pad, HatredSpeechLSTM, lemmatizer, stops

parser = argparse.ArgumentParser(description='Script to run LSTM system for hatred speech detection')
parser.add_argument('--file', type=str, help='The csv file with tweets we want to process')
parser.add_argument('--params', type=str, help='File with NN params', default='params.py')
parser.add_argument('--counter', type=str, help='Path of counter file', default='counter.pickle')
parser.add_argument('--mapping', type=str, help='Path of mapping file', default='vocab.json')
parser.add_argument('--model', type=str, help='Path of model checkpoint file', default='checkpoint.pth')
parser.add_argument('--out', type=str, help='Path of the out file with results', default='net_output.txt')

if __name__ == '__main__':
    args = parser.parse_args()
    params_path = args.params
    counter_path = args.counter
    mapping_path = args.mapping
    model_path = args.model
    result_file = args.out

    test = pd.read_csv(args.file)

    ##### prepare data #####
    # load counter on train data for frequency filtering
    with open(counter_path, 'rb') as inputfile:
        counter = pickle.load(inputfile)

    # clean the data
    test['cleaned'] = awesome_preprocessor(test['tweet'], stops, lemmatizer, counter)

    # open mapping from tokens to indexes built on training data
    with open(mapping_path) as json_file:
        vocab = json.load(json_file)

    max_len = params.max_len

    # map tokens to indexes and pad
    test_padded = index_and_pad(list(test['cleaned']), vocab, max_len)

    # create data loader
    class TestDataset(torch.utils.data.Dataset):
        """Our own dataset class for test data."""

        def __init__(self, padded_data):
            self.data = padded_data

        def __getitem__(self, index):
            x = self.data[index]
            return x

        def __len__(self):
            return len(self.data)

    batch_size = params.batch_size
    dataset = TestDataset(test_padded)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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
    state_dict = torch.load(model_path)
    net.load_state_dict(state_dict)

    if train_on_gpu:
        net.cuda()

    #### run net ####
    def run_on_inputs(net, batch_size, loader, max_len, train_on_gpu):
        val_h = net.init_hidden(batch_size)
        net.eval()
        all_lables = []

        with torch.no_grad():
            for inputs in loader:
                val_h = tuple([each.data for each in val_h])

                if train_on_gpu: inputs = inputs.cuda()

                if inputs.shape[0] != batch_size:  # last batch that is smaller
                    num_missing = batch_size - inputs.shape[0]
                    new_tensors = torch.zeros((num_missing, max_len), dtype=torch.uint8)
                    if train_on_gpu: new_tensors = new_tensors.cuda()
                    inputs = torch.cat((inputs, new_tensors), dim=0)
                    assert inputs.shape[0] == batch_size

                output, val_h = net(inputs, val_h)

                # convert output probabilities to predicted class (0 or 1)
                labels_pred = torch.round(output.squeeze())  # rounds to the nearest integer
                all_lables.extend(labels_pred)
        assert len(all_lables[:-num_missing]) == len(dataset), print(len(all_lables[:-num_missing]), len(dataset))
        return all_lables[:-num_missing]

    result = run_on_inputs(net, batch_size, loader, max_len, train_on_gpu)
    result = [str(int(item)) for item in result]

    # save results in a text file
    with open(result_file, 'w') as outfile:
        outfile.write('\n'.join(result))