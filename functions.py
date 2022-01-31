import re
from collections import Counter
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim.utils import deaccent
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import emoji
import preprocessor as p

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

# create a list of stopwords, without 'not' because it's important for our task
stops = stopwords.words('english')
stops.remove('not')

p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.NUMBER)
# set the pipeline of the tweet-preprocessor object to include only removing URL's, mentions ('@user'), and numbers

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

def awesome_preprocessor(dataset, stopwords, lemmatizer, counter):
  """Preprocessing"""
  REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})")
  REPLACE_WITH_SPACE = re.compile("(<br\s/><br\s/?)|(-)|(/)|(:).")

  cleaned = []
  for tweet in dataset:
    tweet = ' '.join([word for word in tweet.split() if word not in stopwords])
    tweet = emoji.demojize(tweet.replace(':', ' '))  # de-emojizing adds unnecessary colons, like ':red_heart:'
    tweet = p.clean(tweet)  # apply tweet-preprocessor
    tweet = deaccent(tweet)  # de-accents with gensim's deaccent tool
    tweet = REPLACE_NO_SPACE.sub('', tweet.lower())
    tweet = REPLACE_WITH_SPACE.sub(' ', tweet)
    tokens = nltk.word_tokenize(tweet)
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    filtered = []
    for lemma in lemmas:
      if counter[lemma] > 1 and lemma.isalpha() and len(lemma) > 2:
        filtered.append(lemma)
    tweet = ' '.join(filtered).replace('#', '')
    cleaned.append(tweet)

  return cleaned


def index_and_pad(list_tweets, vocab, max_len):
  """
  Return index-mapped and padded tweets.

  list_tweets: the list of cleaned tweets
  vocab: vocabulary that maps tokens to indexes
  max_len: max length of tweets that we allow  (for padding)
  """
  tweets_int = []
  for tweet in list_tweets:
    int_tweet = [vocab[w] if w in vocab else vocab['<unk>'] for w in tweet.split()]
    tweets_int.append(int_tweet)

  features = np.zeros((len(tweets_int), max_len), dtype=int)
  for i, tweet in enumerate(tweets_int):
      tweet_len = len(tweet)

      if tweet_len <= max_len:
          zeroes = list(np.zeros(max_len-tweet_len))
          new = zeroes + tweet
      elif tweet_len > max_len:
          new = tweet[0:max_len]

      features[i, :] = list(new)

  return features


def create_vocab(list_sents):
    """
    Create an indexed vocab in dictionary form.

    list_sents: the list with the text data
    """
    vocab=[]
    data_set = ' '.join(list_sents)
    split_it = data_set.split()
    counter = Counter(split_it)
    #most_occur = counter.most_common(n)
    for word, _ in dict(counter).items():
        vocab.append(word)
    vocab.append('<unk>')
    vocab = sorted(vocab)
    vocab = {w:i for i, w in enumerate(vocab)}
    return vocab


def oversample(train_sample):
  """Oversample minoriity class because of the imbalance"""
  # Separate majority and minority classes in training data for oversampling
  train_majority = train_sample[train_sample['label'] == 0]
  train_minority = train_sample[train_sample['label'] == 1]

  print('majority class before oversample: ', train_majority.shape)
  print('minority class before oversample: ', train_minority.shape)

  # Upsample minority class
  train_minority_oversampled = resample(train_minority,
                                 replace=True,      # sample with replacement
                                 n_samples=train_majority.shape[0],  # to match majority class
                                 random_state=123)  # reproducible results

  # Combine majority class with oversampled minority class
  train_oversampled = pd.concat([train_majority, train_minority_oversampled])
  # shuffling the data
  train_oversampled = train_oversampled.sample(frac=1).reset_index(drop=True)
  # Display new class counts
  print('After oversampling\n', train_oversampled.label.value_counts(), sep='')
  return train_oversampled


class Dataset():
  """Dataset class for creating train, test, and val datasets (prepare the data)."""

  def __init__(self, train, random_state=1, test_size=0.2):
    """Define train, validation, and test data."""

    super().__init__()

    y = train.label.values
    # use 80% for the training and 20% for the test
    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(train.cleaned.values, y, stratify=y,
                                                                            random_state=random_state, test_size=test_size, shuffle=True)
    # now take 10% of the training for validation
    self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, stratify=self.y_train,
                                                                          random_state=1, test_size=0.1, shuffle=True)

  def vectorizer(self):
    """Transform the text data into lists of integers."""

    # create mapping from tokens to indexes on train data
    self.vocab = create_vocab(self.x_train)
    # calculate max len tweet len in train data (in tokens)
    self.max_len = len(max([tweet.split() for tweet in self.x_train], key=len))

    # oversample data and create balanced train dataset
    train_over = oversample(pd.DataFrame({'tweet': self.x_train, 'label': self.y_train}))
    self.y_train = train_over['label'].to_numpy()
    self.x_train = train_over['tweet']

    # convert datasets to integers and pad them
    self.x_train = index_and_pad(self.x_train, self.vocab, self.max_len)
    self.x_val = index_and_pad(self.x_val, self.vocab, self.max_len)
    self.x_test = index_and_pad(self.x_test, self.vocab, self.max_len)

  def loaders(self, batch_size):
    """Create Tensor datasets, return data loaders."""
    #self.x_train=np.vstack(self.x_train).astype(np.int)

    train_data = TensorDataset(torch.from_numpy(self.x_train), torch.from_numpy(self.y_train))
    valid_data = TensorDataset(torch.from_numpy(self.x_val), torch.from_numpy(self.y_val))
    test_data = TensorDataset(torch.from_numpy(self.x_test), torch.from_numpy(self.y_test))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    return train_loader, valid_loader, test_loader, self.vocab


class HatredSpeechLSTM(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob, train_on_gpu=False):
        """Initialize the model by setting up the layers."""

        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.train_on_gpu = train_on_gpu

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()


    def forward(self, x, hidden):
        """Perform a forward pass of our model on some input and hidden state."""
        batch_size = x.size(0)

        # embeddings and lstm_out
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]  # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden


    def init_hidden(self, batch_size):
        """Initialize hidden state."""
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if self.train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden


def train(net, epochs, train_loader, clip, batch_size, criterion, optimizer, train_on_gpu, valid_loader=None):
  net.train()

  loss_train_all_epochs = []
  loss_val_all_epochs = []
  fmeasure_val_all_epochs = []
  best_fmeasure = 0  # 'optimal' f-measure that will be used to perform validation

  # training loop
  for e in range(epochs):
    # initialize hidden state
    h = net.init_hidden(batch_size)

    # Initialize the training loss for the current epoch
    loss_current_epoch = 0

    # batch loop
    for inputs, labels in train_loader:

      # Creating new variables for the hidden state, otherwise
      # we'd backprop through the entire training history
      h = tuple([each.data for each in h])

      # zero accumulated gradients
      net.zero_grad()

      # get the output from the model
      inputs = inputs.type(torch.LongTensor)
      if train_on_gpu: inputs, labels = inputs.cuda(), labels.cuda()

      if inputs.shape[0] != batch_size:  # for the last batch if it's smaller - don't do anything
        continue
      output, h = net(inputs, h)

      # calculate the loss and perform backpropagation
      loss = criterion(output, labels.float())
      loss.backward()
      # clip_grad_norm helps prevent the exploding gradient problem in LSTMs
      nn.utils.clip_grad_norm_(net.parameters(), clip)
      optimizer.step()

      # Add the batch loss to the current epoch loss
      loss_current_epoch += loss.item()

    # at the end of each epoch, record the loss over all batches and the accuracy on the validation set
    loss_train_all_epochs.append(loss_current_epoch)

    if valid_loader is None:  # if we don't have any validation data
      val_loss_current_epoch, fmeasure_current_epoch = 0, 0
    else:
      val_loss_current_epoch, fmeasure_current_epoch = evaluate(net, valid_loader, batch_size, criterion, train_on_gpu)
    loss_val_all_epochs.append(val_loss_current_epoch)
    fmeasure_val_all_epochs.append(fmeasure_current_epoch)

    # print the training and validation loss and validation fmeasure
    print('Epoch [{}/{}],\nOverall training loss: {:.4f};\nMean training loss: {:.4f};\nOverall validation loss: {:.4f};\
    \nMean validation loss: {:.4f};\nMean validation fmeasure: {:.4f}\n'
               .format(e+1, epochs, loss_current_epoch, loss_current_epoch / len(train_loader), val_loss_current_epoch,
                       val_loss_current_epoch / (len(valid_loader) if valid_loader else 1), fmeasure_current_epoch))

    # save the model to variable if the fmeasure is higher than the 'optimal' value
    if fmeasure_current_epoch >= best_fmeasure:
      model_opt = net
      best_fmeasure = fmeasure_current_epoch

    # return to the training mode
    net.train()

  return model_opt, loss_train_all_epochs, loss_val_all_epochs, fmeasure_val_all_epochs


def evaluate(net, valid_loader, batch_size, criterion, train_on_gpu, _print=False):
  val_h = net.init_hidden(batch_size)
  val_loss_current_epoch = 0
  net.eval()  # set to the eval mode
  with torch.no_grad():
    num_tp = 0
    num_fp = 0
    num_fn = 0

    for inputs, labels in valid_loader:

      val_h = tuple([each.data for each in val_h])

      inputs = inputs.type(torch.LongTensor)
      if train_on_gpu: inputs, labels = inputs.cuda(), labels.cuda()

      if inputs.shape[0] != batch_size:  # for the last batch that is smaller - don't do anything
        continue
      output, val_h = net(inputs, val_h)

      val_loss = criterion(output.squeeze(), labels.float())
      val_loss_current_epoch += val_loss.item()  # add the batch loss to the current epoch loss

      # convert output probabilities to predicted class (0 or 1)
      labels_pred = torch.round(output.squeeze())  # rounds to the nearest integer

      tp, fn, fp = return_stats(labels, labels_pred)
      if _print:
        print('batch tp, fn, fp:', tp, fn, fp)
        print('batch labels:', labels_pred)

      num_tp += tp
      num_fn += fn
      num_fp += fp

  f_measure = num_tp / (num_tp + 0.5 * (num_fp + num_fn))

  if _print:
    print('overall tp, fn, fp:', num_tp, num_fn, num_fp)

  return val_loss_current_epoch, f_measure


def return_stats(first, second):
  """Get two boolean tensors, return number of TP, FN, and FP."""
  tp = torch.sum((first==second) * (first==1)).item()
  fn = torch.sum((first!=second) * (first==1)).item()
  fp = torch.sum((first!=second) * (first==0)).item()
  return tp, fn, fp