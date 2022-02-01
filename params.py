batch_size = 200

# model params
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 2
drop_prob = 0.5

# training params
lr = 0.001
criterion = 'BCE'
optimizer = 'Adam'
epochs = 15
clip = 5  # gradient clipping


# other params -- written during training


max_len = 20

vocab_size = 12250
