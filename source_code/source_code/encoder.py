###############################################################################
# Author: Wasi Ahmad
# Project: Learning Vision to Language
# Date Created: 4/02/2017
#
# File Description: This script contains code related to the Encoder class.
###############################################################################

import torch, helper
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


class Encoder(nn.Module):
    """Encoder class of a sequence-to-sequence network"""

    def __init__(self, input_size, config):
        """"Constructor of the class"""
        super(Encoder, self).__init__()
        self.config = config
        self.drop = nn.Dropout(self.config.dropout)
        self.embedding = nn.Embedding(input_size, self.config.emsize)
        self.projection = nn.Linear(self.config.emsize, self.config.emsize)

        if self.config.model in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, self.config.model)(self.config.emsize, self.config.nhid, self.config.nlayers,
                                                      batch_first=True, dropout=self.config.dropout,
                                                      bidirectional=self.config.bidirection)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.config.model]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                         options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(self.config.emsize, self.config.nhid, self.config.nlayers, nonlinearity=nonlinearity,
                              batch_first=True, dropout=self.config.dropout, bidirectional=self.config.bidirection)

    def forward(self, input, hidden):
        """"Defines the forward computation of the encoder"""
        embedded = self.drop(self.embedding(input))
        output = self.projection(embedded.view(-1, embedded.size(2))).view(*embedded.size())
        for i in range(self.config.nlayers):
            output, hidden = self.rnn(output, hidden)
            output = self.drop(output)
        return output, hidden

    def init_weights(self, bsz):
        weight = next(self.parameters()).data
        num_directions = 2 if self.config.bidirection else 1
        if self.config.model == 'LSTM':
            return Variable(weight.new(self.config.nlayers * num_directions, bsz, self.config.nhid).zero_()), Variable(
                weight.new(self.config.nlayers * num_directions, bsz, self.config.nhid).zero_())
        else:
            return Variable(weight.new(self.n_layers * num_directions, bsz, self.config.nhid).zero_())

    def init_embedding_weights(self, dictionary, embeddings_index, embedding_dim):
        """Initialize weight parameters for the embedding layer."""
        pretrained_weight = np.empty([len(dictionary), embedding_dim], dtype=float)
        for i in range(len(dictionary)):
            if dictionary.idx2word[i] in embeddings_index:
                pretrained_weight[i] = embeddings_index[dictionary.idx2word[i]]
            else:
                pretrained_weight[i] = helper.initialize_out_of_vocab_words(embedding_dim)
        # pretrained_weight is a numpy matrix of shape (num_embeddings, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
