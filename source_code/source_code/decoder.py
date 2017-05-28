###############################################################################
# Author: Wasi Ahmad
# Project: Learning Vision to Language
# Date Created: 4/02/2017
#
# File Description: This script contains code related to the Decoder class.
###############################################################################

import torch, numpy
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable


class Decoder(nn.Module):
    """Decoder class of a sequence-to-sequence network"""

    def __init__(self, input_size, config):
        """"Constructor of the class"""
        super(Decoder, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(input_size, self.config.nhid)
        self.linear = nn.Linear(self.config.imgsize, self.config.nhid)
        self.drop = nn.Dropout(self.config.dropout)
        self.out = nn.Linear(self.config.nhid, input_size)
        self.softmax = nn.LogSoftmax()

        if self.config.model in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, self.config.model)(self.config.emsize, self.config.nhid, self.config.nlayers,
                                                      batch_first=True, dropout=self.config.dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.config.model]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                         options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(self.config.emsize, self.config.nhid, self.config.nlayers, nonlinearity=nonlinearity,
                              batch_first=True, dropout=self.config.dropout)

    def forward(self, input, img_features, hidden):
        """"Defines the forward computation of the decoder"""
        embedded = self.drop(self.embedding(input))
        img_embeddings = self.linear(img_features)
        output = torch.div(torch.add(embedded, img_embeddings), 2).unsqueeze(1)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.rnn(output, hidden)
            output = self.drop(output)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_weights(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


class AttentionDecoder(nn.Module):
    """Decoder class (with attention) of a sequence-to-sequence network"""

    def __init__(self, input_size, max_sent_length, config):
        """"Constructor of the class"""
        super(AttentionDecoder, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(input_size, self.config.nhid)
        self.linear = nn.Linear(self.config.imgsize, self.config.nhid)
        self.attn = nn.Linear(self.config.nhid * 2, max_sent_length)
        self.attn_combine = nn.Linear(self.config.nhid * 2, self.config.nhid)
        self.drop = nn.Dropout(self.config.dropout)
        self.out = nn.Linear(self.config.nhid, input_size)

        if self.config.model in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, self.config.model)(self.config.emsize, self.config.nhid, self.config.nlayers,
                                                      batch_first=True, dropout=self.config.dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.config.model]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                         options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(self.config.emsize, self.config.nhid, self.config.nlayers, nonlinearity=nonlinearity,
                              batch_first=True, dropout=self.config.dropout)

    def forward(self, input, img_features, hidden, encoder_outputs):
        """"Defines the forward computation of the attention decoder"""
        embedded = self.drop(self.embedding(input))
        final_layer_index = hidden[0].size(0) - 1
        attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden[0][final_layer_index]), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        attention_combined = self.attn_combine(torch.cat((embedded, attn_applied.squeeze(1)), 1))
        img_embeddings = self.linear(img_features)
        output = torch.div(torch.add(attention_combined, img_embeddings), 2).unsqueeze(1)

        for i in range(self.config.nlayers):
            output = F.relu(output)
            output, hidden = self.rnn(output, hidden)
            output = self.drop(output)

        output = F.log_softmax(self.out(output.squeeze(1)))
        return output, hidden, attn_weights

    def init_weights(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.n_layers, bsz, self.hidden_size).zero_()),
                    Variable(weight.new(self.n_layers, bsz, self.hidden_size).zero_()))
        else:
            return Variable(weight.new(self.n_layers, bsz, self.hidden_size).zero_())


class GloballyAttentiveDecoder(nn.Module):
    """Decoder class (with attention) of a sequence-to-sequence network. This class implements the global attention
    mechanism described in paper - http://aclweb.org/anthology/D15-1166"""

    def __init__(self, input_size, config):
        """"Constructor of the class"""
        super(GloballyAttentiveDecoder, self).__init__()
        self.config = config
        self.weight = nn.Parameter(torch.Tensor(self.config.nhid, self.config.nhid))
        self.embedding = nn.Embedding(input_size, self.config.nhid)
        self.linear = nn.Linear(self.config.imgsize, self.config.nhid)
        self.attn_combine = nn.Linear(self.config.nhid * 2, self.config.nhid)
        self.drop = nn.Dropout(self.config.dropout)
        self.out = nn.Linear(self.config.nhid, input_size)

        # Initializing weights using xavier normal distribution
        init.xavier_normal(self.weight)

        if self.config.model in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, self.config.model)(self.config.emsize, self.config.nhid, self.config.nlayers,
                                                      batch_first=True, dropout=self.config.dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.config.model]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                         options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(self.config.emsize, self.config.nhid, self.config.nlayers, nonlinearity=nonlinearity,
                              batch_first=True, dropout=self.config.dropout)

    def forward(self, input, img_features, hidden, encoder_outputs):
        """"Defines the forward computation of the attention decoder"""
        embedded = self.drop(self.embedding(input))
        img_embeddings = self.linear(img_features)
        output = torch.div(torch.add(embedded, img_embeddings), 2).unsqueeze(1)
        for i in range(self.config.nlayers):
            output, hidden = self.rnn(output, hidden)

        weighted_encoder_output = torch.bmm(self.weight.expand(encoder_outputs.size(0), *self.weight.size()),
                                            torch.transpose(encoder_outputs, 1, 2))
        score = torch.bmm(output, weighted_encoder_output)
        attn_weights = F.softmax(score.squeeze(1))
        context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        attention_combine = self.attn_combine(torch.cat((context_vector.squeeze(1), output.squeeze(1)), 1))
        output = F.log_softmax(self.out(attention_combine))
        return output, hidden, attn_weights

    def init_weights(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.n_layers, bsz, self.hidden_size).zero_()),
                    Variable(weight.new(self.n_layers, bsz, self.hidden_size).zero_()))
        else:
            return Variable(weight.new(self.n_layers, bsz, self.hidden_size).zero_())


class LocallyAttentiveDecoder(nn.Module):
    """Decoder class (with attention) of a sequence-to-sequence network. This class implements the local attention
    mechanism described in paper - http://aclweb.org/anthology/D15-1166"""

    def __init__(self, input_size, config):
        """"Constructor of the class"""
        super(LocallyAttentiveDecoder, self).__init__()
        self.config = config
        self.weight = nn.Parameter(torch.Tensor(self.config.nhid, self.config.nhid))
        self.weight_p = nn.Parameter(torch.Tensor(self.config.nhid, self.config.nhid))
        self.weight_v = nn.Parameter(torch.Tensor(1, self.config.nhid))
        self.embedding = nn.Embedding(input_size, self.config.nhid)
        self.linear = nn.Linear(self.config.imgsize, self.config.nhid)
        self.attn_combine = nn.Linear(self.config.nhid * 2, self.config.nhid)
        self.drop = nn.Dropout(self.config.dropout)
        self.out = nn.Linear(self.config.nhid, input_size)
        self.window_size = 3

        # Initializing weights using xavier normal distribution
        init.xavier_normal(self.weight)
        init.xavier_normal(self.weight_p)
        init.xavier_normal(self.weight_v)

        if self.config.model in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, self.config.model)(self.config.emsize, self.config.nhid, self.config.nlayers,
                                                      batch_first=True, dropout=self.config.dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.config.model]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                         options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(self.config.emsize, self.config.nhid, self.config.nlayers, nonlinearity=nonlinearity,
                              batch_first=True, dropout=self.config.dropout)

    def forward(self, input, img_features, hidden, encoder_outputs):
        """"Defines the forward computation of the attention decoder"""
        embedded = self.drop(self.embedding(input))
        img_embeddings = self.linear(img_features)
        output = torch.div(torch.add(embedded, img_embeddings), 2).unsqueeze(1)
        for i in range(self.config.nlayers):
            output, hidden = self.rnn(output, hidden)

        weighted_encoder_output = torch.bmm(self.weight.expand(encoder_outputs.size(0), *self.weight.size()),
                                            torch.transpose(encoder_outputs, 1, 2))
        score = torch.bmm(output, weighted_encoder_output)
        attn_weights = F.softmax(score.squeeze(1))

        pt_sigmoid = F.sigmoid(
            torch.bmm(self.weight_v.expand(encoder_outputs.size(0), *self.weight_v.size()),
                      F.tanh(torch.bmm(self.weight_p.expand(encoder_outputs.size(0), *self.weight_p.size()),
                                       torch.transpose(output, 1, 2)))))
        pt_sigmoid = torch.mul(pt_sigmoid.squeeze(1).squeeze(1), encoder_outputs.size(1))
        pt_sigmoid = pt_sigmoid.unsqueeze(1).expand(pt_sigmoid.size(0), encoder_outputs.size(1))
        current_s = Variable(torch.range(0, encoder_outputs.size(1) - 1, 1).type_as(pt_sigmoid.data))
        current_s = current_s.expand(encoder_outputs.size(0), current_s.size(0))
        nominator = torch.pow(current_s.sub(pt_sigmoid.expand(*current_s.size())), 2)
        denominator = -2 * (self.window_size / 2) * (self.window_size / 2)
        emphasis = torch.exp(torch.div(nominator, denominator))

        attn_weights = torch.mul(attn_weights, emphasis)
        context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        attention_combine = self.attn_combine(torch.cat((context_vector.squeeze(1), output.squeeze(1)), 1))
        output = F.log_softmax(self.out(attention_combine))
        return output, hidden, attn_weights

    def init_weights(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.n_layers, bsz, self.hidden_size).zero_()),
                    Variable(weight.new(self.n_layers, bsz, self.hidden_size).zero_()))
        else:
            return Variable(weight.new(self.n_layers, bsz, self.hidden_size).zero_())
