###############################################################################
# Author: Wasi Ahmad
# Project: Learning Vision to Language
# Date Created: 4/02/2017
#
# File Description: This script contains code related to the sequence-to-sequence
# network.
###############################################################################

import torch, helper
import torch.nn as nn
from torch.autograd import Variable
from encoder import Encoder
from decoder import GloballyAttentiveDecoder, LocallyAttentiveDecoder


class Sequence2Sequence(nn.Module):
    """Class that classifies question pair as duplicate or not."""

    def __init__(self, dictionary, embedding_index, args):
        """"Constructor of the class."""
        super(Sequence2Sequence, self).__init__()
        self.dictionary = dictionary
        self.embedding_index = embedding_index
        self.config = args
        self.encoder = Encoder(len(self.dictionary), self.config)
        self.decoder = GloballyAttentiveDecoder(len(self.dictionary), self.config)
        self.criterion = nn.NLLLoss()  # Negative log-likelihood loss

        # Initializing the weight parameters for the embedding layer in the encoder.
        self.encoder.init_embedding_weights(self.dictionary, self.embedding_index, self.config.emsize)

    def forward(self, batch_sentence1, batch_sentence2, img_features):
        """"Defines the forward computation of the question classifier."""
        if self.config.model == 'LSTM':
            encoder_hidden, encoder_cell = self.encoder.init_weights(batch_sentence1.size(0))
            output, hidden = self.encoder(batch_sentence1, (encoder_hidden, encoder_cell))
        else:
            encoder_hidden = self.encoder.init_weights(batch_sentence1.size(0))
            output, hidden = self.encoder(batch_sentence1, encoder_hidden)

        # Initialize hidden states of decoder with the last hidden states of the encoder
        decoder_hidden = hidden

        loss = 0
        for idx in range(1, batch_sentence2.size(1)):
            # Use the real target outputs as each next input (teacher forcing)
            input_variable = Variable(torch.zeros(batch_sentence2.size(0)).long())
            target_variable = Variable(torch.zeros(batch_sentence2.size(0)).long())
            if self.config.cuda:
                input_variable = input_variable.cuda()
                target_variable = target_variable.cuda()
            for batch_item_no in range(batch_sentence2.size(0)):
                input_variable[batch_item_no] = batch_sentence2[batch_item_no][idx - 1]
                target_variable[batch_item_no] = batch_sentence2[batch_item_no][idx]

            decoder_output, decoder_hidden, decoder_attention = self.decoder(input_variable, img_features,
                                                                             decoder_hidden, output)
            loss += self.criterion(decoder_output, target_variable)

        return loss
