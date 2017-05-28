###############################################################################
# Author: Wasi Ahmad
# Project: Learning Vision to Language
# Date Created: 4/02/2017
#
# File Description: This script contains code to train the model.
###############################################################################

import time, helper, torch
from util import get_args

import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm


class Train:
    """Train class that encapsulate all functionalities of the training procedure."""

    def __init__(self, model, dictionary, embeddings_index, train_img_features, dev_img_features, config):
        self.model = model
        self.dictionary = dictionary
        self.embeddings_index = embeddings_index
        self.train_img_features = train_img_features
        self.dev_img_features = dev_img_features
        self.config = config
        self.criterion = nn.NLLLoss()  # Negative log-likelihood loss
        self.lr = config.lr

        # Adam optimizer is used for stochastic optimization
        self.optimizer = optim.Adam(self.model.parameters(), self.config.lr)
        self.best_dev_loss = -1
        self.last_best_dev_loss = -1
        self.last_change_dev_loss = 0

    def train_epochs(self, train_batches, dev_batches, n_epochs):
        """Trains model for n_epochs epochs"""
        for epoch in range(n_epochs):
            losses = self.train(train_batches, dev_batches, (epoch + 1))
            helper.save_plot(losses, self.config.save_path + 'training_loss_plot_epoch_{}.png'.format((epoch + 1)))

    def train(self, train_batches, dev_batches, epoch_no):
        # Turn on training mode which enables dropout.
        self.model.train()

        start = time.time()
        plot_losses = []
        print_loss_total = 0
        plot_loss_total = 0

        num_batches = len(train_batches)
        print('epoch %d started' % epoch_no)

        for batch_no in range(num_batches):
            # Clearing out all previous gradient computations.
            self.optimizer.zero_grad()
            train_sentences1, train_sentences2, img_features, max_sent_length = helper.stories_to_tensors(
                train_batches[batch_no], self.train_img_features, self.config.imgsize, self.dictionary)

            assert train_sentences1.size() == train_sentences2.size()
            loss = self.model(train_sentences1, train_sentences2, img_features) / max_sent_length
            # Important if we are using nn.DataParallel()
            if loss.size(0) > 1:
                loss = torch.mean(loss)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
            clip_grad_norm(self.model.parameters(), self.config.clip)
            self.optimizer.step()

            print_loss_total += loss.data[0]
            plot_loss_total += loss.data[0]

            if batch_no % self.config.print_every == 0 and batch_no > 0:
                print_loss_avg = print_loss_total / self.config.print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (
                    helper.show_progress(start, batch_no / num_batches), batch_no,
                    batch_no / num_batches * 100, print_loss_avg))

            if batch_no % self.config.plot_every == 0 and batch_no > 0:
                plot_loss_avg = plot_loss_total / self.config.plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

            if batch_no % self.config.dev_every == 0 and batch_no > 0:
                dev_loss = self.validate(dev_batches)
                print('validation loss = %.4f' % dev_loss)
                if self.best_dev_loss == -1 or self.best_dev_loss > dev_loss:
                    self.best_dev_loss = dev_loss
                else:
                    self.last_change_dev_loss += 1
                    # no improvement in validation loss for last 10 times, so apply learning rate decay
                    if self.last_change_dev_loss == 20:
                        self.last_change_dev_loss = 0
                        self.lr = self.lr * self.config.lr_decay
                        self.optimizer.param_groups[0]['lr'] = self.lr
                        print("Decaying learning rate to %g" % self.lr)

            if batch_no % self.config.save_every == 0 and batch_no > 0:
                if self.last_best_dev_loss == -1 or self.last_best_dev_loss > self.best_dev_loss:
                    self.last_best_dev_loss = self.best_dev_loss
                    helper.save_model_states(self.model, self.last_best_dev_loss, epoch_no, 'model')

        return plot_losses

    def validate(self, dev_batches):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()

        dev_loss = 0
        num_batches = len(dev_batches)
        for batch_no in range(num_batches):
            dev_sentences1, dev_sentences2, img_features, max_sent_length = helper.stories_to_tensors(
                dev_batches[batch_no], self.dev_img_features, self.config.imgsize, self.dictionary)
            assert dev_sentences1.size() == dev_sentences2.size()

            loss = self.model(dev_sentences1, dev_sentences2, img_features) / max_sent_length
            if loss.size(0) > 1:
                loss = torch.mean(loss)
            dev_loss += loss.data[0]

        # Turn on training mode at the end of validation.
        self.model.train()

        return dev_loss / num_batches
