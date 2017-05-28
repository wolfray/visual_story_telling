###############################################################################
# Author: Wasi Ahmad
# Project: Learning Vision to Language
# Date Created: 4/02/2017
#
# File Description: This script provides general purpose utility functions that
# are required at different steps in the experiments.
###############################################################################

import re, os, pickle, string, glob, math, time, util, torch
import numpy as np
from nltk import wordpunct_tokenize
from numpy.linalg import norm
from torch.autograd import Variable
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import OrderedDict

args = util.get_args()


def normalize_word_embedding(v):
    """Normalize pre-trained word embedding of a word."""
    return np.array(v) / norm(np.array(v))


def load_word_embeddings(directory, file):
    """Load pre-trained word embeddings from file."""
    embeddings_index = {}
    f = open(os.path.join(directory, file))
    for line in f:
        try:
            values = line.split()
            word = values[0]
            embeddings_index[word] = normalize_word_embedding([float(x) for x in values[1:]])
        except ValueError as e:
            print(e)
    f.close()
    return embeddings_index


def save_word_embeddings(directory, file, embeddings_index, words):
    """Save selected word embeddings in a file."""
    f = open(os.path.join(directory, file), 'w')
    for word in words:
        if word in embeddings_index:
            f.write(word + '\t' + '\t'.join(str(x) for x in embeddings_index[word]) + '\n')
    f.close()


def save_model_states(model, loss, epoch, tag):
    """Save a deep learning network's states in a file."""
    snapshot_prefix = os.path.join(args.save_path, tag)
    snapshot_path = snapshot_prefix + '_loss_{:.6f}_epoch_{}_model.pt'.format(loss, epoch)
    with open(snapshot_path, 'wb') as f:
        torch.save(model.state_dict(), f)
    for f in glob.glob(snapshot_prefix + '*'):
        if f != snapshot_path:
            os.remove(f)


def load_model_states(model, filename):
    """Load a previously saved model states."""
    filepath = os.path.join(args.save_path, filename)
    with open(filepath, 'rb') as f:
        model.load_state_dict(torch.load(f))


def load_model_states_without_dataparallel(model, filename):
    """Load a previously saved model states."""
    filepath = os.path.join(args.save_path, filename)
    with open(filepath, 'rb') as f:
        state_dict = torch.load(f)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)


def save_object(obj, filename):
    """Save an object into file."""
    with open(filename, 'wb') as output:
        pickle.dump(obj, output)


def load_object(filename):
    """Load object from file."""
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj


def tokenize_and_normalize(s):
    """Tokenize and normalize string."""
    token_list = []
    tokens = wordpunct_tokenize(s.lower())
    token_list.extend([x for x in tokens if not re.fullmatch('[' + string.punctuation + ']+', x)])
    return token_list


def initialize_out_of_vocab_words(dimension):
    """Returns a random vector of size dimension where mean is 0 and standard deviation is 1."""
    return np.random.normal(size=dimension)


def sentence_to_tensor(sentence, max_sent_length, dictionary):
    """Convert a sequence of words to a tensor of word indices."""
    sen_rep = torch.LongTensor(max_sent_length).zero_()
    for i in range(len(sentence)):
        if sentence[i] in dictionary.word2idx:
            sen_rep[i] = dictionary.word2idx[sentence[i]]
        else:
            sen_rep[i] = dictionary.word2idx[dictionary.unknown_token]
    return sen_rep


def stories_to_tensors(batch_stories, image_features, img_size, dictionary, max_sent_length=None):
    """Convert a list of sequences to a list of tensors."""
    if max_sent_length is None:
        max_sent_length = 0
        for story in batch_stories:
            for i in range(len(story.annotations)):
                if max_sent_length < len(story.annotations[i]):
                    max_sent_length = len(story.annotations[i])

    total_instances = len(batch_stories) * batch_stories[0].num_annotations
    all_sentences1 = torch.LongTensor(total_instances, max_sent_length)
    all_sentences2 = torch.LongTensor(total_instances, max_sent_length)
    img_features = torch.FloatTensor(total_instances, img_size)
    index = 0
    for story in batch_stories:
        for i in range(len(story.annotations)):
            if i == 0:
                all_sentences1[index] = torch.LongTensor(max_sent_length).fill_(
                    dictionary.word2idx[dictionary.zero_token])
                all_sentences2[index] = sentence_to_tensor(story.annotations[i], max_sent_length, dictionary)
            else:
                all_sentences1[index] = sentence_to_tensor(story.annotations[i - 1], max_sent_length, dictionary)
                all_sentences2[index] = sentence_to_tensor(story.annotations[i], max_sent_length, dictionary)

            img_features[index] = torch.from_numpy(image_features[story.id][i])
            index += 1

    return Variable(all_sentences1), Variable(all_sentences2), Variable(img_features), max_sent_length


def batchify(data, bsz):
    """Transform data into batches."""
    nbatch = len(data) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data[0:nbatch * bsz]
    # Evenly divide the data across the bsz batches.
    batched_data = [[data[bsz * i + j] for j in range(bsz)] for i in range(nbatch)]
    return batched_data


def save_plot(points, filename):
    """Generate and save the plot."""
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)  # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    ax.plot(points)
    fig.savefig(filename)
    plt.close(fig)  # close the figure


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def convert_to_minutes(s):
    """Converts seconds to minutes and seconds."""
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def show_progress(since, percent):
    """Prints time elapsed and estimated time remaining given the current time and progress in %."""
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (convert_to_minutes(s), convert_to_minutes(rs))


def show_attention_plot(input_sentence, output_words, attentions):
    """Shows attention as a graphical plot"""
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence, rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def save_attention_plot(input_sentence, output_words, attentions, filename):
    """Save attention as a graphical plot"""
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence, rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.savefig(filename)
    plt.close(fig)  # close the figure
