###############################################################################
# Author: Wasi Ahmad
# Project: Learning Vision to Language
# Date Created: 4/02/2017
#
# File Description: This script contains code to read and parse input files.
###############################################################################

import os, json, helper, h5py


class Dictionary(object):
    """Dictionary class that stores all words of train/dev corpus."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        # Create and store three special tokens
        self.pad_token = '<PAD>'
        self.zero_token = '<ZERO>'
        self.start_token = '<SOS>'
        self.end_token = '<EOS>'
        self.unknown_token = '<UNKNOWN>'
        self.idx2word.append(self.pad_token)
        self.word2idx[self.pad_token] = len(self.idx2word) - 1
        self.idx2word.append(self.zero_token)
        self.word2idx[self.zero_token] = len(self.idx2word) - 1
        self.idx2word.append(self.start_token)
        self.word2idx[self.start_token] = len(self.idx2word) - 1
        self.idx2word.append(self.end_token)
        self.word2idx[self.end_token] = len(self.idx2word) - 1
        self.idx2word.append(self.unknown_token)
        self.word2idx[self.unknown_token] = len(self.idx2word) - 1

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def contains(self, word):
        return True if word in self.word2idx else False

    def __len__(self):
        return len(self.idx2word)


class Story(object):
    """Class that wraps a story."""

    def __init__(self, story_id):
        self.id = story_id
        self.annotations = [None, None, None, None, None]
        self.num_annotations = 5

    def add_annotations(self, sentence, sent_id, dictionary, is_test_instance):
        assert self.num_annotations * (1 + self.id) > sent_id
        words = [dictionary.start_token] + helper.tokenize_and_normalize(sentence) + [dictionary.end_token]
        if is_test_instance:
            for i in range(len(words)):
                if not dictionary.contains(words[i]):
                    words[i] = dictionary.unknown_token
        else:
            for word in words:
                dictionary.add_word(word)

        self.annotations[sent_id % self.num_annotations] = words
        return len(words)


class Corpus(object):
    """Corpus class which contains all information about train/dev/test corpus."""

    def __init__(self, path, filename, dictionary, selected_stories=None, is_test_corpus=False):
        self.data = []
        self.max_sent_length = 0
        self.parse(os.path.join(path, filename), dictionary, is_test_corpus, selected_stories)

    def parse(self, path, dictionary, is_test_corpus, selected_stories=None):
        """Parses the content of a file."""
        assert os.path.exists(path)

        stories = {}
        dict_data = json.load(open(path))
        for item in dict_data['annotations']:
            _id = int(item[0]['story_id'])
            if selected_stories and _id not in selected_stories:
                continue
            _scene_id = int(item[0]['storylet_id'])
            _text = item[0]['text']
            if _id in stories:
                num_tokens = stories[_id].add_annotations(_text, _scene_id, dictionary, is_test_corpus)
            else:
                story = Story(_id)
                num_tokens = story.add_annotations(_text, _scene_id, dictionary, is_test_corpus)
                stories[_id] = story

            if self.max_sent_length < num_tokens:
                self.max_sent_length = num_tokens

        # Check if all stories contain five annotations
        for key, value in stories.items():
            if None not in value.annotations:
                self.data.append(value)


def load_image_features(filepath, filename):
    assert os.path.exists(os.path.join(filepath, filename))
    image_features = {}
    with h5py.File(os.path.join(filepath, filename), 'r') as hf:
        for key in hf.keys():
            image_features[int(key)] = hf[key][:]

    return image_features
