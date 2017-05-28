###############################################################################
# Author: Wasi Ahmad
# Project: Learning Vision to Language
# Date Created: 4/02/2017
#
# File Description: This script contains code which starts main execution.
###############################################################################

import data, torch, util, helper, train
from seq2seq import Sequence2Sequence

args = util.get_args()
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

# loading the image features
train_img_features = data.load_image_features(args.data, 'vist_train_image_features.hdf5')
print('Training image features loaded for %d stories.' % len(train_img_features))
dev_img_features = data.load_image_features(args.data, 'vist_dev_image_features.hdf5')
print('Development image features loaded for %d stories.' % len(dev_img_features))

dictionary = data.Dictionary()
train_corpus = data.Corpus(args.data + '/sis/', 'train.story-in-sequence.json', dictionary, train_img_features)
print('Train set size = ', len(train_corpus.data))
dev_corpus = data.Corpus(args.data + '/sis/', 'val.story-in-sequence.json', dictionary, dev_img_features)
print('Development set size = ', len(dev_corpus.data))
print('Vocabulary size = ', len(dictionary))

# save the dictionary object to use during testing
helper.save_object(dictionary, args.save_path + 'dictionary.p')

# embeddings_index = helper.load_word_embeddings(args.data + '/glove/', args.word_vectors_file)
# helper.save_word_embeddings(args.data + '/glove/', 'glove.6B.300d.vist.txt', embeddings_index, dictionary.idx2word)

embeddings_index = helper.load_word_embeddings(args.data + '/glove/', 'glove.6B.300d.vist.txt')
print('Number of OOV words = ', len(dictionary) - len(embeddings_index))

# Splitting the data in batches
train_batches = helper.batchify(train_corpus.data, args.batch_size)
print('Number of train batches = ', len(train_batches))
dev_batches = helper.batchify(dev_corpus.data, args.batch_size)
print('Number of dev batches = ', len(dev_batches))

max_sent_length = train_corpus.max_sent_length if train_corpus.max_sent_length > dev_corpus.max_sent_length \
    else dev_corpus.max_sent_length

print('Max sentence length = ', max_sent_length)

# ###############################################################################
# # Build the model
# ###############################################################################

if args.resume_snapshot:
    dictionary = helper.load_object('dictionary')
    embeddings_index = helper.load_word_embeddings('../data/glove/', 'glove.6B.300d.vist.txt')
    model = Sequence2Sequence(dictionary, embeddings_index, args)
    helper.load_model(model, '')
else:
    model = Sequence2Sequence(dictionary, embeddings_index, args)

# for training on multiple GPUs. use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
if args.cuda:
    model = torch.nn.DataParallel(model).cuda()

# ###############################################################################
# # Train the model
# ###############################################################################

train = train.Train(model, dictionary, embeddings_index, train_img_features, dev_img_features, args)
train.train_epochs(train_batches, dev_batches, args.epochs)
