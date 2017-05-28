###############################################################################
# Author: Wasi Ahmad
# Project: Learning Vision to Language
# Date Created: 4/02/2017
#
# File Description: This script contains code to test the model.
###############################################################################

import data, util, helper, torch, nltk
from torch.autograd import Variable

from seq2seq import Sequence2Sequence

args = util.get_args()


def test(model, single_sentence):
    if model.config.model == 'LSTM':
        encoder_hidden, encoder_cell = model.encoder.init_weights(single_sentence.size(0))
        output, hidden = model.encoder(single_sentence, (encoder_hidden, encoder_cell))
    else:
        encoder_hidden = model.encoder.init_weights(single_sentence.size(0))
        output, hidden = model.encoder(single_sentence, encoder_hidden)

    # Initialize hidden states of decoder with the last hidden states of the encoder
    decoder_hidden = hidden

    sos_token_index = model.dictionary.word2idx[model.dictionary.start_token]
    eos_token_index = model.dictionary.word2idx[model.dictionary.end_token]

    # First input of the decoder is the sentence start token
    decoder_input = Variable(torch.LongTensor([sos_token_index]))
    decoded_words = []
    decoder_attentions = torch.zeros(model.config.max_length, model.config.max_length)

    for di in range(model.config.max_length - 1):
        if model.config.cuda:
            decoder_input = decoder_input.cuda()
        decoder_output, decoder_hidden, decoder_attention = model.decoder(decoder_input, decoder_hidden,
                                                                          output)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == eos_token_index:
            decoded_words.append(model.dictionary.end_token)
            break
        else:
            decoded_words.append(model.dictionary.idx2word[ni])
        decoder_input = Variable(torch.LongTensor([ni]))

    return decoded_words, decoder_attentions[:di + 1]


def evaluate(model, dictionary, sentence):
    """Generates word sequence and their attentions"""
    input_tensor = helper.sentence_to_tensor(sentence, args.max_length, dictionary)
    input_sentence = Variable(input_tensor.view(1, - 1), volatile=True)
    if args.cuda:
        input_sentence = input_sentence.cuda()
    output_words, attentions = test(model, input_sentence)
    return output_words, attentions


def evaluate_and_show_attention(model, dictionary, input_words, filename):
    """Evaluates and shows attention given the input sentence"""
    output_words, attentions = evaluate(model, dictionary, input_words)
    print('input = ', ' '.join(input_words))
    print('output = ', ' '.join(output_words))
    print(nltk.translate.bleu_score.sentence_bleu(input_words, output_words))
    helper.save_attention_plot(input_words, output_words, attentions, args.save_path + filename)


def evaluate_using_bleu(model, test_batches, dictionary, max_sent_length=None):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    num_batches = len(test_batches)
    average_bleu = 0
    for batch_no in range(num_batches):
        if max_sent_length:
            test_sentences1, test_sentences2 = helper.stories_to_tensors(test_batches[batch_no], dictionary,
                                                                         max_sent_length)
        else:
            test_sentences1, test_sentences2, max_sent_length = helper.stories_to_tensors(test_batches[batch_no],
                                                                                          dictionary)
        assert test_sentences1.size() == test_sentences2.size()

        story_idx = 0
        for idx in range(test_sentences1.size(0)):
            input_sentence = test_sentences1[idx].view(1, - 1)
            if args.cuda:
                input_sentence = input_sentence.cuda()
            output_words, attentions = test(model, input_sentence)
            annotation_idx = idx % (test_batches[batch_no][story_idx].num_annotations - 1)
            if idx > 0 and annotation_idx == 0:
                story_idx += 1
            average_bleu += nltk.translate.bleu_score.sentence_bleu(output_words[:-1],
                                                                    test_batches[batch_no][story_idx].annotations[
                                                                        annotation_idx + 1][:-1])

    return average_bleu / num_batches


def evaluate_batches(model, test_batches, dictionary, max_sent_length=None):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    test_loss = 0
    num_batches = len(test_batches)
    for batch_no in range(num_batches):
        if max_sent_length:
            test_sentences1, test_sentences2, max_sent_length_ = helper.stories_to_tensors(test_batches[batch_no],
                                                                                           dictionary,
                                                                                           max_sent_length)
        else:
            test_sentences1, test_sentences2, max_sent_length_ = helper.stories_to_tensors(test_batches[batch_no],
                                                                                           dictionary)
        assert test_sentences1.size() == test_sentences2.size()
        if model.config.cuda:
            test_sentences1 = test_sentences1.cuda()
            test_sentences2 = test_sentences2.cuda()

        loss = model(test_sentences1, test_sentences2)
        if loss.size(0) > 1:
            loss = torch.mean(loss)
        test_loss += loss.data[0] / max_sent_length_

    return test_loss / num_batches


if __name__ == "__main__":
    # Load the saved pre-trained model
    dictionary = helper.load_object(args.save_path + 'dictionary.p')
    embeddings_index = helper.load_word_embeddings('../data/glove/', 'glove.6B.300d.vist.txt')
    model = Sequence2Sequence(dictionary, embeddings_index, 85, args)
    print(model)
    if args.cuda:
        # model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()
    helper.load_model_states_without_dataparallel(model, 'model_loss_2.256082_epoch_7_model.pt')
    print('Model, embedding index and dictionary loaded.')

    test_corpus = data.Corpus(args.data + '/sis/', 'test.story-in-sequence.json', dictionary, is_test_corpus=True)
    print('Test set size = ', len(test_corpus.data))
    test_batches = helper.batchify(test_corpus.data, args.batch_size)
    print('Number of test batches = ', len(test_batches))

    test_loss = evaluate_batches(model, test_batches, dictionary)
    print('Test loss = ', test_loss)
