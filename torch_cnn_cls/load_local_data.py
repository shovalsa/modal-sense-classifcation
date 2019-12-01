import os
from collections import Counter
import logging
import gensim.models.keyedvectors as word2vec
import numpy as np
from torchtext import data
from torchtext.data import Field, TabularDataset
from torchtext.vocab import Vectors
from torch_cnn_cls.utils import clean_str


tokenize = lambda x: x.split() # the text in source files is already tokenized
TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
LABEL = Field(sequential=False, use_vocab=False)

def create_torchtext_data_object(dataset_path):
    train_validation_datafields = [("id", None),
                     ("sentence", TEXT), ("de", LABEL),
                     ("dy", LABEL), ("ep", LABEL)]

    train, valid = TabularDataset.splits(
                   path=dataset_path, # the root directory where the data lies
                   train='train_set.csv', validation="val_set.csv",
                   format='tsv',
                   skip_header=True,
                   fields=train_validation_datafields)

    test_datafields = [("id", None),
                     ("sentence", TEXT), ("de", LABEL),
                     ("dy", LABEL), ("ep", LABEL)]
    test = TabularDataset(
               path=f"{dataset_path}test_set.csv",
               format='tsv',
               skip_header=True,
               fields=test_datafields)

    return train, valid, test





def word2vec_emb_vocab(vocabulary, dim, embed_fp, output_path):
    """
    :param vocabulary: vocabulary constructed from the train & test
    :param dim: dimension of the embeddings
    :param emb_type: glove or w2v
    :return: numpy array w/ shape [size of the vocab, dim]
    """
    PAD = "<PAD>"
    UNK = "<UNK>"

    logging.info("Loading pre-trained w2v binary file...")

    w2v_model = word2vec.KeyedVectors.load_word2vec_format(embed_fp, binary=True)

    emb_w2v = w2v_model.wv.syn0

    logging.info("Building embeddings for this dataset...")
    vocab_size = len(vocabulary)
    embeddings = np.zeros((vocab_size, dim), dtype=np.float32)

    embeddings[vocabulary[PAD],:] = np.zeros((1, dim))
    embeddings[vocabulary[UNK],:] = np.mean(emb_w2v, axis=0).reshape((1, dim))

    counter = 0
    with open(output_path + '.vec', 'w') as vecfile:
        for word in vocabulary:
            try:
                embeddings[vocabulary[word], :] = w2v_model[word].reshape((1, dim))
            except KeyError:
                counter += 1
                embeddings[vocabulary[word], :] = embeddings[vocabulary[UNK],:]

    logging.info("Number of out-of-vocab words: %s from %s" % (counter, vocab_size))

    del emb_w2v
    del w2v_model

    assert len(vocabulary) == embeddings.shape[0]

    return embeddings, vocabulary



def load_dataset(train_data, val, test, embed_fp):
    vectors = Vectors(name=embed_fp, cache='./')
    TEXT.build_vocab(train_data, val, test, vectors=vectors)
    LABEL.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))

    train_data, valid_data = train_data.split() # Further splitting of training_data to create new training_data & validation_data
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test), batch_size=32, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)

    '''Alternatively we can also use the default configurations'''
    # train_iter, test_iter = datasets.IMDB.iters(batch_size=32)

    vocab_size = len(TEXT.vocab)

    return TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter




