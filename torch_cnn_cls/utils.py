import re
import os
import pandas as pd
import gensim.models.keyedvectors as word2vec
from nltk.tokenize import word_tokenize
import logging
import itertools
from collections import Counter
import json


def clean_str(string):
    """
    Tokenization/string cleaning
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def make_ds_csv(dataset_path):
    for batch in ['train', 'test']:
        batch_dfs = []
        for root, dirs, files in os.walk(f'{dataset_path}/{batch}'):
            for file in files:
                if '.txt' in file:
                    df = pd.read_csv(os.path.join(root, file), sep='\t', names=['text', 'label'])
                    df['ep'] = df['label'].apply(lambda x: 1 if x == 'ep' else 0)
                    df['de'] = df['label'].apply(lambda x: 1 if x == 'de' else 0)
                    df['dy'] = df['label'].apply(lambda x: 1 if x == 'dy' else 0)
                    df = df.drop(columns='label')
                    batch_dfs.append(df)
        batch_df = pd.concat(batch_dfs, ignore_index=True)
        if batch == 'train':
            shuffled = batch_df.sample(frac=1)
            splits = np.array_split(shuffled, 10)
            pd.concat(splits[0:8]).to_csv(f'{dataset_path}/train_set.csv', sep='\t')
            pd.concat(splits[0:2]).to_csv(f'{dataset_path}/val_set.csv', sep='\t')
        else:
            batch_df.to_csv(f'{dataset_path}/test_set.csv', sep='\t')


def convert_embed_to_vec(bin_path, vec_path):
    model = word2vec.KeyedVectors.load_word2vec_format(bin_path, binary=True)
    model.save_word2vec_format(vec_path, binary=False)

def export_embedding_file(embedding, vocabulary, output_path):
    with open(output_path + '.vec', 'w') as vecfile:
        for vec, word in zip(embedding, vocabulary.keys()):
            vecfile.write(f'{word} {" ".join([x for x in vec])}\n')

def dump_vocab_for_entire_set(dataset_path, vocab_fp):
    sentences_all_folds = []
    max_length = -1

    for batch in ['train', 'test']:
        for root, dirs, files in os.walk(f'{dataset_path}/{batch}'):
            for file in files:
                if '.txt' in file:
                    path = os.path.join(root, file)
                    with open(path, "r") as f:
                        batch_sentences = f.readlines()
                        sentences_token = [word_tokenize(clean_str(s)) for s in batch_sentences]
                        sentences_all_folds.extend(sentences_token)
                        max_length = max(max_length, max(len(s) for s in sentences_token))

    logging.info("Building vocabulary...")
    word_counts = dict(Counter(itertools.chain(*sentences_all_folds)).most_common())
    word_counts_list = zip(word_counts.keys(), word_counts.values())

    PAD = "<PAD>"
    UNK = "<UNK>"
    vocabulary_inv = [x[0] for x in word_counts_list]
    vocabulary_inv.append(PAD)
    vocabulary_inv.append(UNK)

    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    with open(vocab_fp, 'w') as vf:
        json.dump({'vocabulary':vocabulary}, vf, indent=4)

    return vocabulary, max_length


if __name__ == "__main__":
    bin_path = '/home/shoval/repos/openU/modality/embeddings/GoogleNews-vectors-negative300.bin'
    vec_path = '/home/shoval/repos/openU/modality/embeddings/GoogleNews-vectors-negative300.vec'
    dataset_path = '../data/EPOS_E'
    vocab_filepath = '../vocabs/dataset_vocab.json'
    make_ds_csv(dataset_path=dataset_path)
    dump_vocab_for_entire_set(dataset_path=dataset_path, vocab_fp=vocab_filepath)
    convert_embed_to_vec(bin_path, vec_path)