from collections import Counter
from itertools import chain

from docopt import docopt
import os

def bulid_vocab(file_path, type, lan, freq_cutoff=2):
    corpus = read_corpus(file_path, type)

    word_freq = Counter(chain(*corpus))
    valid_words = [w+"\n" for w, v in word_freq.items() if v >= freq_cutoff]
    print(valid_words[0])
    print(valid_words[1])

    with open('data/vocab_{}.txt'.format(lan), 'w') as f:
        f.writelines(valid_words)


def read_corpus(file_path, source):
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def concatenate_files(files, output_file):
    data = []
    for file in files:
        # print(file)
        with open('data/' + file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # print(len(lines))
            data.extend(lines)

    # print(len(data))
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(data)

def check_vocab(lan):
    with open('data/vocab_{}.txt'.format(lan), 'r') as f:
        lines = f.readlines()
        len_vocab = len(lines)
        print(len_vocab)
    with open('word_emb/{}_embed.txt'.format(lan), 'r') as f:
        lines = f.readlines()
        len_embed = len(lines)
        print(len_embed)
    assert len_vocab == len_embed


if __name__ == '__main__':
    lan = 'en'
    # files = [f for f in os.listdir('data/') if f.find('.' + lan + '.txt') != -1]
    #
    # output_file = 'data/' + lan + '.txt'
    # concatenate_files(files, output_file)
    #
    # bulid_vocab(output_file, 'tgt', lan)

    check_vocab(lan)