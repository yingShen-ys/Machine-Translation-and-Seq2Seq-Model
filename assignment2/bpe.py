# coding=utf-8

"""
Python wrapper for sentencepiece BPE toolkit

Usage:
    bpe.py train --input=<file> [options]
    bpe.py encode --model=<file> [options]
    bpe.py decode --model=<file> [options]

Options:
    -h --help                                   show this screen.
    --input=<file>                              path to input training data file
    --model-prefix=<str>                        the prefix of the output model [default: sp]
    --model=<file>                              the model used for encoding and decoding
    --vocab-size=<int>                          vocab size of the trained model [default: 40000]
    --character-coverage=<float>                character coverage for encodings [default: 1.0]
    --model-type=<str>                          model type: uni-gram, bpe, char or word [default: bpe]
"""

from docopt import docopt
import sentencepiece as spm
import sys

def train(args):
    '''
    Train a BPE model on a given corpus
    '''
    spm.SentencePieceTrainer.Train(f"--input={args['--input']} --model_prefix={args['--model-prefix']} --vocab_size={args['--vocab-size']} --character_coverage={args['--character-coverage']} --model_type={args['--model-type']}")


def encode(args):
    '''
    Encode raw text into sentence pieces
    '''
    sp = spm.SentencePieceProcessor()
    sp.Load(args['--model'])
    for line in sys.stdin:
        print(' '.join(sp.EncodeAsPieces(line.strip())))


def decode(args):
    '''
    Decode BPEs into raw text
    '''
    sp = spm.SentencePieceProcessor()
    sp.Load(args['--model'])
    for line in sys.stdin:
        print(sp.DecodePieces(line.strip().split()))

if __name__ == "__main__":
    args = docopt(__doc__)
    if args['train']:
        train(args)
    elif args['encode']:
        encode(args)
    elif args['decode']:
        decode(args)
