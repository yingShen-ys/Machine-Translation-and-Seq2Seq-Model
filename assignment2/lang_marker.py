# coding=utf-8

"""
add or remove language code marker for each token in this file

Usage:
    lang_marker.py mark --lang-code=<str> [options]
    lang_marker.py unmark [options]

Options:
    -h --help                                   show this screen.
    --lang-code=<str>                           language code for the language in this file
"""

from docopt import docopt
import sentencepiece as spm
import sys


def mark(args):
    for line in sys.stdin:
        line = line.strip().split()
        line = map(lambda x: args['--lang-code'] + '_' + x, line)
        line = ' '.join(line)
        print(line)

def unmark(args):
    for line in sys.stdin:
        line = line.strip().split()
        line = map(lambda x: x[3:], line)
        line = ' '.join(line)
        print(line)

if __name__ == "__main__":
    args = docopt(__doc__)
    if args['mark']:
        mark(args)
    elif args['unmark']:
        unmark(args)
