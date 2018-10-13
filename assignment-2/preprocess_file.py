#!/usr/bin/env python
"""
Preprocess the source and target file.
Right now it can only concatenate the source/auxiliary files and output one single file.

Usage:
    preprocess_file.py --file1=<file> --file2=<file> --output-file=<file>

Options:
    -h --help                  Show this screen.
    --file1=<file>             The first input file.
    --file2=<file>             The second input file.
    --output-file=<file>       The output file.
"""

import os
from docopt import docopt

def concatenate_files(file1, file2, output_file):
    data = []
    for line in open(file1):
        sent = ' '.join(line.strip().split(' '))
        data.append(sent)
    
    for line in open(file2):
        sent = ' '.join(line.strip().split(' '))
        data.append(sent)

    with open(output_file, 'w') as f:
        for line in data:
            f.write(line + '\n')

if __name__ == '__main__':
    args = docopt(__doc__)

    print('read in file 1: %s' % args['--file1'])
    print('read in file 2: %s' % args['--file2'])
    print('read in output file: %s' % args['--output-file'])

    concatenate_files(args['--file1'], args['--file2'], args['--output-file'])
