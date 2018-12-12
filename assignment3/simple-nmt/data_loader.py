import os
from torchtext import data, datasets

PAD, BOS, EOS = 1, 2, 3


class DataLoader():

    def __init__(self,
                 train_fn=None,
                 valid_fn=None,
                 valid_nli_fn=None,
                 exts=None,
                 batch_size=64,
                 device='cpu',
                 max_vocab=99999999,
                 max_length=255,
                 fix_length=None,
                 use_bos=True,
                 use_eos=True,
                 shuffle=True,
                 is_pretrain=False
                 ):

        super(DataLoader, self).__init__()

        self.src = data.Field(sequential=True,
                              use_vocab=True,
                              batch_first=True,
                              include_lengths=True,
                              fix_length=fix_length,
                              init_token=None,
                              eos_token=None
                              )

        self.tgt = data.Field(sequential=True,
                              use_vocab=True,
                              batch_first=True,
                              include_lengths=True,
                              fix_length=fix_length,
                              init_token='<BOS>' if use_bos else None,
                              eos_token='<EOS>' if use_eos else None
                              )
        
        self.premise = data.Field(sequential=True,
                              use_vocab=True,
                              batch_first=True,
                              include_lengths=False,
                              fix_length=fix_length,
                              init_token=None,
                              eos_token=None
                              )

        self.hypothesis = data.Field(sequential=True,
                              use_vocab=True,
                              batch_first=True,
                              include_lengths=False,
                              fix_length=fix_length,
                              init_token=None,
                              eos_token=None
                              )

        self.isSrcPremise = data.LabelField()
        self.labels = data.LabelField()

        if train_fn is not None and valid_fn is not None and exts is not None:
            # if is_pretrain:
            #     train = TranslationDataset(path=train_fn,
            #                             exts=exts[:2],
            #                             fields=[('src', self.src),
            #                                     ('tgt', self.tgt),
            #                                     ],
            #                             max_length=max_length
            #                             )
            # else:
            train = TranslationNLIDataset(path=train_fn,
                                    exts=exts,
                                    fields=[('src', self.src),
                                            ('tgt', self.tgt),
                                            ('premise', self.premise),
                                            ('hypothesis', self.hypothesis),
                                            ('isSrcPremise', self.isSrcPremise),
                                            ('labels', self.labels),
                                            ],
                                    max_length=max_length
                                    )
            valid = TranslationDataset(path=valid_fn,
                                       exts=exts[:2],
                                       fields=[('src', self.src),
                                               ('tgt', self.tgt)
                                               ],
                                       max_length=max_length
                                       )

            self.train_iter = data.BucketIterator(train,
                                                  batch_size=batch_size,
                                                  device='cuda:%d' % device if device >= 0 else 'cpu',
                                                  shuffle=shuffle,
                                                  sort_key=lambda x: len(x.tgt) + (max_length * len(x.src)),
                                                  sort_within_batch=True
                                                  )
            self.valid_iter = data.BucketIterator(valid,
                                                  batch_size=batch_size,
                                                  device='cuda:%d' % device if device >= 0 else 'cpu',
                                                  shuffle=False,
                                                  sort_key=lambda x: len(x.tgt) + (max_length * len(x.src)),
                                                  sort_within_batch=True
                                                  )

            if valid_nli_fn is not None:
                valid_nli = TranslationNLIDataset(path=valid_nli_fn,
                                        exts=exts,
                                        fields=[('src', self.src),
                                                ('tgt', self.tgt),
                                                ('premise', self.premise),
                                                ('hypothesis', self.hypothesis),
                                                ('isSrcPremise', self.isSrcPremise),
                                                ('labels', self.labels),
                                                ],
                                        max_length=max_length
                                        )

                self.valid_nli_iter = data.BucketIterator(valid_nli,
                                                    batch_size=batch_size,
                                                    device='cuda:%d' % device if device >= 0 else 'cpu',
                                                    shuffle=False,
                                                    sort_key=lambda x: len(x.tgt) + (max_length * len(x.src)),
                                                    sort_within_batch=True
                                                    )

            self.src.build_vocab(train, max_size=max_vocab)
            self.tgt.build_vocab(train, max_size=max_vocab)
            if not is_pretrain or valid_nli_fn is not None:
                # self.premise.vocab = self.tgt.vocab
                # self.hypothesis.vocab = self.tgt.vocab
                self.isSrcPremise.build_vocab(train)
                # self.labels.build_vocab(train)

    def load_vocab(self, src_vocab, tgt_vocab):
        self.src.vocab = src_vocab
        self.tgt.vocab = tgt_vocab
        self.premise.vocab = self.tgt.vocab
        self.hypothesis.vocab = self.tgt.vocab
        
    def load_target_vocab(self, tgt_bocab):
        self.tgt.vocab = tgt_bocab

    def load_label_vocab(self, label_bocab):
        self.labels.vocab = label_bocab

class TranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, exts, fields, max_length=None, **kwargs):
        """Create a TranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        if not path.endswith('.'):
            path += '.'

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
        with open(src_path) as src_file, open(trg_path) as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if max_length and max_length < max(len(src_line.split()),
                                                   len(trg_line.split())
                                                   ):
                    continue
                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line], fields))

        super(TranslationDataset, self).__init__(examples, fields, **kwargs)

class TranslationNLIDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, exts, fields, max_length=None, **kwargs):
        """Create a TranslationNLIDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ('src', fields[0]),
                ('trg', fields[1]), 
                ('premise', fields[2]),
                ('hypothesis', fields[3]),
                ('isSrcPremise', fields[4]),
                ('labels', fields[5])
            ]

        if not path.endswith('.'):
            path += '.'

        src_path, trg_path, premise_path, hypothesis_path, is_src_premise_path, label_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
        with open(src_path) as src_file, open(trg_path) as trg_file, open(premise_path) as premise_file, \
            open(hypothesis_path) as hypothesis_path, open(is_src_premise_path) as is_src_premise_path, open(label_path) as label_file:
            for src_line, trg_line, premise_line, hypothesis_line, is_src_premise_line, label_line in zip(src_file, trg_file, premise_file, hypothesis_path, is_src_premise_path, label_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if max_length and max_length < max(len(src_line.split()),
                                                   len(trg_line.split())
                                                   ):
                    continue
                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line, premise_line, hypothesis_line, is_src_premise_line, label_line], fields))

        super(TranslationNLIDataset, self).__init__(examples, fields, **kwargs)

if __name__ == '__main__':
    import sys
    loader = DataLoader(sys.argv[1],
                        sys.argv[2],
                        (sys.argv[3], sys.argv[4]),
                        # (sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8]),
                        batch_size=2,
                        device=-1,
                        is_pretrain=True
                        )

    for batch_index, batch in enumerate(loader.train_iter):
        print(batch.src)
        print(batch.tgt)
        # print(batch.premise)
        # print(batch.hypothesis)
        # print(batch.isSrcPremise)
        # print(batch.labels)

        if batch_index == 0:
            break
