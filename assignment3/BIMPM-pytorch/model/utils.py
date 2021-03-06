from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
import pickle

from nltk import word_tokenize

class SNLI():
    def __init__(self, args):
        self.TEXT = data.Field(batch_first=True, tokenize=word_tokenize, lower=True)
        # self.TEXT = data.Field(batch_first=True, lower=True)
        self.LABEL = data.Field(sequential=False, unk_token=None)
        self.GENRE = data.Field(sequential=False, unk_token=None)

        # self.train, self.dev, self.test = datasets.SNLI.splits(self.TEXT, self.LABEL)
        self.train, self.dev, self.test = datasets.MultiNLI.splits(self.TEXT, self.LABEL, genre_field=self.GENRE)

        vocab = pickle.load(open('iswlt_xnli_multinli_merged_vocab', 'rb'))
        # vocab.vectors = None
        # vocab.load_vectors(GloVe(name='840B', dim=300), unk_init=None, cache=None)
        self.TEXT.vocab = vocab
        # self.LABEL.build_vocab(self.train)
        self.LABEL.vocab = pickle.load(open('multinli_label_vocab', 'rb'))
        self.GENRE.build_vocab(self.test)
        # print(self.GENRE.vocab.itos)

        self.train_iter, self.dev_iter, self.test_iter = \
            data.BucketIterator.splits((self.train, self.dev, self.test),
                                       batch_sizes=[args.batch_size] * 3,
                                       device='cuda:%d' % args.gpu if args.gpu >= 0 else 'cpu')
        
        print(self.test_iter.sort_within_batch, self.test_iter.shuffle)
        self.test_iter.sort_within_batch = False
        self.test_iter.sort = False

        self.max_word_len = max([len(w) for w in self.TEXT.vocab.itos])
        # for <pad>
        self.char_vocab = {'': 0}
        # for <unk> and <pad>
        self.characterized_words = [[0] * self.max_word_len, [0] * self.max_word_len]

        if args.use_char_emb:
            self.build_char_vocab()

    def build_char_vocab(self):
        # for normal words
        for word in self.TEXT.vocab.itos[2:]:
            chars = []
            for c in list(word):
                if c not in self.char_vocab:
                    self.char_vocab[c] = len(self.char_vocab)

                chars.append(self.char_vocab[c])

            chars.extend([0] * (self.max_word_len - len(word)))
            self.characterized_words.append(chars)

    def characterize(self, batch):
        """
        :param batch: Pytorch Variable with shape (batch, seq_len)
        :return: Pytorch Variable with shape (batch, seq_len, max_word_len)
        """
        batch = batch.data.cpu().numpy().astype(int).tolist()
        return [[self.characterized_words[w] for w in words] for words in batch]

class Quora():
    def __init__(self, args):
        self.RAW = data.RawField()
        self.TEXT = data.Field(batch_first=True)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        self.train, self.dev, self.test = data.TabularDataset.splits(
            path='.data/quora',
            train='train.tsv',
            validation='dev.tsv',
            test='test.tsv',
            format='tsv',
            fields=[('label', self.LABEL),
                    ('q1', self.TEXT),
                    ('q2', self.TEXT),
                    ('id', self.RAW)])

        self.TEXT.build_vocab(self.train, self.dev, self.test, vectors=GloVe(name='840B', dim=300))
        self.LABEL.build_vocab(self.train)

        sort_key = lambda x: data.interleave_keys(len(x.q1), len(x.q2))

        self.train_iter, self.dev_iter, self.test_iter = \
            data.BucketIterator.splits((self.train, self.dev, self.test),
                                       batch_sizes=[args.batch_size] * 3,
                                       device=args.gpu,
                                       sort_key=sort_key)

        self.max_word_len = max([len(w) for w in self.TEXT.vocab.itos])
        # for <pad>
        self.char_vocab = {'': 0}
        # for <unk> and <pad>
        self.characterized_words = [[0] * self.max_word_len, [0] * self.max_word_len]

        if args.use_char_emb:
            self.build_char_vocab()

    def build_char_vocab(self):
        # for normal words
        for word in self.TEXT.vocab.itos[2:]:
            chars = []
            for c in list(word):
                if c not in self.char_vocab:
                    self.char_vocab[c] = len(self.char_vocab)

                chars.append(self.char_vocab[c])

            chars.extend([0] * (self.max_word_len - len(word)))
            self.characterized_words.append(chars)

    def characterize(self, batch):
        """
        :param batch: Pytorch Variable with shape (batch, seq_len)
        :return: Pytorch Variable with shape (batch, seq_len, max_word_len)
        """
        batch = batch.data.cpu().numpy().astype(int).tolist()
        return [[self.characterized_words[w] for w in words] for words in batch]