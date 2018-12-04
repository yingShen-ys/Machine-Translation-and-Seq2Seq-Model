import pickle

snli_vocab = pickle.load(open('BIMPM-pytorch/multinli_vocab', 'rb'))
iswlt_vocab = pickle.load(open('simple-nmt/en_vocab_xnli_new', 'rb'))

snli_vocab_set = set(snli_vocab.itos)
iswlt_vocab_set = set(iswlt_vocab.itos)
same = len(iswlt_vocab_set & snli_vocab_set)

print("multinli vocab size:", len(snli_vocab_set))
print("iswlt vocab size:", len(iswlt_vocab_set))

print("same vocab:", same)
print("same vocab percentage in multinli_vocab:", same/len(snli_vocab_set))
print("same vocab percentage in iswlt_vocab:", same/len(iswlt_vocab_set))

iswlt_vocab.extend(snli_vocab)
print(len(iswlt_vocab.itos))
pickle.dump(iswlt_vocab, open('iswlt_xnli_multinli_merged_vocab', 'wb'))
