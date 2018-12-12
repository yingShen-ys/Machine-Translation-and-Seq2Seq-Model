import argparse

import torch
from torch import nn
from torch.autograd import Variable

from model.BIMPM import BIMPM
from model.utils import SNLI, Quora
import pdb
import numpy as np

def test(model, args, data, mode='test', error_indice_suffix=''):
    if mode == 'dev':
        iterator = iter(data.dev_iter)
    else:
        iterator = iter(data.test_iter)

    criterion = nn.CrossEntropyLoss()
    model.eval()
    acc, loss, size = 0, 0, 0
    num_correct_per_label = [0 for _ in range(len(data.LABEL.vocab.itos))]
    num_per_label = [0 for _ in range(len(data.LABEL.vocab.itos))]

    genre = 'genre'
    num_correct_per_genre = [0 for _ in range(len(data.GENRE.vocab.itos))]
    num_per_genre = [0 for _ in range(len(data.GENRE.vocab.itos))]
    error_indice = []
    predicted_label = []

    for batch_idx, batch in enumerate(iterator):
        if args.data_type == 'SNLI':
            s1, s2 = 'premise', 'hypothesis'
        else:
            s1, s2 = 'q1', 'q2'

        genre_label = getattr(batch, genre)
        s1, s2 = getattr(batch, s1), getattr(batch, s2)
        kwargs = {'p': s1, 'h': s2}

        if args.use_char_emb:
            char_p = Variable(torch.LongTensor(data.characterize(s1)))
            char_h = Variable(torch.LongTensor(data.characterize(s2)))

            if args.gpu > -1:
                char_p = char_p.cuda(args.gpu)
                char_h = char_h.cuda(args.gpu)

            kwargs['char_p'] = char_p
            kwargs['char_h'] = char_h

        pred = model(**kwargs)

        batch_loss = criterion(pred, batch.label)
        loss += batch_loss.item()

        _, pred = pred.max(dim=1)
        # print(len(batch))
        # for i in range(len(batch)):
        #     if pred[i] != batch.label[i]:
        #         print(i)
        #         premise = []
        #         hypothesis = []
        #         for word_index in s1[i]:
        #             premise += [data.TEXT.vocab.itos[word_index]]
        #         for word_index in s2[i]:
        #             hypothesis += [data.TEXT.vocab.itos[word_index]]
        #         print(' '.join(premise))
        #         print(' '.join(hypothesis))
        #         print(data.LABEL.vocab.itos[pred[i]], data.LABEL.vocab.itos[batch.label[i]])
        #         pdb.set_trace()
        acc += (pred == batch.label).sum().float()
        size += len(pred)
        predicted_label += pred.tolist()
        for i in range(len(batch)):
            genre_index = genre_label[i]
            num_per_genre[genre_index] += 1

            label_index = batch.label[i]
            num_per_label[label_index] += 1
            if pred[i] == batch.label[i]:
                num_correct_per_genre[genre_index] += 1
                num_correct_per_label[label_index] += 1
            else:
                error_indice.append(batch_idx * len(batch) + i)

    acc /= size
    acc = acc.cpu().item()
    print(data.LABEL.vocab.itos)
    print(np.array(num_correct_per_label)/np.array(num_per_label))

    print(data.GENRE.vocab.itos)
    print(np.array(num_correct_per_genre)/np.array(num_per_genre))
    with open('error_indices_' + error_indice_suffix + '.txt', 'w') as f:
        for i in error_indice:
            f.write(str(i) + '\n')
    
    # with open('predicted_label.txt', 'w') as f:
    #     for i in predicted_label:
    #         f.write(data.LABEL.vocab.itos[i] + '\n')

    return loss, acc

def load_model(args, data, saved_state_dict):
    model = BIMPM(args, data)
    model.load_state_dict(saved_state_dict)

    if args.gpu > -1:
        model.cuda(args.gpu)

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--suffix')

    args = parser.parse_args()
    if args.gpu > -1:
        saved_data = torch.load(args.model_path)
    else:
        saved_data = torch.load(args.model_path, map_location='cpu')

    saved_args = saved_data['args']
    saved_args.gpu = args.gpu

    if saved_args.data_type == 'SNLI':
        print('loading SNLI data...')
        data = SNLI(saved_args)
    elif args.data_type == 'Quora':
        print('loading Quora data...')
        data = Quora(saved_args)

    setattr(saved_args, 'char_vocab_size', len(data.char_vocab))
    setattr(saved_args, 'word_vocab_size', len(data.TEXT.vocab))
    setattr(saved_args, 'class_size', len(data.LABEL.vocab))
    setattr(saved_args, 'max_word_len', data.max_word_len)

    print('loading model...')
    model = load_model(saved_args, data, saved_data['model'])

    _, acc = test(model, saved_args, data, error_indice_suffix=args.suffix)

    print(f'test acc: {acc:.3f}')
