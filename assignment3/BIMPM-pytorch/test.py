import argparse

import torch
from torch import nn
from torch.autograd import Variable

from model.BIMPM import BIMPM
from model.utils import SNLI, Quora


def test(model, args, data, mode='test'):
    if mode == 'dev':
        iterator = iter(data.dev_iter)
    else:
        iterator = iter(data.test_iter)

    criterion = nn.CrossEntropyLoss()
    model.eval()
    acc, loss, size = 0, 0, 0

    for batch in iterator:
        if args.data_type == 'SNLI':
            s1, s2 = 'premise', 'hypothesis'
        else:
            s1, s2 = 'q1', 'q2'

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
        acc += (pred == batch.label).sum().float()
        size += len(pred)

    acc /= size
    acc = acc.cpu().item()
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
    parser.add_argument('--gpu', default=1, type=int)

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

    _, acc = test(model, saved_args, data)

    print(f'test acc: {acc:.3f}')
