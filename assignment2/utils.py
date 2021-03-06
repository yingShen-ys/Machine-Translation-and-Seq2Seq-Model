import math
from typing import List

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, orthogonal_


def lstm_init_(lstm_unit):
    '''
    LSTM initialization:

    1) initialize all biases to 0 except forget gate biases
    (since PyTorch has duplicate biases at every LSTM, each forget gate bias
    is initialized to 1/2 instead).

    2) initialized the hidden2hidden matrix by orthogonal

    3) initialized the input2hidden matrix by xavier_uniform
    '''
    if isinstance(lstm_unit, nn.LSTM):
        for l in range(lstm_unit.num_layers):
            xavier_uniform_(getattr(lstm_unit, "weight_ih_l{}".format(l)).data)
            orthogonal_(getattr(lstm_unit, "weight_hh_l{}".format(l)).data)
            xavier_uniform_(
                getattr(lstm_unit, "weight_ih_l{}{}".format(l, '_reverse')).data)
            orthogonal_(
                getattr(lstm_unit, "weight_hh_l{}{}".format(l, '_reverse')).data)
            getattr(lstm_unit, "bias_ih_l{}".format(l)).data.fill_(0)
            getattr(lstm_unit, "bias_ih_l{}".format(
                l)).data[lstm_unit.hidden_size: 2*lstm_unit.hidden_size] = 1./2
            getattr(lstm_unit, "bias_ih_l{}{}".format(l, '_reverse')).data.fill_(0)
            getattr(lstm_unit, "bias_ih_l{}{}".format(l, '_reverse')
                    ).data[lstm_unit.hidden_size: 2*lstm_unit.hidden_size] = 1./2
            getattr(lstm_unit, "bias_hh_l{}".format(l)).data.fill_(0)
            getattr(lstm_unit, "bias_hh_l{}".format(
                l)).data[lstm_unit.hidden_size: 2*lstm_unit.hidden_size] = 1./2
            getattr(lstm_unit, "bias_hh_l{}{}".format(l, '_reverse')).data.fill_(0)
            getattr(lstm_unit, "bias_hh_l{}{}".format(l, '_reverse')
                    ).data[lstm_unit.hidden_size: 2*lstm_unit.hidden_size] = 1./2
    else:
        print(f"A {type(lstm_unit)} object has been passed to lstm_init_ function instead of torch.nn.LSTM")

def lstm_cell_init_(lstm_cell):
    '''
    Initialize the LSTMCell parameters in a slightly better way
    '''
    if isinstance(lstm_cell, nn.LSTMCell):
        xavier_uniform_(lstm_cell.weight_ih.data)
        orthogonal_(lstm_cell.weight_hh.data)
        lstm_cell.bias_ih.data.fill_(0)
        lstm_cell.bias_hh.data.fill_(0)
        lstm_cell.bias_ih.data[lstm_cell.hidden_size:2*lstm_cell.hidden_size] = 1./2
        lstm_cell.bias_hh.data[lstm_cell.hidden_size:2*lstm_cell.hidden_size] = 1./2
    else:
        print(f"A {type(lstm_cell)} object has been passed to lstm_cell_init_ function instead of torch.nn.LSTMCell")

class LabelSmoothedCrossEntropy(nn.Module):
    '''
    Args:
        - smoothing_coeff: the smoothing coefficient between target dist and uniform

    Input:
        - pred: (N, C, *)
        - target: (N, * )
    '''
    def __init__(self, smoothing_coeff):
        super(LabelSmoothedCrossEntropy, self).__init__()
        self.smoothing_coeff = smoothing_coeff
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, target):
        crossent_with_target = self.ce(pred, target)
        crossent_with_uniform = - F.log_softmax(pred, dim=-1).sum(1) / pred.size(1)
        loss = crossent_with_target * self.smoothing_coeff + crossent_with_uniform * (1 - self.smoothing_coeff)
        return loss

class LabelSmoothedNLL(nn.Module):
    '''
    Args:
        - smoothing_coeff: the smoothing coefficient between target dist and uniform

    Input:
        - pred: (N, C, *)
        - target: (N, * )
    '''
    def __init__(self, smoothing_coeff):
        super(LabelSmoothedNLL, self).__init__()
        self.smoothing_coeff = smoothing_coeff
        self.nll = nn.NLLLoss(reduction='none')

    def forward(self, logprobs, target):
        crossent_with_target = self.nll(logprobs, target)
        crossent_with_uniform = - logprobs.sum(1) / logprobs.size(1)
        loss = crossent_with_target * self.smoothing_coeff + crossent_with_uniform * (1 - self.smoothing_coeff)
        return loss

def input_transpose(sents, pad_token):
    """
    This function transforms a list of sentences of shape (batch_size, token_num) into 
    a list of shape (token_num, batch_size). You may find this function useful if you
    use pytorch
    """
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])

    return sents_t


def read_corpus(file_path, source):
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """
    Given a list of examples, shuffle and slice them into mini-batches
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents
