import math
from typing import List

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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
        loss_1 = self.ce(pred, target)
        loss_2 = - F.log_softmax(pred).sum(1) / pred.size(1)
        loss = loss_1 * self.smoothing_coeff + loss_2 * (1 - self.smoothing_coeff)
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
