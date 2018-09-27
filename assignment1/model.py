# coding=utf-8

"""
Basic seq2seq model with LSTMs and attention
"""

import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, orthogonal_


def dot_attn(a, b): # computes (batch_size, hidden_size) X (batch_size, max_seq_len, hidden_size) >> (batch_size, max_seq_len)
    return torch.einsum('bi,bji->bj', (a, b))

def lstm_cell_init_(lstm_cell):
    '''
    Initialize the LSTMCell parameters in a slightly better way
    '''
    xavier_uniform_(lstm_cell.weight_ih.data)
    orthogonal_(lstm_cell.weight_hh.data)
    lstm_cell.bias_ih.data.fill_(0)
    lstm_cell.bias_hh.data.fill_(0)
    lstm_cell.bias_ih.data[lstm_cell.hidden_size, 2*lstm_cell.hidden_size] = 1./2
    lstm_cell.bias_hh.data[lstm_cell.hidden_size, 2*lstm_cell.hidden_size] = 1./2


class LSTM(nn.Module):
    '''
    An LSTM with recurrent dropout. API is slightly different than default LSTM in PyTorch.
    Refer to "A Theoretically Grounded Applicaiton of Dropout in RNN" Gal et al. for details.
    Doesn't have multi-layer options and hence dropout between stacks of LSTMs.

    Args:
         - input_size: the size of input vectors
         - hidden_size: size of the hidden states h and c
         - rdrop: recurrent dropout rate
    '''

    def __init__(self, input_size, hidden_size, rdrop=0, bidirectional=False, bias=True):
        super(LSTM, self).__init__()
        self.LSTMCell = nn.LSTMCell(input_size, hidden_size, bias=bias)
        lstm_cell_init_(self.LSTMCell)
        if bidirectional:
            self.LSTMCell_rev = nn.LSTMCell(input_size, hidden_size, bias=bias)
            lstm_cell_init_(self.LSTMCell_rev)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rdrop = rdrop
        self.bidirectional = bidirectional
        self.bias = bias

    def lstm_traverse(self, lstm, x, hc=None):
        # collect some info from input and construct h, c and dropout_mask
        batch_size = x.size(0)
        seq_len = x.size(1)
        if self.rdrop and self.training:
            h_dropout_mask = dist.Bernoulli(probs=self.rdrop * x.new_ones(batch_size, self.hidden_size)).sample()
            x_dropout_mask = dist.Bernoulli(probs=self.rdrop * x.new_ones(batch_size, self.input_size)).sample()

        if self.rdrop and self.training: # only apply mask when it is training
            x_tilde = x[:, 0, :] * x_dropout_mask / self.rdrop # inverted dropout here: scale at traning time
        else:
            x_tilde = x[:, 0, :]
        hc = lstm(x_tilde, hc) # first time step

        H = [hc[0]]
        C = [hc[1]]
        for t in range(1, seq_len):
            if self.rdrop:
                h_tilde = hc[0] * h_dropout_mask
                x_tilde = x[:, t, :] * x_dropout_mask
            else:
                h_tilde = hc[0]
                x_tilde = x[:, t, :]
            hc = lstm(x_tilde, (h_tilde, hc[1]))
            H.append(hc[0])
            C.append(hc[1])
        H = torch.stack(H, dim=1)
        C = torch.stack(C, dim=1)
        return H, C

    def forward(self, x, hc=None):
        H, C = self.lstm_traverse(self.LSTMCell, x, hc)
        if self.bidirectional:
            rev_H, rev_C = self.lstm_traverse(self.LSTMCell_rev, x, hc)
            H = torch.cat((H, rev_H), dim=-1)
            C = torch.cat((C, rev_C), dim=-1)
        return H, C


class LSTMSeq2seq(nn.Module):
    '''
    An LSTM based seq2seq model with language as input.

    Args:
         - vocab_size: the size of the vocabulary
         - embedding_size: the size of the embedding
         - hidden_size: the size of the LSTM states, applies to encoder
         - bidirectional: whether or not the encoder is bidirectional
         - rdrop: whether there is a recurrent dropout on the encoder side
    '''

    def __init__(self, vocab_size, embedding_size, hidden_size, bidirectional=True, rdrop=0.3):
        super(LSTMSeq2seq, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.encoder_lstm = LSTM(embedding_size, hidden_size, rdrop=rdrop, bidirectional=bidirectional)
        self.decoder_lstm_cell = nn.LSTMCell(embedding_size, hidden_size * 2 if bidirectional else hidden_size)
        self.decoder_output_layer = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, vocab_size)

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.rdrop = rdrop

    def apply_embedding(self, embedding_matrix):
        '''
        Turn an numpy matrix into torch tensor and plug it into the embedding layer
        '''
        self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_matrix).float())

    def encode(self, src_tokens, src_lens):
        '''
        Encode source sentences into vector representations.

        Args:
             - src_tokens: a torch tensor of a batch of tokens, with shape (batch_size, max_seq_len) >> LongTensor
             - src_lens: a torch tensor of the sentence lengths in the batch, with shape (batch_size,) >> LongTensor
        '''
        src_vectors = self.embedding(src_tokens) # (batch_size, max_seq_len, embedding_size)
        src_states, src_memory = self.encoder_lstm(src_vectors) # both (batch_size, max_seq_len, hidden_size (*2))

        # need to use src_lens to pick out the actual last states of each sequence
        final_states = src_states[:, src_lens, :] # (batch_size, hidden_size (*2))
        return src_states, final_states
    
    def decode(self, src_states, final_states, src_lens, trg_tokens, trg_lens, teacher_forcing=True, search_method='greedy'):
        '''
        Decode with attention and custom decoding.
        
        Args:
             - src_states: the source sentence encoder states at different time steps
             - final_states: the last state of input source sentences
             - src_lens: the lengths of source sentences, helpful in computing attention
             - trg_tokens: target tokens, used for computing log-likelihood as well as teacher forcing (if toggled True)
             - trg_lens: target sentence lengths, helpful in computing the loss
             - teacher_forcing: whether or not the decoder sees the gold sequence in previous steps when decoding
             - search_method: greedy, beam_search, etc. Not yet implemented.
        '''
        if search_method != 'greedy':
            raise NotImplementedError
        
        trg_logits = []

        # dealing with the start token
        h = final_states # (batch_size, hidden_size (*2))
        c = h.new_zeros(h.size(0), h.size(1), requires_grad=False)
        start_token = trg_tokens[..., 0] # (batch_size,)
        vector = self.embedding(start_token) # (batch_size, embedding_size)
        h, c = self.decoder_lstm_cell(vector, (h, c))
        context_vector = LSTMSeq2seq.compute_attention(h, src_states, src_lens, attn_func=dot_attn) # (batch_size, hidden_size (*2))
        curr_logits = self.decoder_output_layer(torch.cat((h, context_vector), dim=-1)) # (batch_size, vocab_size)
        _, prd_token = torch.max(curr_logits, dim=-1) # (batch_size,) the decoded tokens
        trg_logits.append(curr_logits)
        if teacher_forcing:
            prd_token = trg_tokens[..., 1] # feed the gold sequence token to the next time step

        for t, token in enumerate(trg_tokens[1:]):
            vector = self.embedding(token)
            h, c = self.decoder_lstm_cell(vector, (h, c))
            context_vector = LSTMSeq2seq.compute_attention(h, src_states, src_lens, attn_func=dot_attn)
            curr_logits = self.decoder_output_layer(torch.cat(h, context_vector), dim=-1)
            _, prd_token = torch.max(curr_logits, dim=-1)
            trg_logits.append(curr_logits)
            if teacher_forcing:
                prd_token = trg_tokens[..., t+2]
        
        # computing the masked log-likelihood
        trg_logits = torch.stack(trg_logits, dim=1) # (batch_size, max_seq_len, vocab_size)
        log_likelihoods = F.cross_entropy(trg_logits, trg_tokens, reduction=None) # (batch_size, max_seq_len)

        # create masks, assuming padding is AT THE END
        idx = torch.arange(0, src_states.size(1), out=curr_state.new(1).long()).unsqueeze(0)
        mask = (idx < src_lens.unsqueeze(1)).float() # make use of the automatic expansion in comparison
        masked_log_likelihoods = log_likelihoods * mask

        return torch.mean(masked_log_likelihoods)
    
    @staticmethod
    def compute_attention(curr_state, src_states, src_lens, attn_func):
        '''
        Computes the context vector from attention.

        Args:
             - curr_state: the current decoder state
             - src_states: the source states of encoder states
             - src_lens: the lengths of the source sequences
             - attn_func: a callback function that computes unnormalized attention scores
                          attn_scores = attn_func(curr_state, src_states)
        '''
        attn_scores = attn_func(curr_state, src_states) # (batch_size, max_seq_len)

        # create masks, assuming padding is AT THE END
        idx = torch.arange(0, src_states.size(1), out=curr_state.new(1).long()).unsqueeze(0)
        mask = (idx < src_lens.unsqueeze(1)).float() # make use of the automatic expansion in comparison

        # manual softmax with masking
        offset, _ = torch.max(attn_scores, dim=1, keepdim=True) # (batch_size, 1)
        exp_scores = torch.exp(attn_scores - offset) # numerical stability (batch_size, max_seq_len)
        exp_scores = exp_scores * mask
        attn_weights = exp_scores / torch.sum(exp_scores, dim=1, keepdim=True) # (batch_size, max_seq_len)

        context_vector = torch.einsum('bij,bi->bj', (src_states, attn_weights))
        return context_vector
