# coding=utf-8

"""
Basic seq2seq model with LSTMs and attention
"""
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, orthogonal_
from collections import namedtuple
from utils import batch_iter


Hypothesis = namedtuple('Hypothesis', ['value', 'score'])
START_TOKEN_IDX = 1
END_TOKEN_IDX = 2

def pad(idx):
    UNK_IDX = 0 # this is built-in into the vocab.py
    max_len = max(map(len, idx))
    for sent in idx:
        sent += [0] * (max_len - len(sent))
    return idx

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
    lstm_cell.bias_ih.data[lstm_cell.hidden_size:2*lstm_cell.hidden_size] = 1./2
    lstm_cell.bias_hh.data[lstm_cell.hidden_size:2*lstm_cell.hidden_size] = 1./2


class LSTM(nn.Module):
    '''
    An LSTM with recurrent dropout.
    Refer to "A Theoretically Grounded Applicaiton of Dropout in RNN" Gal et al. for details.
    Currently it is fairly slow. May be a good place to start exercising with CUPY for writing
    custom kernels though.

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
            h_dropout_mask = dist.Bernoulli(probs=(1-self.rdrop) * x.new_ones(batch_size, self.hidden_size)).sample()
            x_dropout_mask = dist.Bernoulli(probs=(1-self.rdrop) * x.new_ones(batch_size, self.input_size)).sample()

        if self.rdrop and self.training:
            x_tilde = x[:, 0, :] * x_dropout_mask / self.rdrop
        else:
            x_tilde = x[:, 0, :]
        hc = lstm(x_tilde, hc) # first time step

        H = [hc[0]]
        C = [hc[1]]
        for t in range(1, seq_len):
            if self.rdrop and self.training:
                h_tilde = hc[0] * h_dropout_mask / self.rdrop
                x_tilde = x[:, t, :] * x_dropout_mask / self.rdrop
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
         - vocab: the vocab file from vocab.py
         - embedding_size: the size of the embedding
         - hidden_size: the size of the LSTM states, applies to encoder
         - bidirectional: whether or not the encoder is bidirectional
         - rdrop: whether there is a recurrent dropout on the encoder side
    '''

    def __init__(self, embedding_size, hidden_size, vocab, bidirectional=True, dropout_rate=0.3):
        super(LSTMSeq2seq, self).__init__()
        self.vocab = vocab
        self.src_vocab_size = len(vocab.src)
        self.trg_vocab_size = len(vocab.tgt)
        self.src_embedding = nn.Embedding(self.src_vocab_size, embedding_size)
        self.trg_embedding = nn.Embedding(self.trg_vocab_size, embedding_size)
        self.encoder_lstm = LSTM(embedding_size, hidden_size, rdrop=dropout_rate, bidirectional=bidirectional)
        self.decoder_lstm_cell = nn.LSTMCell(embedding_size, hidden_size * 2 if bidirectional else hidden_size)
        self.decoder_output_layer = nn.Linear(hidden_size * 4 if bidirectional else hidden_size * 2, self.trg_vocab_size)

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.rdrop = dropout_rate

    def forward(self, src_tokens, src_lens, trg_tokens, trg_lens):
        src_states, final_states = self.encode(src_tokens, src_lens)
        ll = self.decode(src_states, final_states, src_lens, trg_tokens, trg_lens)
        return ll

    def encode(self, src_tokens, src_lens):
        '''
        Encode source sentences into vector representations.

        Args:
             - src_tokens: a torch tensor of a batch of tokens, with shape (batch_size, max_seq_len) >> LongTensor
             - src_lens: a torch tensor of the sentence lengths in the batch, with shape (batch_size,) >> LongTensor
        '''
        src_vectors = self.src_embedding(src_tokens) # (batch_size, max_seq_len, embedding_size)
        src_states, _ = self.encoder_lstm(src_vectors) # both (batch_size, max_seq_len, hidden_size (*2))

        # need to use src_lens to pick out the actual last states of each sequence
        batch_idx = torch.arange(0, src_states.size(0), out = src_states.new(0)).long()
        final_states = src_states[batch_idx, src_lens-1, :] # (batch_size, hidden_size (*2))
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
        
        nll = []

        # dealing with the start token
        h = final_states # (batch_size, hidden_size (*2))
        c = h.new_zeros(h.size(0), h.size(1), requires_grad=False)
        start_token = trg_tokens[..., 0] # (batch_size,)
        vector = self.trg_embedding(start_token) # (batch_size, embedding_size)
        h, c = self.decoder_lstm_cell(vector, (h, c))
        context_vector = LSTMSeq2seq.compute_attention(h, src_states, src_lens, attn_func=dot_attn) # (batch_size, hidden_size (*2))
        curr_logits = self.decoder_output_layer(torch.cat((h, context_vector), dim=-1)) # (batch_size, vocab_size)
        neg_log_likelihoods = F.cross_entropy(curr_logits, trg_tokens[..., 1], reduction='none') # (batch_size,)
        nll.append(neg_log_likelihoods)
        _, prd_token = torch.max(curr_logits, dim=-1) # (batch_size,) the decoded tokens
        if teacher_forcing:
            prd_token = trg_tokens[..., 1] # feed the gold sequence token to the next time step

        # input(trg_tokens.shape)
        # TODO: check indexing
        for t in range(trg_tokens.size(-1)-2):
            token = trg_tokens[:, t+1]
            vector = self.trg_embedding(token)
            h, c = self.decoder_lstm_cell(vector, (h, c))
            context_vector = LSTMSeq2seq.compute_attention(h, src_states, src_lens, attn_func=dot_attn)
            curr_logits = self.decoder_output_layer(torch.cat((h, context_vector), dim=-1))
            neg_log_likelihoods = F.cross_entropy(curr_logits, trg_tokens[..., t+2], reduction='none') # (batch_size,)
            nll.append(neg_log_likelihoods)
            _, prd_token = torch.max(curr_logits, dim=-1)
            if teacher_forcing:
                prd_token = trg_tokens[..., t+2]
        
        # computing the masked log-likelihood
        # trg_logits = torch.stack(trg_logits, dim=-1) # (batch_size, max_seq_len, vocab_size)
        # neg_log_likelihoods = F.cross_entropy(trg_logits, trg_tokens[:, 1:], reduction='none') # (batch_size, max_seq_len-1) exclude <s> symbol
        nll = torch.stack(nll, dim=1)

        # create masks, assuming padding is AT THE END
        idx = torch.arange(0, trg_tokens.size(1), out=trg_tokens.new(1).long()).unsqueeze(0)
        mask = (idx < trg_lens.unsqueeze(1)).float() # make use of the automatic expansion in comparison
        masked_log_likelihoods = - nll * mask[:, 1:] # exclude <s> token

        return torch.sum(masked_log_likelihoods) # seems the training code assumes the log-likelihoods are summed per word

    def beam_search(self, src_sent, src_lens, beam_size, max_decoding_time_step):
        '''
        Performs beam search decoding for testing the model. Currently just a fake method and only uses argmax decoding.
        '''
        self.training = False # turn of training
        decoded_idx = []
        scores = 0

        src_states, final_state = self.encode(src_sent, src_lens)
        h = final_state
        c = h.new_zeros(h.size(0), h.size(1), requires_grad=False)
        start_token = src_sent.new_ones((1,)).long() * START_TOKEN_IDX # (batch_size,) should be </s>
        vector = self.trg_embedding(start_token) # (batch_size, embedding_size)
        h, c = self.decoder_lstm_cell(vector, (h, c))
        context_vector = LSTMSeq2seq.compute_attention(h, src_states, src_lens, attn_func=dot_attn) # (batch_size, hidden_size (*2))
        curr_logits = self.decoder_output_layer(torch.cat((h, context_vector), dim=-1)) # (batch_size, vocab_size)
        curr_ll = F.log_softmax(curr_logits, dim=-1) # transform logits into log-likelihoods
        curr_score, prd_token = torch.max(curr_ll, dim=-1) # (batch_size,) the decoded tokens
        decoded_idx.append(prd_token.item())
        scores += curr_score.item()
        # input(decoded_idx)

        decoding_step = 1
        while decoding_step <= max_decoding_time_step and prd_token.item() != END_TOKEN_IDX:
            decoding_step += 1
            vector = self.trg_embedding(prd_token)
            h, c = self.decoder_lstm_cell(vector, (h, c))
            context_vector = LSTMSeq2seq.compute_attention(h, src_states, src_lens, attn_func=dot_attn)
            curr_logits = self.decoder_output_layer(torch.cat((h, context_vector), dim=-1))
            curr_ll = F.log_softmax(curr_logits, dim=-1) # transform logits into log-likelihoods
            curr_score, prd_token = torch.max(curr_ll, dim=-1)
            decoded_idx.append(prd_token.item())
            scores += curr_score.item()
            # input(decoded_idx)

        sentence = list(map(lambda x: self.vocab.tgt.id2word[x], decoded_idx))
        if prd_token.item() == END_TOKEN_IDX:
            sentence = sentence[:-1] # remove the </s> token in final output
        greedy_hyp = Hypothesis(sentence, scores)
        self.training = True # turn training back on
        return [greedy_hyp] * beam_size

    def evaluate_ppl(self, dev_data, batch_size, cuda=True):
        """
        Evaluate perplexity on dev sentences

        Args:
            dev_data: a list of dev sentences
            batch_size: batch size
        
        Returns:
            ppl: the perplexity on dev sentences
        """
        self.training = False
        cum_loss = 0.
        cum_tgt_words = 0.

        # you may want to wrap the following code using a context manager provided
        # by the NN library to signal the backend to not to keep gradient information
        # e.g., `torch.no_grad()`

        if cuda:
            torch.LongTensor = torch.cuda.LongTensor

        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            src_lens = torch.LongTensor(list(map(len, src_sents)))
            trg_lens = torch.LongTensor(list(map(len, tgt_sents)))
            
            # these padding functions modify data in-place
            src_sents = pad(self.vocab.src.words2indices(src_sents))
            tgt_sents = pad(self.vocab.tgt.words2indices(tgt_sents))

            src_sents = torch.LongTensor(src_sents)
            tgt_sents = torch.LongTensor(tgt_sents)
            loss = -self.forward(src_sents, src_lens, tgt_sents, trg_lens).sum()

            loss = loss.item()
            cum_loss += loss
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting the leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)
        self.training = True
        return ppl

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

    @staticmethod
    def load(model_path):
        """
        Load a pre-trained model

        Returns:
            model: the loaded model
        """
        model = torch.load(model_path)
        return model

class OLSTMSeq2seq(nn.Module):
    '''
    An LSTM based seq2seq model with language as input. LSTM is based on original PyTorch implementation.

    Args:
         - vocab: the vocab file from vocab.py
         - embedding_size: the size of the embedding
         - hidden_size: the size of the LSTM states, applies to encoder
         - bidirectional: whether or not the encoder is bidirectional
         - rdrop: whether there is a recurrent dropout on the encoder side
    '''

    def __init__(self, embedding_size, hidden_size, vocab, bidirectional=True, dropout_rate=0.3):
        super(OLSTMSeq2seq, self).__init__()
        self.vocab = vocab
        self.src_vocab_size = len(vocab.src)
        self.trg_vocab_size = len(vocab.tgt)
        self.src_embedding = nn.Embedding(self.src_vocab_size, embedding_size)
        self.trg_embedding = nn.Embedding(self.trg_vocab_size, embedding_size)
        self.encoder_lstm = nn.LSTM(embedding_size, hidden_size, dropout=dropout_rate, bidirectional=bidirectional, num_layers=1, batch_first=True)
        self.decoder_lstm_cell = nn.LSTMCell(embedding_size, hidden_size * 2 if bidirectional else hidden_size)
        self.decoder_output_layer = nn.Linear(hidden_size * 4 if bidirectional else hidden_size * 2, self.trg_vocab_size)

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.rdrop = dropout_rate

    def forward(self, src_tokens, src_lens, trg_tokens, trg_lens):
        src_states, final_states = self.encode(src_tokens, src_lens)
        ll = self.decode(src_states, final_states, src_lens, trg_tokens, trg_lens)
        return ll

    def encode(self, src_tokens, src_lens):
        '''
        Encode source sentences into vector representations.

        Args:
             - src_tokens: a torch tensor of a batch of tokens, with shape (batch_size, max_seq_len) >> LongTensor
             - src_lens: a torch tensor of the sentence lengths in the batch, with shape (batch_size,) >> LongTensor
        '''
        src_vectors = self.src_embedding(src_tokens) # (batch_size, max_seq_len, embedding_size)
        src_states, _ = self.encoder_lstm(src_vectors) # both (batch_size, max_seq_len, hidden_size (*2))

        # need to use src_lens to pick out the actual last states of each sequence
        batch_idx = torch.arange(0, src_states.size(0), out = src_states.new(0)).long()
        final_states = src_states[batch_idx, src_lens-1, :] # (batch_size, hidden_size (*2))
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
        
        nll = []

        # dealing with the start token
        h = final_states # (batch_size, hidden_size (*2))
        c = h.new_zeros(h.size(0), h.size(1), requires_grad=False)
        start_token = trg_tokens[..., 0] # (batch_size,)
        vector = self.trg_embedding(start_token) # (batch_size, embedding_size)
        h, c = self.decoder_lstm_cell(vector, (h, c))
        context_vector = LSTMSeq2seq.compute_attention(h, src_states, src_lens, attn_func=dot_attn) # (batch_size, hidden_size (*2))
        curr_logits = self.decoder_output_layer(torch.cat((h, context_vector), dim=-1)) # (batch_size, vocab_size)
        neg_log_likelihoods = F.cross_entropy(curr_logits, trg_tokens[..., 1], reduction='none') # (batch_size,)
        nll.append(neg_log_likelihoods)
        _, prd_token = torch.max(curr_logits, dim=-1) # (batch_size,) the decoded tokens
        if teacher_forcing:
            prd_token = trg_tokens[..., 1] # feed the gold sequence token to the next time step

        # input(trg_tokens.shape)
        # TODO: check indexing
        for t in range(trg_tokens.size(-1)-2):
            token = trg_tokens[:, t+1]
            vector = self.trg_embedding(token)
            h, c = self.decoder_lstm_cell(vector, (h, c))
            context_vector = LSTMSeq2seq.compute_attention(h, src_states, src_lens, attn_func=dot_attn)
            curr_logits = self.decoder_output_layer(torch.cat((h, context_vector), dim=-1))
            neg_log_likelihoods = F.cross_entropy(curr_logits, trg_tokens[..., t+2], reduction='none') # (batch_size,)
            nll.append(neg_log_likelihoods)
            _, prd_token = torch.max(curr_logits, dim=-1)
            if teacher_forcing:
                prd_token = trg_tokens[..., t+2]
        
        # computing the masked log-likelihood
        # trg_logits = torch.stack(trg_logits, dim=-1) # (batch_size, max_seq_len, vocab_size)
        # neg_log_likelihoods = F.cross_entropy(trg_logits, trg_tokens[:, 1:], reduction='none') # (batch_size, max_seq_len-1) exclude <s> symbol
        nll = torch.stack(nll, dim=1)

        # create masks, assuming padding is AT THE END
        idx = torch.arange(0, trg_tokens.size(1), out=trg_tokens.new(1).long()).unsqueeze(0)
        mask = (idx < trg_lens.unsqueeze(1)).float() # make use of the automatic expansion in comparison
        masked_log_likelihoods = - nll * mask[:, 1:] # exclude <s> token

        return torch.sum(masked_log_likelihoods) # seems the training code assumes the log-likelihoods are summed per word

    def beam_search(self, src_sent, src_lens, beam_size, max_decoding_time_step):
        '''
        Performs beam search decoding for testing the model. Currently just a fake method and only uses argmax decoding.
        '''
        self.training = False # turn of training
        decoded_idx = []
        scores = 0

        src_states, final_state = self.encode(src_sent, src_lens)
        h = final_state
        c = h.new_zeros(h.size(0), h.size(1), requires_grad=False)
        start_token = src_sent.new_ones((1,)).long() * START_TOKEN_IDX # (batch_size,) should be </s>
        vector = self.trg_embedding(start_token) # (batch_size, embedding_size)
        h, c = self.decoder_lstm_cell(vector, (h, c))
        context_vector = LSTMSeq2seq.compute_attention(h, src_states, src_lens, attn_func=dot_attn) # (batch_size, hidden_size (*2))
        curr_logits = self.decoder_output_layer(torch.cat((h, context_vector), dim=-1)) # (batch_size, vocab_size)
        curr_ll = F.log_softmax(curr_logits, dim=-1) # transform logits into log-likelihoods
        curr_score, prd_token = torch.max(curr_ll, dim=-1) # (batch_size,) the decoded tokens
        decoded_idx.append(prd_token.item())
        scores += curr_score.item()
        # input(decoded_idx)

        decoding_step = 1
        while decoding_step <= max_decoding_time_step and prd_token.item() != END_TOKEN_IDX:
            decoding_step += 1
            vector = self.trg_embedding(prd_token)
            h, c = self.decoder_lstm_cell(vector, (h, c))
            context_vector = LSTMSeq2seq.compute_attention(h, src_states, src_lens, attn_func=dot_attn)
            curr_logits = self.decoder_output_layer(torch.cat((h, context_vector), dim=-1))
            curr_ll = F.log_softmax(curr_logits, dim=-1) # transform logits into log-likelihoods
            curr_score, prd_token = torch.max(curr_ll, dim=-1)
            decoded_idx.append(prd_token.item())
            scores += curr_score.item()
            # input(decoded_idx)

        sentence = list(map(lambda x: self.vocab.tgt.id2word[x], decoded_idx))
        if prd_token.item() == END_TOKEN_IDX:
            sentence = sentence[:-1] # remove the </s> token in final output
        greedy_hyp = Hypothesis(sentence, scores)
        self.training = True # turn training back on
        return [greedy_hyp] * beam_size

    def evaluate_ppl(self, dev_data, batch_size, cuda=True):
        """
        Evaluate perplexity on dev sentences

        Args:
            dev_data: a list of dev sentences
            batch_size: batch size
        
        Returns:
            ppl: the perplexity on dev sentences
        """
        self.training = False
        cum_loss = 0.
        cum_tgt_words = 0.

        # you may want to wrap the following code using a context manager provided
        # by the NN library to signal the backend to not to keep gradient information
        # e.g., `torch.no_grad()`

        if cuda:
            torch.LongTensor = torch.cuda.LongTensor

        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            src_lens = torch.LongTensor(list(map(len, src_sents)))
            trg_lens = torch.LongTensor(list(map(len, tgt_sents)))
            
            # these padding functions modify data in-place
            src_sents = pad(self.vocab.src.words2indices(src_sents))
            tgt_sents = pad(self.vocab.tgt.words2indices(tgt_sents))

            src_sents = torch.LongTensor(src_sents)
            tgt_sents = torch.LongTensor(tgt_sents)
            loss = -self.forward(src_sents, src_lens, tgt_sents, trg_lens).sum()

            loss = loss.item()
            cum_loss += loss
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting the leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)
        self.training = True
        return ppl

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

    @staticmethod
    def load(model_path):
        """
        Load a pre-trained model

        Returns:
            model: the loaded model
        """
        model = torch.load(model_path)
        return model
