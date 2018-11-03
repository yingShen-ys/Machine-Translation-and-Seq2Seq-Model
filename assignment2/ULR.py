import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import batch_iter
from typing import List, Dict
from tqdm import tqdm
from model import LSTMSeq2seq, pad
from copy import deepcopy
# from nmt import load_embedding

def load_embedding(embedding_path):
    embedding_dict = dict()
    with open(embedding_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            if len(line) == 2:
                vector_size = int(line[-1])
                continue
            vector = np.array(list(map(float, line[-vector_size:])))
            word = ''.join(line[:-vector_size])
            embedding_dict[word] = vector
    return embedding_dict


class ULREmbedding(nn.Module):
    def __init__(self, vocab, embedding_size, temperature=0.05):
        super(ULREmbedding, self).__init__()
        self.vocab = vocab
        self.src_vocab_size = len(vocab.src)
        self.uni_vocab_size = len(vocab.tgt)
        self.embedding_size = embedding_size
        self.transform_matrix = nn.Parameter(torch.randn(embedding_size, embedding_size)) # A
        self.src_query_embedding = nn.Embedding(self.src_vocab_size, embedding_size) # E^Q
        self.src_mono_embedding = nn.Embedding(self.src_vocab_size, embedding_size) # E^I
        self.uni_key_embedding = nn.Embedding(self.uni_vocab_size, embedding_size) # E^K
        self.uni_val_embedding = nn.Embedding(self.uni_vocab_size, embedding_size) # E^U
        self.alpha_emb = nn.Embedding(self.src_vocab_size, 1) # could this be a neural net
        self.beta = 1.0 # and this be something related to alpha
        self.temperature = temperature

        # fix some embeddings during training
        self.uni_key_embedding.weight.requires_grad = False
        self.src_query_embedding.weight.requires_grad = False
        self.alpha_emb.weight.requires_grad = False

        # some init
        self.alpha_emb.weight.fill_(0)
        torch.nn.init.xavier_uniform_(self.uni_key_embedding.weight)
        torch.nn.init.xavier_uniform_(self.uni_val_embedding.weight)
        torch.nn.init.orthogonal_(self.transform_matrix)

    def init_alpha_table(self, top_tokens):
        print("Initializing the alpha interpolation table")
        for word in top_tokens:
            word_idx = self.vocab.src.word2id[word]
            self.alpha_emb.weight[word_idx, 0] = 1.0

    def init_embeddings(self, src_query_emb_path, src_mono_emb_path, uni_emb_path):
        try:
            src_loaded_counts = 0
            print("Processing pretrained word embeddings for source language query embeddings")
            src_query_emb_dict = load_embedding(src_query_emb_path)
            self.src_query_embedding.weight.requires_grad = False
            for word in tqdm(list(self.vocab.src.word2id.keys())):
                if word not in src_query_emb_dict:
                    continue
                word_idx = self.vocab.src.word2id[word]
                self.src_query_embedding.weight[word_idx, :] = torch.from_numpy(src_query_emb_dict[word]).float()
                src_loaded_counts += 1
            print(f"{src_loaded_counts} words are found for the source language query embeddings")
        except FileNotFoundError:
            print(f"No pretrained embeddings specified for source language query embeddings")


        try:
            src_loaded_counts = 0
            print("Processing pretrained word embeddings for source language monolingual embeddings")
            src_mono_emb_dict = load_embedding(src_mono_emb_path)
            self.src_mono_embedding.weight.requires_grad = False
            for word in tqdm(list(self.vocab.src.word2id.keys())):
                if word not in src_mono_emb_dict:
                    continue
                word_idx = self.vocab.src.word2id[word]
                self.src_mono_embedding.weight[word_idx, :] = torch.from_numpy(src_mono_emb_dict[word]).float()
                src_loaded_counts += 1
            self.src_mono_embedding.weight.requires_grad = True
            print(f"{src_loaded_counts} words are found for the source language monolingual embedding")
        except FileNotFoundError:
            print(f"No pretrained embeddings specified for source language monolingual embeddings")

        try:
            uni_loaded_counts = 0
            print("Processing pretrained word embeddings for universal tokens")
            uni_emb_dict = load_embedding(uni_emb_path)
            self.uni_key_embedding.weight.requires_grad = False
            for word in tqdm(list(self.vocab.tgt.word2id.keys())):
                if word not in uni_emb_dict:
                    continue
                word_idx = self.vocab.tgt.word2id[word]
                self.uni_key_embedding.weight[word_idx, :] = torch.from_numpy(uni_emb_dict[word]).float()
                uni_loaded_counts += 1
            
            self.uni_val_embedding.weight.requires_grad = False
            self.uni_val_embedding.weight.data = deepcopy(self.uni_key_embedding.weight)
            # for word in tqdm(list(self.vocab.tgt.word2id.keys())):
            #     if word not in uni_emb_dict:
            #         continue
            #     word_idx = self.vocab.tgt.word2id[word]
            #     self.uni_val_embedding.weight[word_idx, :] = torch.from_numpy(uni_emb_dict[word]).float()
            self.uni_val_embedding.weight.requires_grad = True
            print(f"{uni_loaded_counts} words are found for the universal tokens")
        except FileNotFoundError:
            print(f"No pretrained embeddings specified for universal token embeddings")

    def forward(self, src_tokens):
        uni_key = self.uni_key_embedding.weight
        uni_val = self.uni_val_embedding.weight
        src_query = self.src_query_embedding(src_tokens) # (batch_size, max_len, emb_dim)
        scores = torch.einsum('bme,ee,ue->bmu', (src_query,self.transform_matrix, uni_key)) # (batch_size, max_len, uni_vocab_size)

        # compute temperature-augmented softmax
        scores = scores / self.temperature
        offset, _ = torch.max(scores, dim=-1, keepdim=True) # (batch_size, max_len, 1)
        scores = scores - offset # (batch_size, max_len, uni_vocab)
        weights = scores.exp() / scores.exp().sum(dim=-1, keepdim=True)

        # compute universal representation of input
        uni_embed = torch.einsum('bmu,ue->bme', (weights, uni_val)) # (batch_size, max_len, emb_dim)

        alpha = self.alpha_emb(src_tokens) # (batch_size, max_len, 1)
        interpol = alpha * self.src_mono_embedding(src_tokens) + self.beta * uni_embed
        return interpol


class ULR(nn.Module):
    def __init__(self, nmt_model, ulr_embed):
        super(ULR, self).__init__()
        self.nmt = nmt_model
        self.ulr_embed = ulr_embed
    
    def forward(self, src_tokens, src_lens, tgt_tokens, tgt_lens, teacher_forcing=0.5):
        src_emb = self.ulr_embed(src_tokens)
        return self.nmt(src_emb, src_lens, tgt_tokens, tgt_lens, teacher_forcing=teacher_forcing, feed_embedding=True)
    
    def greedy_search(self, src_sent, src_lens, beam_size=5, max_decoding_time_step=70, cuda=True):
        src_emb = self.ulr_embed(src_sent)
        return self.nmt.greedy_search(src_sent, src_lens, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step, cuda=True, feed_embedding=True)
    
    def beam_search(self, src_sent, src_lens, beam_size=5, max_decoding_time_step=70, cuda=True):
        src_emb = self.ulr_embed(src_sent)
        return self.nmt.beam_search(src_sent, src_lens, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step, cuda=True, feed_embedding=True)

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

        if cuda:
            torch.LongTensor = torch.cuda.LongTensor

        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting the leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict
            src_lens = torch.LongTensor(list(map(len, src_sents)))
            tgt_lens = torch.LongTensor(list(map(len, tgt_sents)))
            
            # these padding functions modify data in-place
            src_sents = pad(self.nmt.vocab.src.words2indices(src_sents))
            tgt_sents = pad(self.nmt.vocab.tgt.words2indices(tgt_sents))

            src_sents = torch.LongTensor(src_sents)
            tgt_sents = torch.LongTensor(tgt_sents)
            loss = -self.forward(src_sents, src_lens, tgt_sents, tgt_lens).sum()

            loss = loss.item()
            cum_loss += loss

        ppl = np.exp(cum_loss / cum_tgt_words)
        self.training = True
        return ppl

    @staticmethod
    def load(model_path):
        """
        Load a pre-trained model

        Returns:
            model: the loaded model
        """
        model = torch.load(model_path)
        return model
