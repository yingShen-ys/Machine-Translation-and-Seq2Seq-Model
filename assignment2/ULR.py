import torch
import torch.nn as nn
from typing import List, Dict

class ULR(nn.Module):
    '''
    Universal low-resource translation model.
    Args:
         - language_vocabs: a dictionary of ints indicating vocab size for each src language
         - universal_token_vocab: int, indicating vocab size for the universal token (usually just En)
         - embedding_dim: int, all embeddings need to be in the same size
         - alpha: a simple network that determines the interpolation weight for final word representation
         - beta: ditto
    '''
    def __init__(self, language_vocabs: Dict, universal_token_vocab: int, embedding_dim: int, alpha: nn.Module, beta: nn.Module, temperature: float):
        self.universal_token_vocab = universal_token_vocab
        
        self.src_embeddings = nn.ModuleDict({k: nn.Embedding(v, )})
        self.universal_token_key_embedding = nn.Embedding(len(self.universal_token_vocab), embedding_dim) # E^K
        self.universal_token_value_embedding = nn.Embedding(len(self.universal_token_vocab), embedding_dim) # E^U
        self.monolingual_language_specific_embedding = nn.Embedding(len(self.language_specific_vocab), embedding_dim) # E^I
        self.bilinear_transformation = nn.Linear(embedding_dim, embedding_dim)
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

        # fix embedding for E^Q and E^K
        self.language_specific_embedding.weight.requires_grad = False
        self.universal_token_key_embedding.weight.requires_grad = False
    
    def load_pretrain_embeddings(self, pretrain_language_specific_embedding, pretrain_universal_source_token_key_embedding):
        pass

    def load_transformation_matrix_weight(self, weight):
        pass

    def forward(self, batch):
        query = self.language_specific_embedding(batch) # B x N (Embedding_dim)
        weights = torch.mm(self.bilinear_transformation(self.universal_source_token_key_embedding.weight), query).t() # B x M (source_vocab_size)
        weights = torch.exp(weights)/self.temperature # B x M (source_vocab_size)
        weights_sum = torch.sum(weights, dim = 1) # B
        weights = weights/weights_sum # B x M

        value = torch.mm(weights, self.universal_source_token_value_embedding.weight)
        # TODO: implement alpha calculation
        return self.alpha * self.universal_source_token_value_embedding(batch) + self.beta * value
