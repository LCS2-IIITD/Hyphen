import torch
import torch.nn as nn
import torch.nn as nn

from coattention import CoAttention
from hypComEnc import HypComEnc
from hypPostEnc import HypPostEnc
from utils.layers.hyp_layers import *
from utils import manifolds
from utils.manifolds import Euclidean

class Hyphen(nn.Module):

    def __init__(self, embedding_matrix,  word_hidden_size, sent_hidden_size, max_sent_length, max_word_length, device, graph_hidden, num_classes = 2, max_sentence_count = 50 , max_comment_count = 10, batch_size = 32 ,embedding_dim = 100, latent_dim = 100, graph_glove_dim = 100, manifold = "hyper",
    content_module =True, comment_module = True, fourier = False):

        super(Hyphen,self).__init__()
        self.comment_curvature = 1
        self.content_curvature = 1
        self.combined_curvature = 1
        self.fourier = fourier
        self.graph_glove_dim = graph_glove_dim#the dimension of glove embeddings used to initialise the comments amr graph
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.max_sentence_count = max_sentence_count
        self.max_comment_count = max_comment_count
        self.device = device
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length  = max_word_length 
        self.graph_hidden = graph_hidden
        self.manifold = getattr(manifolds, manifold)()
        self.comment_module = comment_module
        self.content_module = content_module 
        self.content_encoder= HypPostEnc(self.word_hidden_size, self.sent_hidden_size, batch_size, num_classes, embedding_matrix, self.max_sent_length, self.max_word_length, self.device, manifold = self.manifold, content_curvature = self.content_curvature)
        self.comment_encoder = HypComEnc(self.graph_glove_dim, self.graph_hidden, num_classes, self.max_comment_count, device= self.device, manifold = self.manifold, content_module = self.content_module, comment_curvature = self.comment_curvature)
        self.coattention = CoAttention(device, latent_dim, manifold = self.manifold,  comment_curvature = self.comment_curvature, content_curvature = self.content_curvature, combined_curvature = self.combined_curvature, fourier = self.fourier)
        
        if self.comment_module and self.content_module: self.fc = nn.Linear(2*latent_dim, num_classes)
        elif self.comment_module: self.fc = nn.Linear(latent_dim, num_classes)
        else: self.fc = nn.Linear(2*self.sent_hidden_size, num_classes)

    def forward(self, content, comment, subgraphs):
        
        #both content and comments modules are on
        if self.comment_module and self.content_module:

            #hyphen-euclidean 
            if isinstance(self.manifold, Euclidean):
                _, content_embedding = self.content_encoder(content)
                comment_embedding = self.comment_encoder(comment, comment.ndata['feat'], subgraphs)
                coatten, As, Ac = self.coattention(content_embedding, comment_embedding)

            else:#hyphen-hyperbolic
                _, content_embedding = self.content_encoder(content)
                comment_embedding = self.comment_encoder(comment, comment.ndata['feat'], subgraphs)
                assert not torch.isnan(content_embedding).any(), "content_embedding is nan"
                assert not torch.isnan(comment_embedding).any(), "comment_embedding is nan"
                coatten, As, Ac = self.coattention(content_embedding, comment_embedding)
            
            preds = self.fc(coatten)
            if torch.isnan(preds).any():
                print(preds, coatten)
                preds = torch.nan_to_num(preds, nan = 0.0)

            assert not torch.isnan(preds).any(), "preds is nan"
            return preds, As, Ac
    
        #only comment module is on
        elif self.comment_module:
            comment_embedding = self.comment_encoder(comment, comment.ndata['feat'], subgraphs)
            comment_embedding = self.manifold.proj(comment_embedding, c = 1.0)
            comment_embedding = self.manifold.logmap0(comment_embedding, c = 1.0)
            preds = self.fc(comment_embedding)
            return preds

        #only content module is on
        else:
            content_embedding, _ = self.content_encoder(content)
            content_embedding = self.manifold.proj(content_embedding, c = 1.0)
            content_embedding = self.manifold.logmap0(content_embedding, c = 1.0)
            preds = self.fc(content_embedding)
            return preds
    