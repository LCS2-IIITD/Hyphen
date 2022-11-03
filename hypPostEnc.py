import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import sys
csv.field_size_limit(sys.maxsize)
import sklearn
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings#ignoring the undefinedmetric warnings -- incase of precision having zero division
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning) 

from utils.manifolds import Euclidean, PoincareBall
from utils.nets import MobiusGRU
from utils.nets import MobiusLinear
from utils.nets import MobiusDist2Hyperplane
from utils.utils import matrix_mul, element_wise_mul

eps = 1e-7

class HypPostEnc(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, num_classes, embedding_matrix, max_sent_length, max_word_length, device, manifold,
    content_curvature):
        super(HypPostEnc, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.manifold = manifold
        self.content_curvature = content_curvature

        if isinstance(self.manifold, Euclidean):
            self.word_att_net = WordAttNet(embedding_matrix, word_hidden_size)
            self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size, num_classes)

        elif isinstance(self.manifold, PoincareBall):
            self.word_att_net = H_WordAttNet(embedding_matrix, word_hidden_size)
            self.sent_att_net = H_SentAttNet(sent_hidden_size, word_hidden_size, num_classes, self.content_curvature)
        
        self._init_hidden_state()
        
    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
        if torch.cuda.is_available() and self.device != torch.device("cpu"):
            self.word_hidden_state = self.word_hidden_state.cuda()
            self.sent_hidden_state = self.sent_hidden_state.cuda()

    def forward(self, input):
        output_list = []
        input = input.permute(1, 0, 2)
        for i in input:
            output, self.word_hidden_state = self.word_att_net(i.permute(1, 0), self.word_hidden_state)
            output_list.append(output)
        output = torch.cat(output_list, 0)
        output, h_output = self.sent_att_net(output, self.sent_hidden_state)
        return output, h_output

def E2Lorentz(input):
    """Function to convert fromm Euclidean space to the Lorentz model"""
    rr = torch.norm(input, p=2, dim=2)
    dd = input.permute(2,0,1) / rr
    cosh_r = torch.cosh(rr)
    sinh_r = torch.sinh(rr)
    output = torch.cat(((dd * sinh_r).permute(1, 2, 0), cosh_r.unsqueeze(0).permute(1, 2, 0)), dim=2)
    return output

def P2Lorentz(input):
    """Function to convert fromm Poincare model to the Lorentz model"""
    rr = torch.norm(input, p=2, dim=2)
    output = torch.cat((2*input, (1+rr**2).unsqueeze(2)),dim=2).permute(2,0,1)/(1-rr**2+eps)
    return output.permute(1,2,0)

def L2Klein(input):
    """Function to convert fromm Lorentz model to the Klein model"""
    dump = input[:, :, -1]
    dump = torch.clamp(dump, eps, 1.0e+16)
    return (input[:, :, :-1].permute(2, 0, 1)/dump).permute(1, 2, 0)

def arcosh(x):
    c0 = torch.log(x)
    c1 = torch.log1p(torch.sqrt(x * x - 1 + eps) / x)
    return c0 + c1

def disLorentz(x, y):
    m = x * y
    prod_minus = -m[:, :, :-1].sum(dim=2) + m[:, :, -1]
    output = torch.clamp(prod_minus, 1.0 + eps, 1.0e+16)
    return arcosh(output)

class H_WordAttNet(nn.Module):
    def __init__(self, embedding_matrix, hidden_size = 50):
        super().__init__()
        # for dot attention
        self.attn = nn.Linear(2*hidden_size, 2*hidden_size, bias=True)
        self.context_weight = nn.Linear(2*hidden_size, 1, bias=False)

        # for Lorentz attention
        self.attn2 = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=True)
        self.Lorentz_centroid = nn.Parameter(torch.Tensor(2*hidden_size))
        self.beta = nn.Parameter(torch.Tensor(1))
        self.c = 1.0

        self.lookup = self.create_embeddeding_layer(embedding_matrix)
        self.gru = nn.GRU(self.embedding_dim, hidden_size, bidirectional = True)
        self._create_weights(mean = 0.0, std = 0.05)

    def _create_weights(self, mean = 0.0, std = 0.05):
        self.Lorentz_centroid.data.normal_(mean, std)
        self.beta.data.normal_(mean, std)
        # self.c.data.normal_(mean, std)

    def forward(self, input, hidden_state):

        output = self.lookup(input)

        self.gru.flatten_parameters()  # Todo{flatten()}
        f_output, h_output = self.gru(output.float(), hidden_state)
    
        ## lorentz attention
        hyp_alpha = torch.tanh_(self.attn2(f_output))
        hyp_alpha = E2Lorentz(hyp_alpha)  # (46,128,101)
        u_w = E2Lorentz(self.Lorentz_centroid.unsqueeze(0).unsqueeze(0)) # (1,1,101)
        dist = disLorentz(hyp_alpha, u_w)
        hyp_alpha = - self.beta * dist - self.c
        hyp_alpha = hyp_alpha - hyp_alpha.max()
        hyp_alpha = F.softmax(hyp_alpha, dim=0)  # Todo{check dim}

        alpha = hyp_alpha

        f_output = E2Lorentz(f_output)
        f_output = L2Klein(f_output)

        dump = 1 - torch.norm(f_output, p=2, dim=2)**2
        dump = torch.clamp(dump, eps, 1-eps)
        dump = torch.sqrt(dump)
        gamma = 1/dump

        gamma = torch.clamp(gamma, 1.0 + eps, 1.0e+16)

        alpha = alpha * gamma
        alpha = alpha / (torch.sum(alpha, dim=0))

        output = torch.sum(alpha * f_output.permute(2,0,1), dim=1).permute(1,0).unsqueeze(0)

        return output, h_output

    def create_embeddeding_layer(self, weights_matrix, trainable=False):
        self.num_embeddings, self.embedding_dim = weights_matrix.shape
        weights_matrix = torch.from_numpy(weights_matrix)
        emb_layer = nn.Embedding(self.num_embeddings, self.embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        emb_layer.weight.requires_grad = trainable
        return emb_layer

class H_SentAttNet(nn.Module):
    def __init__(self, sent_hidden_size=50, word_hidden_size=50, num_classes=2, content_curvature = 1):
        super().__init__()

        self.Lorentz_centroid = nn.Parameter(torch.Tensor(2*sent_hidden_size))
        self.Poincare_centroid = nn.Parameter(torch.Tensor(2*sent_hidden_size))

        self.beta = nn.Parameter(torch.Tensor(1))
        self.c = 1.0

        self.gru_forward = MobiusGRU(2*word_hidden_size, sent_hidden_size)
        self.gru_backward = MobiusGRU(2*word_hidden_size, sent_hidden_size)
        self.hyp_att_projector = MobiusLinear(2*sent_hidden_size, 2*sent_hidden_size, bias=True, c=1.0) #Todo{discard}
        self.dot_att_projector = MobiusLinear(2*sent_hidden_size, 2*sent_hidden_size, bias=True, c=1.0)
        self.dot_att_us = MobiusLinear(2*sent_hidden_size, 1, bias=False, c=1.0)

        self.logit_projector = MobiusLinear(2*sent_hidden_size, sent_hidden_size, bias=True, c=1.0)
        self.logits = MobiusDist2Hyperplane(sent_hidden_size, num_classes)

        self._create_weights(mean = 0.0, std = 0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.Lorentz_centroid.data.normal_(mean, std)
        self.Poincare_centroid.data.normal_(mean, std)
        self.beta.data.normal_(mean, std)

    def forward(self, input, hidden_state):

        input = (input.permute(2,0,1)/(1 + torch.sqrt(1 + torch.norm(input, p=2, dim=2) ** 2))).permute(1,2,0)

        f_output1, h_output1 = self.gru_forward(input, hidden_state[0])
        f_output2, h_output2 = self.gru_backward(torch.flip(input,(0,)), hidden_state[1])

        # on Poincare
        h_output = torch.cat((h_output1, h_output2), 0)
        f_output = torch.cat((f_output1, f_output2), 2)

        # h_K for aggregation
        output = P2Lorentz(f_output)
        output = L2Klein(output)

        # hyp alpha
        hyp_alpha = self.hyp_att_projector(f_output)    # no tanh()
        hyp_alpha = P2Lorentz(hyp_alpha)
        u_w = E2Lorentz(self.Lorentz_centroid.unsqueeze(0).unsqueeze(0))
        dist = disLorentz(hyp_alpha, u_w)
        hyp_alpha = - self.beta * dist - self.c
        hyp_alpha = hyp_alpha - hyp_alpha.max()
        hyp_alpha = F.softmax(hyp_alpha, dim=0)
        alpha = hyp_alpha
        dump = 1 - torch.norm(output, p=2, dim=2) ** 2
        dump = torch.clamp(dump, eps, 1 - eps)
        dump = torch.sqrt(dump)
        gamma = 1 / dump
        gamma = torch.clamp(gamma, 1.0 + eps, 1.0e+16)

        alpha = alpha * gamma
        alpha = alpha / (torch.sum(alpha, dim=0))     #(3,128)
        output = torch.sum(alpha * output.permute(2, 0, 1), dim=1).permute(1, 0)

        #output is in Klein and f_output is in Poincare. Converting Klein to Poincare for output
        output = (output.permute(1,0) / (1 + torch.sqrt(1 + torch.norm(output, p=2, dim=1) ** 2))).permute(1, 0)

        return output, f_output.permute(1, 0, 2)#added sigmoid function

class SentAttNet(nn.Module):
    def __init__(self, sent_hidden_size=50, word_hidden_size=50, num_classes=14):
        super(SentAttNet, self).__init__()
        self.sent_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 2 * sent_hidden_size))
        self.sent_bias = nn.Parameter(torch.Tensor(1, 2 * sent_hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 1))
        self.gru = nn.GRU(2 * word_hidden_size, sent_hidden_size, bidirectional=True)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):
        f_output, h_output = self.gru(input, hidden_state)
        output = matrix_mul(f_output, self.sent_weight, self.sent_bias)
        output = matrix_mul(output, self.context_weight).permute(1, 0)
        output = F.softmax(output, dim = -1)
        output = element_wise_mul(f_output, output.permute(1, 0)).squeeze(0)
        return output, f_output.permute(1, 0, 2) #return none curvature

class WordAttNet(nn.Module):
    def __init__(self, embedding_matrix, hidden_size=50):
        super(WordAttNet, self).__init__()
        self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))
        self.lookup = self.create_embeddeding_layer(embedding_matrix)
        self.gru = nn.GRU(self.embedding_dim, hidden_size, bidirectional=True)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):
        output = self.lookup(input)
        f_output, h_output = self.gru(output.float(), hidden_state)  # feature output and hidden state output
        output = matrix_mul(f_output, self.word_weight, self.word_bias)
        output = matrix_mul(output, self.context_weight).permute(1,0)
        output = F.softmax(output, dim = -1)
        output = element_wise_mul(f_output,output.permute(1,0))
        return output, h_output
        
    def create_embeddeding_layer(self, weights_matrix, trainable=False):
        self.num_embeddings, self.embedding_dim = weights_matrix.shape
        weights_matrix = torch.from_numpy(weights_matrix)
        emb_layer = nn.Embedding(self.num_embeddings, self.embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        emb_layer.weight.requires_grad = trainable
        return emb_layer