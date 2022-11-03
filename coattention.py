import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.layers.hyp_layers import HypAct

class CoAttention(nn.Module):

    """This is the class for Hyperbolic Fourier-coattention mechanism."""
    
    def __init__(self, device,latent_dim = 100,  embedding_dim = 100, manifold = None, content_curvature = None, comment_curvature = None, combined_curvature = None, fourier = False):
        super(CoAttention, self).__init__()

        self.content_curvature = content_curvature
        self.comment_curvature = comment_curvature
        self.manifold = manifold
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.k = 128
        self.Wl = nn.Parameter(torch.Tensor((self.latent_dim, self.latent_dim)))
        self.Wc = nn.Parameter(torch.Tensor((self.k, self.latent_dim)))
        self.Ws = nn.Parameter(torch.Tensor((self.k, self.latent_dim)))
        self.whs = nn.Parameter(torch.Tensor((1, self.k)))
        self.whc = nn.Parameter(torch.Tensor((1, self.k)))
        self.concat_m1 = nn.Parameter(torch.Tensor((1, 1)))
        self.concat_m2 = nn.Parameter(torch.Tensor((1, 1)))
        self.concat_b = nn.Parameter(torch.Tensor((1, self.embedding_dim)))

        #register weights and bias as params
        self.register_parameter("Wl", self.Wl)
        self.register_parameter("Wc", self.Wc)
        self.register_parameter("Ws", self.Ws)
        self.register_parameter("whs", self.whs)
        self.register_parameter("whc", self.whc)

        #concatenation operation for hyperbolic 
        self.register_parameter("concat_m1", self.concat_m1)
        self.register_parameter("concat_m2", self.concat_m2)
        self.register_parameter("concat_b", self.concat_b)

        #initialize data of parameters
        self.Wl.data = torch.randn((self.latent_dim, self.latent_dim))
        self.Wc.data = torch.randn((self.k, self.latent_dim))
        self.Ws.data = torch.randn((self.k, self.latent_dim))
        self.whs.data = torch.randn((1, self.k))
        self.whc.data = torch.randn((1, self.k))
        self.concat_m1.data = torch.randn((1, 1))
        self.concat_m2.data = torch.randn((1, 1))
        self.concat_b.data = torch.randn((1, self.embedding_dim))
        self.c = combined_curvature
        self.map_comments = HypAct(manifold, self.comment_curvature, self.c, act = torch.tanh)
        self.map_content = HypAct(manifold, self.content_curvature, self.c, act = torch.tanh)
        self.fourier = fourier

    def forward(self, sentence_rep, comment_rep):

        """This function will return the shape [batch_size, embedding_dim]."""

        mobius_matvec = self.manifold.mobius_matvec
        proj = self.manifold.proj
        logmap0 = self.manifold.logmap0
        expmap0 = self.manifold.expmap0
        mobius_add = self.manifold.mobius_add
        curv = self.c 

        if self.fourier:
            # KFU
            sentence_rep = logmap0(sentence_rep, c = curv)
            sentence_rep = torch.fft.fft2(sentence_rep).float()

            comment_rep = logmap0(comment_rep, c = curv)
            comment_rep = torch.fft.fft2(comment_rep).float()

            sentence_rep = proj(expmap0(sentence_rep, c = curv), c = curv)
            comment_rep = proj(expmap0(comment_rep, c = curv), c = curv)

        sentence_rep_trans = sentence_rep.transpose(2, 1)#[32, 100, 50]
        comment_rep_trans = comment_rep.transpose(2, 1)#[32, 100, 10]

        comment_rep_trans = proj(comment_rep_trans, curv)
        sentence_rep_trans = proj(sentence_rep_trans, curv)

        # L = torch.tanh(torch.matmul(torch.matmul(comment_rep, self.Wl), sentence_rep_trans))  
        L = mobius_matvec(self.Wl, comment_rep, c= curv)
        L = proj(L, c = curv)
        L_e = logmap0(L, c = curv)
        s_r_t_e = logmap0(sentence_rep_trans, c = curv)
        L = torch.tanh(torch.matmul(L_e, s_r_t_e))
        L = proj(expmap0(L, c = curv), c = curv) #(32, 10, 50)
        assert not torch.isnan(L).any(), "L is nan"

        #Hs = torch.tanh(torch.matmul(self.Ws, sentence_rep_trans) + torch.matmul(torch.matmul(self.Wc, comment_rep_trans), L))
        Hs_a = mobius_matvec(self.Ws, sentence_rep_trans.transpose(-1, -2), c = curv).transpose(-1, -2)
        Hs_a = proj(Hs_a, c = curv)
        Hs_b = mobius_matvec(self.Wc, comment_rep_trans.transpose(-1, -2), c = curv).transpose(-1, -2)
        Hs_b = proj(Hs_b, c = curv)
        Hs_b_e = logmap0(Hs_b, c = curv)
        L_e = logmap0(L, c = curv)
        Hs_b_e = torch.matmul(Hs_b_e, L_e)
        Hs_b = expmap0(Hs_b_e, c = curv)
        Hs_b = proj(Hs_b, c = curv)
        Hs = proj(mobius_add(Hs_a, Hs_b, c = curv), c= curv)
        Hs = proj(expmap0(torch.tanh(logmap0(Hs, c = curv)), c = curv), c = curv) #[32, 80, 50]
        assert not torch.isnan(Hs).any(), "Hs is nan"

        #Hc = torch.tanh(torch.matmul(self.Wc, comment_rep_trans)+ torch.matmul(torch.matmul(self.Ws, sentence_rep_trans), L_trans))
        Hc_a = mobius_matvec(self.Wc, comment_rep_trans.transpose(-1, -2), c = curv).transpose(-1, -2)
        Hc_a = proj(Hc_a, c = curv)
        Hc_b = mobius_matvec(self.Ws, sentence_rep_trans.transpose(-1, -2), c = curv).transpose(-1, -2)
        Hc_b = proj(Hc_b, c = curv)
        Hc_b_e = logmap0(Hc_b, c = curv)
        Hc_b_e = torch.matmul(Hc_b_e, L_e.transpose(-1, -2))
        Hc_b = expmap0(Hc_b_e, c = curv)
        Hc_b = proj(Hc_b, c = curv)
        Hc = proj(mobius_add(Hc_a, Hc_b, c = curv), c= curv)
        Hc = proj(expmap0(torch.tanh(logmap0(Hc, c = curv)), c = curv), c = curv) #[32, 80, 10]
        assert not torch.isnan(Hc).any(), "Hc is nan"

        #As = F.softmax(torch.matmul(self.whs, Hs), dim=2)
        As = mobius_matvec(self.whs, Hs.transpose(-1, -2), c = curv).transpose(-1, -2)#[32, 1, 50]
        As = proj(As, c = curv)
        As = expmap0(F.softmax(logmap0(As, c = curv), dim = -1), c = curv)
        As = proj(As, c = curv)
        assert not torch.isnan(As).any(), "As is nan"

        # Ac = F.softmax(torch.matmul(self.whc, Hc), dim=2)
        Ac = mobius_matvec(self.whc, Hc.transpose(-1, -2), c = curv).transpose(-1, -2)
        Ac = proj(Ac, c = curv)
        Ac = expmap0(F.softmax(logmap0(Ac, c = curv), dim = -1), c = curv)
        Ac = proj(Ac, c = curv) #[32, 1, 10]
        assert not torch.isnan(Ac).any(), "Ac is nan"

        # co_s = torch.matmul(As,sentence_rep) # (1, 100)
        As_e = logmap0(As, c = curv)
        s_r_e = logmap0(sentence_rep, c = curv)
        co_s = torch.matmul(As_e, s_r_e)
        assert not torch.isnan(co_s).any(), "co_s is nan"
        
        # co_c = torch.matmul(Ac, comment_rep) # (1, 100)
        Ac_e = logmap0(Ac, c = curv)
        c_r_e = logmap0(comment_rep, c = curv)
        co_c = torch.matmul(Ac_e, c_r_e)
        assert not torch.isnan(co_c).any(), "co_c is nan"

        co_sc = torch.cat([co_s, co_c], dim = -1)
        co_sc = torch.squeeze(co_sc)

        assert not torch.isnan(co_sc).any(), "co_sc is nan"
        return co_sc, As, Ac # [32, 200], 
