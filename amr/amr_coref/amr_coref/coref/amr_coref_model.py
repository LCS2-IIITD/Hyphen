import os
from   tqdm import tqdm
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from   ..utils.config import Config
from   ..evaluate.pr_scorer import PRScorer
from   .coref_featurizer import SIZE_SPAN, SIZE_WORD, SIZE_FS, SIZE_FP
from   .vocab_embeddings import Vocab, load_vocab_embeddings, load_word_set


# Neural Network model that predicts coreference from AMR graphs
# For debugging...numpy.set_printoptions(edgeitems=1000, linewidth=100000)
# print(train_prediction.detach().cpu().numpy())
class AMRCorefModel(nn.Module):
    def __init__(self, config, graph_vocab, graph_embed_mat, mention_set):
        super().__init__()
        self.config        = config
        self.graph_vocab   = graph_vocab
        self.device        = torch.device(config.device)
        self.mention_set   = mention_set
        # If the embeddings are allowed to train, the original ones needs to be kept in the
        # model so that the span vectors can use them.  Spans are average vectors that can't
        # be trained.
        if self.config.train_embeds:
            self.orig_embeds = numpy.copy(graph_embed_mat)
        else:
            self.orig_embeds = None
        self.build(graph_embed_mat)

    def build(self, graph_embed_mat):
        self.graph_embeds = nn.Embedding.from_pretrained(torch.from_numpy(graph_embed_mat),
                            freeze=not self.config.train_embeds)
        # Define some sizes
        H1 = self.config.h1_size
        H2 = self.config.h2_size
        H3 = self.config.h3_size
        dim_embed     = graph_embed_mat.shape[1]
        dim_men_emb   = (SIZE_SPAN + SIZE_WORD) * dim_embed
        dim_single_in =   dim_men_emb + SIZE_FS
        dim_pair_in   = 2*dim_men_emb + SIZE_FP
        dropout = self.config.dropout
        # Embedding adapter allow for a trainable layer on top of the frozen embedding matrix
        self.embed_adapter = nn.Sequential(nn.Linear(dim_embed, dim_embed), nn.ReLU(), nn.Dropout(dropout))
        # Single network - predict the probability of a mention having no antecedents
        self.single_net = nn.Sequential(nn.Linear(dim_single_in, H1), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(H1, H2), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(H2, H3), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(H3, 1),  nn.Linear(1, 1))
        # Pair network - predict the proability of anaphor / antecedent pairs
        self.pairs_net  = nn.Sequential(nn.Linear(dim_pair_in, H1), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(H1, H2), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(H2, H3), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(H3, 1),  nn.Linear(1, 1))
        # Initializer non-embedding weights and biases
        w = (param.data for name, param in self.named_parameters() if 'weight' in name and 'graph_embeds' not in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for t in w:
            nn.init.xavier_uniform_(t)
        for t in b:
            nn.init.constant_(t, 0)
        # Move model parameters to the GPU
        self = self.to(self.device)

    ###############################################################################################
    #### Run the model
    ###############################################################################################

    # Process a batch.  forward is called by.. Y = model(batch)
    def forward(self, batch):
        net_out = {}
        # Run the single input scores (ie head mention with no antecedents)
        net_out['single_scores'] = self.forward_single_net(batch)
        # If there are pairs (ie head mention with potential antecedents)
        if batch['has_pairs']:
            net_out['pair_scores'] = self.forward_pairs_net(batch)
            net_out['all_scores']  = torch.cat([net_out['single_scores'], net_out['pair_scores']], 1)
        else:
            net_out['all_scores']  = net_out['single_scores']
        return net_out

    # Run the single network to prediction the probability of a mention
    # having no antecedents.
    def forward_single_net(self, batch):
        # Move to device (cpu or cuda)
        sspans          = batch['sspans'].to(self.device, non_blocking=True)           # shape = (batch, 1) span vec
        dspans          = batch['dspans'].to(self.device, non_blocking=True)           # shape = (batch, 1) span vec
        words           = batch['words'].to(self.device, non_blocking=True)            # shape = (batch, W) token indexes
        single_features = batch['single_features'].to(self.device, non_blocking=True)  # shape = (batch, Fs) features
        # Add embedding adaptor to the span vectors
        if self.config.adapt_spans:
            sspans = self.embed_adapter(sspans)
            dspans = self.embed_adapter(dspans)
        # Processing embeddings
        word_embed = self.graph_embeds(words)
        if self.config.adapt_words:
            word_embed = self.embed_adapter(word_embed)                                # shape = (batch, W, emebd_dim)
        word_embed = word_embed.view(words.size()[0], -1)                              # shape = (batch, W*embed_dim)
        # Concatenate layers (layed out horizontally) and run the single network
        single_input  = torch.cat([sspans, dspans, word_embed, single_features], 1)
        single_scores = self.single_net(single_input)
        return single_scores

    # Run the pair network to predict the probability of head mention / antecedent pairs
    # ant => antecedent  ana => anaphor   P (num pairs) == num antecedent / anaphor pairs
    def forward_pairs_net(self, batch):
        ant_sspans    = batch['ant_sspans'].to(self.device,    non_blocking=True)  # shape = (batch, P, S) span vectors
        ant_dspans    = batch['ant_dspans'].to(self.device,    non_blocking=True)  # shape = (batch, P, S) span vectors
        ant_words     = batch['ant_words'].to(self.device,     non_blocking=True)  # shape = (batch, P, W) token indexes
        ana_sspans    = batch['ana_sspans'].to(self.device,    non_blocking=True)  # shape = (batch, P, S) span vectors
        ana_dspans    = batch['ana_dspans'].to(self.device,    non_blocking=True)  # shape = (batch, P, S) span vectors
        ana_words     = batch['ana_words'].to(self.device,     non_blocking=True)  # shape = (batch, P, W) token indexes
        pair_features = batch['pair_features'].to(self.device, non_blocking=True)  # shape = (batch, P, Fp) features
        # Add embedding adaptor to the span vectors
        if self.config.adapt_spans:
            ant_sspans = self.embed_adapter(ant_sspans)
            ant_dspans = self.embed_adapter(ant_dspans)
            ana_sspans = self.embed_adapter(ana_sspans)
            ana_dspans = self.embed_adapter(ana_dspans)
        # Processing embeddings
        batchsize, pairs_num, _ = ant_words.size()
        ant_embed_words = self.graph_embeds(ant_words)
        ana_embed_words = self.graph_embeds(ana_words)
        if self.config.adapt_words:
            ant_embed_words = self.embed_adapter(ant_embed_words)
            ana_embed_words = self.embed_adapter(ana_embed_words)
        ant_embed_words = ant_embed_words.view(batchsize, pairs_num, -1)
        ana_embed_words = ana_embed_words.view(batchsize, pairs_num, -1)
        # Concatenate and run the net
        pair_input = torch.cat([ant_sspans, ant_dspans, ant_embed_words,
                                ana_sspans, ana_dspans, ana_embed_words, pair_features], 2,)
        pair_scores = self.pairs_net(pair_input).squeeze(dim=2)
        return pair_scores

    ###############################################################################################
    #### Training Loss
    ###############################################################################################

    # All pairs and single mentions probabilistic loss
    def all_pair_loss(self, net_out, batch):
        scores    = net_out['all_scores'].to(self.device, non_blocking=True)
        labels    = batch['all_labels'].to(self.device,   non_blocking=True)
        weights   = batch['all_weights'].to(self.device, non_blocking=True)  # (possibly weighted) mask
        num_elems = batch['num_elems']
        # Calculate the loss on each score (single + P antecedent pairs) and sum to a single value
        # Use the masks (aka weight) to apply a 0 weight to padded values.
        # Divide by the true number of elements (omits the padded values) to get the average
        # binary_cross_entropy_with_logits applies a sigmoid layer to the output and then BCE
        loss = F.binary_cross_entropy_with_logits(scores, labels, weight=weights, reduction='sum')
        return loss / num_elems

    # Top pairs probabilistic loss.  Top => the best score from each element in the batch
    def top_pair_loss(self, net_out, batch):
        scores     = net_out['all_scores'].to(self.device, non_blocking=True)   # (batch, P + 1)
        true_ants  = batch['true_ants'].to(self.device,    non_blocking=True)   # (batch, P + 1)
        # Convert scores to probabilities
        epsilon     = 1.0e-7
        s_scores    = torch.sigmoid(scores).clamp(epsilon, 1.0-epsilon)
        # Find the maximum scoring true antecedent from each batch element
        true_pairs  = torch.gather(s_scores, 1, true_ants)
        top_true, _ = torch.log(true_pairs).max(dim=1)  # max(log(p)), p=sigmoid(s)  # (batch,)
        out_score   = torch.sum(top_true).neg()
        num_scores  = top_true.size()[0]    # batchsize
        # Add the top scoring false antecedent for each batch element
        # We have no false antecedents when there are no pairs
        if batch['has_pairs']:
            false_ants   = batch['false_ants'].to(self.device, non_blocking=True)
            false_pairs  = torch.gather(s_scores, 1, false_ants)
            top_false, _ = torch.log(1 - false_pairs).min(dim=1)  # min(log(1-p)), p=sigmoid(s)
            out_score    = out_score + torch.sum(top_false).neg()
            num_scores  = num_scores + top_false.size()[0]  # batchsize
        return out_score / num_scores

    # Slack-rescaled max margin loss
    # costs: FN=> false new,  FL=> false link,  WL=> wrong link
    def ranking_loss(self, net_out, batch):
        scores    = net_out['all_scores'].to(self.device, non_blocking=True)
        true_ants = batch['true_ants'].to(self.device,    non_blocking=True)
        costs     = batch['costs'].to(self.device,        non_blocking=True)
        num_elems = batch['num_elems']
        # Compute slack-rescaled max margin loss
        # Gather scores for the true antecedents and compute 1 + scores - top_true
        true_ant_score = torch.gather(scores, 1, true_ants)
        top_true, _ = true_ant_score.max(dim=1)
        tmp_loss    = scores.add(1).add(top_true.unsqueeze(1).neg())  # (batch, P+1)
        # If looking at pairs, mask off the padded values
        if batch['has_pairs']:
            mask = batch['all_mask'].to(self.device)
            tmp_loss = tmp_loss.mul(mask)
        # Multiply the above loss values by the costs and find the max for each batch item
        tmp_loss  = tmp_loss.mul(costs)
        loss, _   = tmp_loss.max(dim=1)     # (batchsize,)
        out_score = torch.sum(loss)
        return out_score / loss.size()[0]

    ###############################################################################################
    #### Inference / Testing
    ###############################################################################################

    # Run through all batches in the test data loader and accumulate by mention data index
    def process(self, dloader, show_prog=False):
        odict = {'s_labels':{}, 's_probs':{}, 'p_labels':{}, 'p_probs':{}}
        self.eval()
        with torch.no_grad():
            # Loop through all batches in the data_loader
            for batch in tqdm(dloader, ncols=100, disable=not show_prog):
                net_out = self.forward(batch)
                mdata_indexes = batch['mdata_indexes']      # mention data indexing
                # Process the single network values and add to the output dict
                single_labels = batch['single_labels'].cpu().numpy()
                single_probs  = torch.sigmoid(net_out['single_scores']).cpu().numpy()
                assert single_labels.shape == single_probs.shape    # (batch, 1)
                for i, mdi in enumerate(mdata_indexes):
                    assert mdi not in odict['s_labels'] and mdi not in odict['s_probs']
                    odict['s_labels'][mdi] = single_labels[i][0]    # (batch, 1]
                    odict['s_probs'][mdi]  = single_probs[i][0]     # (batch, 1]
                # Process the pairs (if present)
                if 'pair_scores' in net_out:
                    pair_labels = batch['pairs_labels'].cpu().numpy()
                    pair_probs  = torch.sigmoid(net_out['pair_scores']).cpu().numpy()
                    assert pair_labels.shape == pair_probs.shape    # (batch, max_len)
                    assert len(batch['pair_lens']) == pair_probs.shape[0]      # list of batch length
                    # Trims pairs to the correct (non-padded lengths) and add to the output dict
                    for i, mdi in enumerate(mdata_indexes):
                        length = batch['pair_lens'][i]
                        assert mdi not in odict['p_labels'] and mdi not in odict['p_probs']
                        odict['p_labels'][mdi] = pair_labels[i][:length].tolist()
                        odict['p_probs'][mdi]  = pair_probs[i][:length].tolist()
        return odict


    ###############################################################################################
    #### Loading and Saving
    ###############################################################################################

    # Create a new model from various files
    @classmethod
    def from_files(cls, config_fn, graph_embed_fn, mention_set_fn):
        config = Config.load(config_fn)
        graph_vocab,  graph_embed_mat  = load_vocab_embeddings(graph_embed_fn)
        mention_set = load_word_set(mention_set_fn)
        return cls(config, graph_vocab, graph_embed_mat, mention_set)

    # Build the model from it's saved state dict, configuration and vocabulary
    @classmethod
    def from_pretrained(cls, model_dir, model_fn='amr_coref.pt', config_fn='config.json'):
        config     = Config.load(os.path.join(model_dir, config_fn))
        model_dict = torch.load(os.path.join(model_dir, model_fn))
        assert config.graph_num_embeddings == len(model_dict['graph_tokens'])
        graph_mat   = numpy.zeros(shape=(config.graph_num_embeddings,  config.graph_embedding_dim), dtype='float32')
        vocab       = Vocab(model_dict['graph_tokens'])
        mention_set = model_dict['mention_set']
        self        = cls(config, vocab, graph_mat, mention_set)
        self.load_state_dict(model_dict['state_dict'])
        self.optimizer_state_dict = model_dict.get('optimizer_state_dict', None)
        self.orig_embeds = model_dict.get('orig_embeds', None)
        return self

    # Save the model, vocab and config
    def save(self, model_dir, epoch, tphase=None, optimizer=None, model_fn='amr_coref.pt', config_fn='config.json'):
        os.makedirs(model_dir, exist_ok=True)
        self.config.last_epoch            = epoch
        self.config.training_phase        = tphase
        self.config.graph_num_embeddings  = self.graph_embeds.num_embeddings
        self.config.graph_embedding_dim   = self.graph_embeds.embedding_dim
        self.config.save(os.path.join(model_dir, config_fn))
        model_dict = {}
        model_dict['state_dict']   = self.state_dict()
        model_dict['graph_tokens'] = self.graph_vocab.get_embedding_tokens()
        model_dict['mention_set']  = self.mention_set
        model_dict['orig_embeds']  = self.orig_embeds
        if optimizer is not None:
            model_dict['optimizer_state_dict'] = optimizer.state_dict()
        torch.save(model_dict, os.path.join(model_dir, model_fn))

    # Get the vocab for the graph embedding matrix
    def get_graph_vocab(self):
        return self.graph_vocab

    # Get the embedding matrix from the trained model
    def get_graph_embed_mat(self):
        if self.orig_embeds is not None:
            return self.orig_embeds
        else:
            return self.graph_embeds.weight.clone().detach().cpu().numpy()

    ###############################################################################################
    #### Misc
    ###############################################################################################

    # Print the model parameters
    def print_params(self):
        print('Model Parameters')
        for name, param in self.named_parameters():
            print('  %-32s %-32s requires_grad=%s' % (name, param.size(), param.requires_grad))
        num_elements = sum(p.numel() for p in self.parameters())
        print('Total number of elements: {:,}'.format(num_elements))
