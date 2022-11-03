import re
import logging
from   multiprocessing import Pool
from   tqdm import tqdm
import numpy

logger = logging.getLogger(__name__)

# Network data sizes
EMBED_DIM   = 50
SIZE_SPAN   = 2    # number of span vectors (averaged word vectors), one for sent and one for doc
SIZE_WORD   = 5    # number of graph tokens indexes in the mention vector
SIZE_FS     = 215  # number of floats for features of single mentions
SIZE_FP_ANT = 158  # number of floats for features for pairs (ie.. main mention <-> antecedent)
SIZE_FP = SIZE_FP_ANT + SIZE_FS  # number of floats for the antecedents and mention combined

anaphor_types = set(['i', 'you', 'we', 'they', 'he', 'she', 'it', 'thing', 'this', 'that',
                     'other', 'one', 'here', 'there'])


# Class for turning mentino data into features
class CorefFeaturizer(object):
    re_stag = re.compile(r'-\d+$')     # detect or strip sense tags (ie.. -01) from strings
    def __init__(self, mdata, model):
        super().__init__()
        self.mdata           = mdata
        self.graph_vocab     = model.get_graph_vocab()
        self.graph_embed_mat = model.get_graph_embed_mat()
        self.graph_embed_dim = self.graph_embed_mat.shape[1]
        assert self.graph_embed_dim == EMBED_DIM
        self.cache = {}    # for sentence and document averaged vectors
        self.preload_cache()

    ###########################################################################
    #### Span vectors and word indexes
    ###########################################################################

    # Get the averaged vector, for all tokens in the mention's sentence
    # Cache these since they are highly repeated, to save from recomputing them.
    def get_sentence_span_vector(self, mention):
        # Get / cache the sentence and document span vectors so we don't have to keep creating them
        # This assumes that the sent_id / doc_names are unique, which is true for ms-amr data
        if mention.sent_id not in self.cache:
            vector = self.get_average_embedding(self.mdata.get_sentence_tokens(mention))
            self.cache[mention.sent_id] = vector
        return self.cache[mention.sent_id]

    # Get the averaged vector, for all tokens in the mention's document
    # Cache these since they are highly repeated, to save from recomputing them.
    def get_document_span_vector(self, mention):
        if mention.doc_name not in self.cache:
            vector = self.get_average_embedding(self.mdata.get_document_tokens(mention))
            self.cache[mention.doc_name] = vector
        return self.cache[mention.doc_name]

    # Get the graph token indexes for the mention and the ones surrounding it
    def get_word_indexes(self, mention):
        wids = []
        wid = self.graph_vocab.get_index(mention.token)
        wids.append(wid)
        for offset in (-1, +1, -2, +2):
            token = self.mdata.get_token_at_offest(mention, offset)
            wid   = self.graph_vocab.get_index(token)
            wids.append(wid)
        assert len(wids) == SIZE_WORD
        return numpy.array(wids, dtype='int64')


    ###########################################################################
    #### Single features
    ###########################################################################

    # Get the features for the single mention case
    def get_single_features(self, mention):
        # Create empty array to fill
        features = numpy.zeros(SIZE_FS, dtype='float32')
        # Get the features for types of words
        features[0] = 1 if mention.token in anaphor_types else 0
        features[1] = 1 if self.re_stag.search(mention.token) else 0
        features[2] = 1 if mention.token.endswith('-91') else 0
        start_idx = 3
        # Encode the first few sense tag numbers
        match = self.re_stag.search(mention.token)
        stag  = int(match[0][1:]) if match else 0
        start_idx = self.add_single_one_hots(features, stag, start_idx, 5, 6, 'linear')
        # Encode the number of instances of this token's name in the graph
        num_toks = len([t for t in self.mdata.var2token[mention.sent_id].values() if t == mention.token])
        start_idx = self.add_single_one_hots(features, num_toks, start_idx, 5, 6, 'linear')
        # Encode the index of the sentence in the document using one-hots
        # sent_idx:  mean=22  max=226  stdev=26.5  95%CI=75.1
        start_idx = self.add_single_one_hots(features, mention.sent_idx, start_idx, 99, 100, 'linear')
        # Encode the index of the token in the sentence using one-hots
        # tok_idx:  mean=33  max=619  stdev=38.2  95%CI=110.3
        start_idx = self.add_single_one_hots(features, mention.tok_idx, start_idx, 99, 100, 'linear')
        # Error check and return
        assert SIZE_FS == start_idx, '%d != %d' % (SIZE_FS, start_idx)
        return features

    # Encode an integer value, normalized by max_val into num_bins and add it to the features at start_idx
    # return the next starting index. This assumes the features array is already 0s for these features.
    def add_single_one_hots(self, features, value, start_idx, max_val, num_bins, oh_type):
        bin     = self.get_one_hot_bin(value, max_val, num_bins, oh_type)
        features[start_idx+bin] = 1.0
        return start_idx + num_bins

    ###########################################################################
    #### Pair features
    ###########################################################################

    # Get the pair features give a mention and list of antecedents
    def get_pair_features(self, mention, antecedents):
        if len(antecedents) == 0:
            return None
        num_p    = len(antecedents)
        features = numpy.zeros(shape=(num_p, SIZE_FP_ANT), dtype='float32')
        # Get T/F if graph tokens match exactly or match with the -0x removed
        for i, antecedent in enumerate(antecedents):
            features[i,0] = mention.token == antecedent.token
            features[i,1] = self.re_stag.sub('', mention.token) == self.re_stag.sub('', antecedent.token)
            # Features if mention is in the same graph
            features[i,2] = mention.sent_idx == antecedent.sent_idx
            features[i,3] = mention.sent_idx >  antecedent.sent_idx
            features[i,4] = mention.sent_idx <  antecedent.sent_idx
            # Features for variables / subgraphs
            features[i,5] = self.mdata.are_variables_the_same(mention, antecedent)
            features[i,6] = self.mdata.are_subgraphs_the_same(mention, antecedent)
            # Indicate if the pair candidate we're looking at is the single mention
            # This will always be false when defining antecents = mentions[:midx]
            features[i,7] = mention.mdata_idx ==  antecedent.mdata_idx
        start_idx = 8
        # Add the difference in sentence indexes between mention and antecedent as a one-hot
        # sidx_diff:  mean=5  max=10  stdev=3.4  95%CI=12.2
        values    = [abs(mention.sent_idx - a.sent_idx) for a in antecedents]
        start_idx = self.add_pair_one_hots(features, values, start_idx, 9, 10, 'linear')
        # Add the token from the single mention to the antecdent as a normalized one-hot vector
        # doc_idx_diff:  mean=1424  max=13568  stdev=1936.9  95%CI=5298.8
        doc_idx   = self.mdata.get_doc_tok_idx(mention)
        values    = [abs(doc_idx - self.mdata.get_doc_tok_idx(a)) for a in antecedents]
        start_idx = self.add_pair_one_hots(features, values, start_idx, 8000, 100, 'curve_a')
        # Add the normalized number of mentions between then single mention and antecedent
        # This is different than tokens, as only tokens that have variables in the graph are mentions
        # men_idx_diff  mean=215  max=1846  stdev=267.2  95%CI=749.7
        values = [abs(mention.mdata_idx - a.mdata_idx) for a in antecedents]
        start_idx = self.add_pair_one_hots(features, values, start_idx, 1000, 40, 'curve_a')
        assert SIZE_FP_ANT == start_idx, '%d != %d' % (SIZE_FP_ANT, start_idx)
        return features

    # Encode an integer value, normalized by max_val into num_bins and add it to the features at start_idx
    # return the next starting index.  This assumes the features array is already 0s for these features.
    def add_pair_one_hots(self, features, values, start_idx, max_val, num_bins, oh_type):
        assert len(values) == features.shape[0]
        bins = [self.get_one_hot_bin(val, max_val, num_bins, oh_type) for val in values]
        for i, bin in enumerate(bins):
            features[i, start_idx+bin] = 1.0
        return start_idx + num_bins

    ###########################################################################
    #### Build targets
    ###########################################################################

    # label data is (P+1,)
    # For the single (aka head) mention the label is defined as the probability of the mention having
    #   no antecedents, ie.. it is not in a cluster, or none of the antecedent match it's cluster id
    # For each antecedent/head-mention pair, the label is 1 if they have a cluster id and in the same one
    def build_targets(self, mention, antecedents):
        # If the mention is not in a cluster, then it has no antecedents
        if not mention.cluster_id:  # None or ''
            single_labels = [1]
            pairs_labels  = [0] * len(antecedents)
        else:
            pairs_labels   = [1 if ant.cluster_id==mention.cluster_id else 0 for ant in antecedents]
            no_antecedents = not any(pairs_labels)
            single_labels  = [1 if no_antecedents else 0]
        # Convert to numpy and return
        single_labels = numpy.array(single_labels, dtype='float32')
        if len(pairs_labels) > 0:
            pairs_labels = numpy.array(pairs_labels, dtype='float32')
        else:
            pairs_labels = None
        return single_labels, pairs_labels

    ###########################################################################
    #### Misc Functions
    ###########################################################################

    # Convert an integer to a one hot from 0 to max_val over num_bins.
    # linear is a linear distribution for 0 to val. For all integer values, use num_bins = max_val+1
    # curve_a is a non-linear (arc'd) distribution so that max_val/2 is in bin num_bins-1
    # Return the bin that is "hot" (1).
    @staticmethod
    def get_one_hot_bin(val, max_val, num_bins, oh_type):
        if oh_type == 'linear':
            norm = numpy.clip((val/max_val), 0.0, 1.0)
            bin  = int(numpy.round(norm * (num_bins-1)))
        elif oh_type == 'curve_a':
            a = (max_val/num_bins)
            y = 1 - a/(a + val)
            bin = int(round(num_bins*y))
            bin = numpy.clip(bin, 0, num_bins-1) # clip returns from min, up to and including, max
            # Prevent low values from spanning multiple bins
            if bin > val:
                bin = int(numpy.ceil(val))
        return bin

    # Get the averaged token embeddings for a list of tokens
    def get_average_embedding(self, tokens):
        vector = numpy.zeros(self.graph_embed_dim, dtype='float32')
        for token in tokens:
            index = self.graph_vocab.get_index(token)
            vector += self.graph_embed_mat[index]
        return vector / len(tokens)

    # Loop through the training data and call get__x_span_vector() to calculate and
    # cache the sentence and document vectors
    def preload_cache(self):
        for doc_name, mentions in self.mdata.mentions.items():
            for mention in mentions:
                if mention.doc_name not in self.cache or mention.sent_id not in self.cache:
                    self.get_sentence_span_vector(mention)
                    self.get_document_span_vector(mention)


###############################################################################
#### Build the single data and the 2D matrix of head-mention -> antecedent pair
###############################################################################
gfeaturizer, gmax_dist = None, None    # for multiprocessing
def build_coref_features(mdata, model, **kwargs):
    chunksize = kwargs.get('feat_chunksize',          200)
    maxtpc    = kwargs.get('feat_maxtasksperchild',   200)
    processes = kwargs.get('feat_processes',         None)    # None = use os.cpu_count()
    show_prog = kwargs.get('show_prog',              True)
    global gfeaturizer, gmax_dist
    gfeaturizer = CorefFeaturizer(mdata, model)
    gmax_dist   = model.config.max_dist if model.config.max_dist is not None else 999999999
    # Build the list of doc_names and mention indexes for multiprocessing and the output container
    idx_keys = [(dn, idx) for dn, mlist in gfeaturizer.mdata.mentions.items() for idx in range(len(mlist))]
    feat_data = {}
    for dn, mlist in gfeaturizer.mdata.mentions.items():
        feat_data[dn] = [None]*len(mlist)
    # Loop through and get the pair features for all antecedents
    pbar = tqdm(total=len(idx_keys), ncols=100, disable=not show_prog)
    with Pool(processes=processes, maxtasksperchild=maxtpc) as pool:
        for fdata in pool.imap_unordered(worker, idx_keys, chunksize=chunksize):
            dn, midx, sspans, dspans, words, sfeats, pfeats, slabels, plabels = fdata
            feat_data[dn][midx] = {'sspans':sspans,   'dspans':dspans, 'words':words,
                                   'sfeats':sfeats,   'pfeats':pfeats,
                                   'slabels':slabels, 'plabels':plabels}
            pbar.update(1)
    pbar.close()
    # Error check
    for dn, feat_list in feat_data.items():
        assert None not in feat_list
    return feat_data


def worker(idx_key):
    global gfeaturizer, gmax_dist, gfrozen_embeds
    doc_name, midx = idx_key
    mlist       = gfeaturizer.mdata.mentions[doc_name]
    mention     = mlist[midx]               # the head mention
    antecedents = mlist[:midx]              # all antecedents up to (not including) head mention
    antecedents = antecedents[-gmax_dist:]  # truncate earlier value so list is only max_dist long
    # Process the single and pair data data
    sspan_vector = gfeaturizer.get_sentence_span_vector(mention)
    dspan_vector = gfeaturizer.get_document_span_vector(mention)
    word_indexes = gfeaturizer.get_word_indexes(mention)
    sfeats       = gfeaturizer.get_single_features(mention)
    pfeats       = gfeaturizer.get_pair_features(mention, antecedents)
    # Build target labels.  Note that if there are no clusters in the mention data this will still
    # return a list of targets, though all singles will be 1 and pairs 0
    slabels, plabels = gfeaturizer.build_targets(mention, antecedents)
    return doc_name, midx, sspan_vector, dspan_vector, word_indexes, sfeats, pfeats, slabels, plabels
