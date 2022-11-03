import logging
import functools

logger = logging.getLogger(__name__)


# Class to build and hold a list of clusters
# cr_data is a dictionary with keys doc_gids, gdata and clusters (for training data only)
class CorefMentionData(object):
    def __init__(self, cr_data, mention_set):
        self.mention_set  = mention_set
        self.doc_sids     = cr_data['doc_gids']   # [doc_name] = [sent_id, sent_id,..] (ordered list)
        self.gtokens      = {}   # [sent_id] tokenized, serialized penman graph of concepts/edges
        self.gvars        = {}   # [sent_id] tokenized, serialized penman graph of variables (only)
        self.var2token    = {}   # [sent_id] variable to concept dictionary
        self.sg_vars      = {}   # [sent_id] list of sets of variables in each subgraph
        self.mentions     = {}   # [doc_name] = list of sorted Mention objects
        self.sent_lens    = {}   # [doc_name] = list of sentence lengths (sorted by sentence index)
        self._build_mdata(cr_data['gdata'])
        self.has_clusters = self._add_cluster_ids(cr_data.get('clusters', {}))
        # This defines an indexing system for the data used throughout the code.
        # Here mdata[idx] represents a unique document / mention index for the data.
        self.idx2_dn_midx = [(d, i) for d in sorted(self.mentions.keys()) for i in range(len(self.mentions[d]))]
        self.iter_index   = 0       # used in __getitem__() method below

    #######################################################################
    #### Public methods                                                ####
    #######################################################################

    # Get the token for the mention
    def get_token(self, mention):
        return self.gtokens[mention.sent_id][mention.tok_idx]

    # Get the token at a offset from the mention
    def get_token_at_offest(self, mention, offset):
        idx = mention.tok_idx + offset
        if idx < 0 or idx >= len(self.gtokens[mention.sent_id])-1:
            return None     # Will get turned into an <none> token
        return self.gtokens[mention.sent_id][idx]

    # Get the position of the mention in the document
    def get_doc_tok_idx(self, mention):
        prior_slens = sum(self.sent_lens[mention.doc_name][:mention.sent_idx])
        return prior_slens + mention.tok_idx

    # Get all tokens for the sentence the mention is in
    def get_sentence_tokens(self, mention):
        return self.gtokens[mention.sent_id]

    # Get all tokens for the document the mention is in
    def get_document_tokens(self, mention):
        tokens = []
        for sent_id in self.doc_sids[mention.doc_name]:
            tokens += self.gtokens[sent_id]
        return tokens

    # Get a dictionary of all the clusters for a given document
    def get_clusters(self, doc_name):
        mdict = {}
        for m in self.mentions[doc_name]:
            if m.cluster_id is None: continue
            mdict[m.cluster_id] = mdict.get(m.cluster_id, []) + [m]
        return mdict

    # Get the document name and mention index for an index in the dataset
    def get_dn_and_midx(self, mdata_index):
        doc_name, midx = self.idx2_dn_midx[mdata_index]
        return doc_name, midx

    # Check if variables are the same when in the same graph.  This happens because of referenced
    # (aka re-entrant) variables.  Variable names are only unqiue within a graph so, by definition,
    # they're not equal if they're not in the same graph.
    def are_variables_the_same(self, men1, men2):
        if men1.sent_idx != men2.sent_idx:
            return False
        return men1.variable == men2.variable

    # Check if the 2 mentions variables are in the same subgraph (multi-sentence graphs split)
    # By definition, if graphs are not the same, subgraphs are not.
    # Note that the naming is not quite accurate.  Generally variables are not shared (reference)
    # across subgraphs. However, there are a few instances where they are, so using this method you
    # can't 100% gaurentee the subgraphs are the same but this doesn't require alignment between
    # graphs and subgraphs (so it's simple) and as a feature, should provide good information.
    def are_subgraphs_the_same(self, men1, men2):
        if men1.sent_idx != men2.sent_idx:
            return False
        # If in the same graph, and there are no subgraphs, then subgraphs are the same
        var_sets = self.sg_vars[men1.sent_id]
        if len(var_sets) == 0:
            return True
        # Loop through the subgraphs to see if both variables share a subgraph
        for v1, v2 in zip(men1.variable, men2.variable):
            for var_set in var_sets:
                if v1 in var_set and v2 in var_set:
                    return True
        return False

    #######################################################################
    #### Private methods                                               ####
    #######################################################################

    # Build the mentions from "graph data"
    def _build_mdata(self, gdata_dict):
        # separate out and tokenize the graph tokens and variables
        for sent_id, gdata in gdata_dict.items():
            self.gtokens[sent_id]   = gdata['sgraph'].split()
            self.gvars[sent_id]     = [v if v != '_' else None for v in gdata['variables'].split()]
            assert len(self.gtokens[sent_id]) == len(self.gvars[sent_id])
            self.var2token[sent_id] = gdata['var2concept']
            self.sg_vars[sent_id]   = [set(vstr.split()) for vstr in gdata.get('sg_vars', [])]
        # Compute sentence lengths (used for computing document token index differences)
        for doc_name, sent_ids in self.doc_sids.items():
            self.sent_lens[doc_name] = [len(self.gtokens[sent_id]) for sent_id in sent_ids]
        # Add a mention object for any item that fits the criteria which is..
        # 1 - Any node in the graph (the token for every variable) and..
        # 2 - Only tokens on the self.mention_set are allowed (if the set is not None)
        # ie.. no edges, attributes or parens, just concept node tokens in the mention_set
        for doc_name, sent_ids in self.doc_sids.items():
            self.mentions[doc_name] = set()
            # Loop through all sentences and then the token index in each to add mentions
            # note that gvars and gtokens are list of identical lengths (None used for tokens with no vars)
            for sidx, sent_id in enumerate(sent_ids):
                for tidx, (variable, token) in enumerate(zip(self.gvars[sent_id], self.gtokens[sent_id])):
                    if (variable is not None) and (self.mention_set is None or token in self.mention_set):
                        mobj = Mention(doc_name, sent_id, token, variable, sidx, tidx, cluster_id=None)
                        self.mentions[doc_name].add(mobj)
            # Convert the set to a list sorted by the mention's appearence in the document
            self.mentions[doc_name] = sorted(self.mentions[doc_name])
            # Now go back and add the mentions index in the list to the mention object itself
            for i, mention in enumerate(self.mentions[doc_name]):
                mention.mdata_idx = i

    # Add cluster ids to existing mentions from the cluster dictionary
    def _add_cluster_ids(self, cluster_dict):
        has_clusters = False
        for doc_name, clusters in cluster_dict.items():
            # Build a lookup so we don't have to recurse the existing mentions list multiple times
            cluster_lu = {}     # cluster_lu[(sent_id, variable)] = cluster_id
            for cluster_id, mentions in clusters.items():
                for mention in mentions:
                    key = (mention['id'], mention['variable'])
                    # Error check that the concept/variable listed match what's in the lookup
                    assert self.var2token[mention['id']].get(mention['variable']) == mention['concept']
                    # There shouldn't be multiple entries but the amr-ms data has a few
                    if key in cluster_lu:
                        logger.warning('Duplicate mention in cluster data: %s' % str(key))
                    cluster_lu[key] = cluster_id
                    has_clusters = True
            # Loop through existing mentions and add cluster ids.  Because the same variable can
            # occur at multiple tokens indexes (due to references), a cluster_id may map to multiple
            # mentions. Keep track of which ones were used just to make sure we consumed them all.
            keys_used = set()
            for mention in self.mentions[doc_name]:
                key = (mention.sent_id, mention.variable)
                if  key in cluster_lu:
                    mention.cluster_id = cluster_lu[key]
                    keys_used.add(key)
            # We should have consumed all the cluster ids in the lookup.  If not, there's an issue.
            # It could be that the mention_set doesn't contain all the tokens in the gold clusters
            # and so the list of mentions is missing some needed values.
            assert keys_used == set(cluster_lu.keys())
        return has_clusters


    #######################################################################
    #### Class data access methods for iteration, etc..                ####
    #######################################################################

    # Return a list of mentions from the index back to the beginning.
    # This is done to facilitate use as training data and prevents accidentally
    # using list of mentions that crosses document boundaries.
    def __getitem__(self, index):
        doc_name, midx = self.idx2_dn_midx[index]
        return self.mentions[doc_name][:midx+1]

    def __len__(self):
        return len(self.idx2_dn_midx)

    def __iter__(self):
        self.iter_index = 0
        return self

    def __next__(self):
        if self.iter_index >= len(self):
            raise StopIteration
        mentions = self[self.iter_index]
        self.iter_index += 1
        return mentions


# Data containter for mentions
# This is to facilitates data accumulation, sorting and uniqueness of the mentions
@functools.total_ordering
class Mention(object):
    def __init__(self, doc_name, sent_id, token, variable, sent_index, token_index, cluster_id=None):
        self.mdata_idx  = None          # set later (index in mdata's mention list)
        self.doc_name   = doc_name      # name of the document
        self.sent_id    = sent_id       # string id for the sentence
        self.token      = token         # token in graph, only node or edge names
        self.variable   = variable      # variable for node (mentions are always nodes)
        self.sent_idx   = sent_index    # sentence index in the document
        self.tok_idx    = token_index   # token index in the sentence
        self.cluster_id = cluster_id    # string id for the cluster of mentions

    # This definition purposely only relies on the sentence and token indexes.
    # It doesn't gaurentee tokens, etc.. are the same.
    # The assumption (and enforced uniqueness) is that there is only one mention
    # for each location in the document.
    def __hash__(self):
        return hash((self.doc_name, self.sent_idx, self.tok_idx))

    # Equal defined same as hash above.  Comparision between documents is invalid.
    def __eq__(self, other):
        if self.doc_name != other.doc_name:
            raise NotImplementedError
        if self.sent_idx == other.sent_idx and self.tok_idx == other.tok_idx:
            return True
        else:
           return False

    # Used for sorting
    def __lt__(self, other):
        if self.doc_name != other.doc_name:
            raise NotImplementedError
        if self.sent_idx < other.sent_idx:
            return True
        elif self.sent_idx > other.sent_idx:
            return False
        else:
            return self.tok_idx < other.tok_idx

    def __str__(self):
        string = '%s : %s : sidx=%d  tidx=%d  tok=%s  var=%s  cid=%s' % \
            (self.doc_name, self.sent_id, self.sent_idx, self.tok_idx, self.token,
             self.variable, self.cluster_id)
        return string
