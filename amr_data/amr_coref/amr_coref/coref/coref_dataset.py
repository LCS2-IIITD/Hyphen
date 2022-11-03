import logging
import numpy
from   torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# A Torch Dataset used for feeding data to the training routine via the DataLoader
# Returns a dict of a single training sample.
class CorefDataset(Dataset):
    def __init__(self, feat_data, mention_data, costs, max_dist=None):
        super().__init__()
        self.fdata    = feat_data
        self.mdata    = mention_data
        self.costs    = costs
        self.max_dist = max_dist if max_dist is not None else 999999999

    def __len__(self):
        return len(self.mdata)

    def __getitem__(self, ds_idx):
        # Ge the document name and mention index for a given dataset index
        doc_name, midx = self.mdata.idx2_dn_midx[ds_idx]
        fdict = self.fdata[doc_name]
        # Copy over the single features for this data sample
        odict = {'mdata_indexes': ds_idx}
        odict.update(self.get_single(midx, fdict))
        # Copy over the pair data (head mention --> antecedent)
        # the antecedent indexes are all the values before mention index
        if midx == 0:
            odict['has_pairs'] = False
        else:
            odict.update(self.get_pairs(midx, fdict))
            odict['has_pairs'] = True
        # Get the target data
        odict['single_labels'] = fdict[midx]['slabels']     # list with 1 entry (at this point)
        if fdict[midx]['plabels'] is not None:
            odict['pairs_labels'] = fdict[midx]['plabels'][-self.max_dist:]    # truncate for max_dist
            odict['false_ants']  = self.get_false_ants(odict['single_labels'], odict['pairs_labels'])
            # Fix the single label for truncated pairs. We might have removed the only antecedent(s) that
            # are part of the mention's cluster and so might need to change the single label
            # For the single (aka head mention) the label is 1 if there no antecedents (no non-zero labels)
            if numpy.count_nonzero(odict['pairs_labels']) == 0:
                odict['single_labels'][0] = 1
        # Get some data build on top of the target labels
        odict['costs']     = self.get_costs(     odict['single_labels'], odict.get('pairs_labels', None))
        odict['true_ants'] = self.get_true_ants( odict['single_labels'], odict.get('pairs_labels', None))
        return odict

    # Transfer the head mention's data (mention at midx) to the keys for the single network input
    def get_single(self, midx, fdict):
        return {'sspans': fdict[midx]['sspans'], 'dspans': fdict[midx]['dspans'],
                'words':fdict[midx]['words'], 'single_features':fdict[midx]['sfeats']}

    # Build the pairs network input data by extracting each antecedent's mention data and arraying
    # the head mention's (aka anaphor) data across the spans too
    def get_pairs(self, midx, fdict):
        start = 0 if midx <= self.max_dist else midx - self.max_dist
        ant_indexes = range(start, midx)    # truncate for max_dist
        # There are up to "P" antecedents for a given mention
        # "P" varies in size for each training data element and will be padded in dataloader
        # a_xspans(P, 1) a_words(P, W) features(P, Fp)
        num_p        = len(ant_indexes)
        ant_sspans   = numpy.array([fdict[aidx]['sspans']  for aidx in ant_indexes])
        ant_dspans   = numpy.array([fdict[aidx]['dspans']  for aidx in ant_indexes])
        ant_words    = numpy.array([fdict[aidx]['words']   for aidx in ant_indexes])
        ant_features = fdict[midx]['pfeats'][-num_p:,:]     # truncate for max_dist
        # Anaphora.  These are the single (head mention) spans and words arrays, repeated "P" times (number of pairs)
        ana_sspans   = numpy.tile(fdict[midx]['sspans'],   (num_p, 1))
        ana_dspans   = numpy.tile(fdict[midx]['dspans'],   (num_p, 1))
        ana_words    = numpy.tile(fdict[midx]['words'],   (num_p, 1))
        ana_features = numpy.tile(fdict[midx]['sfeats'],  (num_p, 1))
        # The pair features is a concatenation of all the antecedent features and the repeated anphor features
        pair_features = numpy.concatenate((ant_features, ana_features), axis=1)
        # Add to the dictionary
        return {'ant_sspans':ant_sspans, 'ant_dspans':ant_dspans, 'ant_words':ant_words,
                'ana_sspans':ana_sspans, 'ana_dspans':ana_dspans, 'ana_words':ana_words,
                'pair_features':pair_features}

    # Build the costs array (here's how I interpret these, but I could be wrong)
    # FN=> false new (single, 1 == no ants) cost applied to single prediction if labels says there are antecedents
    # FL=> false link (pairs) cost for (1-pair label) if single labels says there are antecedents
    # WL=> wrong link (pairs) cost for pairs label if single label says no antecedents
    def get_costs(self, label, pairs_labels):
        single_label = label[0]    # list with one value
        if pairs_labels is None:
            costs = numpy.array([(1 - single_label) * self.costs['FN']], dtype='float32')
        else:
            if single_label == 0:
                costs = numpy.concatenate(([self.costs['FN']], self.costs['WL'] * (1 - pairs_labels)))
            else:
                costs = numpy.concatenate(([0], self.costs['FL'] * numpy.ones_like(pairs_labels)))
            assert costs.shape == (pairs_labels.shape[0] + 1,)
        return costs.astype('float32')

    # Get the indexes of the single/antecedents who's labels are 1 (True)
    def get_true_ants(self, label, pairs_labels):
        if pairs_labels is None:
            # set zero as the index of the true_ant, indicating the head mention (ie the single mention)
            true_ants = numpy.array([0], dtype='float32')
        else:
            labels_stack = numpy.concatenate((label, pairs_labels), axis=0)
            assert labels_stack.shape == (pairs_labels.shape[0] + 1,)
            true_ants_unpad = numpy.flatnonzero(labels_stack)   # all non-zero indexes (flattened)
            assert len(true_ants_unpad) != 0  # the head mention (single) will be 1 with no antecedents
            true_ants = numpy.pad(true_ants_unpad, (0, len(pairs_labels) + 1 - len(true_ants_unpad)), "edge")
            assert true_ants.shape == (pairs_labels.shape[0] + 1,)
        return true_ants.astype('int64')

    # Get the indexes of the antecedent's who's labels are 0 (False)
    # There are no false ants when their are no pairs so this array is invalid (None) under those conditions
    def get_false_ants(self, label, pairs_labels):
        assert pairs_labels is not None
        labels_stack = numpy.concatenate((label, pairs_labels), axis=0)
        assert labels_stack.shape == (pairs_labels.shape[0] + 1,)
        false_ants_unpad = numpy.flatnonzero(1 - labels_stack)
        assert len(false_ants_unpad) != 0  # the head mention must be zero if none of the antecedents are
        false_ants = numpy.pad(false_ants_unpad, (0, len(pairs_labels) + 1 - len(false_ants_unpad)), "edge")
        assert false_ants.shape == (pairs_labels.shape[0] + 1,)
        return false_ants.astype('int64')
