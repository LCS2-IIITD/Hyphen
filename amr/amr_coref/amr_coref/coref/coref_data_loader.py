import random
import numpy
import torch
from   torch.utils.data import DataLoader, Sampler
from   .coref_featurizer import build_coref_features
from   .coref_dataset import CorefDataset
from   .coref_mention_data import CorefMentionData
from   ..utils.data_utils import load_json


# get_data_loader_x() are the main functions for loading data for training, test and inference
#   cr_data is the json file / data-format created by build_coref_tdata()
#   The model is passed because it contains the config, vocab,.. which are needed for featurizing
# The class hierarchy can get confusing.  Here's how it works..
# CorefMentionData is the bottom level.  It creates a list of mentions from the data and also
#   controls the overall indexing by mapping documents / mentions to a single index value.
# CoRefFeaturizer takes the mention data and extracts features from it.  This is somewhat expensive
#   and is done via a multi-tasking function once prior to iterating over the data in training.
# CorefDataset is a torch Dataset that contains both the mention and feature data. When an index
#   is retrieved, it re-organizes the feature data into single and pair data required by the model.
#   This is done "online" while looping during training because pair data is a 2D matrix of mentions
#   by mentions and we don't want to (or need to) keep this huge matrix in RAM.
# The dataset is then wrapped into a torch DataLoader. Its purpose is to wrap up the batch
#   iteration, data collation, etc...  It contains a custom..
#   * CorefBatchSampler that controls how the batch samples are drawn. This is needed to group
#     batches together that are roughly the same length, saving training time.
#   * CoRefCollator handles collating data samples drawn but the data_loader.  Basically
#     this converts a list of dictionaries to a dictionary of lists.  A custom class is needed
#     here as the default Torch collator doesn't handle the variable length data samples.
# All of this revolves arround what torch calls a map-style dataset.  This means that the data
#   supports the random access __getitem__ method and not just sequential iteration (__iter__).
def get_data_loader_from_file(cr_data_fn, model, **kwargs):
    cr_data = load_json(cr_data_fn)
    return get_data_loader_from_data(cr_data, model, **kwargs)

def get_data_loader_from_data(cr_data, model, **kwargs):
    # Allow overriding shuffle for testing
    shuffle = kwargs.pop('shuffle') if 'shuffle' in kwargs else model.config.shuffle
    # Load the raw coreference data, featurize it and drop it into a custom torch Dataset
    mdata   = CorefMentionData(cr_data, model.mention_set)
    fdata   = build_coref_features(mdata, model, **kwargs)
    dataset = CorefDataset(fdata, mdata, model.config.costs, model.config.max_dist)
    # Extract relevant arguments for dataloader from kwargs. Then, create and return the object
    if 'num_workers' not in kwargs:
        kwargs['num_workers'] = model.config.num_workers
    dl_arg_list = ('num_workers', 'pin_memory', 'drop_last', 'timeout', 'prefetch_factor', 'persistent_workers')
    kwa_dl = {k:v for k, v in kwargs.items() if k in dl_arg_list}
    # pin_memory speeds memory transfer up but the default is False, so set it True if not specified
    if 'pin_memory' not in kwa_dl:
        kwa_dl['pin_memory'] = True
    # Create a custom Sampler and collator functions, then drop everything into the torch DataLoader
    sampler  = CorefBatchSampler(dataset, model.config.batch_size, shuffle)
    collator = CoRefCollator(model.config.all_pair_weights)
    dloader  = DataLoader(dataset, collate_fn=collator.collate_fn, batch_sampler=sampler, **kwa_dl)
    return dloader


# Combine the keys from a batched list of dictionaries into a single dictionary with lists for
# batch data and convert to torch.tensors.  This is custom because of the variable lengths of
# the antecedents and because I have some keys in here that are informational, and not lists.
class CoRefCollator(object):
    def __init__(self, all_pair_weights):
        self.all_pair_weights = all_pair_weights

    def collate_fn(self, batch):
        odict = {}
        # Remove batch level keys and error check these are all the same
        odict['has_pairs']   = self.check_and_get_same(batch, 'has_pairs')
        # Convert from a list of dictionaries to a dictionary of lists
        keys  = [k for k in batch[0].keys() if k not in ('has_pairs',)]
        for key in keys:
            odict[key] = [entry[key] for entry in batch]
        # Single mentions don't contain sequences so the don't need to be padded
        # Convert them from list to numpy arrays for simplicity keeping the data type
        for key in ('sspans', 'dspans', 'words', 'single_features'):
            odict[key] = numpy.array(odict[key])
        # The pair objects are variable length and must be padded so they're all the same length
        # First step is to get the pair lengths for each batch entry (all pair lengths are the same)
        # Not all batches will have antecedents so check for the key first
        max_pair_len = 0  # used below
        odict['num_singles'] = len(batch)     # every batch has a single element
        odict['num_elems']   = odict['num_singles']
        if odict['has_pairs']:
            # Keep pair_lens as a list so it isn't converted to a torch object below
            odict['pair_lens'] = [entry.shape[0] for entry in odict['ant_sspans']]
            odict['num_pairs'] = sum(odict['pair_lens'])
            odict['num_elems'] += odict['num_pairs']
            max_pair_len = max(odict['pair_lens'])
            # Next pad them to be max_pair_len
            for key in ('ant_sspans', 'ant_dspans', 'ant_words',
                        'ana_sspans', 'ana_dspans', 'ana_words', 'pair_features'):
                odict[key] = self.array_list_to_3D(odict[key], max_pair_len)
        # Convert targets
        # The single labels are all length 1 and don't get padded
        odict['single_labels'] = numpy.array(odict['single_labels'])
        odict['costs']         = self.array_list_to_2D(odict['costs'],     max_pair_len+1)
        odict['true_ants']     = self.array_list_to_2D(odict['true_ants'], max_pair_len+1)
        # Pad the pairs to the same length as the input pairs above
        if odict['has_pairs']:
            odict['pairs_labels'] = self.array_list_to_2D(odict['pairs_labels'], max_pair_len)
            odict['all_labels']   = numpy.concatenate([odict['single_labels'], odict['pairs_labels']], 1)
            # Create a mask for the padded values and pass it into the loss as weights
            odict['all_mask'] = numpy.zeros(shape=(len(batch), max_pair_len+1), dtype='float32')
            for i, pair_len in enumerate(odict['pair_lens']):
                odict['all_mask'][i,:1+pair_len] = 1     # single + pairs
            # False antecedents don't exist for arrays with pairs
            odict['false_ants'] = self.array_list_to_2D(odict['false_ants'], max_pair_len+1)
        else:
            odict['all_labels']  = numpy.array(odict['single_labels'], dtype='float32')
            odict['all_mask'] = numpy.ones(shape=(len(batch), 1),   dtype='float32')
        # Add weights to the binary mask (true_pair_loss uses this whether weighted or not)
        odict['all_weights'] = numpy.copy(odict['all_mask'])
        if self.all_pair_weights is not None:
            odict['all_weights'] = self.add_weights(odict['all_weights'], odict['all_labels'])
        # Convert everything that's in a numpy array to a torch tensor
        # Other items like lists (ie... pair_lens), ints or boolens are left that way
        for key in odict.keys():
            if isinstance(odict[key], numpy.ndarray):
                odict[key] = torch.from_numpy(odict[key])
        # Add in a few additional keys for easy use later
        odict['bshape_0'] = len(batch)
        odict['bshape_1'] = 1 + max_pair_len    # single + max pair length
        # Add back in the batch level keys (which are standard python types) and return
        return odict

    # Convert a list of numpy arrays to a padded numpy 2D array
    @staticmethod
    def array_list_to_2D(array_list, max_len):
        dtype = array_list[0].dtype
        new_array = numpy.zeros(shape=(len(array_list), max_len), dtype=dtype)
        for i, a in enumerate(array_list):
            new_array[i,:a.shape[0]] = a
        return new_array

    # Convert a list of numpy arrays to a padded numpy 3D array
    @staticmethod
    def array_list_to_3D(array_list, max_len):
        dtype = array_list[0].dtype
        dim3  = array_list[0].shape[1]
        new_array = numpy.zeros(shape=(len(array_list), max_len, dim3), dtype=dtype)
        for i, a in enumerate(array_list):
            new_array[i,:a.shape[0],:] = a
        return new_array

    # Simple function to assert that all values for a key in a list of dicts are the same
    # and return the consistant value.
    @staticmethod
    def check_and_get_same(batch, key):
        key_val = batch[0][key]
        for entry in batch:
            entry_val = entry[key]
            assert key_val == entry_val, 'Inconsistant %s != %s' % (str(key_val), str(entry_val))
        return key_val

    # Add weights to the weight array.  Up to this point the weight array is a mask of 1s and 0s
    # to mask out the pairs padded values.  Add a weight to the sample, based on the label.
    def add_weights(self, all_weights, all_labels):
        for i in range(all_weights.shape[0]):
            for j in range(all_weights.shape[1]):
                if all_labels[i,j] > 0.5:
                    weight = self.all_pair_weights['single_1'] if i == 0 else self.all_pair_weights['pair_1']
                    all_weights[i,j] *= weight
                else:
                    weight = self.all_pair_weights['single_0'] if i == 0 else self.all_pair_weights['pair_0']
                    all_weights[i,j] *= weight
        return all_weights


# This class is used by dataloader to get a list of indexes for each batch.
# It's custom because we want to group data into batches that are close to the same length.
# This minimizes the amount of padding required and cuts down on traininging. time.
# The class also prevents mixing entries with with no antecedents with those that have them,
# since the assumtion in the model is that your batch either has "pairs" with every batch entry
# or none of the entries have pairs.
class CorefBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle):
        super().__init__(dataset)
        self.mdata      = dataset.mdata
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.lengths    = []
        self.batches    = self.create_batches()
        random.shuffle(self.batches)
        self.iter_index = 0

    def create_batches(self):
        lengths = [(i, len(sample)) for i, sample in enumerate(self.mdata)]
        lengths = sorted(lengths, key=lambda x:x[1])    # sort by length
        # Extract the length 1 samples and simplfy to a list of indexes
        # length==1 are the mentions without antecedents and need to be kept separate.
        single_indexes = [l[0] for l in lengths if l[1]==1]
        pair_indexes   = [l[0] for l in lengths if l[1]>1]
        # Save batch lenghts for debug / info
        self.lengths = [l[1] for l in lengths]
        # Now chunk based on batch size
        single_batches = list(self.chunk(single_indexes, self.batch_size))
        pair_batches   = list(self.chunk(pair_indexes, self.batch_size))
        # Combine them
        return single_batches + pair_batches

    # Yield successive n-sized chunks from lst.
    @staticmethod
    def chunk(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        self.iter_index = 0
        if self.shuffle:
            random.shuffle(self.batches)
        return self

    def __next__(self):
        if self.iter_index >= len(self):
            raise StopIteration
        batch_indexes = self.batches[self.iter_index]
        self.iter_index += 1
        return batch_indexes
