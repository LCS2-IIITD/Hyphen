import os
import torch
from   .amr_coref_model import AMRCorefModel
from   .coref_data_loader import get_data_loader_from_file
from   ..evaluate.pr_scorer import PRScorer


class Tester(object):
    def __init__(self, model, test_dloader, **kwargs):
        self.model        = model
        self.test_dloader = test_dloader
        self.config       = self.model.config
        self.mdata        = self.test_dloader.dataset.mdata
        self.show_prog    = kwargs.get('show_prog',  True)

    ###########################################################################
    #### Instantiation methods
    ###########################################################################

    # Load model from a file
    @classmethod
    def from_file(cls, model_dir, test_fn, **kwargs):
        model = AMRCorefModel.from_pretrained(model_dir)
        return cls.from_model(model, test_fn, **kwargs)

    # Setup the tester from a preloaded model
    @classmethod
    def from_model(cls, model, test_fn, **kwargs):
        # overide max_dist is in kwargs
        if 'max_dist' in kwargs:
            model.config.max_dist = kwargs['max_dist']
        test_dloader = get_data_loader_from_file(test_fn, model, **kwargs)
        return cls(model, test_dloader, **kwargs)

    ###########################################################################
    #### Test methods
    ###########################################################################

    # Run through all batches in the test data loader and accumulate by mention data index
    def run_test(self):
        results = self.model.process(self.test_dloader, self.show_prog)
        return results

    # Get the precision/recall score from the reulst dictionary, obtained in run_test() above
    def get_precision_recall_scores(self, results):
        single_scores = PRScorer()
        for s_label, s_prob in zip(results['s_labels'].values(), results['s_probs'].values()):
            single_scores.add_score(s_label, s_prob)    # add single value
        pair_scores = PRScorer()
        for p_labels, p_probs in zip(results['p_labels'].values(), results['p_probs'].values()):
            pair_scores.add_scores(p_labels, p_probs)   # add list of values
        return single_scores, pair_scores
