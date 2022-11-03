import os
import uuid
import penman
from   penman.models.noop import NoOpModel
from   .amr_coref_model import AMRCorefModel
from   .coref_data_loader import get_data_loader_from_data
from   .build_coref_tdata import get_serialized_graph_data
from   .clustering import get_predicted_clusters


class Inference(object):
    def __init__(self, model_dir, amr_list,show_prog=False, greedyness=0.0, **kwargs):
        self.model        = AMRCorefModel.from_pretrained(model_dir)
        # overide max_dist is in kwargs
        if 'max_dist' in kwargs:
            self.model.config.max_dist = kwargs['max_dist']
        self.config        = self.model.config
        self.mdata         = None
        self.show_prog     = show_prog
        self.greedyness    = greedyness
        self.cluster_dicts = {}     # saved for debug
        self.amr_list = amr_list
        self.amr_dict = {}

    # Coreference graph strings or penman graphs
    # !!! Note that if loading penman graphs, they must have been encoded using the NoOpModel
    def coreference(self, doc_graphs, doc_name = 'doc_001'):
        # Convert to penman graphs if needed
        assert len(doc_graphs) > 0
        if isinstance(doc_graphs[0], penman.Graph):
            pgraphs = doc_graphs
        elif isinstance(doc_graphs[0], str):
            pgraphs = [penman.decode(gstring, model=NoOpModel()) for gstring in doc_graphs]
        else:
            raise ValueError('Invalid input type of %s' % type(doc_graphs[0]))
        # Create document ids and covert to a dictionary
        doc_gids, pgraph_dict = {doc_name:[]}, {}
        lv = 0
        for pgraph in pgraphs:
            gid    = str(uuid.uuid4())
            doc_gids[doc_name].append(gid)
            pgraph_dict[gid] = pgraph
            self.amr_dict[gid] = self.amr_list[lv]
            lv+=1
        ret_clusters = {}

        # Serialize the graphs and extract relevant data
        gdata_dict = get_serialized_graph_data(pgraph_dict)
        # Add empty clusters to mention_data
        clusters = {doc_name:{}}
        # combine everything and save to a temporary file
        tdata_dict = {'clusters':clusters, 'doc_gids':doc_gids, 'gdata':gdata_dict}
        # Create the data loader
        self.test_dloader = get_data_loader_from_data(tdata_dict, self.model, show_prog=self.show_prog, shuffle=False)
        self.mdata        = self.test_dloader.dataset.mdata
        # Run the model and cluster the data
        results = self.model.process(self.test_dloader, self.show_prog)
        cluster_dicts = get_predicted_clusters(self.mdata, results['s_probs'], results['p_probs'], self.greedyness)
        # Class only does one document at a time so return the first (and only) cluster_dict
        assert len(cluster_dicts) == 1
        # Save for debug, etc..
        self.cluster_dicts[doc_name] = cluster_dicts[0]['pred']
        # Variables may appear multiple times in a sentence so there will be multiple mentions for
        # them. Only keep one instance of the sent_idx.variable.
        for relation, mentions in cluster_dicts[0]['pred'].items():
            cid_set = set((m.sent_idx, m.variable) for m in mentions)
            ret_clusters[relation] = sorted(cid_set)
        
        print(self.amr_dict[gid])

        return ret_clusters