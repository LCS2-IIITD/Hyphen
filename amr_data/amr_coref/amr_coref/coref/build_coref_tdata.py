import re
import os
import logging
from   copy import deepcopy
from   tqdm import tqdm
import penman
from   .multi_sentence_amr import MSAMRFiles, MSAMR
from   .penman_serializer import PenmanSerializer
from   .penman_multisentence import is_multi_sentence, split_multi_sentence
from   ..utils.data_utils import dump_json

logger = logging.getLogger(__name__)


def build_coref_tdata(amr3_dir, is_train):
    # Get the cluster data and graph ids by doc_name and the penman graphs by graph id
    cdata_dict, gids_dict, pgraph_dict = get_amr3_ms_graphs(amr3_dir, is_train)
    # Get the serialized penman graphs and associated data
    gdata_dict = get_serialized_graph_data(pgraph_dict)
    # combine everything and return the data
    all_data = {'clusters':cdata_dict, 'doc_gids':gids_dict, 'gdata':gdata_dict}
    return all_data


# Loop through all the multi-sentence files and load the associated graphs
def get_amr3_ms_graphs(amr3_dir, is_train):
    cdata_dict, gids_dict, pgraph_dict = {}, {}, {}
    msfns = MSAMRFiles(amr3_dir, is_train=is_train)
    for i in tqdm(range(len(msfns)), ncols=100, leave=False, desc='Loading'):
        doc_name  = msfns.get_test_name(i)
        ms_fpath  = msfns.get_ms_fpath(i)
        amr_fpath = msfns.get_amr_fpath(i)
        msdoc     = MSAMR(ms_fpath)
        # For now just look at the ident chains and the mentions in them
        cluster_out = {}
        for cluster_id, cluster_data in  msdoc.ident_chain_dict.items():
            mentions =  [cd for cd in cluster_data if cd['type'] == 'mention']
            if len(mentions) < 2:
                continue
            cluster_out[cluster_id] = mentions
        # Load the associated AMR3 aligned graphs in the multi-sentence XML files as an ordered dict
        # of IDs to graphs.  Graph IDs are unique across documents so condense them to a single list.
        pgraph_doc = msdoc.load_amrs(amr_fpath)
        pgraph_dict.update(pgraph_doc)
        # Accumulate the cluster data for the document
        assert doc_name not in cdata_dict
        gids_dict[doc_name]  = list(pgraph_doc.keys())
        cdata_dict[doc_name] = cluster_out
    return cdata_dict, gids_dict, pgraph_dict


# Get the serialized graph data for all graphs
def get_serialized_graph_data(pgraph_dict):
    gdata_dict = {}
    for gid, pgraph in pgraph_dict.items():
        # There are some graphs (100+) in multiple documents, though the ids are unique
        # in the corpus. I'm not sure if these are errors or if documents should overlap.
        # In AMR3 the graph(aka sentence) ids are unique so there's no reason to duplicate
        # them. They can be top-level entries.  No need to keep them under the doc_names.
        if gid in gdata_dict:
            logger.warning('Duplicate gid: %s' % (gid))
        serializer  = PenmanSerializer(pgraph)
        sgraph      = serializer.get_graph_string()
        vstring     = serializer.get_variables_string()
        var2concept = serializer.get_var_to_concept()
        # Error check here verify serializer operation
        # Note that loading penman graphs without using the NoOpModel() could cause
        # some of these issues.
        sg_toks     = sgraph.split()
        ve_toks     = vstring.split()
        assert len(ve_toks) == len(sg_toks), '%s : %d != %d' % (gid, len(ve_toks), len(sg_toks))
        v_set_a = set([v for v in ve_toks if v != '_'])
        v_set_b = set(var2concept.keys())
        assert v_set_a == v_set_b, '%s : %s / %s' % (gid, str(v_set_a), str(v_set_b))
        gdata_dict[gid] = {'sgraph':sgraph, 'variables':vstring, 'var2concept':var2concept}
        # check if the graph itself is "multi-sentence" and if so split into pieces
        if is_multi_sentence(pgraph):
            subgraphs    = split_multi_sentence(pgraph)
            sub_gstrings = [pgraph_to_gstring(pg) for pg in subgraphs]
            gdata_dict[gid]['sg_vars']   = [' '.join(sorted(g.variables())) for g in subgraphs]
    return gdata_dict


# Convert a penman graph to a one line string (no metadata)
def pgraph_to_gstring(pgraph):
    pgraph  = deepcopy(pgraph)
    pgraph.metadata = {}
    gstring = penman.encode(pgraph, indent=0)
    gstring = gstring.replace('\n', ' ')
    gstring = re.sub(' +', ' ', gstring)
    return gstring
