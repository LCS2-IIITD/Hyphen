import os
import json
import logging

logger = logging.getLogger(__name__)


# Convert cluster dictionary to scorch data format
# Clustering is based on the graph's variable and because of re-entracy, there can be multiple
# instances in the serialized graph.  Get rid of these by saving as a set of sent_idx/variable
def save_sdata(directory, doc_name, cluster_dict):
    clusters_sfmt = {}
    for cid, mlist in cluster_dict.items():
        clusters_sfmt[cid] = sorted(set(['%d.%s' % (m.sent_idx, m.variable) for m in mlist]))
    # Single cluster can cause erroneously high CoNLL scores.  Previous logic should have removed
    # them but error check in case it hasn't and remove them if present.
    for rel in list(clusters_sfmt.keys()):
        cluster_set = clusters_sfmt[rel]
        if len(cluster_set) < 2:
            del clusters_sfmt[rel]
            logger.error('Removing: invalid cluster for %s/%s/%s : %s' % (directory, doc_name, rel, str(cluster_set)))
    sdata = {'name':doc_name, 'type':'clusters', 'clusters':clusters_sfmt}
    fpath = os.path.join(directory, doc_name + '.json')
    with open(fpath, 'w') as f:
        json.dump(sdata, f, indent=4)


# Take in the gold mention data and the predicted single and pair probs,
# process and write the results to results_dir
def cluster_and_save_sdata(mdata, s_probs, p_probs, results_dir, greedyness=0.0):
    # Setup dictories
    gold_dir   = os.path.join(results_dir, 'gold')
    pred_dir   = os.path.join(results_dir, 'pred')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(gold_dir,    exist_ok=True)
    os.makedirs(pred_dir,    exist_ok=True)
    # Remove any previous files
    for dirname in [gold_dir, pred_dir]:
        for fn in os.listdir(dirname):
            os.remove(os.path.join(dirname, fn))
    # Collate the mention data and model probabilities
    all_data = collate_mdata(mdata, s_probs, p_probs)
    # Loop through each document
    cluster_dicts = []
    for doc_name, doc_data in all_data.items():
        gold_cluster_dict = mdata.get_clusters(doc_name)
        save_sdata(gold_dir, doc_name, gold_cluster_dict)
        # Get and save prediction data
        pred_cluster_dict = build_clusters(doc_data, greedyness)
        save_sdata(pred_dir, doc_name, pred_cluster_dict)
        cluster_dicts.append({'doc_name':doc_name, 'gold':gold_cluster_dict, 'pred':pred_cluster_dict})
    return gold_dir, pred_dir, cluster_dicts


# Convert the predicted probabilities into a list of clusters
def get_predicted_clusters(mdata, s_probs, p_probs, greedyness=0.0):
    cluster_dicts = []
    all_data = collate_mdata(mdata, s_probs, p_probs)
    for doc_name, doc_data in all_data.items():
        pred_cluster_dict = build_clusters(doc_data, greedyness)
    cluster_dicts.append({'doc_name':doc_name, 'pred':pred_cluster_dict})
    return cluster_dicts


# Probabilities from the model are indexed by CorefMention index
# Separate these into documents and get the original mentions from that class
def collate_mdata(mdata, s_probs, p_probs):
    # Get the data indexes.  Should be a list 0 to N. This is probably not required but
    # it's expected, and there might be something wrong if the assert below fail.
    mdata_indexes = sorted(s_probs.keys())
    assert mdata_indexes[0]  == 0
    assert mdata_indexes[-1] == len(mdata_indexes)-1
    # Group the model prediction data into documents
    all_data = {}
    for mdata_index in mdata_indexes:
        doc_name, midx = mdata.get_dn_and_midx(mdata_index)
        # Create the dict and grab all the mentions when we get a new document name
        # We assume no order to the indexing
        if doc_name not in all_data:
            all_data[doc_name] = {}
            all_data[doc_name]['mentions'] = mdata.mentions[doc_name]
            all_data[doc_name]['s_probs']  = [None] * len(mdata.mentions[doc_name])
            all_data[doc_name]['p_probs']  = [None] * len(mdata.mentions[doc_name])
        # Get the single probs.  This is a list of single values
        all_data[doc_name]['s_probs'][midx] = s_probs[mdata_index]
        # Get the pair probs. These are list from every pair up the mention index
        # At the first mention index there aren't any pairs
        all_data[doc_name]['p_probs'][midx] = p_probs.get(mdata_index, None)
    return all_data


# Convert the document data (formatted in collate_mdata) into a set of clusters
def build_clusters(doc_data, greedyness):
    mentions = doc_data['mentions']
    num_mentions = len(mentions)
    # Compute scores
    best_scores = [None] * num_mentions
    best_ants   = [None] * num_mentions
    # Loop through all mentions
    for i in range(num_mentions):
        # Get the score for mention has no antecedents
        best_scores[i] = doc_data['s_probs'][i] - greedyness
        best_ants[i]   = i      # no antecedent
        # Loop through all possible antecedents and get the pair score
        # There is no antedents for the first item in the list
        if i==0: continue
        # doc_data['mentions'] comes from coref_mention_data and this will include all mentions in the
        # in sentence. However, doc_data['p_probs'] comes from the model and could have the antecents
        # list truncated for ones that are too far away (See max_dist in coref_dataset.py).  Compute
        # the offset and create an array padded at the beginning, with 0 for the uncalculated probs.
        ant_offset = i - len(doc_data['p_probs'][i])
        pair_probs = [0]*ant_offset + list(doc_data['p_probs'][i])
        for j in range(i):
            p_prob = pair_probs[j]
            if p_prob > best_scores[i]:
                best_scores[i] = p_prob
                best_ants[i]   = j
    # Build into clusters by looping though the list of best anaphor->antecent indexes
    # and chain matching indexes together.
    # Loop through a list of pairs(mention idx, antecedent idx), skipping any where the
    # indexes match (this means the best antecedent is the anaphor, aka no antecedents)
    cluster_sets = []
    for midx, aidx in [(midx, aidx) for midx, aidx in enumerate(best_ants) if midx != aidx]:
        found = False
        # See if either index is in any of the existing clusters and if so, add the
        # add the other
        for cluster_set in cluster_sets:
            if midx in cluster_set or aidx in cluster_set:
                cluster_set.update([midx, aidx])
                found = True
                break
        # If we didn't find anything, add a new cluster for both indexes
        if not found:
            cluster_sets.append(set([midx, aidx]))
    # Filter out singles since these aren't valid
    cluster_sets = [s for s in cluster_sets if len(s) > 1]
    # Now convert the list of sets with indexes to a dictionary with lists of mentions
    # Up to this point clusters are by mention, however the final clusters are a set based on
    # sent_idx and variable.  Since a variable can appear more than one time in a graph, weed out
    # redudant values. Also check for clusters with only 1 item. These cause erroneously high
    # CoNLL scores and can appear when weeding out redundant variables.
    pred_clusters = {}
    for cluster_set in cluster_sets:
        # Get sets of relations with more than 1 relation
        mentions = [doc_data['mentions'][idx] for idx in cluster_set]
        relation_set = set(['%d.%s' % (m.sent_idx, m.variable) for m in mentions])
        if len(relation_set) < 2:
            continue
        key = 'rel-%d' % len(pred_clusters)
        pred_clusters[key] = sorted(mentions)
    return pred_clusters
