import json
import glob
import amrlib
import pandas as pd
import penman
from penman import constant
from amrlib.graph_processing.annotator import add_lemmas
from amrlib.alignments.rbw_aligner import RBWAligner
from penman.models.noop import NoOpModel
import ast
import pickle
import os
import argparse
parser = argparse.ArgumentParser()

# Adding the required arguments
parser.add_argument('--dataset', choices = ['antivax', 'politifact', 'gossipcop', 'figlang_twitter', 'figlang_reddit', 'twitter16', 'rumoureval', 'pheme', 'twitter15', 'hasoc'], help='Specify the dataset for which you want to run the experiments.')
# Parse the argument
args = parser.parse_args()


def add_edge(graph, source, role, target):

    """Function to add an edge between two previously existing nodes in the graph.
    
    Here, source and target node instances already exist in "graph" and we simply add an edge with relation "rel"
    between the two. The purpose of this is to add :COREF and :SAME edges
    
    TODO: Modify the epidata while adding a new edge"""


    edges= [(source, role, target)]#adding the new edge
    edges.extend(graph.edges())

    #modified amr after adding the required edge
    modified = penman.Graph(graph.instances() + edges + graph.attributes())
    return modified

def coreference_edges(merged_amr, name, amr_coref):

    """ Function to add coreference edges to the merged graph. The input is the combined AMR graph of all smaller
    comment AMR graphs."""
    d = amr_coref[name]
    for relation, cluster in d.items():
        var = [i[1] for i in cluster]
        #amr_coref is sorted according to time (i.e. comments appearing first temporally appear before) by default
        source = var[0] #the directed edge will start from the comment appearing first: following temporal fashion
        for target in var[1:]:#add :COREF edges from the source word to all words in the cluster
            added = add_edge(merged_amr, source, ":COREF", target)
            merged_amr = added
    return merged_amr

def concept_merge(modified):
    normalised_graph = normalise_graph(modified)
    word2var = generate_word2var(normalised_graph)
    for word, var in word2var.items():
        head_node = var[0]
        if len(var)>1:
            for j in var[1:]:
                added = add_edge(normalised_graph, head_node, ":SAME", j)
                normalised_graph = added
    return normalised_graph

def generate_word2var(normalised_graph):
    """This function returns a word-to-variableNames dictionary.
    
    A word might be present on multiple nodes. This returns the dictionary storing the nodes for every word."""
    word2var = {}# a dictionary mapping the words to nodes; example: the word 'name' might belong to 2 nodes

    for (source, role, target) in normalised_graph.instances():
        if target in word2var:
            word2var[target].append(source)
        else:
            word2var[target] = [source]
    return word2var

def normalise_graph(modified):

    """A function to convert the concepts in the amr to meaningful form so that we can apply glove embedding later on
    to find their node representations. 
    
    Note: Removing the part after the hyphen (-) in many of the concept names."""
    normalised_instances = []
    for (source, role, target) in modified.instances():
        if "-" in target:#for example: "save-01" concept is converted to "save". 
            normalised_instances.append((source, role, target[:target.rfind("-")]))
        else:
            normalised_instances.append((source, role, target))
    normalised_graph = penman.Graph(normalised_instances + modified.edges() + modified.attributes())
    return normalised_graph

amr_list = glob.glob(f"{args.dataset}_amr/{args.dataset}_amr_csv/{args.dataset}*.csv")

amr_coref = json.load(open(f"{args.dataset}_amr/{args.dataset}_amr_coref.json", "r"))#load the coref json with cluster information
amr_coref_names = [i[:i.find(".")] for i in list(amr_coref.keys())]
lv=0
for q in amr_list:
    df = pd.read_csv(q)
    #modify all Comment AMRs belonging to one news article
    name = q[q.rfind('/')+1:q.rfind('.')]#name of current file containing the amrs

    if name not in amr_coref_names:
        print("Skipping")
        continue

    #reading the modified amrs passed as input for coreference resolution
    modified_amr_list = penman.load(f"{args.dataset}_amr/{args.dataset}_amr_coref/{name}.amr.penman", model = NoOpModel())

    #adding dummy node and :COMMENT edges
    instances, edges, attributes = [('d', ':instance', 'dummy')], [], []
    metadata, epidata= {'snt': '', 'lemmas' : [], 'tokens' : []}, {('d', ':instance', 'dummy') : []}
    
    subgraph_list = []

    for graph in modified_amr_list:

        #for now we are ignoring the :coref and :same-as nodes to be a part of any subgraph - later on they can be added
        node_list = [source for source, _, _ in graph.instances()]#maintained for creating subgraphs later on
        subgraph_list.append(node_list)

        edges.append(('d', ':COMMENT', graph.top))
        instances.extend(graph.instances())
        edges.extend(graph.edges())
        attributes.extend(graph.attributes())
        metadata['snt']+= "{} ".format(graph.metadata['snt'])
        metadata['lemmas'].extend(ast.literal_eval(graph.metadata['lemmas']))
        metadata['tokens'].extend(ast.literal_eval(graph.metadata['tokens']))
        #adding epidata for the added edge
        epidata[('d', ':COMMENT', graph.top)] = [penman.layout.Push(graph.top)]
        #Adding the epidata from all amrs
        epidata.update(graph.epidata)

    metadata['tokens'] = json.dumps(metadata['tokens']) #convert metadata tokens to string format
    metadata['lemmas'] = json.dumps(metadata['lemmas'])
    
    #final modified graph consisting of all comment AMRs corresponding to one news piece
    modified = penman.Graph(instances + edges + attributes)
    modified.metadata = metadata
    modified.epidata = epidata

    #adding the coreference edges to the merged amr graph
    modified = coreference_edges(modified, "{}.amr.penman".format(name), amr_coref)
    
    #adding the concept merging edges to the merged amr graph
    try:
        modified = concept_merge(modified)
    except:
        continue

    modified.metadata['subgraphs'] = json.dumps(subgraph_list)#adding the subgraph information
    #here every subgraph corresponds to a particular comment on the news article and the merged amr is the overall combination of all such subgraphs

    # print(penman.encode(modified))
    #storing the final AMR graph (merged + coreference + concept merging)
    dst_dir = f"{args.dataset}_amr/{args.dataset}_amr_merge/{name}.amr.penman"
    os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
    penman.dump([modified], dst_dir, model = NoOpModel())

    print("Done", lv, q)
    lv+=1

    # break