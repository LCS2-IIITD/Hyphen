import glob
import networkx as nx
import penman
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
import dgl
import json
import numpy as np
from sklearn.model_selection import train_test_split

import argparse
parser = argparse.ArgumentParser()

# Adding the required arguments
parser.add_argument('--dataset', choices = ['antivax', 'politifact', 'gossipcop', 'figlang_twitter', 'figlang_reddit', 'twitter16', 'rumoureval', 'pheme', 'twitter15', 'hasoc'], help='Specify the dataset for which you want to run the experiments.')
parser.add_argument('--test-split', type = float, default= 0.1, help = "Specify the required test data ratio.")

# Parse the argument
args = parser.parse_args()

merged_amr = glob.glob(f"{args.dataset}_amr/{args.dataset}_amr_merge/*.amr.penman")
df = pd.read_csv(f'{args.dataset}_amr/{args.dataset}.csv')

def var2word(p_graph):
    v2w = {}
    for (source, _, target) in p_graph.instances():
        v2w[source] = target
    return v2w

def get_glove():
    glove = {}
    f = open('/home/karish19471/glove/glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        glove[word] = coefs
    return glove

def to_dict(d):
    return {i: {'feat':j} for i,j in d.items()}

def id2label(df):
    return dict(zip(df['id'], zip(df['labels'], df['text'])))

glove = get_glove()
i2l = id2label(df)
EMBEDDING_DIM = 100
dataset = []
lv = 0

for curr in merged_amr:
    print(lv)
    p_graph = penman.load(curr, model = NoOpModel())[0]
    name = curr[curr.rfind('/')+1:curr.rfind('.amr')]
    v2w = var2word(p_graph)
    nx_graph = nx.MultiDiGraph()
    nx_graph.add_edges_from([(s, t) for s, _, t in p_graph.edges()])#TODO: Add edges from instances as well

    #-----------------------------------extracting subgraphs----------------------------------------------------
    #sorted ordering is a must in order to preserve the node order in case of using from_networkx
    temp= nx.convert_node_labels_to_integers(nx_graph, ordering = 'sorted', label_attribute= 'original')
    original2new  = {temp.nodes[i]['original']:i for i in temp.nodes}
    subgraphs = [[ original2new[j] for j in i] for i in eval(p_graph.metadata['subgraphs'])]
    #-----------------------------------------------------------------------------------------------------------

    MAP = {i:glove.get(v2w[i], [0]*EMBEDDING_DIM) for i in nx_graph.nodes()}
    attr= to_dict(MAP)
    nx.set_node_attributes(nx_graph, attr)
    try:
        dgl_graph = dgl.from_networkx(nx_graph, node_attrs=['feat'])
    except:
        continue
    (source, target) = dgl_graph.edges()
    label, content = i2l[name]
    sample = {'label':label, 'graph': dgl_graph, 'content': content, 'id': name, 'subgraphs':subgraphs}
    dataset.append(sample)
    lv+=1

with open(f"{args.dataset}_amr/{args.dataset}.pkl", "wb") as f:
    pickle.dump(dataset, f)

with open(f"{args.dataset}_amr/{args.dataset}.pkl", "rb") as f:
    d =pickle.load(f)

train, test = train_test_split(d, stratify=np.array([i['label'] for i in d]), test_size=args.test_split)

with open(f"{args.dataset}_amr/{args.dataset}_train.pkl", "wb") as f:
    pickle.dump(train, f)
with open(f"{args.dataset}_amr/{args.dataset}_test.pkl", "wb") as f:
    pickle.dump(test, f)