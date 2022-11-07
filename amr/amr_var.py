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


#************************************************************************************
# Adding dummy source node and COMMENT edges to multiple AMRs and merge into one.
#************************************************************************************

def modify_variables(amr, i):

    """This function takes an AMR as input and modifies the variables of the AMR depending on the serial number
    of the AMR. Here, (i) refers to the (i)th comment on a particular news piece.
    
    Returns: The modifed penman Graph of the AMR string.
    
    Note: This function does not modify the edpidata or metadata of the input AMR. We just modify the variable names 
    in this function. Since our ultimate goal is to merge several AMR graphs, it is highly likely that different
    amrs have the same variable names. Thus to distinguish between variables of different amrs we assign unique names
    to different variables."""

    g = penman.decode(amr)

    g_meta = add_lemmas(amr, snt_key='snt')#adding lemmas , tokens to the AMR string
    
    #create a dictionary for mapping old variables to new variable names

    var, d = list(g.variables()), {}

    for j in range(len(var)):
        d[var[j]] = "c{}-{}".format(i, j)
        
    #modify the variable names of instances, edges, attributes of the original amr graph 
    instances, edges, attributes, epidata = [], [], [], {}
    for source, role, target in g.instances():#modify the instances
        instances.append((d[source], role, target))
    for source, role, target in g.edges():#modify the edges
        edges.append((d[source], role, d[target]))

    for source, role, target in g.attributes():#modify the attributes
        attributes.append((d[source], role, target))

    for (source, role, target) in g.epidata.keys():#modify the attributes
        
        push_pop = g.epidata[(source, role, target)]
        
        modified_epi = []
        for p in push_pop:
            if isinstance(p, penman.layout.Push):  modified_epi.append(penman.layout.Push(d[p.variable]))
            elif isinstance(p, penman.layout.Pop):  modified_epi.append(p)
            else: print(p)
    
        #if the epidata key is either an instance or attribute triple
        if (source, role, target) in g.instances() or (source, role, target) in g.attributes(): 
            epidata[(d[source], role, target)] = modified_epi
        
        elif (source, role, target) in g.edges(): 
            epidata[(d[source], role, d[target])] = modified_epi
        else:
            print((source, role, target))
        
    modified  = penman.Graph(instances + edges + attributes)#return the modifies graph 
    
    modified.metadata = g_meta.metadata #using the metadata from the original graph
    
    modified.epidata = epidata #using the epidata from the original graph -- name changed

    assert len(eval(modified.metadata['lemmas']))==len(eval(modified.metadata['tokens'])), "Length of tokens must be equal to lemmas"

    return modified


amr_list = glob.glob(f"data/{args.dataset}/{args.dataset}_amr_csv/{args.dataset}*.csv")

lv, ignored = 1, []
for q in amr_list:
    df = pd.read_csv(q)
    #removing any comment with null amr - possible case when language is not english
    df = df.dropna()
    print("Processing", q, end = " ")
    # modify all Comment AMRs belonging to one news article
    modified_amr_list = []
    try:
        for i, j in df.iterrows():
            var_mod = modify_variables(j['amr'], i+1)
            name = q[q.rfind('/')+1:q.rfind('.')]
            modified_amr_list.append(var_mod)
        
        dst_dir = f'data/{args.dataset}/{args.dataset}_amr_coref/{name}.amr.penman'
        os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
        penman.dump(modified_amr_list, dst_dir, model = NoOpModel())
    
    except AssertionError:
        print("**************Ignoring the file {}******************".format(q))
        ignored.append(q)

    except:
        print("**************Ignoring the file decode error{}******************".format(q))
        ignored.append(q)

    lv+=1 
    print("Done", lv, len(modified_amr_list))
print("Files ignored:\n{}".format(ignored))
print("{} filed ignored".format(len(ignored)))


