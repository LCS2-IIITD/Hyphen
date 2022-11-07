#!/usr/bin/python3
import os
import pickle
import glob
import penman
from   penman.models.noop import NoOpModel
from   amr_coref.coref.inference import Inference
import json
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def parse_default_args():
    parser = argparse.ArgumentParser(description='AMR_Coref')
    parser.add_argument('--dataset', type=str, default='politifact')
    args, _ = parser.parse_known_args()
    args = parser.parse_args()
    return args


# function to add to JSON
def write_json(new_data, p_name, filename):
    temp = p_name[p_name.rfind('/')+1:]
    with open(filename,'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data[temp] = new_data
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)

if __name__ == '__main__':

    args = parse_default_args()

    model_dir = 'amr/amr_coref/data/model_coref-v0.1.0/'

    json.dump({}, open(f'data/{args.dataset}/{args.dataset}_amr_coref.json','w'), indent = 4)

    # Load the model and test data
    print('Loading model from %s' % model_dir)
    
    amr_list = glob.glob(f"data/{args.dataset}/{args.dataset}_amr_coref/{args.dataset}*.amr.penman")

    print(f"Found{len(amr_list)} files")

    ordered_pgraphs = []
    for i in amr_list:
        ordered_pgraphs.append(penman.load(i, model = NoOpModel()))
    
    for i in range(len(ordered_pgraphs)):

        # Get test data
        print(amr_list[i])
        print('Loading test data')

        inference = Inference(model_dir, amr_list)

        print('Clustering')
        cluster_dict = {}
        if len(ordered_pgraphs[i]) >= 2:
            try:
                cluster_dict = inference.coreference(ordered_pgraphs[i])
            except:
                print("________________****************_______________ignore")
        print()

        # Print out the clusters
        print('Clusters')

        write_json(cluster_dict, amr_list[i], filename = f'data/{args.dataset}/{args.dataset}_amr_coref.json')

        print("DONE", i)