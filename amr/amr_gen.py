import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import amrlib
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES']= "2"
import argparse
parser = argparse.ArgumentParser()

# Adding the required arguments
parser.add_argument('--dataset', default = 'politifact', choices = ['antivax', 'politifact', 'gossipcop', 'figlang_twitter', 'figlang_reddit', 'twitter16', 'rumoureval', 'pheme', 'twitter15', 'hasoc'], help='Specify the dataset for which you want to run the experiments.')
parser.add_argument('--max-comments', type = int, default = 50, help = "Specify the maximum number of comments per post that you wish to convert to AMR.")

# Parse the argument
args = parser.parse_args()

stog = amrlib.load_stog_model()
print("Loaded stog")
df = pd.read_csv(f'amr_data/{args.dataset}_amr/{args.dataset}.csv')
df['comments'] = df['comments'].fillna(" ")
total = df.shape[0]
comments = [df.iloc[i]['comments'].split('::')[:args.max_comments] for i in range(df.shape[0])]
num_comments = np.sum(np.array([len(i) for i in comments]))
done_comments = 0
ids = list(df['id'])
for lv in range(len(ids)):
    graphs = stog.parse_sents(comments[lv])
    data = pd.DataFrame()
    data['comment'] = np.array(comments[lv])
    data['amr'] = np.array(graphs)
    dest_dir = f"amr_data/{args.dataset}_amr/{args.dataset}_amr_csv/"
    os.makedirs(dest_dir, exist_ok = True)
    data.to_csv(dest_dir+ ids[lv]+'.csv', index = False)
    done_comments+=len(comments[lv])
    print(f"Done {lv}/{total}\tDone {done_comments}/{num_comments}")