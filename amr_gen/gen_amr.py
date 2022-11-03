import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import amrlib
import os
os.environ['CUDA_VISIBLE_DEVICES']= "0"
dataset = 'figlang_twitter'
import os
import warnings
warnings.filterwarnings("ignore")
stog = amrlib.load_stog_model()
print("Loaded stog")
df = pd.read_csv(f'{dataset}_amr/{dataset}.csv')
df['comments'] = df['comments'].fillna(" ")
total = df.shape[0]
comments = [df.iloc[i]['comments'].split('::')[:50] for i in range(df.shape[0])]
num_comments = np.sum(np.array([len(i) for i in comments]))
done_comments = 0
ids = list(df['id'])
for lv in range(len(ids)):
    graphs = stog.parse_sents(comments[lv])
    data = pd.DataFrame()
    data['comment'] = np.array(comments[lv])
    data['amr'] = np.array(graphs)
    dest_dir = f"{dataset}_amr/{dataset}_amr_csv/"
    os.makedirs(dest_dir, exist_ok = True)
    data.to_csv(dest_dir+ ids[lv]+'.csv', index = False)
    done_comments+=len(comments[lv])
    print(f"Done {lv}/{total}\tDone {done_comments}/{num_comments}")