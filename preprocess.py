import pickle
import numpy as np
import re
from nltk import tokenize
from tensorflow.keras.utils import to_categorical
import argparse
parser = argparse.ArgumentParser()
from sklearn import preprocessing

# Adding the required arguments
parser.add_argument('--dataset', choices = ['antivax', 'politifact', 'gossipcop', 'figlang_twitter', 'figlang_reddit', 'twitter16', 'rumoureval', 'pheme', 'twitter15', 'hasoc'], help='Specify the dataset for which you want to run the experiments.')
glove = '/home/karish19471/glove/glove.6B.100d.txt'
# Parse the argument
args = parser.parse_args()

def get_data(DATA):

    def train_test_split(data):
        contents, comments, labels, ids, subgraphs= [], [], [], [], []
        for idx in range(len(data)):
            try:
                text = data[idx]['content']
                contents.append(tokenize.sent_tokenize(text))
                ids.append(data[idx]['id'])
                labels.append(data[idx]['label'])
                comments.append(data[idx]['graph'])#this is the comment graph -- merged amr graph corresponding to each 
                subgraphs.append(data[idx]['subgraphs'])
            except:#to remove the null samples - samples with no comments
                print("error", idx)

        labels = np.asarray(labels)
        le = preprocessing.LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
        labels = to_categorical(labels)
        return contents, comments, labels, ids, subgraphs

    # dataset used for training
    train = pickle.load(open(f"data/{DATA}/{DATA}_train.pkl", 'rb'))
    test = pickle.load(open(f"data/{DATA}/{DATA}_test.pkl", 'rb'))
    
    #add shuffling 
    np.random.shuffle(train)
    np.random.shuffle(test)

    x_train, c_train, y_train, id_train, sub_train = train_test_split(train)
    x_val, c_val, y_val, id_val, sub_val = train_test_split(test)

    return {'train': {'id': id_train, 'x':x_train, 'c': c_train, 'y': y_train, 'subgraphs':sub_train}, 'val': {'id': id_val, 'x':x_val, 'c': c_val, 'y': y_val, 'subgraphs':sub_val}}

props = get_data(args.dataset)
file = open(f'data/{args.dataset}_preprocessed.pkl', 'wb')
pickle.dump(props, file)