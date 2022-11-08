import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import pickle
import argparse
import pickle
parser = argparse.ArgumentParser()
from main import HyphenModel

parser.add_argument('--manifold', choices=['PoincareBall', 'Euclidean'], default = 'PoincareBall', help='Choose the underlying manifold for Hyphen')
parser.add_argument('--no-fourier', default=True, action='store_false', help='If you want to remove the Fourier sublayer from Hyphen\'s co-attention module.')
parser.add_argument('--no-comment', default=True, action='store_false', help='If you want to remove the comment module from Hyphen i.e. just consider news content as the only input modality.')
parser.add_argument('--no-content', default=True, action='store_false', help='If you want to remove the content module from Hyphen, i.e. just consider user comments as the only input modality.')
parser.add_argument('--log-path', type = str, default = "logging/run", help='Specify the path of the Log file for Tensorboard.')
parser.add_argument('--lr', default = 0.001, type = float, help='Specify the learning rate for Hyphen.')
parser.add_argument('--dataset', default= 'politifact', choices = ['antivax', 'politifact', 'gossipcop', 'figlang_twitter', 'figlang_reddit', 'twitter16', 'rumoureval', 'pheme', 'twitter15', 'hasoc'], help='Specify the dataset for which you want to run the experiments.')
parser.add_argument('--max-coms', type = int, default= 10, help='Specify the maximum number of user comments (per post) you want to consider.')
parser.add_argument('--max-sents', type = int, default= 20, help='Specify the maximum number of news sentences from the social media post that you want to consider.')
parser.add_argument('--max-com-len', type = int, default= 10, help='Specify the maximum length of a user comment to feed in Hyphen.')
parser.add_argument('--max-sent-len', type = int, default = 10, help='Specify the maximum length of a news sentence.')
parser.add_argument('--batch-size', type = int,  default = 32,  help='Specify the batch size of the dataset.')
parser.add_argument('--epochs', type = int, default= 5, help='The number of epochs to train Hyphen.')

args = parser.parse_args()

file = open(f'data/{args.dataset}_preprocessed.pkl', 'rb')
props = pickle.load(file)

id_train, id_test = props['train']['id'], props['val']['id']
x_train, x_val = props['train']['x'], props['val']['x']
y_train, y_val = props['train']['y'], props['val']['y']
c_train, c_val = props['train']['c'], props['val']['c']
sub_train, sub_val = props['train']['subgraphs'], props['val']['subgraphs']

hyphen = HyphenModel(args.dataset, args.max_sent_len, args.max_com_len, args.max_sents, args.max_coms, manifold= args.manifold, lr = args.lr, log_path=args.log_path,
comment_module =args.no_comment, content_module = args.no_content, fourier = args.no_fourier)

hyphen.train(x_train, y_train, c_train, c_val, x_val, y_val, sub_train, sub_val, batch_size= args.batch_size, epochs=args.epochs)