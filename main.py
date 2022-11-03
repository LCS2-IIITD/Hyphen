import os
import time
import gc
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import keras_preprocessing
import tqdm
import dgl
from tensorboardX import SummaryWriter
from utils.radam import RiemannianAdam
from utils.metrics import Metrics
from hyphen import Hyphen
from utils.dataset import FakeNewsDataset
from utils.utils import get_evaluation

class HyphenModel():
    def __init__(self, platform, max_sen_len, max_com_len, max_sents, max_coms, manifold, log_path, lr, content_module, comment_module, fourier):
        self.model = None
        self.max_sen_len = max_sen_len
        self.max_sents = max_sents
        self.max_coms = max_coms
        self.max_com_len = max_com_len
        self.vocab_size = 0
        self.word_embedding = None
        self.model = None
        self.word_attention_model = None
        self.sentence_comment_co_model = None
        self.tokenizer = None
        self.metrics = Metrics()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
        self.log_path = log_path
        self.manifold = manifold
        self.lr = lr
        self.content_module = content_module
        self.comment_module = comment_module
        self.fourier = fourier
        self.platform = platform
        
    def _fit_on_texts(self, train_x, val_x):
        """
        Creates vocabulary set from the news content and the comments
        """
        texts = []
        texts.extend(train_x)
        texts.extend(val_x)
        self.tokenizer = keras_preprocessing.text.Tokenizer(num_words=30000)
        all_text = []

        all_sentences = []
        for text in texts:
            for sentence in text:
                all_sentences.append(sentence)

        all_text.extend(all_sentences)
        self.tokenizer.fit_on_texts(all_text)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self._create_reverse_word_index()
        pickle.dump(self.tokenizer, open("tokenizer.pkl", 'wb'))
        print("saved tokenizer")

    def _create_reverse_word_index(self):
        '''
            create a dictionary with index as key and corresponding word as value pair.
            e.g.

            reverse_word_index = {1: 'the', 2: 'to', 3: 'a', 4: 'and', 5: 'of', 6: 'is', 7: 'in', 8: 'that', 9: 'i', ....}
        '''
        self.reverse_word_index = {value: key for key, value in self.tokenizer.word_index.items()}

    def _build_model(self, n_classes=2, batch_size = 12,embedding_dim=100):
        '''
            This function is used to build Hyphen model.
        '''
        embeddings_index = {}

        self.glove_dir = "/home/karish19471/glove/glove.6B.100d.txt"

        f = open(self.glove_dir, encoding="utf-8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        # get word index
        word_index = self.tokenizer.word_index
        embedding_matrix = np.random.random((len(word_index)+1, embedding_dim))

        # create embedding matrix.
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        self.word_hidden_size, self.sent_hidden_size, self.max_sent_length, self.max_word_length, self.graph_hidden= 50, 50, 50, 50, 100
        model = Hyphen(embedding_matrix, self.word_hidden_size, self.sent_hidden_size, self.max_sent_length, self.max_word_length, 
        self.device, graph_hidden = self.graph_hidden, batch_size = batch_size, num_classes = n_classes, max_comment_count= self.max_coms,
        max_sentence_count=self.max_sents, manifold = self.manifold, comment_module = self.comment_module,
        content_module = self.content_module, fourier = self.fourier)

        model = model.to(self.device)

        if self.manifold == "Euclidean": #choose the manifold
            self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        elif self.manifold == "PoincareBall":
            self.optimizer = RiemannianAdam(model.parameters(), lr = self.lr)

        self.criterion = nn.CrossEntropyLoss()

        return model

    def _encode_texts(self, texts):
        """
        Pre process the news content sentences to equal length for feeding to GRU
        :param texts:
        :return:
        """
        encoded_texts = np.zeros((len(texts), self.max_sents, self.max_sen_len), dtype='int32')
        for i, text in enumerate(texts):
            encoded_text = np.array(pad_sequences(
                self.tokenizer.texts_to_sequences(text),
                maxlen=self.max_sen_len, padding='post', truncating='post', value=0))[:self.max_sents]
            encoded_texts[i][:len(encoded_text)] = encoded_text

        return encoded_texts

    def test(self, train_x, train_y, train_c, val_c, val_x, val_y, sub_train, sub_val, batch_size = 9):
        
        self.tokenizer = pickle.load(open("tokenizer.pkl", 'rb'))
        print("Building model....")
        self.model = self._build_model(n_classes=train_y.shape[-1], batch_size= batch_size, embedding_dim=100)
        print("Model built.")

        print("Encoding texts....")
        # Create encoded input for content and comments
        encoded_train_x = self._encode_texts(train_x)
        encoded_val_x = self._encode_texts(val_x)
        print("preparing dataset....")

        #adding self loops in the dgl graphs
        train_c= [dgl.add_self_loop(i) for i in train_c]
        val_c= [dgl.add_self_loop(i) for i in val_c]

        train_dataset = FakeNewsDataset(encoded_train_x, train_c, train_y, sub_train, self.glove_dir,  self.max_sent_length, self.max_word_length)
        val_dataset = FakeNewsDataset(encoded_val_x, val_c, val_y, sub_val, self.glove_dir,  self.max_sent_length, self.max_word_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn = train_dataset.collate_fn, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn = val_dataset.collate_fn, shuffle=True, drop_last=True)

        self.dataset_sizes = {'train': train_dataset.__len__(), 'val': val_dataset.__len__()}
        self.dataloaders = {'train': train_loader, 'val': val_loader}
        print("Dataset prepared.")

        self.model.load_state_dict(torch.load(f"saved_models/{self.platform}/best_model_{self.manifold}.pt"))

        print("Loaded state dict")

        self.model.eval()
        loss_ls= []
        total_samples= 0
        As_batch, Ac_batch, predictions_batch = [], [], []
        for i, sample in enumerate(self.dataloaders['val']):

            content, comment, label, subgraphs = sample
            num_sample = len(label)#last batch size
            total_samples+=num_sample

            comment = comment.to(self.device)
            content = content.to(self.device)
            label = label.to(self.device)

            self.model.content_encoder._init_hidden_state(num_sample)

            predictions, As, Ac= self.model(content, comment, subgraphs)

            te_loss = self.criterion(predictions, label)
            loss_ls.append(te_loss * num_sample)

            _, predictions =  torch.max(torch.softmax(predictions, dim = -1), 1)
            _, label =  torch.max(label, 1)

            As_batch.extend(As.detach().cpu().numpy())
            Ac_batch.extend(Ac.detach().cpu().numpy())
            predictions_batch.extend(predictions.detach().cpu().numpy())
        return predictions_batch, As_batch, Ac_batch

    def train(self, train_x, train_y, train_c, val_c, val_x, val_y, sub_train, sub_val, batch_size=9, epochs=5):

        self.writer = SummaryWriter(self.log_path)

        # Fit the vocabulary set on the content and comments
        self._fit_on_texts(train_x, val_x)

        print("Building model....")
        self.model = self._build_model(n_classes=train_y.shape[-1], batch_size= batch_size, embedding_dim=100)
        print("Model built.")

        print("Encoding texts....")
        # Create encoded input for content and comments
        encoded_train_x = self._encode_texts(train_x)
        encoded_val_x = self._encode_texts(val_x)
        print("preparing dataset....")

        #adding self loops in the dgl graphs
        train_c= [dgl.add_self_loop(i) for i in train_c]
        val_c= [dgl.add_self_loop(i) for i in val_c]

        train_dataset = FakeNewsDataset(encoded_train_x, train_c, train_y, sub_train, self.glove_dir,  self.max_sent_length, self.max_word_length)
        val_dataset = FakeNewsDataset(encoded_val_x, val_c, val_y, sub_val, self.glove_dir,  self.max_sent_length, self.max_word_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn = train_dataset.collate_fn, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn = val_dataset.collate_fn, shuffle=True, drop_last=True)

        self.dataset_sizes = {'train': train_dataset.__len__(), 'val': val_dataset.__len__()}
        self.dataloaders = {'train': train_loader, 'val': val_loader}
        print("Dataset prepared.")

        #train model for given epoch
        self.run_epoch(epochs)

        self.writer.close()

    def run_epoch(self, epochs):
        '''
        Function to train model for given epochs
        '''

        since = time.time()
        clip = 5#modify clip

        best_f1 = 0.0

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 100)
            self.metrics.on_train_begin()
            self.model.train()
            
            num_iter_per_epoch = len(self.dataloaders['train'])
            for iter, sample in enumerate(tqdm.tqdm(self.dataloaders['train'])):
                self.optimizer.zero_grad()
            
                content, comment, label, subgraphs = sample

                comment = comment.to(self.device)
                content = content.to(self.device)
                label = label.to(self.device)
                self.model.content_encoder._init_hidden_state(len(label))
                predictions, As, Ac = self.model(content, comment, subgraphs) #As and Ac are the attention weights we are returning
                loss = self.criterion(predictions, label)
                loss.backward()
                self.optimizer.step()

                training_metrics = get_evaluation(torch.max(label, 1)[1].cpu().numpy(), predictions.cpu().detach().numpy(), list_metrics=["accuracy"])
                self.writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + iter)
                self.writer.add_scalar('Train/Accuracy', training_metrics["accuracy"], epoch * num_iter_per_epoch + iter)
            

            self.model.eval()
            loss_ls= []
            total_samples= 0
            for i, sample in enumerate(self.dataloaders['val']):

                content, comment, label, subgraphs = sample
                num_sample = len(label)#last batch size
                total_samples+=num_sample

                comment = comment.to(self.device)
                content = content.to(self.device)
                label = label.to(self.device)

                self.model.content_encoder._init_hidden_state(num_sample)

                predictions, As, Ac= self.model(content, comment, subgraphs)

                te_loss = self.criterion(predictions, label)
                loss_ls.append(te_loss * num_sample)

                _, predictions =  torch.max(torch.softmax(predictions, dim = -1), 1)
                _, label =  torch.max(label, 1)

                print(predictions)
                predictions= predictions.detach().cpu().numpy()
                label = label.detach().cpu().numpy()

                self.metrics.on_batch_end(epoch, i, predictions, label)

            acc_, f1 = self.metrics.on_epoch_end(epoch) 
            if f1 > best_f1:
                print(f"Best F1: {f1}")
                print("Saving best model!")
                dst_dir = f'saved_models/{self.platform}/'
                os.makedirs(dst_dir, exist_ok  = True)
                torch.save(self.model.state_dict(), f'{dst_dir}best_model_{self.manifold}.pt')
                best_model = self.model
                best_f1 = f1

            te_loss = sum(loss_ls) / total_samples
            self.writer.add_scalar('Test/Loss', te_loss, epoch)
            self.writer.add_scalar('Test/Accuracy', acc_, epoch)
            self.writer.add_scalar('Test/F1', f1, epoch)
        
        print(f"Best F1: {best_f1}")
        print("Training  end")
        print('-'*100)    

    def process_atten_weight(self, encoded_text, content_word_level_attentions, sentence_co_attention):
        '''
            Process attention weights for sentence
        '''
        no_pad_text_att = []
        for k in range(len(encoded_text)):
            tmp_no_pad_text_att = []
            cur_text = encoded_text[k]
            for i in range(len(cur_text)):
                sen = cur_text[i]
                no_pad_sen_att = []
                if sum(sen) == 0:
                    continue
                for j in range(len(sen)):
                    wd_idx = sen[j]
                    if wd_idx == 0:
                        continue
                    wd = self.reverse_word_index[wd_idx]
                    no_pad_sen_att.append((wd, content_word_level_attentions[k][i][j]))

                tmp_no_pad_text_att.append((no_pad_sen_att, sentence_co_attention[k][i]))

            no_pad_text_att.append(tmp_no_pad_text_att)

        # Normalize without padding tokens
        no_pad_text_att_normalize = None
        for npta in no_pad_text_att:
            if len(npta) == 0:
                continue
            sen_att, sen_weight = list(zip(*npta))
            new_sen_weight = [float(i) / sum(sen_weight) for i in sen_weight]
            new_sen_att = []
            for sw in sen_att:
                word_list, att_list = list(zip(*sw))
                att_list = [float(i) / sum(att_list) for i in att_list]
                new_wd_att = list(zip(word_list, att_list))
                new_sen_att.append(new_wd_att)
            no_pad_text_att_normalize = list(zip(new_sen_att, new_sen_weight))

        return no_pad_text_att_normalize

    def process_atten_weight_com(self, encoded_text, sentence_co_attention):
        '''
            Process attention weight for comments
        '''
        
        no_pad_text_att = []
        for k in range(len(encoded_text)):
            tmp_no_pad_text_att = []
            cur_text = encoded_text[k]
            for i in range(len(cur_text)):
                sen = cur_text[i]
                no_pad_sen_att = []
                if sum(sen) == 0:
                    continue
                for j in range(len(sen)):
                    wd_idx = sen[j]
                    if wd_idx == 0:
                        continue
                    wd = self.reverse_word_index[wd_idx]
                    no_pad_sen_att.append(wd)
                tmp_no_pad_text_att.append((no_pad_sen_att, sentence_co_attention[k][i]))

            no_pad_text_att.append(tmp_no_pad_text_att)

        return no_pad_text_att