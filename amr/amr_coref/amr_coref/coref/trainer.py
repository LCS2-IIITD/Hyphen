#!/usr/bin/python3
import os
import random
import time
from   datetime import datetime
import numpy
import torch
from   torch.optim import AdamW
from   torch.utils.tensorboard import SummaryWriter
from   tqdm import tqdm
from   ..utils.config import Config
from   ..utils.log_splitter import LogSplitter
from   ..evaluate import scorch
from   .tester import Tester
from   .amr_coref_model import AMRCorefModel
from   .coref_data_loader import get_data_loader_from_file
from   .clustering import cluster_and_save_sdata


# See random generators for consistant results in testing
random.seed(0)
torch.manual_seed(0)
numpy.random.seed(0)


class Trainer(object):
    all_pair = 'all_pair'
    top_pair = 'top_pair'
    ranking  = 'ranking'
    def __init__(self, model_dir, model, logname):
        self.model_dir    = model_dir
        self.model        = model
        self.config       = model.config
        self.ls           = LogSplitter(logname, model_dir)
        tbdir             = os.path.join(model_dir, 'tb', datetime.now().strftime('%Y-%m-%dT%H:%M'))
        os.makedirs(tbdir, exist_ok=True)
        self.writer       = SummaryWriter(log_dir=tbdir)
        self.tester       = None
        self.show_pbar    = True

    # Create a new model
    @classmethod
    def from_scratch(cls, model_dir, config_fn, graph_embed_fn, mention_set_fn, logname='train.log'):
        os.makedirs(model_dir, exist_ok=True)
        print('Loading config and embeddings')
        model = AMRCorefModel.from_files(config_fn, graph_embed_fn, mention_set_fn)
        print()
        self = cls(model_dir, model, logname)
        return self

    # Load a previously trained model
    @classmethod
    def from_pretrained(cls, model_dir, logname='train_cont.log'):
        print('Loading model from %s' % model_dir)
        model = AMRCorefModel.from_pretrained(model_dir)
        print()
        self = cls(model_dir, model, logname)
        return self

    # Setup the optimizer
    def set_optimizer(self, learning_rate, l2_normalization):
        self.optimizer = AdamW(self.model.parameters(), learning_rate, weight_decay=l2_normalization)

    # Setup the train data loader
    def setup_train_data(self, train_fn, **kwargs):
        self.ls.print('Loading and featurizing training data')
        self.train_dloader = get_data_loader_from_file(train_fn, self.model, **kwargs)
        lengths = self.train_dloader.batch_sampler.lengths
        self.ls.print('The train data is length {:,}'.format(len(self.train_dloader)))
        self.ls.print('There are {:,} training samples'.format(len(lengths)))
        self.ls.print('The max length is {:,} and {:,} are singles'.format(max(lengths), lengths.count(1)))
        self.ls.print()

    # Setup the test data loader
    def setup_test_data(self, test_fn, **kwargs):
        self.ls.print('Loading and featurizing test data')
        self.tester = Tester.from_model(self.model, test_fn)
        lengths = self.tester.test_dloader.batch_sampler.lengths
        self.ls.print('The test data is length {:,}'.format(len(self.tester.test_dloader)))
        self.ls.print('There are {:,} test samples'.format(len(lengths)))
        self.ls.print('The max length is {:,} and {:,} are singles'.format(max(lengths), lengths.count(1)))
        self.ls.print()

    # Train the model
    def train(self, loss_type, num_epochs, start_epoch=1):
        # Setup the loss function (ie.. model.all_pair_loss())
        loss_function = getattr(self.model, loss_type + '_loss')
        # Loop through training epochs
        self.ls.print('Training with the %s loss function' % loss_type)
        self.ls.print('{:,} batches/epoch for {:,} epochs'.format(len(self.train_dloader), num_epochs))
        batch_size = len(self.train_dloader)
        writer_index, run_loss_list = 0, []
        for epoch in range(start_epoch, start_epoch+num_epochs):
            epoch_st = time.time()
            self.ls.print('Epoch %d' % epoch)
            epoch_loss = 0
            self.model.train()
            pbar = tqdm(self.train_dloader, ncols=100, disable=not self.show_pbar)
            self.set_pbar_description(pbar, None, None)
            # Loop throgh the batch
            for bnum, batch in enumerate(pbar, start=1):
                self.optimizer.zero_grad()
                Y = self.model(batch)
                loss = loss_function(Y, batch)
                # Detect issues and quit if something goes wrong
                if torch.isnan(loss):
                    self.ls.print('!!! nan in loss - exiting.')
                    exit()
                # Backpropagate and update
                batch_loss  = loss.item()
                epoch_loss += batch_loss
                run_loss_list.append(batch_loss)
                loss.backward()
                self.optimizer.step()
                if self.config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.set_pbar_description(pbar, batch_loss, (epoch_loss/bnum))
                # Write to tensorboard
                # self.writer.add_scalar('Batch loss', batch_loss, writer_index)
                # self.writer.add_scalar('Running loss', numpy.mean(run_loss_list), writer_index)
                run_loss_list = run_loss_list[-batch_size:]
                writer_index += 1
            # Summarize the batches
            pbar.close()
            epoch_loss = epoch_loss/batch_size
            self.ls.print('Epoch loss: %8.3e' %  epoch_loss)
            self.writer.add_scalar('Epoch loss', epoch_loss, epoch)
            # Test every so often
            if 0 == epoch % self.config.test_interval and self.tester is not None:
                try:
                    self.ls.print('Testing')
                    results = self.tester.run_test()
                    # Precision / Recall
                    # Write to the screen and to the log file
                    single_scores, pair_scores = self.tester.get_precision_recall_scores(results)
                    self.ls.print('   Single: %s' % str(single_scores))
                    self.ls.print('   Pair:   %s' % str(pair_scores))
                    # Write to the Tensorboard writer
                    precision, recall, f1 = single_scores.get_precision_recall_f1()
                    self.writer.add_scalar('Test Single Precision', precision, epoch)
                    self.writer.add_scalar('Test Single Recall', recall, epoch)
                    precision, recall, f1 = pair_scores.get_precision_recall_f1()
                    self.writer.add_scalar('Test Pair Precision', precision, epoch)
                    self.writer.add_scalar('Test Pair Recall', recall, epoch)
                    # Scoring Scoring
                    results_dir = os.path.join(self.model_dir, 'coref_test')
                    scores_fn   = os.path.join(results_dir, 'scores.txt')
                    gold_dir, pred_dir, _ = cluster_and_save_sdata(self.tester.mdata, results['s_probs'],
                                            results['p_probs'], results_dir, self.config.greedyness)
                    results = scorch.get_scores(gold_dir, pred_dir)
                    if 'CoNll-2012' in results:
                        conll = results['CoNll-2012']
                        self.ls.print('   CoNLL-2012 average score: %.3f' % conll)
                        self.writer.add_scalar('CoNLL-2012', conll, epoch)
                    else:
                        self.ls.print('No results from scorch testing')
                except:
                    self.ls.print('Exception while testing')
            # number of epochs left
            epochs_left = num_epochs - (epoch - start_epoch) - 1
            # Save every so often, always save on last epoch
            if (0 == epoch % self.config.save_interval) or (epochs_left == 0):
                self.ls.print('Saving model to %s' % self.model_dir)
                self.model.save(self.model_dir, epoch, loss_type, self.optimizer)
            # Print epoch timing stats
            if epochs_left > 0:
                dur         = time.time() - epoch_st
                remain      = dur * epochs_left
                remain_h    = remain // 3600
                remain_m    = int(round((remain - remain_h * 3600) / 60.0))
                self.ls.print('Epoch duration was %d seconds. %dh%02dm remaining for %d epochs.' % \
                    (dur, remain_h, remain_m, epochs_left))

    # Update tqdm progress bar description
    @staticmethod
    def set_pbar_description(pbar, loss, av_loss):
        desc = 'Loss: '
        if loss is None:
            desc += ' '*8
        elif isinstance(loss, str):
            desc += '%-8s' % loss
        else:
            desc += '%8.3e' % loss
        desc += '  av='
        if av_loss is None:
            desc += ' '*8
        else:
            desc += '%8.3e' % av_loss
        pbar.set_description(desc)
