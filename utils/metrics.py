from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
import numpy as np

class Metrics():
    
    def on_train_begin(self):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_auc = []
        self.val_acc = []

    def on_batch_end(self,epoch, batch ,val_predict, val_targ):

        _val_f1 = f1_score(val_targ, val_predict, average = 'weighted')
        _val_recall = recall_score(val_targ, val_predict, average = 'weighted')
        _val_precision = precision_score(val_targ, val_predict, average = 'weighted')
        _val_acc = accuracy_score(val_targ, val_predict)

        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_acc.append(_val_acc)

        print("Epoch: %d - batch: %d- val_accuracy: % f - val_precision: % f - val_recall % f val_f1: %f " % (
            epoch, batch ,_val_acc, _val_precision, _val_recall, _val_f1))

    def on_epoch_end(self, epoch):

        f1 = np.mean(self.val_f1s)
        recall = np.mean(self.val_recalls)
        precision = np.mean(self.val_precisions)
        acc = np.mean(self.val_acc)
        
        print('-' * 100)
        print("Epoch: %d - val_accuracy: % f - val_precision: % f - val_recall % f val_f1: %f " % (
            epoch, acc, precision, recall, f1))
        print('-' * 100)
        return acc, f1