import numpy


# Precision, Recall, F1 Scores
# Precision = tp/(tp+fp)    tp=>true positive
# Recall    = tp/(tp+fn)    fn=>false negative
class PRScorer(object):
    def __init__(self, thresh=0.5):
        self.thresh = thresh  # val > thresh ==> positive
        self.y_true = []
        self.y_pred = []

    def reset(self):
        self.y_true.clear()
        self.y_pred.clear()

    # Add a single score
    def add_score(self, y_true, y_pred):
        self.y_true.append(int(y_true > self.thresh))
        self.y_pred.append(int(y_pred > self.thresh))

    # Add a list scores (float values)
    def add_scores(self, y_true, y_pred):
        assert len(y_true) > 0
        assert len(y_true) == len(y_pred)
        self.y_true += [int(val > self.thresh) for val in y_true]
        self.y_pred += [int(val > self.thresh) for val in y_pred]

    def get_precision_recall_f1(self):
        # from sklearn.metrics import precision_score, recall_score
        # precision = precision_score(self.y_true, self.y_pred)
        # recall    = recall_score(   self.y_true, self.y_pred)
        tp, tn, fp, fn = self.get_counts()
        if (tp + fp) == 0 or (tp + fn) == 0:
            return 0, 0, 0
        precision = tp / (tp + fp)
        recall    = tp / (tp + fn)
        if precision + recall > 0:
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0
        return precision, recall, f1

    # return tp, tn, fp, fn
    def get_counts(self):
        y_true = numpy.array(self.y_true, dtype='int')
        y_pred = numpy.array(self.y_pred, dtype='int')
        cm = numpy.zeros(shape=(2,2), dtype='int')
        numpy.add.at(cm, (y_true, y_pred), 1)       # cm[true,pred] = count
        return int(cm[1,1]), int(cm[0,0]), int(cm[0,1]), int(cm[1,0])

    def __str__(self):
        precision, recall, f1 = self.get_precision_recall_f1()
        string  = ''
        string += 'Precision: {:5.2f}   Recall: {:5.2f}   F1: {:5.2f}'.format(100.*precision, 100.*recall, 100.*f1)
        return string


# Use set definition to create precision/recall scores
class PRScorerForSets(object):
    def __init__(self):
        self.gold_set = set()
        self.pred_set = set()

    def add_gold(self, gold):
        self.gold_set.add(gold)

    def add_pred(self, pred):
        self.pred_set.add(pred)

    def get_precision_recall_f1(self):
        len_gold, len_pred, num_intersect, _ = self.get_counts()
        precision = num_intersect / len_pred
        recall    = num_intersect / len_gold
        f1        = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    def get_counts(self):
        num_intersect = len(self.gold_set.intersection(self.pred_set))
        num_missing   = len(self.gold_set.difference(self.pred_set))
        return len(self.gold_set), len(self.pred_set), num_intersect, num_missing

    def __str__(self):
        precision, recall, f1 = self.get_precision_recall_f1()
        string  = ''
        string += 'Precision: {:5.2f}   Recall: {:5.2f}   F1: {:5.2f}'.format(100.*precision, 100.*recall, 100.*f1)
        return string
