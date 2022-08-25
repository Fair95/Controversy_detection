import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, jaccard_score, precision_score, recall_score, f1_score, classification_report
import torch

from constants import topic_list
class MyMetric:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.y_pred = []
        self.y_true = []

        self.target_names = topic_list
        self.sigmoid = torch.nn.Sigmoid()

    def logits_to_class(self, tensor):
        return self.sigmoid(tensor)

    def tensor_to_numpy(self, tensor):
        # next, use threshold to turn them into integer predictions
        npy = np.zeros(tensor.shape)
        npy[np.where(tensor.cpu() >= self.threshold)] = 1
        return npy

    def update(self, y_pred, y_true):
        if isinstance(y_pred, torch.Tensor):
            y_pred = self.logits_to_class(y_pred)
            y_pred = self.tensor_to_numpy(y_pred)
            y_true = self.tensor_to_numpy(y_true)
        self.y_pred.append(y_pred)
        self.y_true.append(y_true)

    def cal_statistics_and_update(self, y_pred, y_true, method='subset_accuracy'):
        self.update(y_pred, y_true)
        result = self.cal_running_statistics(peroid=len(y_pred), method=method)
        return result

    def cal_running_statistics(self, peroid=0, method='subset_accuracy'):
        y_true = np.vstack(self.y_true[-peroid:])
        y_pred = np.vstack(self.y_pred[-peroid:])
        return eval(method)(y_true, y_pred)

    def reset(self):
        self.y_pred = []
        self.y_true = []

    def __repr__(self):
        assert len(self.y_pred) > 0, 'No data recorded'
        return classification_report(np.vstack(self.y_true), np.vstack(self.y_pred), target_names=self.target_names)

def sample_accuracy(y_true, y_pred):
    return jaccard_score(y_true, y_pred, average='samples')

def sample_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average='samples')

def sample_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average='samples')

def sample_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='samples')

def subset_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def micro_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average='micro')

def micro_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average='micro')

def micro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')
  
def macro_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average='macro')

def macro_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average='macro')
  
def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

