import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


def auc_performance(score, labels):
    score = np.squeeze(score)
    roc_auc = roc_auc_score(labels, score)
    ap = average_precision_score(labels, score)
    return roc_auc, ap


def f1_performance(score, target):
    score = np.squeeze(score)
    normal_ratio = (target == 0).sum() / len(target)
    threshold = np.percentile(score, 100 * normal_ratio)
    pred = np.zeros(len(score))
    pred[score > threshold] = 1
    precision, recall, f1, _ = precision_recall_fscore_support(target, pred, average="binary")
    return float(f1)
