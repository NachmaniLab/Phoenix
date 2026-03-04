import numpy as np


def compute_f1(y_true, y_pred, label, beta=1):
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    tp = int(np.sum((yt == label) & (yp == label)))
    fp = int(np.sum((yt != label) & (yp == label)))
    fn = int(np.sum((yt == label) & (yp != label)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if precision + recall == 0:
        return 0

    f1 = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)
    return f1


def compute_recall(y_true, y_pred, label):
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    tp = int(np.sum((yt == label) & (yp == label)))
    fn = int(np.sum((yt == label) & (yp != label)))
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return recall


def normalized_inverse_class_frequency(y_true):
    """Compute normalized inverse class frequency (NICF) for each class"""
    yt = np.asarray(y_true)
    labels, counts = np.unique(yt, return_counts=True)
    freqs = counts / len(yt)
    inverse = 1.0 / freqs
    normalized = inverse / inverse.sum()
    return dict(zip(labels.tolist(), normalized.tolist()))


def weighted_metric_using_icf(y_true, y_pred, compute_metric, metric_args=None):
    metric_args = metric_args or {}
    normalized_icf = normalized_inverse_class_frequency(y_true)
    weighted_metric = 0.0
    for label, weight in normalized_icf.items():
        metric = compute_metric(y_true, y_pred, label, **metric_args)
        weighted_metric += weight * metric
    return weighted_metric
