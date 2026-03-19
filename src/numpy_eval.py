import numpy as np


def confusion_matrix_np(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        cm[yt, yp] += 1
    return cm


def per_class_precision_recall_f1(cm):
    n_classes = cm.shape[0]
    precision = np.zeros(n_classes, dtype=np.float64)
    recall = np.zeros(n_classes, dtype=np.float64)
    f1 = np.zeros(n_classes, dtype=np.float64)
    support = cm.sum(axis=1)

    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        precision[i] = p
        recall[i] = r
        f1[i] = f

    return precision, recall, f1, support


def macro_metrics(cm):
    precision, recall, f1, _ = per_class_precision_recall_f1(cm)
    return {
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
    }


def accuracy_np(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def weighted_f1(cm):
    _, _, f1, support = per_class_precision_recall_f1(cm)
    total = support.sum()
    if total == 0:
        return 0.0
    return float(np.sum(f1 * support) / total)


def balanced_accuracy(cm):
    _, recall, _, _ = per_class_precision_recall_f1(cm)
    return float(np.mean(recall))


def print_classification_report(cm, class_names):
    precision, recall, f1, support = per_class_precision_recall_f1(cm)

    print("\nClassification report (NumPy):")
    print(f"{'class':<12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    for i, name in enumerate(class_names):
        print(f"{name:<12} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {int(support[i]):>10}")
