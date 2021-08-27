"""
Compute metrics during training
"""
import torch
from .categories import extended_hate_categories
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_hate_metrics(pred):
    """
    Compute metrics for hatespeech classifier
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    _, _, macro_f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'macro_f1': macro_f1,
        'precision': precision,
        'recall': recall
    }


def compute_category_metrics(pred):
    """
    Compute metrics for hatespeech category classifier
    """

    labels = pred.label_ids
    preds = torch.sigmoid(torch.Tensor(pred.predictions)).round()

    ret = {
    }
    """
    Calculo F1 por cada posición. Asumo que cada categoría está alineada correctamente en la i-ésima posición
    """
    f1s = []
    precs = []
    recalls = []
    for i, cat in enumerate(extended_hate_categories):
        cat_labels, cat_preds = labels[:, i], preds[:, i]
        precision, recall, f1, _ = precision_recall_fscore_support(
            cat_labels, cat_preds, average='binary'
        )

        f1s.append(f1)
        precs.append(precision)
        recalls.append(recall)

        ret[cat.lower()+"_f1"] = f1

    ret["mean_f1"] = torch.Tensor(f1s).mean()
    ret["mean_precision"] = torch.Tensor(precs).mean()
    ret["mean_recall"] = torch.Tensor(recalls).mean()
    return ret


def compute_extended_category_metrics(dataset, pred):
    """
    Add F1 for Task A
    """

    metrics = compute_category_metrics(pred)
    hate_true = dataset["HATEFUL"]
    hate_pred = ((pred.predictions[:, 1:] > 0).sum(axis=1) > 0).astype(int)

    prec, recall, f1, _ = precision_recall_fscore_support(hate_true, hate_pred, average="binary")

    metrics.update({
        "hate_precision": prec,
        "hate_recall": recall,
        "hate_f1": f1,
    })
    return metrics
