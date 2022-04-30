from cmath import isnan
import torch
import numpy as np
from typing import Tuple, Union
from vocab import Vocab
from sklearn.metrics import roc_auc_score,f1_score, average_precision_score
    
np.seterr(invalid='ignore')

def multi_hot_to_codes(multi_hot: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    # print(f"[multi_hot_to_codes] multi_hot.shape: {multi_hot.shape}")
    assert len(multi_hot.shape) == 1
    code_list = torch.where(multi_hot == 1)[0]
    return code_list.detach().numpy()


def multi_label_metric(y_true: torch.Tensor, y_pred: torch.Tensor, y_pred_prob:torch.Tensor):

    def jaccard(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        score = []
        for i, target in enumerate(y_true):
            target = multi_hot_to_codes(target)
            predict = multi_hot_to_codes(y_pred[i])
            intersection = set(predict) & set(target)
            union = set(predict) | set(target)
            jaccard_score = 0 if len(union) == 0 else len(intersection) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def average_precision_recall_f1(y_true: torch.Tensor, y_pred: torch.Tensor) -> Tuple[float, float, float]:
        precision_scores = []
        recall_scores = []
        f1_scores = []
        for i, target in enumerate(y_true):
            target = multi_hot_to_codes(target)
            predict = multi_hot_to_codes(y_pred[i])
            intersection = set(predict) & set(target)
            if len(predict) == 0:
                # print("[multi_label_metric::average_precision] predicted : ", i)
                prc_score = 0
            else:
                prc_score = len(intersection) / len(predict)
            precision_scores.append(prc_score)

            if len(target) == 0:
                # print("[multi_label_metric::average_recall] something weird with idx: ", i)
                recall_score = 0
            else:
                recall_score = len(intersection) / len(target)
            recall_scores.append(recall_score)
            
            if prc_score + recall_score == 0:
                f1_scores.append(0)
            else:
                f1_scores.append(2*prc_score*recall_score / (prc_score + recall_score))      
            
        return np.mean(precision_scores), np.mean(recall_scores), np.mean(f1_scores)

    def precision_at_k(y_true:torch.Tensor, y_pred_prob: torch.Tensor, k=3) -> float:
        precision = 0
        sort_index = torch.argsort(y_pred_prob, axis=-1)[:, -k:]
        for i, target in enumerate(y_true):
            true_positives = 0
            assert k == len(sort_index[i]), "Ideally they should be equal"
            # if y_true[i, sort_index[i, 0]] == 1:
            #     print("[precision_at_k]: Highest matching prob:", y_pred_prob[i, sort_index[i, 0]])
            for j in range(k):
                if y_true[i, sort_index[i, j]] == 1:
                    true_positives += 1
            precision += true_positives / k
        return precision / len(y_true)

    # sklearn based metrics:    
    def sklearn_f1(y_true, y_pred):
        macro_f1 = []
        for idx, target in enumerate(y_true):
            macro_f1.append(f1_score(target, y_pred[idx], average='macro'))
        return np.mean(macro_f1)

    def sklearn_roc_auc(y_true: np.array, y_pred_prob: np.array) -> float:
        roc_auc = []
        for idx, target in enumerate(y_true):
            roc_auc.append(roc_auc_score(target, y_pred_prob[idx], average='macro'))
        return np.mean(roc_auc)

    def sklearn_pr_auc(y_true: np.array, y_pred_prob: np.array):
        precision_score = []
        for idx, target in enumerate(y_true):
            precision_score.append(average_precision_score(target, y_pred_prob[idx])) #  TODO: need to check whether this is reqd? average='macro'
        return np.mean(precision_score)


    j_accard = jaccard(y_true, y_pred)
    avg_recall, avg_prc, avg_f1 = average_precision_recall_f1(y_true, y_pred)
    # p_1 = precision_at_k(y_true, y_pred_prob, k=1)
    # p_3 = precision_at_k(y_true, y_pred_prob, k=3)
    # p_5 = precision_at_k(y_true, y_pred_prob, k=5)
    # roc_auc = sklearn_roc_auc(y_true.detach().numpy(), y_pred_prob.detach().numpy())
    pr_auc = sklearn_pr_auc(y_true.detach().numpy(), y_pred_prob.detach().numpy())
    if isnan(pr_auc):
        pr_auc = 0

    return j_accard, pr_auc, avg_f1
