import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F

class PurePyTorchGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(PurePyTorchGCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        return output

class PurePyTorchGCN(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, dropout=0.3):
        super(PurePyTorchGCN, self).__init__()
        self.gc1 = PurePyTorchGCNLayer(in_features, hidden_dim)
        self.gc2 = PurePyTorchGCNLayer(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

def build_normalized_adj(adj_matrix):
    adj_dense = adj_matrix.toarray() if hasattr(adj_matrix, "toarray") else adj_matrix
    A_tilde = adj_dense + np.eye(adj_dense.shape[0])
    d = np.sum(A_tilde, axis=1)
    d_inv_sqrt = np.power(d, -0.5, where=d > 0)
    d_inv_sqrt[d == 0] = 0.0
    D_tilde = np.diag(d_inv_sqrt)
    norm_adj = D_tilde @ A_tilde @ D_tilde
    return torch.tensor(norm_adj, dtype=torch.float32)

def evaluate_predictions(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    metrics = {
        'Accuracy': round(float(acc), 4),
        'Precision': round(float(prec), 4),
        'Recall': round(float(rec), 4),
        'F1 Score': round(float(f1), 4)
    }
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
            metrics['ROC_AUC'] = round(float(auc), 4)
        except Exception:
            metrics['ROC_AUC'] = None
    return metrics
