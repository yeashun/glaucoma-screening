import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve,roc_curve,auc,confusion_matrix

def compute_metrics(y,p):
    y=np.asarray(y,dtype=int)
    p=np.asarray(p,dtype=float)
    if not np.isfinite(p).all():
        raise ValueError('Nonfinite predictions')
    precision,recall,_=precision_recall_curve(y,p)
    fpr,tpr,threshold=roc_curve(y,p)
    out={'auroc':float(roc_auc_score(y,p)),'ap':float(average_precision_score(y,p)),
         'pr_auc_trapezoid':float(auc(recall,precision)),
         'confusion_05':confusion_matrix(y,p>=.5,labels=[0,1]).tolist(),
         'n':int(len(y)),'positive':int(y.sum())}
    for specificity in (.95,.90):
        allowed=np.flatnonzero(fpr<=1-specificity+1e-12)
        best=allowed[np.argmax(tpr[allowed])]
        out[f'sensitivity_at_specificity_{specificity}']=float(tpr[best])
        out[f'threshold_at_specificity_{specificity}']=float(threshold[best]) if np.isfinite(threshold[best]) else None
    return out
