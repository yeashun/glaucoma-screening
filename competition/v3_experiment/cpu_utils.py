"""CPU-only utilities: this module never imports torch or initializes CUDA."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get('GLAUCOMA_DATA_ROOT', str(ROOT/'data/official')))
try:
    import numpy as np
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT / '.inference_deps'))
    import numpy as np


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def metrics(y, p):
    y, p = np.asarray(y, dtype=np.int8), np.asarray(p, dtype=np.float64)
    if len(y) != len(p) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid labels or probabilities')
    if set(np.unique(y)) != {0,1}:
        return {'n':len(y),'positive':int(y.sum()),'auroc':None,'ap':None,
                'note':'AUC is undefined for a single-class stratum'}
    order = np.argsort(-p, kind='stable')
    scores, truth = p[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores)-1]
    tp, fp = truth.cumsum()[ends], (1-truth).cumsum()[ends]
    recall, precision = tp/y.sum(), tp/(tp+fp)
    tpr, fpr = np.r_[0., recall], np.r_[0., fp/(y==0).sum()]
    threshold = np.r_[np.inf,scores[ends]]
    tn0=int(((y==0)&(p<.5)).sum()); fp0=int(((y==0)&(p>=.5)).sum())
    fn0=int(((y==1)&(p<.5)).sum()); tp0=int(((y==1)&(p>=.5)).sum())
    output={'n':len(y),'positive':int(y.sum()),'auroc':float(np.trapezoid(tpr,fpr)),
            'ap':float((np.diff(np.r_[0.,recall])*precision).sum()),
            'pr_auc_trapezoid':float(np.trapezoid(np.r_[1.,precision],np.r_[0.,recall])),
            'confusion_05':[[tn0,fp0],[fn0,tp0]],'recall_05':tp0/max(1,tp0+fn0),
            'precision_05':tp0/max(1,tp0+fp0)}
    for spec in (.95,.98):
        eligible=np.flatnonzero(fpr<=1-spec+1e-12)
        index=eligible[np.argmax(tpr[eligible])]
        output[f'recall_at_spec_{spec:.2f}']=float(tpr[index])
        output[f'threshold_at_spec_{spec:.2f}']=float(threshold[index]) if np.isfinite(threshold[index]) else None
    return output


def bootstrap_counts(y, repeats=2000, seed=20261103):
    rng=np.random.default_rng(seed)
    positive,negative=np.flatnonzero(y==1),np.flatnonzero(y==0)
    counts=np.zeros((repeats,len(y)),dtype=np.uint16)
    for i in range(repeats):
        counts[i,positive]=np.bincount(rng.integers(0,len(positive),len(positive)),minlength=len(positive))
        counts[i,negative]=np.bincount(rng.integers(0,len(negative),len(negative)),minlength=len(negative))
    return counts


def weighted_auc(y,p,counts):
    order=np.argsort(p,kind='stable')
    starts=np.r_[0,np.flatnonzero(np.diff(p[order]))+1]
    truth=y[order]
    output=[]
    for offset in range(0,len(counts),100):
        w=counts[offset:offset+100,order].astype(np.float64)
        pos=np.add.reduceat(w*(truth==1)[None,:],starts,axis=1)
        neg=np.add.reduceat(w*(truth==0)[None,:],starts,axis=1)
        wins=(pos*(neg.cumsum(axis=1)-.5*neg)).sum(axis=1)
        output.extend((wins/(pos.sum(axis=1)*neg.sum(axis=1))).tolist())
    return np.asarray(output)


def difference_summary(candidate, reference, observed):
    delta=candidate-reference
    return {'observed_delta_auroc':float(observed),'bootstrap_mean_delta':float(delta.mean()),
            'ci95_low':float(np.quantile(delta,.025)),'ci95_high':float(np.quantile(delta,.975)),
            'repeats':len(delta),'seed':20261103,'paired':True,'stratified':True,'unit':'image',
            'limitation':'Conditional on previous validation-based checkpoint selection; not patient-level independent evidence.'}


def average_ranks(p):
    order=np.argsort(p,kind='stable')
    starts=np.r_[0,np.flatnonzero(np.diff(p[order]))+1]
    ends=np.r_[starts[1:],len(p)]
    result=np.empty(len(p))
    for start,end in zip(starts,ends):
        result[order[start:end]]=(start+end-1)/2+1
    return result


def ordering_change(y,old,new):
    pos,neg=np.flatnonzero(y==1),np.flatnonzero(y==0)
    before=np.sign(old[pos,None]-old[neg][None,:])
    after=np.sign(new[pos,None]-new[neg][None,:])
    credit=(after.astype(float)-before)*.5
    return {'corrected_inverted_pairs':int(((before<0)&(after>0)).sum()),
            'new_inverted_pairs':int(((before>0)&(after<0)).sum()),
            'net_pair_credit_gain':float(credit.sum()),
            'positive_images_improved':int((credit.sum(axis=1)>0).sum()),
            'positive_images_worsened':int((credit.sum(axis=1)<0).sum()),
            'positive_misses_corrected_05':int(((y==1)&(old<.5)&(new>=.5)).sum()),
            'new_positive_misses_05':int(((y==1)&(old>=.5)&(new<.5)).sum()),
            'negative_false_positives_corrected_05':int(((y==0)&(old>=.5)&(new<.5)).sum()),
            'new_negative_false_positives_05':int(((y==0)&(old<.5)&(new>=.5)).sum())}
