"""Fit a preregistered simple score+Age stack on complete image-only OOF scores."""
from cpu_utils import *
import argparse
from analyze_age import age_value
from oof_utils import load_verified_oof,load_official_valid


def logistic_fit(x,y,penalty=.001):
    weights=np.zeros(x.shape[1])
    regularizer=np.full(x.shape[1],penalty);regularizer[0]=0
    def objective(w):
        z=x@w
        return float((np.logaddexp(0,z)-y*z).mean()+.5*np.sum(regularizer*w*w))
    for step in range(100):
        z=np.clip(x@weights,-40,40);p=1/(1+np.exp(-z))
        gradient=x.T@(p-y)/len(y)+regularizer*weights
        if np.max(np.abs(gradient))<1e-8:break
        hessian=(x.T*(p*(1-p)))@x/len(y)+np.diag(regularizer+1e-10)
        direction=np.linalg.solve(hessian,gradient);old=objective(weights);rate=1.
        while rate>1e-8 and objective(weights-rate*direction)>old:rate*=.5
        weights-=rate*direction
    return weights


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--oof-root',required=True,help='Contains fold0..fold4 model run directories')
    parser.add_argument('--valid-predictions',required=True,help='Frozen full-training image-only model predictions')
    parser.add_argument('--output',required=True)
    parser.add_argument('--data-root',default=str(DATA_ROOT))
    args=parser.parse_args()
    train=read_csv(Path(args.data_root)/'train.csv')
    valid=read_csv(Path(args.data_root)/'valid.csv')
    train_probability,provenance=load_verified_oof(args.oof_root,require_image_only=True,data_root=args.data_root)
    valid_probability=load_official_valid(args.valid_predictions,data_root=args.data_root)
    valid_ages=np.array([age_value(row) for row in train if age_value(row) is not None])
    median,mean,std=float(np.median(valid_ages)),float(valid_ages.mean()),float(valid_ages.std())
    def features(rows,probabilities):
        p=np.clip(np.array(probabilities),1e-6,1-1e-6)
        age=np.array([median if age_value(row) is None else age_value(row) for row in rows])
        missing=np.array([age_value(row) is None for row in rows],dtype=float)
        return np.column_stack([np.log(p/(1-p)),(age-mean)/std,missing])
    x=features(train,train_probability)
    mu,scale=x.mean(axis=0),x.std(axis=0);scale=np.where(scale<1e-8,1.,scale)
    x=np.column_stack([np.ones(len(x)),(x-mu)/scale])
    y=np.array([int(row['True_Label']) for row in train])
    coefficient=logistic_fit(x,y)
    vx=features(valid,valid_probability)
    vx=np.column_stack([np.ones(len(vx)),(vx-mu)/scale])
    probability=1/(1+np.exp(-np.clip(vx@coefficient,-40,40)))
    output=Path(args.output)
    if (output/'stack.json').exists():raise FileExistsError('Do not overwrite frozen stacking results')
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'stack.json',{'coefficients':coefficient.tolist(),'feature_mean':mu.tolist(),
               'feature_std':scale.tolist(),'age_stats':{'median':median,'mean':mean,'std':std},
               'features':['intercept','image_logit','standardized_age','age_missing'],
               'l2_penalty':.001,'fitted_on':'complete official-train image-only OOF scores and train labels',
               'official_valid_for_fit':False,'test_used':False,
               'oof_provenance':provenance,'valid_csv_sha256':sha256(args.valid_predictions),
               'validation':metrics(np.array([int(row['True_Label']) for row in valid]),probability)})
    with (output/'valid_stack.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['Image_name','True_Label','Probability'])
        writer.writerows((row['Image_name'],row['True_Label'],p) for row,p in zip(valid,probability))


if __name__=='__main__':main()
