"""Fixed-L2 logistic model fusion fitted only on complete training OOF scores."""
from cpu_utils import *
import argparse
from fit_age_stack import logistic_fit
from oof_utils import load_verified_oof,load_official_valid


def named_paths(items):
    result={}
    for item in items:
        name,separator,path=item.partition('=')
        if not separator or not name or not path or name in result:
            raise ValueError('Use unique NAME=PATH entries')
        result[name]=Path(path)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--oof-root',action='append',required=True,help='Repeat NAME=OOF_ROOT')
    parser.add_argument('--valid-predictions',action='append',required=True,help='Repeat NAME=VALID_CSV')
    parser.add_argument('--output',required=True)
    parser.add_argument('--data-root',default=str(DATA_ROOT))
    args=parser.parse_args()
    roots=named_paths(args.oof_root);valid_paths=named_paths(args.valid_predictions)
    assert set(roots)==set(valid_paths) and len(roots)>=2
    names=list(roots)
    train=read_csv(Path(args.data_root)/'train.csv')
    valid=read_csv(Path(args.data_root)/'valid.csv')
    train_y=np.array([int(row['True_Label']) for row in train])
    valid_y=np.array([int(row['True_Label']) for row in valid])
    oof=[];validation=[];provenance={}
    for name in names:
        probability,records=load_verified_oof(roots[name],data_root=args.data_root)
        oof.append(probability)
        validation.append(load_official_valid(valid_paths[name],data_root=args.data_root))
        assert np.std(validation[-1])>0,'Constant validation scores cannot support correlation analysis'
        provenance[name]={'oof_folds':records,'valid_csv_sha256':sha256(valid_paths[name])}
    def logits(p):
        p=np.clip(p,1e-6,1-1e-6)
        return np.log(p/(1-p))
    x=logits(np.column_stack(oof))
    mean=x.mean(axis=0);scale=x.std(axis=0);scale=np.where(scale<1e-8,1.,scale)
    design=np.column_stack([np.ones(len(x)),(x-mean)/scale])
    coefficient=logistic_fit(design,train_y,penalty=.001)
    # No validation labels are accessed by feature scaling or coefficient fitting.
    vx=logits(np.column_stack(validation))
    vx=np.column_stack([np.ones(len(vx)),(vx-mean)/scale])
    fused=1/(1+np.exp(-np.clip(vx@coefficient,-40,40)))
    counts=bootstrap_counts(valid_y)
    fused_bootstrap=weighted_auc(valid_y,fused,counts)
    single={name:metrics(valid_y,p) for name,p in zip(names,validation)}
    fused_metrics=metrics(valid_y,fused)
    comparisons={name:difference_summary(fused_bootstrap,weighted_auc(valid_y,p,counts),
                    fused_metrics['auroc']-single[name]['auroc']) for name,p in zip(names,validation)}
    pearson=np.corrcoef(np.column_stack(validation),rowvar=False)
    spearman=np.corrcoef(np.column_stack([average_ranks(p) for p in validation]),rowvar=False)
    output=Path(args.output)
    if (output/'stack.json').exists():raise FileExistsError('Do not overwrite frozen stacking results')
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'stack.json',{'model_order':names,'coefficients':coefficient.tolist(),
        'logit_mean':mean.tolist(),'logit_std':scale.tolist(),'l2_penalty':.001,
        'fitted_on':'complete official-train OOF scores and train labels',
        'validation_for_fit':False,'test_used':False,'gpu_used':False,
        'validation_single':single,'validation_stack':fused_metrics,
        'paired_bootstrap':comparisons,'pearson':pearson.tolist(),'spearman':spearman.tolist(),
        'ordering_vs_single':{name:ordering_change(valid_y,p,fused) for name,p in zip(names,validation)},
        'provenance':provenance,
        'limitations':['Model family choice and checkpoint selection remain conditional on prior validation use.',
            'OOF input models saw less training data than full models; score-distribution shift remains possible.',
            'Patient independence and deployment latency have not been established.']})
    with (output/'valid_stack.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['Image_name','True_Label','Probability'])
        writer.writerows((row['Image_name'],row['True_Label'],p) for row,p in zip(valid,fused))


if __name__=='__main__':main()
