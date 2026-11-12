"""Validation-only Age ablations; no fitting, threshold search, or test inference."""
import sys
if __name__=='__main__' and '--enable-gpu' not in sys.argv:
    raise SystemExit('Age GPU evaluation is disabled until an allocated GPU is explicitly enabled.')
from common import *
import argparse
from core import FundusClassifier,encode_age
from data_v3 import make_dataset
from torch.utils.data import DataLoader


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--enable-gpu',action='store_true',required=True)
    parser.add_argument('--physical-gpu',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--cache-root')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=args.physical_gpu:raise RuntimeError('Allocated GPU mapping mismatch')
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert state['config']['use_age'],'Age sensitivity requires an Age model'
    model=FundusClassifier(state['config'],False).cuda();model.load_state_dict(state['model'],strict=True);model.eval()
    dataset=make_dataset(args.data_root,'valid',state['age_stats'],args.cache_root,state['config']['image_size'])
    loader=DataLoader(dataset,batch_size=32,shuffle=False,num_workers=4,pin_memory=True)
    rows=read_rows(args.data_root,'valid')
    original_age=torch.stack([encode_age(row,state['age_stats']) for row in rows])
    permutation=np.random.default_rng(20261103).permutation(len(rows))
    shuffled_age=original_age[torch.from_numpy(permutation)]
    median=encode_age({'Age':state['age_stats']['median']},state['age_stats'])
    missing=encode_age({'Age':''},state['age_stats'])
    variants={key:[] for key in ('real_age','permuted_age','median_observed','all_missing')}
    names=[];labels=[];offset=0
    with torch.inference_mode():
        for batch in loader:
            count=len(batch['name']);image=batch['image'].cuda(non_blocking=True)
            ages={'real_age':batch['age'],'permuted_age':shuffled_age[offset:offset+count],
                  'median_observed':median.expand(count,-1),'all_missing':missing.expand(count,-1)}
            with torch.autocast('cuda',dtype=torch.float16):
                image_feature=model.encoder(image)
                for key,age in ages.items():
                    feature=model.age_mlp(age.cuda(non_blocking=True))
                    z=model.head(torch.cat([image_feature,feature],dim=1)).flatten()
                    variants[key].extend(z.float().sigmoid().cpu().tolist())
            names.extend(batch['name']);labels.extend(batch['label'].int().tolist());offset+=count
    assert names==[row['Image_name'] for row in rows]
    from cpu_utils import metrics
    original=np.array(variants['real_age'])
    result={'checkpoint_sha256':sha256(args.checkpoint),'age_stats':state['age_stats'],
        'permutation_seed':20261103,'validation_only':True,'test_used':False,'parameters_fitted':False,
        'metrics':{key:metrics(np.array(labels),np.array(p)) for key,p in variants.items()},
        'mean_absolute_probability_change':{key:float(np.abs(np.array(p)-original).mean()) for key,p in variants.items()},
        'interpretation':'Ablations measure sensitivity, not clinical causal effects. Missing all ages is a distribution shift; sensitivity alone does not establish harmful reliance.'}
    output=Path(args.output)
    if (output/'metrics.json').exists():raise FileExistsError('Do not overwrite an existing sensitivity result')
    output.mkdir(parents=True,exist_ok=True);write_json(output/'metrics.json',result)
    with (output/'valid_age_sensitivity.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['Image_name','True_Label',*variants])
        writer.writerows(zip(names,labels,*variants.values()))


if __name__=='__main__':main()
