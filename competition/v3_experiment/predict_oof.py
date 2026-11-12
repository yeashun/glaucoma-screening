"""Freeze inner-selected weights and predict one untouched outer OOF fold."""
import sys
if __name__=='__main__' and '--enable-gpu' not in sys.argv:
    raise SystemExit('OOF GPU inference is disabled until an allocated GPU is explicitly enabled.')
from common import *
import argparse
from data_v3 import make_dataset
from core import FundusClassifier
from torch.utils.data import DataLoader


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--enable-gpu',action='store_true',required=True)
    parser.add_argument('--physical-gpu',required=True)
    parser.add_argument('--run-root',required=True)
    parser.add_argument('--split-plan',required=True)
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--cache-root')
    parser.add_argument('--weights',choices=('original','ema'),default='original')
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=args.physical_gpu:raise RuntimeError('Allocated GPU mapping mismatch')
    root=Path(args.run_root);plan=json.loads(Path(args.split_plan).read_text())
    partition=json.loads((root/'data_partition.json').read_text())
    assert partition['valid_source']=='inner_train_holdout'
    assert not partition.get('smoke',False),'Smoke outputs cannot generate formal OOF predictions'
    complete=json.loads((root/'complete.json').read_text())
    assert not complete.get('smoke',False)
    assert partition['split_plan_sha256']==sha256(args.split_plan)
    assert plan['train_csv_sha256']==sha256(Path(args.data_root)/'train.csv')
    checkpoint_file='best_ema.pt' if args.weights=='ema' else 'best.pt'
    checkpoint_hash=sha256(root/checkpoint_file)
    state=torch.load(root/checkpoint_file,map_location='cpu',weights_only=False)
    assert complete['epochs']==state['config']['epochs'],'Finish the full training schedule before outer inference'
    assert state['split_plan_sha256']==sha256(args.split_plan)
    model=FundusClassifier(state['config'],False).cuda();model.load_state_dict(state['model'],strict=True);model.eval()
    data=make_dataset(args.data_root,'train',state['age_stats'],args.cache_root,state['config']['image_size'],
                      training=False,names=plan['outer_holdout_names'])
    loader=DataLoader(data,batch_size=32,shuffle=False,num_workers=4,pin_memory=True)
    names,labels,probabilities=[],[],[]
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast('cuda',dtype=torch.float16):
                z=model(batch['image'].cuda(non_blocking=True),batch['age'].cuda(non_blocking=True))
            names.extend(batch['name']);labels.extend(batch['label'].int().tolist())
            probabilities.extend(z.float().sigmoid().cpu().tolist())
    assert set(names)==set(plan['outer_holdout_names']) and len(names)==len(set(names))
    assert checkpoint_hash==sha256(root/checkpoint_file),'Checkpoint changed during outer inference'
    output=root/'outer_oof.csv'
    if output.exists():raise FileExistsError('Do not overwrite frozen outer predictions')
    with output.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['Image_name','True_Label','Probability','outer_fold'])
        writer.writerows((name,y,p,plan['outer_fold']) for name,y,p in zip(names,labels,probabilities))
    write_json(root/'outer_oof.metadata.json',{'split_plan_sha256':sha256(args.split_plan),
               'checkpoint_sha256':checkpoint_hash,'checkpoint_file':checkpoint_file,
               'training_complete':True,'smoke_training':False,
               'weight_variant':args.weights,'prediction_sha256':sha256(output),
               'train_csv_sha256':plan['train_csv_sha256'],'outer_fold':plan['outer_fold'],
               'checkpoint_selected_using':'inner_valid_only','official_valid_used':False,'use_age':state['config']['use_age']})


if __name__=='__main__':main()
