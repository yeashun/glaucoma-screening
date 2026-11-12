"""Validation-only horizontal-flip TTA, enabled only after explicit allocation."""
import sys
if __name__=='__main__' and '--enable-gpu' not in sys.argv:
    raise SystemExit('TTA GPU inference is disabled until a GPU is allocated; pass --enable-gpu explicitly.')
from common import *
import argparse
from core import FundusClassifier
from data_v3 import make_dataset
from torch.utils.data import DataLoader


@torch.inference_mode()
def evaluate_variants(model,loader,device):
    model.eval()
    names,labels,original,flipped=[],[],[],[]
    for batch in loader:
        image=batch['image'].to(device,non_blocking=True)
        age=batch['age'].to(device,non_blocking=True)
        with torch.autocast('cuda',dtype=torch.float16,enabled=device.type=='cuda'):
            z=model(image,age)
            flip_z=model(image.flip(-1),age)
        names.extend(batch['name']);labels.extend(batch['label'].int().tolist())
        original.extend(z.float().sigmoid().cpu().tolist())
        flipped.extend(flip_z.float().sigmoid().cpu().tolist())
    original,flipped=np.asarray(original),np.asarray(flipped)
    return names,np.asarray(labels),original,flipped,.5*(original+flipped)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--enable-gpu',action='store_true',required=True)
    parser.add_argument('--physical-gpu',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--cache-root')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=args.physical_gpu:
        raise RuntimeError('CUDA_VISIBLE_DEVICES must match the allocated physical GPU')
    from cpu_utils import bootstrap_counts,weighted_auc,difference_summary,metrics
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=FundusClassifier(state['config'],False).cuda();model.load_state_dict(state['model'],strict=True)
    dataset=make_dataset(args.data_root,'valid',state['age_stats'],args.cache_root,state['config']['image_size'])
    loader=DataLoader(dataset,batch_size=32,shuffle=False,num_workers=4,pin_memory=True)
    names,y,p,q,tta=evaluate_variants(model,loader,torch.device('cuda'))
    original_metrics,tta_metrics=metrics(y,p),metrics(y,tta)
    counts=bootstrap_counts(y)
    paired=difference_summary(weighted_auc(y,tta,counts),weighted_auc(y,p,counts),tta_metrics['auroc']-original_metrics['auroc'])
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    with (output/'valid_tta.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['Image_name','True_Label','Original','Flip','Probability'])
        writer.writerows(zip(names,y.tolist(),p.tolist(),q.tolist(),tta.tolist()))
    write_json(output/'metrics.json',{'original':original_metrics,'tta':tta_metrics,'paired_bootstrap':paired,
               'checkpoint_sha256':sha256(args.checkpoint),'official_valid_only':True,'test_predicted':False,
               'deployment_eligible':False,'reason':'T4 batch1 end-to-end latency must be verified separately'})


if __name__=='__main__':main()
