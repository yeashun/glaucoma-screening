import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from glaucoma.data import FundusDataset
from glaucoma.models import load_checkpoint
from glaucoma.roi import box_from_map

p=argparse.ArgumentParser()
p.add_argument('--global-checkpoint',required=True)
p.add_argument('--data-root',required=True)
p.add_argument('--output',default='data/rois.csv')
p.add_argument('--cam-checkpoint',default='runs/roi/cam.pt')
p.add_argument('--fraction',type=float,default=.45)
p.add_argument('--batch-size',type=int,default=16)
args=p.parse_args()
torch.set_num_threads(4)
torch.manual_seed(42)
model,checkpoint=load_checkpoint(args.global_checkpoint)
model.cuda().eval()
size=checkpoint['metrics']['image_size']
stats=checkpoint['age_stats']
root=Path(args.data_root)
excluded=json.loads((root/'excluded_train.json').read_text()) if (root/'excluded_train.json').exists() else []
train=FundusDataset(root,'train',stats,size,False,excluded=excluded)
loader=DataLoader(train,batch_size=args.batch_size,num_workers=8,pin_memory=True)
features=[]
labels=[]
with torch.inference_mode():
    for i,batch in enumerate(loader):
        with torch.autocast('cuda',dtype=torch.float16):
            pooled=model.encoder(batch['image'].cuda(non_blocking=True))
        features.append(pooled.float().cpu())
        labels.append(batch['label'])
        if i%100==0:
            print(json.dumps({'stage':'cam_features','batch':i,'batches':len(loader)}),flush=True)
x=torch.cat(features).cuda()
y=torch.cat(labels).cuda()
cam=torch.nn.Linear(x.shape[1],1).cuda()
optimizer=torch.optim.AdamW(cam.parameters(),lr=1e-3,weight_decay=1e-4)
for step in range(500):
    indices=torch.randint(len(x),(512,),device='cuda')
    loss=torch.nn.functional.binary_cross_entropy_with_logits(cam(x[indices]).flatten(),y[indices])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
weight=cam.weight.detach().float().clone()
cam_path=Path(args.cam_checkpoint)
cam_path.parent.mkdir(parents=True,exist_ok=True)
cam_state={'weight':weight.cpu(),'bias':cam.bias.detach().cpu(),'input_size':size,'fraction':args.fraction,
           'global_checkpoint':str(Path(args.global_checkpoint).resolve()),
           'global_sha256':hashlib.sha256(Path(args.global_checkpoint).read_bytes()).hexdigest(),
           'method':'Train-label-only linear CAM; no gradients/updates at inference.'}
torch.save(cam_state,cam_path)
del x,y,features,labels
output=Path(args.output)
output.parent.mkdir(parents=True,exist_ok=True)
with output.open('w',newline='',encoding='utf-8') as f:
    writer=csv.writer(f)
    writer.writerow(['split','Image_name','x0','y0','x1','y1'])
    for split in ['train','valid']:
        dataset=FundusDataset(root,split,stats,size,False,excluded=excluded if split=='train' else None)
        loader=DataLoader(dataset,batch_size=args.batch_size,num_workers=8,pin_memory=True)
        with torch.inference_mode():
            for i,batch in enumerate(loader):
                with torch.autocast('cuda',dtype=torch.float16):
                    _,maps=model.encoder(batch['image'].cuda(non_blocking=True),return_map=True)
                for j,name in enumerate(batch['name']):
                    box=box_from_map(maps[j].float(),weight,batch['bbox'][j].tolist(),size,args.fraction)
                    writer.writerow([split,name,*box])
                if i%100==0:
                    print(json.dumps({'stage':'roi','split':split,'batch':i,'batches':len(loader)}),flush=True)
print(json.dumps({'roi_csv':str(output),'cam_checkpoint':str(cam_path)}),flush=True)
