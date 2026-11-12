import argparse
import csv
import json
import hashlib
import signal
import time
from pathlib import Path
import numpy as np
import torch
from glaucoma.data import open_rgb,valid_bbox,image_tensor,encode_age,read_rows
from glaucoma.models import load_checkpoint
from glaucoma.roi import box_from_map

class Predictor:
    def __init__(self,checkpoint,device='cuda',cam_checkpoint=None):
        self.device=torch.device(device)
        self.model,self.checkpoint=load_checkpoint(checkpoint,'cpu')
        self.model.to(self.device).eval()
        self.config=self.checkpoint['config']
        self.size=self.checkpoint.get('metrics',{}).get('image_size',self.config.get('image_size',384))
        self.cam=None
        if self.config.get('kind') in ('local','fusion'):
            cam_path=cam_checkpoint or self.config.get('cam_checkpoint')
            if not cam_path:
                raise ValueError('Local/fusion inference requires its frozen CAM checkpoint')
            if cam_checkpoint is None and not Path(cam_path).is_absolute():
                cam_path=Path(checkpoint).resolve().parent/cam_path
            self.cam=torch.load(cam_path,map_location='cpu',weights_only=False)
            self.size=self.cam['input_size']

    @torch.inference_mode()
    def __call__(self,path,row):
        image=open_rgb(path)
        bbox=valid_bbox(image)
        tensor=image_tensor(image.crop(bbox),self.size)[None].to(self.device)
        age=encode_age(row,self.checkpoint['age_stats'])[None].to(self.device)
        with torch.autocast(device_type=self.device.type,dtype=torch.float16,enabled=self.device.type=='cuda'):
            if self.cam is None:
                logits=self.model(tensor,age)
            else:
                global_features,fmap=self.model.encoder(tensor,return_map=True)
                box=box_from_map(fmap[0].float(),self.cam['weight'],bbox,self.size,self.cam['fraction'])
                local=image_tensor(image.crop(box),self.config.get('local_size',320))[None].to(self.device)
                local_features=self.model.local_encoder(local)
                logits=self.model.classify_features(global_features,age,local_features)
            probability=logits.float().sigmoid().item()
        if not np.isfinite(probability):
            raise ValueError('Nonfinite prediction')
        return probability

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data-root',required=True)
    p.add_argument('--split',default='test',choices=['train','valid','test'])
    p.add_argument('--output',default='submission.csv')
    p.add_argument('--cam-checkpoint')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(4)
    predictor=Predictor(args.checkpoint,cam_checkpoint=args.cam_checkpoint)
    root=Path(args.data_root)
    paths={f.name.casefold():f for f in (root/args.split).iterdir() if f.is_file()}
    rows=read_rows(root,args.split)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_suffix(output.suffix+'.partial')
    fingerprint_path=output.with_suffix(output.suffix+'.metadata.json')
    fingerprint={'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                 'input_csv_sha256':hashlib.sha256((root/f'{args.split}.csv').read_bytes()).hexdigest(),
                 'image_size':predictor.size,'total_rows':len(rows),'split':args.split}
    start=0
    if args.resume and partial.exists():
        if json.loads(fingerprint_path.read_text())!=fingerprint:
            raise ValueError('Partial predictions belong to different inputs/checkpoint')
        with partial.open(newline='',encoding='utf-8') as f:
            previous=list(csv.DictReader(f))
        for i,row in enumerate(previous):
            if row['Image_name']!=rows[i]['Image_name'] or not 0<=float(row['Probability'])<=1:
                raise ValueError('Invalid partial prediction prefix')
        start=len(previous)
    elif partial.exists():
        raise FileExistsError('Partial predictions exist; use --resume or a new output path')
    fingerprint_path.write_text(json.dumps(fingerprint,indent=2))
    progress={'predicted':start,'total':len(rows)}
    def interrupted(signum,frame):
        print(json.dumps({'event':'signal','signal':signum,**progress}),flush=True)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGHUP,interrupted)
    with partial.open('a' if start else 'w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        if not start:
            writer.writerow(['Image_name','Probability','Pred_Label'])
            f.flush()
        for i in range(start,len(rows)):
            row=rows[i]
            probability=predictor(paths[row['Image_name'].casefold()],row)
            writer.writerow([row['Image_name'],format(probability,'.9g'),int(probability>=.5)])
            progress['predicted']=i+1
            if i%25==0:
                f.flush()
            if i%100==0:
                print(json.dumps({'predicted':i+1,'total':len(rows)}),flush=True)
    partial.replace(output)
    print(json.dumps({'output':str(output),'rows':len(rows)}))

if __name__=='__main__':
    main()
