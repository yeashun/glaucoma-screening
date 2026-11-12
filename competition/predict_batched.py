"""Fixed eval-mode batch inference with the same preprocessing as Predictor."""
import argparse
import csv
import hashlib
import json
import signal
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Subset
from glaucoma.data import FundusDataset
from predict import Predictor

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data-root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(4)
    predictor=Predictor(args.checkpoint)
    if predictor.model.kind!='global':
        raise ValueError('This batch path is for fixed global models only')
    dataset=FundusDataset(args.data_root,'test',predictor.checkpoint['age_stats'],predictor.size,False)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_suffix('.csv.partial')
    meta=output.with_suffix('.csv.metadata.json')
    fingerprint={'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                 'test_csv_sha256':hashlib.sha256((Path(args.data_root)/'test.csv').read_bytes()).hexdigest(),
                 'image_size':predictor.size,'batch_size':args.batch_size,'mode':'frozen_eval_batch_fp16','total_rows':len(dataset)}
    start=0
    if args.resume and partial.exists():
        if json.loads(meta.read_text())!=fingerprint:
            raise ValueError('Cannot mix different inference settings/checkpoints')
        with partial.open(newline='') as f:
            old=list(csv.DictReader(f))
        for i,row in enumerate(old):
            if row['Image_name']!=dataset.rows[i]['Image_name']:
                raise ValueError('Partial name order mismatch')
        start=len(old)
    elif partial.exists():
        raise FileExistsError('Partial output exists; specify --resume')
    meta.write_text(json.dumps(fingerprint,indent=2))
    progress={'predicted':start,'total':len(dataset)}
    def stopped(signum,frame):
        print(json.dumps({'event':'signal','signal':signum,**progress}),flush=True)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stopped)
    signal.signal(signal.SIGHUP,stopped)
    loader=DataLoader(Subset(dataset,range(start,len(dataset))),batch_size=args.batch_size,shuffle=False,
                      num_workers=args.workers,pin_memory=True,persistent_workers=False)
    with partial.open('a' if start else 'w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        if not start:
            writer.writerow(['Image_name','Probability','Pred_Label'])
            f.flush()
        with torch.inference_mode():
            for i,batch in enumerate(loader):
                image=batch['image'].cuda(non_blocking=True)
                age=batch['age'].cuda(non_blocking=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    scores=predictor.model(image,age).float().sigmoid().cpu().tolist()
                for name,score in zip(batch['name'],scores):
                    writer.writerow([name,format(score,'.9g'),int(score>=.5)])
                f.flush()
                progress['predicted']+=len(scores)
                print(json.dumps(progress),flush=True)
    partial.replace(output)
    print(json.dumps({'completed':str(output),'rows':len(dataset),'batch_size':args.batch_size}),flush=True)

if __name__=='__main__':
    main()
