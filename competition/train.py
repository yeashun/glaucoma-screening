import argparse
import csv
import fcntl
import json
import math
import os
import platform
import random
import signal
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from glaucoma.data import FundusDataset,StratifiedBatchSampler,fit_age,read_rows,seed_worker
from glaucoma.models import FundusClassifier
from glaucoma.metrics import compute_metrics

STOP=False
def stop_handler(*_):
    global STOP
    STOP=True

def write_json(path,value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)

def save_predictions(path,names,labels,probabilities):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['Image_name','True_Label','Probability'])
        writer.writerows(zip(names,labels,probabilities))

@torch.inference_mode()
def evaluate(model,loader,device):
    model.eval()
    names=[]
    labels=[]
    probabilities=[]
    for batch in loader:
        image=batch['image'].to(device,non_blocking=True)
        age=batch['age'].to(device,non_blocking=True)
        local=batch['local'].to(device,non_blocking=True) if 'local' in batch else None
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            logits=model(image,age,local)
        names.extend(batch['name'])
        labels.extend(batch['label'].int().tolist())
        probabilities.extend(logits.float().sigmoid().cpu().tolist())
    return compute_metrics(labels,probabilities),names,labels,probabilities

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--resume')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--init-global')
    parser.add_argument('--init-local')
    parser.add_argument('--roi-csv')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    run=Path(args.run)
    run.mkdir(parents=True,exist_ok=True)
    lock=(run/'train.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (run/'complete.json').exists() and not args.resume:
        raise RuntimeError('Completed run already exists; choose a new run directory')
    signal.signal(signal.SIGTERM,stop_handler)
    signal.signal(signal.SIGINT,stop_handler)
    seed=config.get('seed',42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(config.get('cpu_threads',4))
    torch.backends.cudnn.benchmark=True
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda' and not args.smoke:
        raise RuntimeError('Full training requires CUDA')
    rows=read_rows(args.data_root,'train')
    excluded=[]
    exclusion_path=Path(args.data_root)/'excluded_train.json'
    if exclusion_path.exists():
        excluded=json.loads(exclusion_path.read_text())
    fitted_rows=[r for r in rows if r['Image_name'] not in set(excluded)]
    stats=fit_age(fitted_rows)
    checkpoint=None
    if args.resume:
        checkpoint=torch.load(args.resume,map_location='cpu',weights_only=False)
        if checkpoint['config']['backbone']!=config['backbone'] or checkpoint['config']['use_age']!=config['use_age']:
            raise ValueError('Resume backbone/age architecture mismatch')
        stats=checkpoint['age_stats']
    model=FundusClassifier(config,pretrained=config.get('pretrained',True) and not args.resume).to(device)
    if args.init_global:
        initial=torch.load(args.init_global,map_location='cpu',weights_only=False)
        encoder_state={k.removeprefix('encoder.'):v for k,v in initial['model'].items() if k.startswith('encoder.')}
        model.encoder.load_state_dict(encoder_state)
    if args.init_local:
        initial=torch.load(args.init_local,map_location='cpu',weights_only=False)
        local_state={k.removeprefix('local_encoder.'):v for k,v in initial['model'].items() if k.startswith('local_encoder.')}
        model.local_encoder.load_state_dict(local_state)
    backbone=[]
    head=[]
    for name,param in model.named_parameters():
        (backbone if name.startswith(('encoder.','local_encoder.')) else head).append(param)
    weight_decay = float(config.get('weight_decay', 1e-4))

    optimizer = torch.optim.AdamW(
        [
            {'params': backbone, 'lr': config.get('backbone_lr', 3e-5)},
            {'params': head, 'lr': config.get('head_lr', 3e-4)}
        ],
        weight_decay=weight_decay
    )
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    start_epoch=0
    best=-1.
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch=checkpoint['epoch']
        best=checkpoint['best_auroc']
        if 'rng' in checkpoint:
            random.setstate(checkpoint['rng']['python'])
            np.random.set_state(checkpoint['rng']['numpy'])
            torch.set_rng_state(checkpoint['rng']['torch'])
            if device.type=='cuda':
                torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])
        prefix=f'epoch_{start_epoch:03d}_best'
        parent=Path(args.resume).parent
        source_best=parent/f'{prefix}.pt'
        if not source_best.exists():
            source_best=parent/'best.pt'
        if parent!=run and source_best.exists():
            # Parent validation winner is an eligible candidate for the branch.
            import shutil
            parent_best=torch.load(source_best,map_location='cpu',weights_only=False)
            if parent_best['epoch']<=start_epoch:
                shutil.copy2(source_best,run/'best.pt')
                for suffix,target in [('_valid.csv','best_valid.csv'),('_metrics.json','best_metrics.json')]:
                    source=parent/f'{prefix}{suffix}'
                    if source.exists():
                        shutil.copy2(source,run/target)
    if args.smoke:
        config=dict(config,epochs=1,warmup_epochs=0)
        start_epoch=0
        best=-1
    environment={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
                 'device':torch.cuda.get_device_name() if device.type=='cuda' else 'cpu','visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                 'data_root':str(Path(args.data_root).resolve()),'smoke':args.smoke,'excluded_train':len(excluded)}
    write_json(run/'config.json',config)
    write_json(run/'environment.json',environment)
    write_json(run/'age_stats.json',stats)
    epochs=config.get('epochs',14)
    started=time.time()
    generator=torch.Generator().manual_seed(seed)
    for epoch in range(start_epoch,epochs):
        generator.manual_seed(seed+epoch)
        size=config.get('image_size',384)
        batch_size=config.get('batch_size',16)
        if config.get('switch_epoch') is not None and epoch>=config['switch_epoch']:
            size=config.get('high_size',512)
            batch_size=config.get('high_batch_size',8)
        train=FundusDataset(args.data_root,'train',stats,size,True,64 if args.smoke else None,args.roi_csv,config.get('local_size',320),excluded)
        valid=FundusDataset(args.data_root,'valid',stats,size,False,64 if args.smoke else None,args.roi_csv,config.get('local_size',320))
        labels=np.array([int(r['True_Label']) for r in train.rows])
        pi=float(labels.mean())
        workers=config.get('workers',8) if not args.smoke else 2
        options=dict(num_workers=workers,pin_memory=device.type=='cuda',worker_init_fn=seed_worker,generator=generator,
                     persistent_workers=False)
        stratified=config.get('stratified_from') is not None and epoch>=config['stratified_from']
        if stratified:
            sampler=StratifiedBatchSampler(labels,batch_size,seed)
            sampler.epoch=epoch
            trainloader=DataLoader(train,batch_sampler=sampler,**options)
            q=sampler.npos/batch_size
        else:
            trainloader=DataLoader(train,batch_size=batch_size,shuffle=True,**options)
            q=pi
        validloader=DataLoader(valid,batch_size=batch_size*2,shuffle=False,**options)
        warmup=epoch<config.get('warmup_epochs',1)
        for name,param in model.named_parameters():
            if name.startswith('encoder.'):
                param.requires_grad_(not warmup and not config.get('freeze_global',False))
            elif name.startswith('local_encoder.'):
                param.requires_grad_(not warmup and not config.get('freeze_local',False))
        model.train()
        if warmup:
            model.encoder.eval()
            if model.local_encoder:
                model.local_encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        accum=max(1,config.get('effective_batch',32)//batch_size)
        epoch_start=time.time()
        running=0.
        seen=0
        rank_lambda=config.get('rank_lambda',0.) if stratified else 0.
        for step,batch in enumerate(trainloader):
            progress=(epoch+step/max(1,len(trainloader)))/epochs
            factor=.05+.95*(1+math.cos(math.pi*progress))/2
            optimizer.param_groups[0]['lr']=config.get('backbone_lr',3e-5)*factor
            optimizer.param_groups[1]['lr']=config.get('head_lr',3e-4)*factor
            image=batch['image'].to(device,non_blocking=True)
            age=batch['age'].to(device,non_blocking=True)
            target=batch['label'].to(device,non_blocking=True)
            local=batch['local'].to(device,non_blocking=True) if 'local' in batch else None
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                logits=model(image,age,local)
                losses=nn.functional.binary_cross_entropy_with_logits(logits.float(),target,reduction='none')
                if stratified:
                    weights=torch.where(target==1,pi/q,(1-pi)/(1-q))
                    loss=(losses*weights).mean()
                else:
                    loss=losses.mean()
                rank=torch.zeros((),device=device)
                if rank_lambda:
                    positive=logits[target==1].float()
                    negative=logits[target==0].float()
                    rank=nn.functional.softplus(negative[:,None]-positive[None,:]).mean()
                    loss=loss+rank_lambda*rank
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            group_start=(step//accum)*accum
            divisor=min(accum,len(trainloader)-group_start)
            scaler.scale(loss/divisor).backward()
            if (step+1)%accum==0 or step+1==len(trainloader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(),5.)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running+=float(loss.detach())*len(target)
            seen+=len(target)
            if step%50==0 or step+1==len(trainloader):
                print(json.dumps({'event':'train','run':run.name,'epoch':epoch+1,'step':step+1,'steps':len(trainloader),
                                  'image_size':size,'loss':running/seen,'images_s':seen/max(.001,time.time()-epoch_start),
                                  'rank_loss':float(rank.detach()),'gpu_memory_gb':torch.cuda.max_memory_allocated()/1e9 if device.type=='cuda' else 0}),flush=True)
            if STOP:
                write_json(run/'interrupted.json',{'epoch':epoch+1,'step':step+1,'message':'Restart from last completed epoch checkpoint'})
                return
        metrics,names,ys,ps=evaluate(model,validloader,device)
        metrics.update(epoch=epoch+1,image_size=size,train_loss=running/seen,elapsed_s=time.time()-started,smoke=args.smoke)
        improved=metrics['auroc']>best
        if improved:
            best=metrics['auroc']
        state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
               'config':config,'age_stats':stats,'epoch':epoch+1,'best_auroc':best,'metrics':metrics,
               'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
                      'cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}}
        temporary=run/'last.pt.tmp'
        torch.save(state,temporary)
        temporary.replace(run/'last.pt')
        if improved or not (run/'best.pt').exists():
            import shutil
            shutil.copy2(run/'last.pt',run/'best.pt')
            save_predictions(run/'best_valid.csv',names,ys,ps)
            write_json(run/'best_metrics.json',metrics)
        if epoch+1 in (8,10):
            import shutil
            shutil.copy2(run/'last.pt',run/f'epoch_{epoch+1:03d}.pt')
            shutil.copy2(run/'best.pt',run/f'epoch_{epoch+1:03d}_best.pt')
            shutil.copy2(run/'best_valid.csv',run/f'epoch_{epoch+1:03d}_best_valid.csv')
            shutil.copy2(run/'best_metrics.json',run/f'epoch_{epoch+1:03d}_best_metrics.json')
        with (run/'metrics.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(metrics)+'\n')
        write_json(run/'progress.json',metrics)
        print(json.dumps({'event':'validation','run':run.name,'best_auroc':best,**metrics}),flush=True)
        del trainloader,validloader
    write_json(run/'complete.json',{'best_auroc':best,'elapsed_s':time.time()-started,'smoke':args.smoke})

if __name__=='__main__':
    main()
