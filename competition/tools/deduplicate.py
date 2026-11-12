"""Detect exact train/valid file duplicates using size filtering and SHA-256."""
import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--data-root',required=True)
args=p.parse_args()
root=Path(args.data_root)
groups={}
for split in ['train','valid']:
    with (root/f'{split}.csv').open(newline='',encoding='utf-8-sig') as f:
        labels={r['Image_name'].casefold():int(r['True_Label']) for r in csv.DictReader(f)}
    groups[split]=defaultdict(list)
    for entry in os.scandir(root/split):
        if entry.is_file():
            groups[split][entry.stat().st_size].append((entry.path,labels[entry.name.casefold()]))
common=groups['train'].keys()&groups['valid'].keys()
candidates=[(split,path,y) for size in common for split in ['train','valid'] for path,y in groups[split][size]]
print(json.dumps({'candidate_files':len(candidates),'all_train_valid_files':51396+5849}),flush=True)
def digest(item):
    split,path,label=item
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while block:=f.read(1024*1024):
            h.update(block)
    return split,path,label,h.hexdigest()
hashed=defaultdict(lambda:defaultdict(list))
with ThreadPoolExecutor(max_workers=4) as pool:
    for i,(split,path,label,h) in enumerate(pool.map(digest,candidates)):
        hashed[h][split].append({'file':Path(path).name,'label':label})
        if i%1000==0:
            print(json.dumps({'hashed':i,'total':len(candidates)}),flush=True)
matches=[]
excluded=[]
for h,items in hashed.items():
    if 'train' in items and 'valid' in items:
        matches.append({'sha256':h,**items})
        excluded.extend(x['file'].rsplit('.',1)[0]+'.jpg' for x in items['train'])
(root/'excluded_train.json').write_text(json.dumps(sorted(set(excluded))))
Path('data/train_valid_duplicate_audit.json').write_text(json.dumps({'matches':matches,'excluded_train':sorted(set(excluded)),
    'method':'All exact byte duplicates across train/valid, size-filtered then SHA-256. No perceptual or patient-level matching.'},indent=2))
print(json.dumps({'duplicate_groups':len(matches),'excluded_training_images':len(set(excluded))}),flush=True)
