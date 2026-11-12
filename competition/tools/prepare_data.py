"""Extract the supplied archive into an isolated directory and audit CSV linkage."""
import argparse
import csv
import json
import os
import time
import zipfile
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--archive',default='data/glaucoma_first.zip.upload')
p.add_argument('--expected-bytes',type=int,default=33243465372)
args=p.parse_args()
archive=Path(args.archive)
while not archive.exists() or archive.stat().st_size!=args.expected_bytes:
    print(json.dumps({'waiting_for_archive':str(archive),'bytes':archive.stat().st_size if archive.exists() else 0,'expected':args.expected_bytes}),flush=True)
    time.sleep(15)
destination=Path('data/official').resolve()
destination.mkdir(parents=True,exist_ok=True)
ready=destination/'ready.json'
if not ready.exists():
    with zipfile.ZipFile(archive) as z:
        members=z.infolist()
        # Refuse traversal and symbolic links before extracting any entry.
        for info in members:
            target=(destination/info.filename).resolve()
            if not target.is_relative_to(destination) or (info.external_attr>>16)&0o170000==0o120000:
                raise ValueError(f'Unsafe archive entry: {info.filename}')
        for i,info in enumerate(members):
            z.extract(info,destination)
            if i%2500==0:
                print(json.dumps({'extracted':i,'total':len(members)}),flush=True)
    candidates=list(destination.rglob('train.csv'))
    if len(candidates)!=1:
        raise ValueError(f'Expected one official training CSV: {candidates}')
    root=candidates[0].parent
    report={}
    for split,expected in [('train',51396),('valid',5849),('test',5855)]:
        with (root/f'{split}.csv').open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        files={entry.name.casefold() for entry in os.scandir(root/split) if entry.is_file()}
        names=[r['Image_name'] for r in rows]
        if len(rows)!=expected or len(names)!=len(set(names)) or set(n.casefold() for n in names)!=files:
            raise ValueError(f'CSV/directory integrity failed for {split}')
        report[split]={'rows':len(rows),'files':len(files)}
        if split!='test':
            labels=[int(r['True_Label']) for r in rows]
            if set(labels)!={0,1}:
                raise ValueError('Invalid labels')
            report[split]['positive']=sum(labels)
    report['root']=str(root)
    ready.write_text(json.dumps(report,indent=2))
print(ready.read_text(),flush=True)
