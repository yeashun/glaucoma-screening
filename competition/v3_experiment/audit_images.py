"""Full CPU SHA256 and perceptual duplicate-risk audit; no split is altered."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from cpu_utils import *
import argparse
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import time
from PIL import Image, ImageOps

_dct=np.cos(np.pi*(np.arange(32)[None,:]+.5)*np.arange(8)[:,None]/32)


def fingerprint(image):
    gray=np.asarray(image.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float64)
    coefficients=(_dct@gray@_dct.T).ravel()
    bits=coefficients>np.median(coefficients[1:])
    bits[0]=False
    phash=int.from_bytes(np.packbits(bits).tobytes(),'big')
    small=np.asarray(image.convert('L').resize((9,8),Image.Resampling.LANCZOS))
    dhash=int.from_bytes(np.packbits(small[:,1:]>small[:,:-1]).tobytes(),'big')
    return f'{phash:016x}',f'{dhash:016x}',float(gray.std())


def cropped(image):
    thumb=image.copy();thumb.thumbnail((160,160))
    values=np.asarray(thumb);ys,xs=np.nonzero(values.max(axis=2)>8)
    if len(xs)<20:return image
    sx,sy=image.width/thumb.width,image.height/thumb.height
    pad=max(2,int(min(image.size)*.025))
    box=(max(0,int(xs.min()*sx)-pad),max(0,int(ys.min()*sy)-pad),
         min(image.width,int((xs.max()+1)*sx)+pad),min(image.height,int((ys.max()+1)*sy)+pad))
    return image.crop(box)


def inspect(job):
    split,row,path=job
    stat=Path(path).stat()
    value={'split':split,'Image_name':row['Image_name'],'label':int(row['True_Label']) if 'True_Label' in row else None,
           'age':row.get('Age'),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'sha256':sha256(path)}
    try:
        with Image.open(path) as source:
            value.update(width=source.width,height=source.height,format=source.format,
                         exif_tag_ids=sorted(int(x) for x in source.getexif().keys()),
                         metadata_keys=sorted(source.info.keys()))
            source.draft('RGB',(512,512))
            image=ImageOps.exif_transpose(source).convert('RGB')
            image.thumbnail((512,512))
        a,b,std=fingerprint(image);ca,cb,cstd=fingerprint(cropped(image))
        value.update(phash=a,dhash=b,pixel_std=std,crop_phash=ca,crop_dhash=cb,crop_pixel_std=cstd)
    except Exception as error:
        value['decode_error']=f'{type(error).__name__}: {error}'
    return value


class BKTree:
    def __init__(self):self.root=None
    def query(self,value,radius):
        stack=[] if self.root is None else [self.root]
        while stack:
            node=stack.pop();distance=(value^node[0]).bit_count()
            if distance<=radius:yield node[0],distance
            stack.extend(child for edge,child in node[1].items() if distance-radius<=edge<=distance+radius)
    def add(self,value):
        if self.root is None:self.root=[value,{}];return
        node=self.root
        while True:
            distance=(value^node[0]).bit_count()
            if not distance:return
            if distance not in node[1]:node[1][distance]=[value,{}];return
            node=node[1][distance]


def vector_neighbors(hashes,dhashes,eligible,index):
    """Exact Hamming radius search using NumPy popcount; bounded linear memory."""
    distances=np.bitwise_count(hashes[:index]^hashes[index])
    indices=np.flatnonzero((distances<=4)&eligible[:index])
    horizontal=np.bitwise_count(dhashes[indices]^dhashes[index])
    keep=horizontal<=8
    return zip(indices[keep].tolist(),distances[indices[keep]].tolist(),horizontal[keep].tolist())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',default=str(DATA_ROOT))
    parser.add_argument('--output',default=str(ROOT/'results/V3_0/cpu/P0'))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--search-engine',choices=('auto','vectorized','bktree'),default='auto')
    args=parser.parse_args()
    root,output=Path(args.data_root),Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    manifest={};jobs=[];paths={}
    for split in ('train','valid','test'):
        rows=read_csv(root/f'{split}.csv')
        available={p.name.casefold():p for p in (root/split).iterdir() if p.is_file()}
        assert len(rows)==len(set(r['Image_name'] for r in rows))
        assert {r['Image_name'].casefold() for r in rows}==set(available)
        manifest[split]={'rows':len(rows),'csv_sha256':sha256(root/f'{split}.csv'),'columns':list(rows[0])}
        for row in rows:
            path=str(available[row['Image_name'].casefold()])
            jobs.append((split,row,path));paths[(split,row['Image_name'])]=path
    if args.limit:jobs=jobs[:args.limit]
    write_json(output/'input_manifest.json',{'splits':manifest,'full_audit':args.limit is None,
                'hash_algorithm':'SHA256 raw bytes; 64-bit DCT pHash with zero DC + horizontal dHash; full and black-border-cropped views',
                'perceptual_decode':'Pillow JPEG draft <=512 followed by resize32; perceptual hashes are not pixel-equality proofs',
                'patient_grouping':'Age is never used as a patient identifier'})
    journal=output/'image_fingerprints.jsonl'
    cached={}
    if journal.exists():
        for line in journal.read_text(encoding='utf-8').splitlines():
            if line.strip():
                record=json.loads(line);cached[(record['split'],record['Image_name'])]=record
    pending=[];records=[]
    for job in jobs:
        key=(job[0],job[1]['Image_name']);previous=cached.get(key);stat=Path(job[2]).stat()
        if previous and previous.get('mtime_ns')==stat.st_mtime_ns and previous.get('bytes')==stat.st_size:
            records.append(previous)
        else:pending.append(job)
    started=time.monotonic()
    with journal.open('a',encoding='utf-8') as handle, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(inspect,pending,chunksize=16):
            records.append(record);handle.write(json.dumps(record,ensure_ascii=False)+'\n')
            if len(records)%256==0 or len(records)==len(jobs):
                handle.flush()
                progress={'stage':'hashing_and_perceptual_features','done':len(records),'total':len(jobs),
                          'elapsed_this_process_s':time.monotonic()-started,'workers':args.workers}
                write_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
    records.sort(key=lambda row:({'train':0,'valid':1,'test':2}[row['split']],row['Image_name']))
    exact=defaultdict(list)
    for i,row in enumerate(records):exact[row['sha256']].append(i)
    duplicate_groups=[indices for indices in exact.values() if len(indices)>1]
    write_json(output/'exact_duplicates.json',{
        'groups':[[records[i] for i in indices] for indices in duplicate_groups],
        'cross_split_groups':sum(len({records[i]['split'] for i in group})>1 for group in duplicate_groups),
        'within_or_cross_groups':len(duplicate_groups)})
    if args.limit:
        write_json(output/'benchmark.json',{'images':len(records),'elapsed_s':time.monotonic()-started,
                    'images_per_s':len(records)/max(.001,time.monotonic()-started)})
        return
    parents=list(range(len(records)))
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b:parents[max(a,b)]=min(a,b)
    for group in duplicate_groups:
        for i in group[1:]:union(group[0],i)
    near=output/'perceptual_candidates.csv'
    seen=set();near_count=0;cross_count=0;strict_edges=0
    decoded={}
    def thumbnail(index,view):
        key=(index,view)
        if key not in decoded:
            with Image.open(paths[(records[index]['split'],records[index]['Image_name'])]) as source:
                source.draft('RGB',(512,512));image=ImageOps.exif_transpose(source).convert('RGB')
                image.thumbnail((512,512))
            if view:image=cropped(image)
            decoded[key]=np.asarray(image.resize((32,32),Image.Resampling.LANCZOS),dtype=np.float32)
        return decoded[key]
    def thumbnail_pair(index):
        with Image.open(paths[(records[index]['split'],records[index]['Image_name'])]) as source:
            source.draft('RGB',(512,512));image=ImageOps.exif_transpose(source).convert('RGB')
            image.thumbnail((512,512))
        full=np.asarray(image.resize((32,32),Image.Resampling.LANCZOS),dtype=np.float32)
        crop=np.asarray(cropped(image).resize((32,32),Image.Resampling.LANCZOS),dtype=np.float32)
        return index,full,crop
    thumbnail_pool=ThreadPoolExecutor(max_workers=args.workers)
    with near.open('w',encoding='utf-8-sig',newline='') as handle:
        fields=['split_a','name_a','split_b','name_b','view','phash_distance','dhash_distance','rgb_mae32','strict_risk_edge','same_sha256']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for view in (False,True):
            tree=BKTree();buckets=defaultdict(list)
            pkey='crop_phash' if view else 'phash';dkey='crop_dhash' if view else 'dhash'
            stdkey='crop_pixel_std' if view else 'pixel_std'
            vectorized=args.search_engine=='vectorized' or (args.search_engine=='auto' and hasattr(np,'bitwise_count'))
            if vectorized:
                hashes=np.array([int(r.get(pkey,'0'),16) for r in records],dtype=np.uint64)
                dhashes=np.array([int(r.get(dkey,'0'),16) for r in records],dtype=np.uint64)
                eligible=np.array([pkey in r and r[stdkey]>=5 for r in records],dtype=bool)
            for i,row in enumerate(records):
                if vectorized and i%256==0:
                    needed=set()
                    for q in range(i,min(i+256,len(records))):
                        if not eligible[q]:continue
                        for j,_,_ in vector_neighbors(hashes,dhashes,eligible,q):
                            if (j,q) in seen:continue
                            for index in (j,q):
                                if (index,view) not in decoded:needed.add(index)
                    for index,full,crop in thumbnail_pool.map(thumbnail_pair,sorted(needed)):
                        decoded[(index,False)]=full;decoded[(index,True)]=crop
                if pkey not in row or row[stdkey]<5:continue
                value=int(row[pkey],16)
                if vectorized:
                    matches=vector_neighbors(hashes,dhashes,eligible,i)
                else:
                    matches=((j,dist,(int(row[dkey],16)^int(records[j][dkey],16)).bit_count())
                             for other_hash,dist in tree.query(value,4) for j in buckets[other_hash])
                for j,dist,d in matches:
                        pair=(j,i)
                        if pair in seen:continue
                        if d>8:continue
                        seen.add(pair)
                        mae=float(np.abs(thumbnail(i,view)-thumbnail(j,view)).mean())
                        strict=d<=4 and mae<=5
                        if strict:union(i,j);strict_edges+=1
                        cross=records[i]['split']!=records[j]['split']
                        near_count+=1;cross_count+=cross
                        writer.writerow(dict(zip(fields,[records[j]['split'],records[j]['Image_name'],row['split'],row['Image_name'],
                            'crop' if view else 'full',dist,d,mae,strict,row['sha256']==records[j]['sha256']])))
                if not vectorized:buckets[value].append(i);tree.add(value)
                if i%1024==0:
                    handle.flush();write_json(output/'progress.json',{'stage':'perceptual_radius_search',
                        'view':'crop' if view else 'full','done':i,'total':len(records),'candidate_pairs':near_count,
                        'elapsed_this_process_s':time.monotonic()-started})
    thumbnail_pool.shutdown(wait=True)
    clusters=defaultdict(list)
    for i in range(len(records)):clusters[find(i)].append(i)
    with (output/'duplicate_risk_groups.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['split','Image_name','risk_group','group_size','True_Label'])
        for group in clusters.values():
            group_id=hashlib.sha256('|'.join(records[i]['split']+'/'+records[i]['Image_name'] for i in group).encode()).hexdigest()[:20]
            for i in group:writer.writerow([records[i]['split'],records[i]['Image_name'],group_id,len(group),records[i]['label']])
    summary={'images':len(records),'split_counts':dict(Counter(r['split'] for r in records)),
             'raw_sha256_complete':True,'decode_errors':[r for r in records if 'decode_error' in r],
             'exact_duplicate_groups':len(duplicate_groups),
             'exact_cross_split_groups':sum(len({records[i]['split'] for i in g})>1 for g in duplicate_groups),
             'perceptual_candidate_pairs':near_count,'perceptual_cross_split_candidate_pairs':cross_count,
             'strict_similarity_risk_edges':strict_edges,'risk_clusters_multiple_images':sum(len(g)>1 for g in clusters.values()),
             'exif_tag_ids':dict(Counter(str(tag) for r in records for tag in r.get('exif_tag_ids',[]))),
             'elapsed_this_process_s':time.monotonic()-started,'gpu_used':False,
             'perceptual_search_engine':'numpy_vectorized_exact_hamming' if vectorized else 'BK_tree_exact_hamming',
             'patient_ids_available':False,'age_used_as_patient_id':False,
             'limitations':['Perceptual candidates and risk clusters require visual confirmation; similarity is not patient identity.',
                            'Absence of duplicate images does not rule out fellow-eye or longitudinal patient overlap.',
                            'No official split or label was changed. Test images used only for integrity audit, never model fitting.']}
    write_json(output/'summary.json',summary)
    write_json(output/'progress.json',{'stage':'complete',**summary})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
