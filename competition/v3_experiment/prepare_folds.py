"""Training-only duplicate-risk group stratification; no Age or test grouping."""
from cpu_utils import *
from collections import defaultdict


def assign(groups,labels,n_splits,seed):
    rng=np.random.default_rng(seed)
    grouped=defaultdict(list)
    for i,group in enumerate(groups):grouped[group].append(i)
    sequence=list(grouped.items());rng.shuffle(sequence)
    sequence.sort(key=lambda item:(-len(item[1]),-sum(labels[i] for i in item[1])))
    totals=np.bincount(labels,minlength=2).astype(float)
    counts=np.zeros((n_splits,2));sizes=np.zeros(n_splits)
    result=np.empty(len(labels),dtype=int)
    diagonal=np.arange(n_splits)
    for group,indices in sequence:
        vector=np.bincount(labels[indices],minlength=2)
        trial=np.broadcast_to(counts,(n_splits,n_splits,2)).copy()
        trial[diagonal,diagonal]+=vector
        scores=np.std(trial/np.maximum(totals,1)[None,None,:],axis=1).mean(axis=1)
        chosen=min(range(n_splits),key=lambda fold:(scores[fold],sizes[fold],fold))
        counts[chosen]+=vector;sizes[chosen]+=len(indices)
        result[indices]=chosen
    return result


def main():
    audit=ROOT/'results/V3_0/cpu/P0'
    if not (audit/'summary.json').is_file():raise SystemExit('Complete the P0 audit before creating folds.')
    summary=json.loads((audit/'summary.json').read_text())
    if summary['decode_errors']:raise SystemExit('Resolve image decode errors before freezing folds.')
    train=read_csv(ROOT/'glaucoma_first/glaucoma_first/train.csv')
    names=[row['Image_name'] for row in train];indices={name:i for i,name in enumerate(names)}
    y=np.array([int(row['True_Label']) for row in train]);parents=list(range(len(train)))
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parents[max(a,b)]=min(a,b)
    by_hash={}
    for line in (audit/'image_fingerprints.jsonl').read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        if row['split']!='train':continue
        index=indices[row['Image_name']]
        if row['sha256'] in by_hash:union(index,by_hash[row['sha256']])
        by_hash[row['sha256']]=index
    for row in read_csv(audit/'perceptual_candidates.csv'):
        if row['split_a']==row['split_b']=='train' and row['strict_risk_edge']=='True':
            union(indices[row['name_a']],indices[row['name_b']])
    groups=np.array([find(i) for i in range(len(train))])
    outer=assign(groups,y,5,20261103)
    output=ROOT/'results/V3_0/cpu/OOF_plans';output.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for fold in range(5):
        hold=np.flatnonzero(outer==fold);remain=np.flatnonzero(outer!=fold)
        inner=assign(groups[remain],y[remain],10,20261103+fold+1)
        inner_valid=remain[inner==0];training=remain[inner!=0]
        assert not (set(groups[hold])&set(groups[training]))
        assert not (set(groups[inner_valid])&set(groups[training]))
        assert not (set(groups[inner_valid])&set(groups[hold]))
        plan={'outer_fold':fold,'train_names':[names[i] for i in training],
              'inner_valid_names':[names[i] for i in inner_valid],'outer_holdout_names':[names[i] for i in hold],
              'train_csv_sha256':sha256(ROOT/'glaucoma_first/glaucoma_first/train.csv'),
              'grouping':'Training-training exact bytes + strict perceptual-risk edges; not patient IDs',
              'age_used_for_grouping':False,'official_valid_used':False,'test_used_for_grouping':False,
              'outer_holdout_used_for_checkpoint_selection':False}
        write_json(output/f'fold{fold}.json',plan)
        summaries.append({'fold':fold,'train':len(training),'inner_valid':len(inner_valid),
                           'outer_holdout':len(hold),'outer_positive':int(y[hold].sum())})
    with (output/'outer_assignments.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['Image_name','True_Label','risk_group','outer_fold'])
        writer.writerows(zip(names,y.tolist(),groups.tolist(),outer.tolist()))
    write_json(output/'summary.json',{'folds':summaries,'gpu_used':False,'oof_models_trained':0,
               'patient_independence_verified':False,'risk_cluster_visual_confirmation_required':True})
    print(json.dumps(summaries))


if __name__=='__main__':main()
