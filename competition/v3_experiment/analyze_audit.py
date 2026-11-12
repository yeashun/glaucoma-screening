"""Summarize similarity risk without asserting patient identity or fixing labels."""
from cpu_utils import *
from collections import Counter


def main():
    output=ROOT/'results/V3_0/cpu/P0'
    summary=json.loads((output/'summary.json').read_text())
    fingerprints={}
    with (output/'image_fingerprints.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            row=json.loads(line);fingerprints[(row['split'],row['Image_name'])]=row
    totals=Counter();strict=Counter();conflicts=Counter();very_close=Counter();examples=[]
    with (output/'perceptual_candidates.csv').open(encoding='utf-8-sig',newline='') as handle:
        for row in csv.DictReader(handle):
            pair='--'.join(sorted((row['split_a'],row['split_b'])))
            totals[pair]+=1
            if row['strict_risk_edge']!='True':continue
            strict[pair]+=1
            a=fingerprints[(row['split_a'],row['name_a'])]
            b=fingerprints[(row['split_b'],row['name_b'])]
            conflict=a['label'] is not None and b['label'] is not None and a['label']!=b['label']
            if conflict:conflicts[pair]+=1
            close=(float(row['rgb_mae32'])<1 and int(row['dhash_distance'])<=2 and
                   (a['width'],a['height'])==(b['width'],b['height']))
            if close:
                very_close[pair]+=1
                if conflict:examples.append({**row,'label_a':a['label'],'label_b':b['label'],
                    'age_a':a['age'],'age_b':b['age']})
    groups={}
    for row in read_csv(output/'duplicate_risk_groups.csv'):
        groups[row['risk_group']]=int(row['group_size'])
    write_json(output/'risk_analysis.json',{'candidate_pairs_by_split':dict(totals),
        'strict_risk_pairs_by_split':dict(strict),'strict_pairs_with_disagreeing_known_labels':dict(conflicts),
        'very_close_heuristic_definition':'Strict edge; RGB_MAE32<1; dHash<=2; identical original dimensions. Not patient identity.',
        'very_close_pairs_by_split':dict(very_close),'very_close_known_label_conflicts':examples,
        'multiple_image_groups':sum(size>1 for size in groups.values()),
        'largest_risk_cluster_images':max(groups.values()),
        'images_in_multiple_image_groups':sum(size for size in groups.values() if size>1),
        'exif_tag_ids':summary['exif_tag_ids'],'gpu_used':False,'labels_changed':False,
        'limitation':'All similarity thresholds are screening heuristics. Different eyes can match; visual/source confirmation is required.'})
    print(json.dumps({'strict_by_split':dict(strict),'very_close_by_split':dict(very_close),
          'very_close_known_label_conflict_pairs':len(examples),'largest_risk_cluster_images':max(groups.values())}))


if __name__=='__main__':main()
