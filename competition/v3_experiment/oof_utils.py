"""Strict coverage and provenance checks for future CPU-only stacking."""
from cpu_utils import *


def load_verified_oof(folder, require_image_only=False, data_root=None):
    folder=Path(folder)
    data_root=Path(data_root) if data_root is not None else DATA_ROOT
    train_path=data_root/'train.csv'
    train=read_csv(train_path)
    train_hash=sha256(train_path)
    scores={};provenance=[]
    for fold in range(5):
        run=folder/f'fold{fold}'
        metadata=json.loads((run/'outer_oof.metadata.json').read_text())
        plan_path=ROOT/f'results/V3_0/cpu/OOF_plans/fold{fold}.json'
        plan=json.loads(plan_path.read_text())
        assert metadata['checkpoint_selected_using']=='inner_valid_only'
        assert metadata['training_complete'] and not metadata['smoke_training']
        assert not metadata['official_valid_used']
        assert metadata['train_csv_sha256']==train_hash
        assert metadata['split_plan_sha256']==sha256(plan_path)
        assert metadata['prediction_sha256']==sha256(run/'outer_oof.csv')
        checkpoint_file=metadata.get('checkpoint_file','best.pt')
        assert checkpoint_file in ('best.pt','best_ema.pt')
        assert metadata['checkpoint_sha256']==sha256(run/checkpoint_file)
        assert metadata['outer_fold']==fold
        if require_image_only:assert not metadata['use_age'],'Image-only OOF scores are required'
        rows=read_csv(run/'outer_oof.csv')
        assert len(rows)==len(plan['outer_holdout_names'])
        assert {row['Image_name'] for row in rows}==set(plan['outer_holdout_names'])
        for row in rows:
            name=row['Image_name'];assert name not in scores
            assert int(row['outer_fold'])==fold
            probability=float(row['Probability'])
            assert np.isfinite(probability) and 0<=probability<=1
            scores[name]=(probability,int(row['True_Label']))
        provenance.append(metadata)
    assert set(scores)=={row['Image_name'] for row in train}
    assert all(scores[row['Image_name']][1]==int(row['True_Label']) for row in train)
    return np.array([scores[row['Image_name']][0] for row in train]),provenance


def load_official_valid(path,data_root=None):
    data_root=Path(data_root) if data_root is not None else DATA_ROOT
    valid=read_csv(data_root/'valid.csv')
    rows=read_csv(path)
    assert [row['Image_name'] for row in rows]==[row['Image_name'] for row in valid]
    assert all(row['True_Label']==official['True_Label'] for row,official in zip(rows,valid))
    probability=np.array([float(row['Probability']) for row in rows])
    assert np.isfinite(probability).all() and ((0<=probability)&(probability<=1)).all()
    return probability
