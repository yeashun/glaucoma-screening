from pathlib import Path
import csv
import random
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms as T

def read_rows(root, split):
    with (Path(root)/f'{split}.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'Empty split: {split}')
    names=[r['Image_name'] for r in rows]
    if len(names)!=len(set(names)):
        raise ValueError(f'Duplicate names in {split}')
    return rows

def age_value(row):
    try:
        age=float(row['Age'])
        return age if np.isfinite(age) and 0<age<=120 else None
    except (ValueError,TypeError,KeyError):
        return None

def fit_age(rows):
    values=[age_value(r) for r in rows if age_value(r) is not None]
    median=float(np.median(values))
    filled=np.array([age_value(r) if age_value(r) is not None else median for r in rows])
    return {'median':median,'mean':float(filled.mean()),'std':max(float(filled.std()),1.)}

def encode_age(row,stats):
    age=age_value(row)
    return torch.tensor([((stats['median'] if age is None else age)-stats['mean'])/stats['std'],float(age is None)],dtype=torch.float32)

def valid_bbox(image):
    thumb=image.copy()
    thumb.thumbnail((160,160))
    array=np.asarray(thumb)
    mask=array.max(axis=2)>8
    ys,xs=np.nonzero(mask)
    if len(xs)<20:
        return (0,0,image.width,image.height)
    sx=image.width/thumb.width
    sy=image.height/thumb.height
    pad=max(2,int(min(image.size)*.025))
    return (max(0,int(xs.min()*sx)-pad),max(0,int(ys.min()*sy)-pad),
            min(image.width,int((xs.max()+1)*sx)+pad),min(image.height,int((ys.max()+1)*sy)+pad))

def open_rgb(path):
    with Image.open(path) as source:
        image=ImageOps.exif_transpose(source).convert('RGB')
        image.load()
    return image

import random
from PIL import Image, ImageOps
import torchvision.transforms as T


def image_tensor(image, size, training=False):
    """
    眼底图像预处理与数据增强

    Args:
        image: PIL Image
        size: 输入图像尺寸
        training: 是否处于训练阶段

    Returns:
        标准化后的图像 Tensor
    """

    # 1. 确保 RGB 格式
    image = image.convert("RGB")

    # 2. 保持长宽比例，缩放并填充至指定尺寸
    image = ImageOps.pad(
        image,
        (size, size),
        method=Image.Resampling.BILINEAR,
        color=(0, 0, 0)
    )

    if training:

        # 3. 随机水平翻转
        if random.random() < 0.5:
            image = ImageOps.mirror(image)

        # 4. 随机旋转
        image = image.rotate(
            random.uniform(-15, 15),
            resample=Image.Resampling.BILINEAR,
            fillcolor=(0, 0, 0)
        )

        # 5. 颜色增强
        image = T.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10
        )(image)

        # 6. Gamma 随机校正
        if random.random() < 0.2:
            gamma = random.uniform(0.9, 1.1)

            image = T.functional.adjust_gamma(
                image,
                gamma=gamma,
                gain=1.0
            )

    # 7. 转换为 Tensor
    tensor = T.functional.to_tensor(image)

    # 8. ImageNet 标准化
    tensor = T.functional.normalize(
        tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    return tensor
class FundusDataset(Dataset):
    def __init__(self,root,split,stats,size=384,training=False,limit=None,roi_csv=None,local_size=320,excluded=None):
        self.root=Path(root)
        self.split=split
        self.rows=read_rows(root,split)
        if excluded:
            self.rows=[r for r in self.rows if r['Image_name'] not in excluded]
        if limit:
            # Deterministic spread, preserving enough positives for smoke metrics.
            positives=[r for r in self.rows if r.get('True_Label')=='1']
            negatives=[r for r in self.rows if r.get('True_Label')!='1']
            rng=random.Random(42)
            npos=min(len(positives),max(2,round(limit*.1))) if positives else 0
            self.rows=rng.sample(positives,npos)+rng.sample(negatives,min(len(negatives),limit-npos))
            rng.shuffle(self.rows)
        self.stats=stats
        self.size=size
        self.training=training
        self.local_size=local_size
        self.paths={p.name.casefold():p for p in (self.root/split).iterdir() if p.is_file()}
        missing=[r['Image_name'] for r in self.rows if r['Image_name'].casefold() not in self.paths]
        if missing:
            raise FileNotFoundError(f'{split}: {len(missing)} images missing; first={missing[:3]}')
        self.rois={}
        if roi_csv:
            with Path(roi_csv).open(newline='',encoding='utf-8') as f:
                self.rois={r['Image_name']:tuple(int(r[k]) for k in ['x0','y0','x1','y1']) for r in csv.DictReader(f) if r['split']==split}
            absent=[r['Image_name'] for r in self.rows if r['Image_name'] not in self.rois]
            if absent:
                raise ValueError(f'Missing ROI: {absent[:3]}')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]
        original=open_rgb(self.paths[row['Image_name'].casefold()])
        bbox=valid_bbox(original)
        item={'image':image_tensor(original.crop(bbox),self.size,self.training),
              'age':encode_age(row,self.stats),'label':torch.tensor(float(row.get('True_Label',-1))),
              'name':row['Image_name'],'bbox':torch.tensor(bbox),'original_size':torch.tensor(original.size)}
        if self.rois:
            item['local']=image_tensor(original.crop(self.rois[row['Image_name']]),self.local_size,self.training)
        return item

class StratifiedBatchSampler(Sampler):
    def __init__(self,labels,batch_size,seed=42):
        self.pos=np.flatnonzero(np.asarray(labels)==1)
        self.neg=np.flatnonzero(np.asarray(labels)==0)
        self.batch_size=batch_size
        self.npos=max(1,batch_size//8)
        self.nneg=batch_size-self.npos
        self.steps=(len(labels)+batch_size-1)//batch_size
        self.seed=seed
        self.epoch=0
        if not len(self.pos) or not len(self.neg):
            raise ValueError('Both classes required')

    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch)
        for _ in range(self.steps):
            indices=np.concatenate([rng.choice(self.pos,self.npos,replace=True),rng.choice(self.neg,self.nneg,replace=True)])
            rng.shuffle(indices)
            yield indices.tolist()

    def __len__(self):
        return self.steps

def seed_worker(worker_id):
    seed=torch.initial_seed()%2**32
    np.random.seed(seed)
    random.seed(seed)
