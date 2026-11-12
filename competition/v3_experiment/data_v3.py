"""Size-specific deterministic RGB cache with online V2 augmentations."""
from common import *
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF
from torchvision import transforms as T
from torch.utils.data import Dataset


def transform(image, size, training, gamma):
    if image.size != (size, size):
        image = ImageOps.pad(image, (size, size), method=Image.Resampling.BILINEAR, color=(0, 0, 0))
    if training:
        if random.random() < .5:
            image = ImageOps.mirror(image)
        image = image.rotate(random.uniform(-15, 15), resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
        image = T.ColorJitter(brightness=.15, contrast=.15, saturation=.10)(image)
        if gamma and random.random() < .2:
            image = TF.adjust_gamma(image, random.uniform(.9, 1.1))
    tensor = TF.to_tensor(image)
    return TF.normalize(tensor, [.485, .456, .406], [.229, .224, .225])


class CachedDataset(Dataset):
    def __init__(self, root, split, stats, cache_root, size, training=False, gamma=False, limit=None):
        from core import encode_age
        self.encode_age = encode_age
        self.root, self.split = Path(root), split
        self.rows = read_rows(root, split)
        if limit:
            positive = [r for r in self.rows if r.get('True_Label') == '1']
            negative = [r for r in self.rows if r.get('True_Label') != '1']
            rng = random.Random(42)
            npos = min(len(positive), max(2, round(limit * .1)))
            self.rows = rng.sample(positive, npos) + rng.sample(negative, min(len(negative), limit - npos))
            rng.shuffle(self.rows)
        self.stats, self.size, self.training, self.gamma = stats, size, training, gamma
        self.cache_root = Path(cache_root)
        manifest = json.loads((self.cache_root / f'{split}.json').read_text())
        all_rows = read_rows(root, split)
        self.full_length=len(all_rows)
        if manifest['image_size'] != size or manifest['names'] != [r['Image_name'] for r in all_rows]:
            raise ValueError('Cache size/order does not match official split')
        if manifest['csv_sha256'] != sha256(Path(root) / f'{split}.csv'):
            raise ValueError('Cache source CSV differs')
        self.indices = {row['Image_name']: i for i, row in enumerate(all_rows)}
        self.boxes = np.asarray(manifest['boxes'], dtype=np.int64)
        self.original_sizes = np.asarray(manifest['original_sizes'], dtype=np.int64)
        self.array, self.array_pid = None, None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.array is None or self.array_pid != os.getpid():
            self.array = np.load(self.cache_root / f'{self.split}.npy', mmap_mode='r')
            if self.array.shape!=(self.full_length,self.size,self.size,3) or self.array.dtype!=np.uint8:
                raise ValueError('Cache array shape or dtype is incompatible')
            self.array_pid = os.getpid()
        row = self.rows[index]
        source_index = self.indices[row['Image_name']]
        image = Image.fromarray(self.array[source_index])
        return {'image': transform(image, self.size, self.training, self.gamma),
                'age': self.encode_age(row, self.stats),
                'label': torch.tensor(float(row.get('True_Label', -1))),
                'name': row['Image_name'], 'bbox': torch.from_numpy(self.boxes[source_index].copy()),
                'original_size': torch.from_numpy(self.original_sizes[source_index].copy())}


class RawDataset(Dataset):
    def __init__(self,root,split,stats,size,training=False,gamma=False,limit=None):
        from core import encode_age
        self.encode_age=encode_age
        self.root,self.split,self.stats=Path(root),split,stats
        self.rows=read_rows(root,split)
        if limit:
            positive=[r for r in self.rows if r.get('True_Label')=='1']
            negative=[r for r in self.rows if r.get('True_Label')!='1']
            rng=random.Random(42);n=min(len(positive),max(2,round(limit*.1)))
            self.rows=rng.sample(positive,n)+rng.sample(negative,min(len(negative),limit-n))
            rng.shuffle(self.rows)
        self.size,self.training,self.gamma=size,training,gamma
        self.paths={p.name.casefold():p for p in (self.root/split).iterdir() if p.is_file()}
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        from core import open_rgb,valid_bbox
        row=self.rows[index];image=open_rgb(self.paths[row['Image_name'].casefold()]);box=valid_bbox(image)
        return {'image':transform(image.crop(box),self.size,self.training,self.gamma),
                'age':self.encode_age(row,self.stats),'label':torch.tensor(float(row.get('True_Label',-1))),
                'name':row['Image_name'],'bbox':torch.tensor(box),'original_size':torch.tensor(image.size)}


def make_dataset(root,split,stats,cache_root,size,training=False,gamma=False,limit=None,names=None):
    available=cache_root and all((Path(cache_root)/f'{split}{suffix}').exists() for suffix in ('.json','.npy'))
    constructor_limit=None if names is not None else limit
    dataset=(CachedDataset(root,split,stats,cache_root,size,training,gamma,constructor_limit) if available else
             RawDataset(root,split,stats,size,training,gamma,constructor_limit))
    if names is not None:
        chosen=set(names);dataset.rows=[row for row in dataset.rows if row['Image_name'] in chosen]
        if len(dataset.rows)!=len(chosen):raise ValueError('Split plan names do not match dataset')
        if limit:
            positive=[r for r in dataset.rows if r.get('True_Label')=='1']
            negative=[r for r in dataset.rows if r.get('True_Label')!='1']
            rng=random.Random(42);n=min(len(positive),max(2,round(limit*.1)))
            dataset.rows=rng.sample(positive,n)+rng.sample(negative,min(len(negative),limit-n))
            rng.shuffle(dataset.rows)
    return dataset
