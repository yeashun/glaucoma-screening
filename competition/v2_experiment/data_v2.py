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
            self.array_pid = os.getpid()
        row = self.rows[index]
        source_index = self.indices[row['Image_name']]
        image = Image.fromarray(self.array[source_index])
        return {'image': transform(image, self.size, self.training, self.gamma),
                'age': self.encode_age(row, self.stats),
                'label': torch.tensor(float(row.get('True_Label', -1))),
                'name': row['Image_name'], 'bbox': torch.from_numpy(self.boxes[source_index].copy()),
                'original_size': torch.from_numpy(self.original_sizes[source_index].copy())}
