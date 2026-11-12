"""Lossless cache of deterministic RGB crop/pad; original random transforms remain."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from glaucoma.data import read_rows, encode_age, image_tensor


class CachedFundusDataset(Dataset):
    def __init__(self, root, split, stats, cache_root, size=384, training=False):
        self.rows = read_rows(root, split)
        self.stats, self.size, self.training = stats, size, training
        self.cache_root, self.split = Path(cache_root), split
        manifest = json.loads((self.cache_root / f'{split}.json').read_text())
        if manifest['image_size'] != size or manifest['names'] != [r['Image_name'] for r in self.rows]:
            raise ValueError('Cached images do not match the requested split/size/order')
        import hashlib
        import PIL
        import glaucoma.data
        if manifest['csv_sha256'] != hashlib.sha256((Path(root) / f'{split}.csv').read_bytes()).hexdigest():
            raise ValueError('Official CSV differs from the cache source')
        if manifest['preprocessing_source_sha256'] != hashlib.sha256(Path(glaucoma.data.__file__).read_bytes()).hexdigest():
            raise ValueError('Original preprocessing source changed after cache generation')
        if manifest['pillow'] != PIL.__version__:
            raise ValueError('Pillow version differs from the cache generator')
        self.boxes = np.array(manifest['boxes'], dtype=np.int64)
        self.original_sizes = np.array(manifest['original_sizes'], dtype=np.int64)
        self.array = None
        self.array_pid = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.array is None or self.array_pid != os.getpid():
            self.array = np.load(self.cache_root / f'{self.split}.npy', mmap_mode='r')
            if self.array.shape != (len(self.rows), self.size, self.size, 3) or self.array.dtype != np.uint8:
                raise ValueError('Invalid lossless cache shape/dtype')
            self.array_pid = os.getpid()
        image = Image.fromarray(self.array[index])
        row = self.rows[index]
        # Padding an already 384x384 image is an exact pixel identity, consumes no RNG.
        return {'image': image_tensor(image, self.size, self.training),
                'age': encode_age(row, self.stats), 'label': torch.tensor(float(row.get('True_Label', -1))),
                'name': row['Image_name'], 'bbox': torch.from_numpy(self.boxes[index].copy()),
                'original_size': torch.from_numpy(self.original_sizes[index].copy())}
