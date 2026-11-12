"""Build exact deterministic RGB caches and benchmark them without model updates."""
import argparse
import concurrent.futures
import os
import random
import shutil
import time
from common import *
from PIL import ImageOps
from torch.utils.data import DataLoader, Subset, Dataset
from glaucoma.data import open_rgb, valid_bbox, image_tensor
from cached_data import CachedFundusDataset


def precompute(job):
    index, path = job
    image = open_rgb(path)
    original_size = image.size
    bbox = valid_bbox(image)
    image = ImageOps.pad(image.crop(bbox), (384, 384), method=__import__('PIL').Image.Resampling.BILINEAR,
                         color=(0, 0, 0))
    return index, np.asarray(image), bbox, original_size


def cache_manifest(root, split, rows, boxes, original_sizes, elapsed):
    import PIL
    import glaucoma.data
    return {'split': split, 'image_size': 384, 'names': [r['Image_name'] for r in rows],
            'boxes': boxes, 'original_sizes': original_sizes,
            'csv_sha256': sha256(Path(root) / f'{split}.csv'),
            'preprocessing_source_sha256': sha256(glaucoma.data.__file__),
            'pillow': PIL.__version__, 'dtype': 'uint8', 'build_elapsed_s': elapsed,
            'random_augmentation_cached': False}


def build_split(root, split, cache_root, workers=16):
    rows = read_rows(root, split)
    final = cache_root / f'{split}.npy'
    manifest_path = cache_root / f'{split}.json'
    if manifest_path.exists() and final.exists():
        return
    paths = {p.name.casefold(): p for p in (Path(root) / split).iterdir() if p.is_file()}
    temporary = cache_root / f'{split}.partial.npy'
    required = len(rows)*384*384*3
    if shutil.disk_usage(cache_root).free < required + 2_000_000_000:
        raise RuntimeError('Insufficient private NVMe cache space')
    array = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.uint8, shape=(len(rows), 384, 384, 3))
    boxes, sizes = [None]*len(rows), [None]*len(rows)
    start = time.monotonic()
    jobs = [(i, str(paths[row['Image_name'].casefold()])) for i, row in enumerate(rows)]
    # Submit bounded batches so pending decoded images cannot consume unbounded RAM.
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for offset in range(0, len(rows), workers*4):
            for index, image, box, size in executor.map(precompute, jobs[offset:offset+workers*4], chunksize=1):
                array[index] = image
                boxes[index], sizes[index] = box, size
            if offset % (workers*4*10) == 0 or offset+workers*4 >= len(rows):
                array.flush()
                done = min(len(rows), offset+workers*4)
                progress = {'stage': 'building_lossless_cache', 'split': split, 'images': done,
                            'total': len(rows), 'elapsed_s': time.monotonic()-start,
                            'images_per_s': done/max(.001, time.monotonic()-start)}
                write_json(HERE / 'performance/cache_build_progress.json', progress)
                print(json.dumps(progress), flush=True)
    array.flush()
    del array
    temporary.replace(final)
    os.chmod(final, 0o600)
    write_json(manifest_path, cache_manifest(root, split, rows, boxes, sizes, time.monotonic()-start))


def benchmark(root, workers=4):
    """Time deterministic preparation and full 4-worker train loaders on 256 images."""
    torch.set_num_threads(1)
    state = torch.load(HERE / 'checkpoints/baseline.pt', map_location='cpu', weights_only=False)
    original = FundusDataset(root, 'train', state['age_stats'], size=384, training=True)
    indices = np.linspace(0, len(original)-1, 256, dtype=int).tolist()
    temp = HERE / 'performance/benchmark_cache'
    temp.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deterministic_times = []
    for i in indices:
        t = time.perf_counter()
        _, image, _, _ = precompute((i, str(original.paths[original.rows[i]['Image_name'].casefold()])))
        deterministic_times.append(time.perf_counter()-t)
        np.save(temp / f'{i}.npy', image)
    class Prepared(Dataset):
        def __len__(self):
            return len(indices)
        def __getitem__(self, index):
            i = indices[index]
            image = __import__('PIL').Image.fromarray(np.load(temp / f'{i}.npy'))
            return image_tensor(image, 384, True)
    reference = Subset(original, indices)
    ready = Prepared()
    results = {}
    for label, data in [('original', reference), ('cached', ready)]:
        generator = torch.Generator().manual_seed(42)
        loader = DataLoader(data, batch_size=16, shuffle=False, num_workers=workers,
                            worker_init_fn=seed_worker, generator=generator)
        started = time.monotonic()
        count = 0
        for batch in loader:
            count += len(batch['image'] if label == 'original' else batch)
        results[label] = {'samples': count, 'elapsed_s': time.monotonic()-started,
                          'samples_s': count/(time.monotonic()-started)}
    results['deterministic_preparation_mean_s'] = float(np.mean(deterministic_times))
    results['loader_speedup'] = results['original']['elapsed_s']/results['cached']['elapsed_s']
    write_json(HERE / 'performance/cpu_cache_benchmark.json', results)
    print(json.dumps(results), flush=True)


def verify(root, cache_root):
    """Pixel/augmentation/worker-order equality against the historical data path."""
    torch.set_num_threads(1)
    state = torch.load(HERE / 'checkpoints/baseline.pt', map_location='cpu', weights_only=False)
    records = {}
    for split in ('train', 'valid'):
        for training in ([False, True] if split == 'train' else [False]):
            original = FundusDataset(root, split, state['age_stats'], size=384, training=training)
            cached = CachedFundusDataset(root, split, state['age_stats'], cache_root, training=training)
            indices = np.linspace(0, len(original)-1, 64, dtype=int).tolist()
            for i in indices:
                random.seed(42+i)
                torch.manual_seed(42+i)
                a = original[i]
                random.seed(42+i)
                torch.manual_seed(42+i)
                b = cached[i]
                for key in ('image', 'age', 'label', 'bbox', 'original_size'):
                    if not torch.equal(a[key], b[key]):
                        raise RuntimeError(f'Lossless cache mismatch: {split} {training} {i} {key}')
            records[f'{split}_training_{training}'] = {'sample_tensors_exactly_equal': len(indices)}
    old = FundusDataset(root, 'train', state['age_stats'], size=384, training=True)
    new = CachedFundusDataset(root, 'train', state['age_stats'], cache_root, training=True)
    first_four_hashes = []
    for data in (old, new):
        generator = torch.Generator().manual_seed(42)
        loader = DataLoader(data, batch_size=16, shuffle=True, num_workers=4,
                            worker_init_fn=seed_worker, generator=generator)
        digest = hashlib.sha256()
        iterator = iter(loader)
        for _ in range(4):
            digest.update(next(iterator)['image'].numpy().tobytes())
        first_four_hashes.append(digest.hexdigest())
        del iterator, loader
    legacy = json.loads((HERE / 'training_logs/AUC-C0_s42/epoch01.json').read_text())
    if not (first_four_hashes[0] == first_four_hashes[1] == legacy['natural_first_four_batches_tensor_sha256']):
        raise RuntimeError('Cache changed first four natural batches relative to completed C0')
    records['first_four_batch_hash'] = first_four_hashes[1]
    records['matches_completed_C0'] = True
    write_json(HERE / 'performance/cache_equivalence_checks.json', records)
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['benchmark', 'build', 'verify'])
    p.add_argument('--data-root', required=True)
    p.add_argument('--cache-root', default='/tmp/glaucoma_E_AUC01_cache')
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()
    (HERE / 'performance').mkdir(exist_ok=True)
    cache = Path(args.cache_root)
    if args.phase == 'benchmark':
        benchmark(args.data_root)
    elif args.phase == 'build':
        cache.mkdir(mode=0o700, exist_ok=True)
        if cache.stat().st_uid != os.getuid():
            raise RuntimeError('Cache directory must be owned by this account')
        os.chmod(cache, 0o700)
        for split in ('train', 'valid'):
            build_split(args.data_root, split, cache, args.workers)
        print(json.dumps(verify(args.data_root, cache)), flush=True)
        write_json(HERE / 'performance/cache_ready.json', {'cache_root': str(cache),
                   'ready': True, 'training_augmentation_unchanged': True,
                   'completed_unix': time.time()})
    else:
        print(json.dumps(verify(args.data_root, cache)), flush=True)


if __name__ == '__main__':
    main()
