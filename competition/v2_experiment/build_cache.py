"""Build an exact deterministic crop/pad cache for one requested resolution."""
from common import *
import argparse
import concurrent.futures
import shutil
import time
from PIL import ImageOps
from core import open_rgb, valid_bbox


def prepare(job):
    index, path, size = job
    image = open_rgb(path)
    original_size = image.size
    bbox = valid_bbox(image)
    image = ImageOps.pad(image.crop(bbox), (size, size), method=__import__('PIL').Image.Resampling.BILINEAR,
                         color=(0, 0, 0))
    return index, np.asarray(image), bbox, original_size


def build(root, split, cache_root, size, workers):
    rows = read_rows(root, split)
    final, manifest_path = cache_root / f'{split}.npy', cache_root / f'{split}.json'
    if final.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['image_size'] == size and manifest['csv_sha256'] == sha256(Path(root) / f'{split}.csv'):
            return
        raise RuntimeError('Existing cache has incompatible metadata')
    required = len(rows) * size * size * 3
    if shutil.disk_usage(cache_root).free < required + 5_000_000_000:
        raise RuntimeError('Insufficient private NVMe space for cache')
    paths = {p.name.casefold(): p for p in (Path(root) / split).iterdir() if p.is_file()}
    temporary = cache_root / f'{split}.partial.npy'
    array = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.uint8, shape=(len(rows), size, size, 3))
    boxes, original_sizes = [None] * len(rows), [None] * len(rows)
    jobs = [(i, str(paths[row['Image_name'].casefold()]), size) for i, row in enumerate(rows)]
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for offset in range(0, len(jobs), workers * 4):
            for index, image, bbox, original_size in pool.map(prepare, jobs[offset:offset + workers * 4], chunksize=1):
                array[index] = image
                boxes[index], original_sizes[index] = bbox, original_size
            if offset % (workers * 40) == 0 or offset + workers * 4 >= len(jobs):
                array.flush()
                print(json.dumps({'event': 'cache', 'size': size, 'split': split,
                                  'images': min(len(jobs), offset + workers * 4), 'total': len(jobs),
                                  'elapsed_s': time.monotonic() - started}), flush=True)
    array.flush()
    del array
    temporary.replace(final)
    os.chmod(final, 0o600)
    write_json(manifest_path, {'image_size': size, 'split': split,
               'names': [row['Image_name'] for row in rows], 'boxes': boxes,
               'original_sizes': original_sizes, 'csv_sha256': sha256(Path(root) / f'{split}.csv'),
               'dtype': 'uint8', 'build_elapsed_s': time.monotonic() - started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    target = Path(args.cache_root)
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.stat().st_uid != os.getuid():
        raise RuntimeError('Cache directory is not owned by current user')
    os.chmod(target, 0o700)
    for split in ('train', 'valid'):
        build(args.data_root, split, target, args.size, args.workers)
    write_json(target / 'ready.json', {'size': args.size, 'ready': True, 'completed_unix': time.time()})
