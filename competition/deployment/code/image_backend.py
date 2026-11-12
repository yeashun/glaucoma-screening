"""Activate the offline row-parallel Pillow build before torchvision imports PIL."""
import ctypes
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

INFO = {'enabled': False, 'implementation': 'standard_pillow'}


def activate():
    global INFO
    if os.environ.get('GLAUCOMA_DISABLE_NATIVE_IMAGE') == '1':
        INFO['reason'] = 'explicitly disabled'
        return INFO
    tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
    if sys.platform != 'linux' or platform.machine() not in ('x86_64', 'AMD64') or tag not in ('cp311','cp312'):
        INFO['reason'] = 'native image backend targets Linux x86_64 CPython 3.11/3.12'
        return INFO
    root = Path(__file__).resolve().parent / '_image_backend'
    if not (root/'manifest.json').is_file():
        INFO['reason'] = 'Public source checkout: optional native runtime is not bundled'
        return INFO
    if any(name == 'PIL' or name.startswith('PIL.') for name in sys.modules):
        raise RuntimeError('Import code/predict.py before any PIL/torchvision import to select the verified backend')
    root = Path(__file__).resolve().parent / '_image_backend'
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['runtime_files'].items():
        if relative.startswith('cp') and not relative.startswith(tag+'/'):
            continue
        file = root / relative
        actual = hashlib.sha256(file.read_bytes()).hexdigest()
        if actual != expected['sha256']:
            raise RuntimeError(f'Native image dependency hash mismatch: {relative}')
    # These are dependency libraries, not images or outputs. Every prediction still reads its file.
    for name in manifest['preload_libraries']:
        ctypes.CDLL(str(root/'lib'/name), mode=ctypes.RTLD_GLOBAL)
    os.environ['PILLOW_RESAMPLE_THREADS'] = '4'
    sys.path.insert(0, str(root/tag))
    import PIL
    if PIL.__version__ != '12.3.0':
        raise RuntimeError('Unexpected Pillow backend')
    INFO = {'enabled': True, 'implementation': 'pillow12.3_exact_integer_parallel_rows',
            'resize_threads': 4, 'crop_threads': 4, 'python_tag': tag,
            'module': PIL.__file__, 'pixel_algorithm_changed': False}
    return INFO
