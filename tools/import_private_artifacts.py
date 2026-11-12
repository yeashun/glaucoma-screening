"""Copy trusted local competition artifacts into ignored deployment directories."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def install(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(target) != sha(source):
            raise FileExistsError(f'Refusing to replace different artifact: {target.name}')
        return
    shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--native-runtime', action='store_true')
    args = parser.parse_args()
    source = args.source.resolve()
    target = ROOT / 'competition/deployment'
    selection = json.loads((source / 'model/selection.json').read_text(encoding='utf-8'))
    assert sha(source / 'model/stack.json') == selection['stack_sha256']
    names = ['selection.json', 'stack.json']
    for model in selection['models']:
        name = model['name'] + '_s42.pt'
        assert sha(source / 'model' / name) == model['export_sha256']
        names.append(name)
    for name in names:
        install(source / 'model' / name, target / 'model' / name)
    if (source / 'submission.csv').is_file():
        install(source / 'submission.csv', target / 'submission.csv')
    if args.native_runtime:
        native = source / 'code/_image_backend'
        manifest = json.loads((native / 'manifest.json').read_text(encoding='utf-8'))
        for name, record in manifest['runtime_files'].items():
            f = (native / name).resolve()
            assert f.is_relative_to(native.resolve()), 'Invalid runtime manifest path'
            assert sha(f) == record['sha256'], name
            install(f, target / 'code/_image_backend' / name)
        install(native / 'manifest.json', target / 'code/_image_backend/manifest.json')
    print(json.dumps({'model_files': len(names), 'native_runtime_imported': args.native_runtime,
                      'artifacts_directory_gitignored': True}))


if __name__ == '__main__':
    main()
