"""Check the public tree for restricted artifacts, private paths and credentials."""
import csv
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKED = {'.pt', '.pth', '.ckpt', '.joblib', '.pkl', '.zip', '.gz', '.pdf', '.xlsx',
           '.jpg', '.jpeg', '.png', '.npy', '.npz', '.so', '.pem', '.key', '.jsonl'}
PATTERNS = {
    'private IPv4 address': r'\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b',
    'private account path': r'/data/' + r'home/[^/\s]+/|[CD]:[\\/]Users[\\/]|[D]:[\\/]',
    'GitHub credential': r'\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b',
    'private key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'fixed hardware UUID': r'GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
}


def candidates():
    if (ROOT / '.git').exists():
        # Untracked local artifacts are not publication inputs.
        out = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '-z'])
        return [ROOT / n.decode('utf-8') for n in out.split(b'\0') if n]
    return [f for f in ROOT.rglob('*') if f.is_file() and '__pycache__' not in f.parts]


def main():
    errors = []
    files = candidates()
    for f in files:
        name = f.relative_to(ROOT).as_posix()
        if f.suffix.lower() in BLOCKED or f.name.startswith('.env'):
            errors.append((name, 'restricted artifact type'))
        if f.stat().st_size > 1000000:
            errors.append((name, 'unexpected large file'))
        if f.suffix == '.csv':
            if not name.startswith('docs/results/'):
                errors.append((name, 'non-aggregate CSV'))
            with f.open(encoding='utf-8', newline='') as handle:
                fields = next(csv.reader(handle), [])
            if set(fields) & {'Image_name', 'sample_id', 'image_path', 'True_Label'}:
                errors.append((name, 'sample-level CSV fields'))
        text = f.read_text(encoding='utf-8')
        for reason, pattern in PATTERNS.items():
            if re.search(pattern, text, flags=re.IGNORECASE):
                errors.append((name, reason))
    if errors:
        for name, reason in errors:
            print(f'FAIL {name}: {reason}')
        raise SystemExit(1)
    print(f'PASS: {len(files)} public files; no restricted artifacts, credentials or private paths detected.')


if __name__ == '__main__':
    main()
