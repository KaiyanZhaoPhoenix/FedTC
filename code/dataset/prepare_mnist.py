"""Download or copy the exact, hash-verified FedAA MNIST partition used in the studies."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
import urllib.request


HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=HERE / 'mnist')
    p.add_argument('--from-checkout', type=Path, help='Existing FedAA checkout containing dataset/mnist/')
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--jobs', type=int, default=4)
    args = p.parse_args()
    if args.jobs < 1:
        p.error('--jobs must be positive')
    spec = json.loads((HERE / 'mnist_manifest.json').read_text())
    destination = args.output.expanduser().resolve()
    source = args.from_checkout.expanduser().resolve() / 'dataset/mnist' if args.from_checkout else None

    def prepare(item):
        relative, expected = item
        target = destination / relative
        target.resolve().relative_to(destination)
        if target.exists():
            if sha256(target) != expected:
                raise RuntimeError(f'Existing file differs from the study: {target}. Use a different --output directory.')
            return 'verified'
        if args.verify_only:
            raise FileNotFoundError(f'Missing study data: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name+'.', suffix='.part', delete=False) as f:
            temporary = Path(f.name)
        try:
            if source:
                shutil.copyfile(source / relative, temporary)
            else:
                url = f'https://raw.githubusercontent.com/Gp1g/FedAA/{spec["commit"]}/dataset/mnist/{relative}'
                for attempt in range(3):
                    try:
                        request = urllib.request.Request(url, headers={'User-Agent': 'FedTC-data-reproduction'})
                        with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as output:
                            shutil.copyfileobj(response, output)
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(attempt+1)
            if sha256(temporary) != expected:
                raise RuntimeError(f'Download/copy hash mismatch: {relative}')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return 'prepared'

    count = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(prepare, item) for item in spec['files'].items()]
        for future in as_completed(futures):
            future.result()
            count += 1
            if count % 25 == 0 or count == len(futures):
                print(f'Verified {count}/{len(futures)} files', flush=True)
    print(f'Exact study partition ready: {destination}')


if __name__ == '__main__':
    main()
