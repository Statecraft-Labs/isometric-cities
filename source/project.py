"""Project paths, recipe discovery and durable local artifact files."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]


def normalise(value):
    return ' '.join(re.findall(r'[^\W_]+', unicodedata.normalize('NFKD', str(value)).casefold()))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def interpreter():
    runtime = ROOT / '.venv/bin/python'
    return runtime if runtime.is_file() else Path(sys.executable)


def recipe_path(value):
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    requested = normalise(value)
    for path in sorted((ROOT / 'cities').glob('*.json')):
        city = read(path)
        names = [path.stem, city['id'], city['title'], *city.get('aliases', [])]
        if requested in map(normalise, names):
            return path
    raise ValueError(f'No recipe for {value!r}; use atlas.py make to resolve a new place')


@contextmanager
def ownership(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f'Another process owns {folder}; resume after it finishes') from None
        yield
