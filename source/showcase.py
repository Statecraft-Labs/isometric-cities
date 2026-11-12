"""Export a small continuous artwork, native Deep Zoom tiles and an offline explorer."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import html
import io
import math
import re
from pathlib import Path
import shutil
import sys
import webbrowser
from urllib.parse import urlsplit

from PIL import Image

from project import ROOT, digest, read, write, ownership
from tiles import build as build_tiles


def script_json(value):
    return json.dumps(value, ensure_ascii=True).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')


def data_url(path, *, preview=False):
    if not preview:
        return 'data:image/png;base64,' + base64.b64encode(Path(path).read_bytes()).decode('ascii')
    with Image.open(path) as image:
        image.thumbnail((1600, 1600), Image.Resampling.BOX)
        stream = io.BytesIO()
        image.save(stream, format='WEBP', quality=85, method=4)
    return 'data:image/webp;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')


def validate(image, source, metadata):
    if not isinstance(metadata, dict):
        raise ValueError('Metadata must be a JSON object')
    json.dumps(metadata, allow_nan=False)
    sizes = []
    for path, key in ((image, 'image_sha256'), (source, 'source_sha256')):
        with Image.open(path) as opened:
            if opened.format != 'PNG' or opened.mode != 'RGB' or opened.getexif().get(274, 1) != 1:
                raise ValueError('Use RGB PNGs with no pending orientation transform')
            if min(opened.size) < 32 or max(opened.size) > 32768 or opened.width * opened.height > 64_000_000:
                raise ValueError('Showcase images must fit within 32..32768 pixels per side and 64 megapixels')
            opened.load()
            sizes.append(opened.size)
        if key in metadata and metadata[key] != digest(path):
            raise ValueError(f'Metadata {key} differs from the actual file')
    if sizes[0] != sizes[1] or min(sizes[0]) < 32 or max(sizes[0]) > 32768:
        raise ValueError('Source and art must have identical native dimensions, 32..32768 pixels')
    width, height = sizes[0]
    for key, value in (('width', width), ('height', height)):
        if key in metadata and metadata[key] != value:
            raise ValueError(f'Metadata {key} differs from the native image')
    regions, ids = metadata.get('regions', []), set()
    if not isinstance(regions, list):
        raise ValueError('Regions must be a list')
    for region in regions:
        if not isinstance(region, dict) or not isinstance(region.get('name'), str) or not region['name'].strip():
            raise ValueError('Each region needs a name and a box [x,y,width,height]')
        box = region.get('box')
        if not isinstance(box, list) or len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in box):
            raise ValueError('Region boxes must contain four finite numbers')
        x, y, w, h = box
        if min(x, y) < 0 or min(w, h) <= 0 or x+w > width or y+h > height:
            raise ValueError('Region boxes must lie inside the native image')
        identity = region.get('id', region['name'])
        if not isinstance(identity, str) or identity in ids:
            raise ValueError('Region IDs must be unique strings')
        ids.add(identity)
    if metadata.get('opening_region') is not None and metadata['opening_region'] not in ids:
        raise ValueError('opening_region must name an existing region ID')
    navigation = metadata.get('navigation', [])
    if not isinstance(navigation, list):
        raise ValueError('Navigation must be a list of relative city-page links')
    for link in navigation:
        if not isinstance(link, dict) or not isinstance(link.get('name'), str) or not link['name'].strip() or not isinstance(link.get('href'), str):
            raise ValueError('Each city link needs a name and relative HTML href')
        href = link['href']
        url = urlsplit(href)
        if url.scheme or url.netloc or href.startswith(('/', '\\')) or '\\' in href or not url.path.endswith('.html') or any(ord(c) < 32 for c in href):
            raise ValueError('City navigation links must be relative local HTML pages')
    points = metadata.get('landmarks', [])
    if not isinstance(points, list):
        raise ValueError('Landmarks must be a list')
    for point in points:
        if not isinstance(point, dict) or any(type(point.get(k)) not in (int, float) or not math.isfinite(point[k]) for k in ('x', 'y')):
            raise ValueError('Landmark anchors must have finite x and y')
        if not 0 <= point['x'] <= width or not 0 <= point['y'] <= height:
            raise ValueError('Landmark anchors must lie inside the native image')
    point_ids = [point.get('id', str(index+1)) for index, point in enumerate(points)]
    if any(not isinstance(value, str) or not value for value in point_ids) or len(set(point_ids)) != len(point_ids):
        raise ValueError('Landmark IDs must be unique nonempty strings')
    if metadata.get('opening_landmark') is not None and metadata['opening_landmark'] not in point_ids:
        raise ValueError('opening_landmark must name an existing landmark')
    for value in [metadata.get('hero_zoom', 1), *[point.get('zoom', 1) for point in points]]:
        if type(value) not in (int, float) or not math.isfinite(value) or not .05 <= value <= 4:
            raise ValueError('Hero zoom must be finite in .05..4')
    if any(type(point.get('view_width', 1024)) not in (int, float) or not math.isfinite(point.get('view_width', 1024)) or not 128 <= point.get('view_width', 1024) <= 32768 for point in points):
        raise ValueError('Landmark view_width must be finite in 128..32768 pixels')
    if any(type(point.get('look_up', 0)) not in (int, float) or not math.isfinite(point.get('look_up', 0)) or abs(point.get('look_up', 0)) > height for point in points):
        raise ValueError('Landmark look_up must be a finite offset inside the image height')
    if not metadata.get('credits'):
        raise ValueError('Supply source credits in metadata; they travel with every viewer')
    return width, height


HTML = Path(__file__).with_name('explorer.html').read_text()


def page(image, source, metadata, *, tiled=False):
    width, height = validate(Path(image), Path(source), metadata)
    record = dict(metadata, width=width, height=height,
                  image_sha256=digest(image), source_sha256=digest(source))
    record.pop('viewer_tiles', None)
    if tiled:
        record['viewer_tiles'] = [record['deep_zoom'], record['source_deep_zoom']]
    values = {'ART': script_json(data_url(image, preview=tiled)), 'SOURCE': script_json(data_url(source, preview=tiled)),
              'DATA': script_json(record), 'TITLE': html.escape(str(record.get('title', 'City')) + ' · Isometric Atlas')}
    return re.sub(r'__(ART|SOURCE|DATA|TITLE)__', lambda match: values[match[1]], HTML)


def export(image, source, metadata, out):
    image, source, out = map(lambda p: Path(p).resolve(), (image, source, out))
    width, height = validate(image, source, metadata)
    if any(path == out or out in path.parents for path in (image, source)):
        raise ValueError('Export directory cannot contain either input')
    with ownership(out):
        if any(path.name != '.lock' for path in out.iterdir()):
            raise ValueError('Export directory must be new or empty; existing work is unchanged')
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image, out/'map.png')
        shutil.copyfile(source, out/'source.png')
        record = dict(metadata, width=width, height=height, image_sha256=digest(image), source_sha256=digest(source))
        record['export'] = {'method': 'Unchanged native PNG bytes; tiles cut after the artwork is complete',
                            'resampling': 'box at lower levels; native level unchanged', 'native_upscale': 1}
        record['deep_zoom'] = build_tiles(out/'map.png', out/'map.dzi', out/'map_files')
        record['source_deep_zoom'] = build_tiles(out/'source.png', out/'source.dzi', out/'source_files')
        write(out/'manifest.json', record)
        (out/'viewer.html').write_text(page(image, source, record, tiled=True))
        (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=viewer.html"><a href="viewer.html">Open map</a>')
        write(out/'export-proof.json', {str(path.relative_to(out)): digest(path)
              for path in sorted(out.rglob('*')) if path.is_file() and path.name != '.lock'})
        return {'viewer': str(out/'viewer.html'), 'map': str(out/'map.png'), 'tiles': str(out/'map.dzi'),
                'dimensions': [width, height], 'image_sha256': record['image_sha256'], 'source_sha256': record['source_sha256']}


def demo_main(argv=None):
    parser = argparse.ArgumentParser(description='Export both bundled examples for offline exploration. No model or network required.')
    parser.add_argument('--out', type=Path, help='Explicit immutable export folder; default selects the current example edition')
    parser.add_argument('--open', action='store_true', help='Open the finished offline explorer in your default browser')
    parser.add_argument('--serve', action='store_true', help='Serve only the exported examples over local HTTP')
    parser.add_argument('--host', default='127.0.0.1', help='Use 0.0.0.0 for an iPhone on the same Wi-Fi')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('--port must be in 1..65535')
    if args.out is None:
        inputs=[digest(ROOT/f'examples/{city}{suffix}') for city in ('dc','edinburgh') for suffix in ('.png','-source.png','.json')]
        edition=hashlib.sha256(json.dumps([inputs,HTML,Image.__version__,digest(__file__)]).encode()).hexdigest()[:12]
        args.out=ROOT/'outputs/demo'/edition
    results = {}
    for city in ('dc', 'edinburgh'):
        image, source = ROOT/f'examples/{city}.png', ROOT/f'examples/{city}-source.png'
        metadata, out = read(ROOT/f'examples/{city}.json'), args.out.resolve()/city
        if (out/'manifest.json').is_file():
            with ownership(out):
                record = read(out/'manifest.json')
                if any(record.get(key) != value for key, value in metadata.items()):
                    raise ValueError(f'Existing example metadata at {out} differs; use a fresh --out')
                if record.get('image_sha256') != digest(image) or record.get('source_sha256') != digest(source) or digest(out/'map.png') != digest(image) or digest(out/'source.png') != digest(source):
                    raise ValueError(f'Existing example at {out} differs; use a fresh --out')
                if (out/'viewer.html').read_text() != page(image, source, record, tiled=True):
                    raise ValueError(f'Existing explorer at {out} differs; use a fresh --out')
                proof = read(out/'export-proof.json')
                if any((out/name).resolve().is_relative_to(out) is False or digest(out/name) != expected
                       for name, expected in proof.items()):
                    raise ValueError(f'Existing example files at {out} changed; use a fresh --out')
                results[city] = {'viewer': str(out/'viewer.html'), 'cached': True}
        else:
            results[city] = export(image, source, metadata, out)
    (args.out.resolve()/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=dc/viewer.html"><a href="dc/viewer.html">DC</a> <a href="edinburgh/viewer.html">Edinburgh</a>')
    print(json.dumps(results, indent=2))
    if args.serve:
        from functools import partial
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
        handler = partial(SimpleHTTPRequestHandler, directory=str(args.out.resolve()))
        with ThreadingHTTPServer((args.host, args.port), handler) as server:
            url=f'http://localhost:{server.server_port}/'
            print(f'Explorer: {url}', flush=True)
            if args.host == '0.0.0.0':
                print(f'iPhone: open http://YOUR-MAC-LAN-ADDRESS:{server.server_port}/ on the same Wi-Fi. Ctrl-C stops sharing.', flush=True)
            if args.open:
                webbrowser.open(url)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    elif args.open:
        webbrowser.open((args.out.resolve()/'index.html').as_uri())
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(export(args.image, args.source, json.loads(args.metadata.read_text()), args.out), indent=2))
        return 0
    except (ValueError, OSError) as error:
        print(json.dumps({'status': 'failed', 'reason': str(error)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
