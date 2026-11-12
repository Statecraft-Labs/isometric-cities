"""Resolve one requested place, acquire a bounded source and keep one resume receipt."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from project import ROOT, digest, normalise, recipe_path, write
from local_style import prepare_source, queue_style, style_plan

NOMINATIM = 'https://nominatim.openstreetmap.org/search'
USER_AGENT = 'IsometricCities/0.3 (single-user local Mac renderer; https://github.com/Statecraft-Labs/isometric-cities)'


def slug(text):
    value = normalise(text).replace(' ', '-')[:72]
    return value if value and value[0].isalpha() else 'location-' + (value or 'point')


def geocode(query, cache, *, offline=False, endpoint=None, opener=None, clock=None, sleep=None):
    """A single explicit lookup, cached and serialized across this repository's processes."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    endpoint = endpoint or os.environ.get('ATLAS_GEOCODER_URL', NOMINATIM)
    opener, clock, sleep = opener or urlopen, clock or time.time, sleep or time.sleep
    key = hashlib.sha256((endpoint + '\n' + normalise(query)).encode()).hexdigest()
    path = cache / (key + '.json')
    with (cache / '.request.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            record = json.loads(path.read_text())
            if record.get('error'):
                raise ValueError(record['error'] + f' Cached lookup: {path}')
            return record
        if offline:
            raise ValueError('No cached geocode. Use LAT,LON for deterministic offline resolution, or resolve this place online once.')
        stamp = cache / '.last-request'
        if stamp.exists():
            sleep(max(0, 1.05 - (clock() - float(stamp.read_text()))))
        url = endpoint + '?' + urlencode({'q': query, 'format': 'jsonv2', 'limit': 1})
        request = Request(url, headers={'User-Agent': USER_AGENT, 'Accept': 'application/json'})
        stamp.write_text(str(clock()))
        try:
            with opener(request, timeout=30) as response:
                payload = json.load(response)
        except (HTTPError, URLError, OSError, ValueError) as error:
            # Cache transport failures too. An agent must inspect before deliberately retrying.
            problem = f'Geocoder unavailable: {error}. Use explicit LAT,LON, a known recipe, or inspect the cached error before retrying.'
            write(path, {'query': query, 'endpoint': endpoint, 'error': problem})
            raise ValueError(problem) from error
        if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict) or not all(k in payload[0] for k in ('lat', 'lon')):
            problem = f'Geocoder found no usable result for {query!r}. Use a more specific location or LAT,LON.'
            write(path, {'query': query, 'endpoint': endpoint, 'error': problem})
            raise ValueError(problem)
        result = {'query': query, 'endpoint': endpoint, 'result': payload[0],
                  'attribution': '© OpenStreetMap contributors, ODbL 1.0', 'policy': 'https://operations.osmfoundation.org/policies/nominatim/'}
        write(path, result)
        return result


def starter(query, lat, lon):
    if not all(math.isfinite(v) for v in (lat, lon)) or not (-80 <= lat <= 80 and -179 <= lon <= 179):
        raise ValueError('Use local coordinates within latitude ±80 and longitude ±179; polar/date-line areas need a reviewed recipe')
    from geometry import SceneCoordinates
    scene = SceneCoordinates(lat, lon)
    west, south = scene.lonlat(-900, -900)
    east, north = scene.lonlat(900, 900)
    bbox = [float(south), float(west), float(north), float(east)]
    identity = hashlib.sha256(f'{lat:.7f},{lon:.7f}'.encode()).hexdigest()[:8]
    return {'id': slug(query) + '-' + identity, 'title': query, 'subtitle': 'Bounded local source starter',
            'origin': {'lat': lat, 'lon': lon}, 'bbox': bbox, 'patch_bbox': bbox,
            'terrain': {'provider': 'skadi'},
            'camera': {'view_quadrant': 'SE', 'pixels_per_metre': 2, 'pixel_upscale': 1,
                       'terrain_cell_metres': 4, 'patch_terrain_cell_metres': 4},
            'landmarks': [{'id': 'requested-centre', 'name': query, 'lat': lat, 'lon': lon}],
            'source_quality': {'kind': 'osm-terrain-starter', 'photographic_coverage': 'not established',
                               'next': 'Inspect the requested centre, choose a nearby landmark cluster, and find primary licensed aerial/oblique/3D sources. See docs/workflow.md.'}}


def resolve_location(query, cache, *, offline=False, geocoder=None):
    try:
        path = recipe_path(query)
    except ValueError:
        path = None
    if path:
        return {'method': 'recipe', 'query': query, 'recipe': json.loads(path.read_text()),
                'recipe_source': {'path': str(path), 'sha256': digest(path)}}
    match = re.fullmatch(r'\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*', query)
    if match:
        lat, lon = map(float, match.groups())
        return {'method': 'coordinates', 'query': query, 'recipe': starter(query, lat, lon)}
    record = (geocoder or geocode)(query, cache, offline=offline)
    try:
        lat, lon = float(record['result']['lat']), float(record['result']['lon'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('Cached geocoder result has invalid coordinates; inspect the cache or use LAT,LON') from error
    return {'method': 'geocoder', 'query': query, 'geocoder': record, 'recipe': starter(query, lat, lon)}


def render_source(receipt, out, *, offline=False):
    """The acquisition feature owns source providers and their file conventions."""
    from source_build import build_source
    recipe = receipt['location']['recipe']
    snapshot = out/'recipe.json'
    if snapshot.is_file() and json.loads(snapshot.read_text()) != recipe:
        raise ValueError('Recipe edits detected. Copy recipe.json and use a new --out; your edits were preserved.')
    write(snapshot, recipe)
    result = build_source(recipe, out/'capture', offline=offline)
    receipt['source_command'] = result.get('command')
    receipt['source_quality'] = result.get('quality')
    write(out/'receipt.json', receipt)
    return result['image'], result['manifest']


def check_review(receipt, review_path):
    from local_style import completed_art
    completed_art(receipt)
    if digest(receipt['source']['image']) != receipt['source']['sha256']:
        raise ValueError('Prepared source changed; its review no longer applies')
    review = json.loads(Path(review_path).read_text())
    hashes = {item['sha256'] for item in receipt.get('art', [])}
    checks = ('landmarks', 'materials', 'geometry', 'native_detail')
    if receipt['state'] not in ('art-ready', 'reviewed') or review.get('image_sha256') not in hashes or review.get('source_sha256') != receipt['source']['sha256']:
        raise ValueError('Review must bind the exact prepared source and completed art hashes')
    findings = review.get('findings')
    if not review.get('reviewer') or review.get('decision') not in ('accept', 'reject') or not isinstance(findings, list) or not findings or any(not isinstance(item, str) or not item.strip() for item in findings) or any(review.get('checks', {}).get(key) not in ('pass', 'fail') for key in checks):
        raise ValueError('Review needs reviewer, accept/reject, concrete findings and landmark/material/geometry/native_detail checks')
    if review['decision'] == 'accept' and any(review['checks'][key] != 'pass' for key in checks):
        raise ValueError('An accepted review needs four passing checks')
    receipt.update(state='reviewed', visual_acceptance=review['decision'], review={'path': str(Path(review_path).resolve()), 'sha256': digest(review_path), 'record': review})


def summary(receipt, path):
    source = receipt.get('source', {})
    result = {'receipt': str(path), 'state': receipt['state'], 'location': receipt['location']['query']}
    if source:
        result.update(source=source['image'], dimensions=source['conditioning']['dimensions'])
    if receipt.get('style_plan'):
        result.update(backend=receipt['style_plan']['backend'], style_ready=receipt['style_plan']['ready'])
        if not receipt['style_plan']['ready']:
            result['next'] = receipt['style_plan']['bootstrap_reason']
    if receipt.get('job'):
        result['job'] = receipt['job']['id']
    if receipt.get('art'):
        result['result'] = receipt['art']
        result['viewer'] = str(Path(path).parent / 'viewer.html')
        result['visual_acceptance'] = receipt['visual_acceptance']
    if receipt.get('next') and not source:
        result['next'] = receipt['next']
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog='atlas.py make', description=__doc__)
    parser.add_argument('location', help='Known place, plain-language location, recipe path, or LAT,LON')
    parser.add_argument('--source', type=Path, help='Use a local RGB image instead of acquiring/rendering sources')
    parser.add_argument('--manifest', type=Path, help='Optional genuine source camera/provenance JSON')
    parser.add_argument('--crop', nargs=4, type=int, metavar=('X', 'Y', 'WIDTH', 'HEIGHT'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--plan-only', action='store_true', help='Resolve and prepare provided pixels; acquire/render nothing and run no model')
    parser.add_argument('--style', action='store_true', help='Submit or resume one pinned local image job')
    parser.add_argument('--wait', action='store_true', help='Poll the existing job for up to 45 seconds; repeat safely')
    parser.add_argument('--backend', choices=('auto', 'imagine', 'mflux'), default='auto')
    parser.add_argument('--caption-file', type=Path)
    parser.add_argument('--scale', type=float, help='One of the selected preset\'s tested scales')
    parser.add_argument('--preset', type=Path, help='Pinned local style preset JSON')
    parser.add_argument('--review', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.manifest and not args.source:
            raise ValueError('--manifest requires --source')
        if args.plan_only and (args.style or args.wait or args.review):
            raise ValueError('--plan-only runs no model and cannot be combined with --style, --wait or --review')
        out = (args.out or ROOT / 'outputs/locations' / (slug(args.location) + '-' + hashlib.sha256(args.location.encode()).hexdigest()[:8])).resolve()
        out.mkdir(parents=True, exist_ok=True)
        with (out / '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = out / 'receipt.json'
            supplied = {'path': str(args.source.resolve()), 'sha256': digest(args.source)} if args.source else None
            identity = {'location': args.location, 'source': supplied, 'manifest': {'path': str(args.manifest.resolve()), 'sha256': digest(args.manifest)} if args.manifest else None, 'crop': args.crop}
            if Path(args.location).is_file():
                identity['recipe'] = {'path': str(Path(args.location).resolve()), 'sha256': digest(Path(args.location))}
            if path.exists():
                receipt = json.loads(path.read_text())
                if receipt['request'] != identity:
                    raise ValueError('This output belongs to different inputs. Resume with the same request or use a new --out directory.')
                if receipt.get('source') and digest(receipt['source']['image']) != receipt['source']['sha256']:
                    raise ValueError('Prepared source changed; use a new output directory')
                snapshot = out / 'recipe.json'
                if snapshot.is_file() and json.loads(snapshot.read_text()) != receipt['location']['recipe']:
                    raise ValueError('Recipe edits detected. Copy recipe.json to a separate file and run make with that path and a new --out. Your edits were preserved.')
            else:
                # Local images are allowed without a geocoding call: their place label is not registration.
                if args.source:
                    try:
                        location = resolve_location(args.location, ROOT / 'outputs/cache/geocoding', offline=True)
                    except ValueError:
                        location = {'method': 'image-label', 'query': args.location, 'recipe': None}
                else:
                    location = resolve_location(args.location, ROOT / 'outputs/cache/geocoding', offline=args.offline)
                receipt = {'schema_version': 1, 'state': 'resolved', 'request': identity, 'location': location, 'visual_acceptance': 'unreviewed'}
                write(path, receipt)
            if 'source' not in receipt:
                if args.source:
                    source, manifest = args.source, args.manifest
                elif args.plan_only:
                    recipe = out / 'recipe.json'
                    write(recipe, receipt['location']['recipe'])
                    receipt['next'] = 'Repeat without --plan-only to acquire OSM/terrain or configured photographic sources. Review richer source availability before styling.'
                    receipt['recipe_path'] = str(recipe)
                    write(path, receipt)
                    print(json.dumps(summary(receipt, path), indent=2))
                    return 0
                else:
                    source, manifest = render_source(receipt, out, offline=args.offline)
                receipt['source'] = prepare_source(source, out, crop=args.crop, manifest=manifest, allow_large=True)
                receipt['state'] = 'source-ready'
                write(path, receipt)
            recipe = receipt['location'].get('recipe') or {}
            recipe_caption = recipe.get('pixel_art_caption')
            if recipe.get('caption_file') and not args.caption_file:
                caption_path = Path(recipe['caption_file']).expanduser()
                recipe_caption = (caption_path if caption_path.is_absolute() else ROOT/caption_path).read_text().strip()
            caption = args.caption_file.read_text().strip() if args.caption_file else recipe_caption or f'A detailed isometric pixel-art map of the visible {args.location} scene in the reference. Crisp dark outlines, flat cel-shaded daylight, readable building facades and natural roof-material differences. Buildings, streets, parks and open courtyards follow the reference camera and geometry.'
            candidate = style_plan(receipt['source'], out, caption, args.scale, backend=args.backend, preset_path=args.preset)
            if receipt.get('job'):
                previous = receipt['style_plan']
                from local_style import same_style_job
                if not same_style_job(previous, candidate):
                    raise ValueError('A queued job retains its caption, scale, backend and model preset. Resume unchanged or use a new output directory.')
            else:
                receipt['style_plan'] = candidate
            write(path, receipt)
            if args.wait and not args.style and not receipt.get('job'):
                raise ValueError('No existing job to wait for. Use --style to submit one local job.')
            if args.style or args.wait:
                queue_style(receipt, path, wait=args.wait)
            if args.review:
                check_review(receipt, args.review)
                write(path, receipt)
                from local_style import export_viewer
                export_viewer(receipt, out)
            print(json.dumps(summary(receipt, path), indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(json.dumps({'status': 'failed', 'reason': str(error)}, indent=2), file=sys.stderr)
        return 1
