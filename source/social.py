#!/usr/bin/env python3
"""Cut a short portrait feed film from a registered map: keyed shots, source-to-art wipes, no text."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw

from media import ROOT, Viewport, catmull, clamp, credits_text
from project import digest

Image.MAX_IMAGE_PIXELS = 300_000_000
MAX_UPSCALE = 1.25
EASES = {'linear': lambda t: t, 'inout': lambda t: t*t*(3-2*t), 'in': lambda t: t*t, 'out': lambda t: t*(2-t)}


def default_recipe(manifest):
    """Any place: wipe in on the opening landmark, glide the film route, return so it loops."""
    marks = [point['id'] for point in manifest['landmarks']]
    if not marks:
        raise ValueError('Supply manifest landmarks or use --recipe with map-coordinate shots')
    route = [key for key in manifest.get('film', {}).get('route', []) if key in marks] or marks
    lead = manifest.get('opening_landmark') if manifest.get('opening_landmark') in marks else route[0]
    stops = [lead, *[key for key in route if key != lead][:5], lead]
    glides = [{'seconds': 4, 'path': [{'at': a}, {'at': b}]} for a, b in zip(stops, stops[1:])]
    return {'size': [1080, 1350], 'fps': 30, 'loop': True, 'shots': [
        {'seconds': 2.6, 'path': [{'at': lead}], 'wipe': [.5, 1.02]}, *glides,
        {'seconds': 1.6, 'path': [{'at': lead}], 'wipe': [1.02, .5]}]}


class SocialPlan:
    """Shots cut hard unless neighbours share a key. `wipe` is the art-covered share of the width.
    Views stay within 1.25x of native unless the recipe sets `pixel_zoom`, up to 3."""

    def __init__(self, manifest, recipe):
        self.width, self.height = recipe.get('size', [1080, 1350])
        self.fps = recipe.get('fps', 30)
        if min(self.width, self.height) < 90 or self.width % 2 or self.height % 2 or self.fps <= 0:
            raise ValueError('Use even dimensions of at least 90 pixels and a positive frame rate')
        self.map_width, self.map_height = manifest['width'], manifest['height']
        self.aspect = self.width / self.height
        self.widest = min(self.map_width, self.map_height * self.aspect)
        self.loop = bool(recipe.get('loop'))
        self.closest = self.width / float(recipe.get('pixel_zoom', MAX_UPSCALE))
        if not self.width / 3 <= self.closest <= self.width:
            raise ValueError('pixel_zoom must lie in 1..3')
        anchors = {point['id']: (float(point.get('pixel_x', point.get('x'))), float(point.get('pixel_y', point.get('y'))))
                   for point in manifest.get('landmarks', [])}
        self.shots, first = [], 0
        for shot in recipe['shots']:
            frames = round(shot['seconds'] * self.fps)
            if frames < 1 or shot.get('ease', 'inout') not in EASES or not shot.get('path'):
                raise ValueError('Each shot needs a path, a positive duration and a known ease')
            self.shots.append({'first': first, 'frames': frames, 'ease': shot.get('ease', 'inout'),
                               'wipe': shot.get('wipe'), 'keys': [self.key(key, anchors) for key in shot['path']]})
            first += frames
        self.count = first

    def key(self, key, anchors):
        at = key['at']
        if isinstance(at, str) and at not in anchors:
            raise ValueError(f'Unknown landmark id: {at}')
        x, y = anchors[at] if isinstance(at, str) else map(float, at)
        dx, dy = key.get('offset', (0, 0))
        width = float(key.get('width', self.width))
        if not self.closest <= width <= self.widest:
            raise ValueError(f'View width {width:g} is outside {self.closest:g}..{self.widest:g} map pixels')
        return (*self.inside(x + dx, y + dy, width), math.log(width))

    def inside(self, x, y, width):
        height = width / self.aspect
        return clamp(x, width/2, self.map_width - width/2), clamp(y, height/2, self.map_height - height/2)

    def frame(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        shot = next(shot for shot in reversed(self.shots) if shot['first'] <= index)
        t = (index - shot['first']) / max(1, shot['frames'] - 1)
        keys = shot['keys']
        position = EASES[shot['ease']](t) * (len(keys) - 1)
        segment = min(int(position), max(0, len(keys) - 2))
        knots = [keys[clamp(segment + offset, 0, len(keys) - 1)] for offset in (-1, 0, 1, 2)]
        x, y, zoom = (catmull(*(knot[axis] for knot in knots), position - segment) for axis in range(3))
        width = clamp(math.exp(zoom), self.closest, self.widest)
        x, y = self.inside(x, y, width)
        view = Viewport(x - width/2, y - width/self.aspect/2, width, width/self.aspect, self.width, self.height)
        wipe = shot['wipe']
        return view, None if wipe is None else wipe[0] + (wipe[1] - wipe[0]) * EASES['inout'](t)

    def verify(self):
        upscale, wipes = 0.0, 0
        for index in range(self.count):
            view, wipe = self.frame(index)
            if min(view.left, view.top) < -1e-6 or view.left + view.width > self.map_width + 1e-6 \
                    or view.top + view.height > self.map_height + 1e-6:
                raise AssertionError(f'Frame {index} leaves the map')
            upscale = max(upscale, 1 / view.scale)
            wipes += wipe is not None and 0 < wipe < 1
        first, last = self.frame(0), self.frame(self.count - 1)
        closure = max(abs(a - b) for a, b in zip((*first[0].__dict__.values(), 1 if first[1] is None else first[1]),
                                               (*last[0].__dict__.values(), 1 if last[1] is None else last[1])))
        if upscale > self.width / self.closest + 1e-9 or self.loop and closure > 1e-6:
            raise AssertionError('Film zooms past its limit or its last frame does not return to its first')
        return {'result': 'pass', 'frames_checked': self.count, 'seconds': self.count / self.fps,
                'maximum_on_screen_words': 0, 'text_or_overlay_layers': 'none; one unlabelled wipe edge',
                'all_viewports_inside_map': True, 'maximum_square_pixel_zoom': upscale, 'source_wipe_frames': wipes,
                'wipe_layers_share_one_viewport': True, 'loop_closure_error': closure if self.loop else None,
                'shot_first_frames': [shot['first'] for shot in self.shots]}


def sample(image, view):
    """One viewport transform for every layer. Zoomed-in pixels stay square, never smeared."""
    size = (view.output_width, view.output_height)
    if view.scale >= 1:
        return image.resize(size, Image.Resampling.LANCZOS, box=(view.left, view.top, view.left + view.width, view.top + view.height))
    left, top = math.floor(view.left), math.floor(view.top)
    crop = image.crop((left, top, math.ceil(view.left + view.width), math.ceil(view.top + view.height)))
    crop = crop.resize((crop.width * 4, crop.height * 4), Image.Resampling.NEAREST)
    box = [(value - origin) * 4 for value, origin in ((view.left, left), (view.top, top), (view.left + view.width, left), (view.top + view.height, top))]
    return crop.resize(size, Image.Resampling.BOX, box=box)


def render(art, source, view, wipe):
    image = sample(art, view)
    if wipe is None or wipe >= 1:
        return image
    edge = max(0, round(wipe * view.output_width))
    image.paste(sample(source, view).crop((edge, 0, *image.size)), (edge, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((edge - 4, 0, edge + 3, image.height), fill=(22, 30, 34))
    draw.rectangle((edge - 2, 0, edge + 1, image.height), fill=(255, 248, 224))
    return image


def export(args):
    manifest = json.loads(args.manifest.read_text())
    image = args.image or args.manifest.with_suffix('.png')
    source = args.source or args.manifest.with_name(args.manifest.stem + '-source.png')
    recipe_path = args.recipe or args.manifest.with_name(args.manifest.stem + '-social.json')
    recipe = json.loads(recipe_path.read_text()) if recipe_path.exists() else default_recipe(manifest)
    output = args.output or ROOT/'outputs/media'/f"{manifest.get('city_id', 'city')}-social.mp4"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f'Output exists; choose a fresh path or use --overwrite: {output}')
    if digest(image) != manifest['image_sha256'] or digest(source) != manifest['source_sha256']:
        raise ValueError('Map or source pixels do not match the manifest hashes')
    art, original = (Image.open(path).convert('RGB') for path in (image, source))
    if not art.size == original.size == (manifest['width'], manifest['height']):
        raise ValueError('Map, source and manifest dimensions must match for a registered wipe')
    plan = SocialPlan(manifest, recipe)
    proof = plan.verify()
    credit = credits_text(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.credits.txt').write_text(credit)
    output.with_suffix('.plan.json').write_text(json.dumps({
        'recipe': recipe, 'recipe_path': str(recipe_path) if recipe_path.exists() else 'default from manifest landmarks',
        'image_sha256': manifest['image_sha256'], 'source_sha256': manifest['source_sha256'], 'verification': proof}, indent=2) + '\n')
    render(art, original, *plan.frame(0)).save(output.with_suffix('.poster.png'))
    picks = sorted({shot['first'] + offset for shot in plan.shots for offset in (0, shot['frames'] // 2, shot['frames'] - 1)})
    sheet = Image.new('RGB', (360 * min(6, len(picks)), 450 * math.ceil(len(picks) / 6)))
    for cell, index in enumerate(picks):
        tile = render(art, original, *plan.frame(index)).resize((360, 450), Image.Resampling.LANCZOS)
        sheet.paste(tile, (cell % 6 * 360, cell // 6 * 450))
    sheet.save(output.with_suffix('.contact.jpg'), quality=90)
    if args.plan_only:
        print(json.dumps(proof, indent=2))
        return proof
    temporary = output.with_name(output.stem + '.rendering.mp4')
    encoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{plan.width}x{plan.height}', '-r', str(plan.fps), '-i', '-', '-an',
        '-vf', 'scale=out_color_matrix=bt709:out_range=tv,format=yuv420p', '-c:v', 'libx264', '-preset', 'medium',
        '-crf', '17', '-threads', '6', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
        '-movflags', '+faststart', '-metadata', f"title={manifest.get('title', 'City')}", '-metadata', f'comment={credit}',
        str(temporary)], stdin=subprocess.PIPE)
    try:
        for index in range(plan.count):
            encoder.stdin.write(render(art, original, *plan.frame(index)).tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError('ffmpeg failed')
    finally:
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait()
    temporary.replace(output)
    print(output, flush=True)
    return proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'examples/dc.json')
    parser.add_argument('--image', type=Path, help='Default: the manifest path with .png')
    parser.add_argument('--source', type=Path, help='Default: the manifest stem plus -source.png')
    parser.add_argument('--recipe', type=Path, help='Shot list; default MANIFEST-social.json, else landmarks')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plan-only', action='store_true', help='Verify every frame; write poster and contact sheet only')
    parser.add_argument('--overwrite', action='store_true')
    try:
        export(parser.parse_args(argv))
        return 0
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
