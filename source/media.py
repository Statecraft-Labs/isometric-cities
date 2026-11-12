#!/usr/bin/env python3
"""Export a city fly-through with sparse, geographically anchored annotations."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile

from project import digest, write

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
Image.MAX_IMAGE_PIXELS = 300_000_000
MAX_WORDS = 7


@dataclass(frozen=True)
class Viewport:
    left: float
    top: float
    width: float
    height: float
    output_width: int
    output_height: int

    @property
    def scale(self):
        return self.width / self.output_width

    def project(self, x, y):
        return ((x - self.left) / self.scale, (y - self.top) / self.scale)

    def unproject(self, x, y):
        return (self.left + x * self.scale, self.top + y * self.scale)


@dataclass(frozen=True)
class Annotation:
    kind: str
    text: str
    opacity: float
    source_anchor: tuple[float, float] | None = None


@dataclass(frozen=True)
class FilmFrame:
    index: int
    seconds: float
    viewport: Viewport
    annotation: Annotation | None


def word_count(text):
    return len(re.findall(r"\w+(?:['’-]\w+)*", text, flags=re.UNICODE))


def font(size):
    for path in (
        Path('/System/Library/Fonts/Supplemental/Arial.ttf'),
        Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def catmull(a, b, c, d, t):
    return .5 * ((2 * b) + (-a + c) * t + (2*a - 5*b + 4*c - d) * t*t + (-a + 3*b - 3*c + d) * t*t*t)


def clamp(value, low, high):
    return min(max(value, low), high)


class FilmPlan:
    def __init__(self, manifest, seconds=90, fps=60, width=1920, height=1080, callouts=False, source_credit_band=None):
        if not math.isfinite(seconds) or seconds <= 0 or fps <= 0 or width < 160 or height < 90 or width % 2 or height % 2:
            raise ValueError('Use a positive duration/fps and even dimensions of at least 160×90')
        self.manifest = manifest
        self.seconds, self.fps = seconds, fps
        self.width, self.height = width, height
        self.count = max(2, round(seconds * fps))
        self.callouts = callouts
        band = 0 if source_credit_band is None else source_credit_band
        if band < 0:
            raise ValueError('Source credit band cannot be negative')
        self.safe_bounds = (2.0, 2.0, float(manifest['width'] - 2), float(manifest['height'] - band))
        left, top, right, bottom = self.safe_bounds
        if right <= left or bottom <= top:
            raise ValueError('Map is too small after excluding the source credit strip')
        self.aspect = width / height
        film = manifest.get('film', {})
        zoom = film.get('zoom_fraction', .06)
        if type(zoom) not in (int, float) or not 0 <= zoom <= .15 or not math.isfinite(zoom):
            raise ValueError('film.zoom_fraction must be a finite number in 0..0.15')
        self.zoom_fraction = zoom
        self.opening_credit = film.get('opening_credit', '')
        if self.opening_credit:
            raise ValueError('Opening text is not supported; source credits belong in metadata and the accompanying description')
        requested_width = None
        if 'view_width_pixels' in film:
            value = film['view_width_pixels']
            if type(value) not in (int, float) or value < 1:
                raise ValueError('film.view_width_pixels must be a finite number of at least one source pixel')
            try:
                requested_width = float(value)
            except OverflowError as error:
                raise ValueError('film.view_width_pixels exceeds the supported numeric range') from error
            if not math.isfinite(requested_width):
                raise ValueError('film.view_width_pixels must be finite')
            self.base_width = min(requested_width, (right-left)/(1+zoom), (bottom-top)*self.aspect/(1+zoom))
        else:
            self.base_width = min(2700, (right-left)*.42, (bottom-top)*self.aspect*.82) / (1+zoom)
        self.framing = {
            'mode': 'automatic' if requested_width is None else 'explicit',
            'requested_view_width_pixels': requested_width,
            'nominal_view_width_pixels': self.base_width,
            'zoom_range_view_width_pixels': [self.base_width*(1-zoom), self.base_width*(1+zoom)],
            'zoom_fraction': zoom,
            'clamped_to_source': requested_width is not None and self.base_width < requested_width,
        }
        points = list(manifest.get('landmarks', [])) or [{'name': '', 'x': manifest['width']/2, 'y': manifest['height']/2}]
        requested=film.get('route',[])
        if requested:
            if not isinstance(requested,list) or any(not isinstance(key,str) for key in requested) or len(set(requested))!=len(requested):
                raise ValueError('Film route must contain unique landmark IDs')
            available={point.get('id'):point for point in points}
            selected=[available[key] for key in requested if key in available]
            if selected:
                points=selected
        self.stops = []
        half_width = self.base_width * (1+zoom) / 2
        half_height = half_width / self.aspect
        for point in points:
            x, y = float(point.get('pixel_x', point.get('x'))), float(point.get('pixel_y', point.get('y')))
            if not all(map(math.isfinite, (x, y))):
                raise ValueError('Landmark coordinates must be finite')
            self.stops.append({
                'id': point.get('id'), 'name': str(point.get('name', '')),
                'anchor_x': x, 'anchor_y': y,
                'camera_x': clamp(x, left + half_width, right - half_width),
                'camera_y': clamp(y, top + half_height, bottom - half_height),
            })
        if callouts:
            for index in range(2, len(self.stops), 3):
                if word_count(self.stops[index]['name']) > MAX_WORDS:
                    raise ValueError(f'Landmark label exceeds seven words: {self.stops[index]["name"]}')

    def frame(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        elapsed = index / self.fps
        t = index / (self.count - 1)
        position = t * len(self.stops)
        segment = min(int(position), len(self.stops)-1)
        local = position - segment
        indices = [(segment + offset) % len(self.stops) for offset in (-1, 0, 1, 2)]
        knots = [self.stops[i] for i in indices]
        x = catmull(*(p['camera_x'] for p in knots), local)
        y = catmull(*(p['camera_y'] for p in knots), local)
        view_width = self.base_width * (1 + self.zoom_fraction * math.cos(2 * math.pi * t))
        view_height = view_width / self.aspect
        left, top, right, bottom = self.safe_bounds
        x = clamp(x, left + view_width/2, right - view_width/2)
        y = clamp(y, top + view_height/2, bottom - view_height/2)
        viewport = Viewport(x-view_width/2, y-view_height/2, view_width, view_height, self.width, self.height)
        annotation = None
        if self.callouts:
            for stop_index in range(2, len(self.stops), 3):
                stop = self.stops[stop_index]
                centre = stop_index / len(self.stops) * (self.count-1) / self.fps
                distance = abs(elapsed - centre)
                if distance <= 1.5 and stop['name']:
                    anchor = (stop['anchor_x'], stop['anchor_y'])
                    px, py = viewport.project(*anchor)
                    if self.width*.1 < px < self.width*.9 and self.height*.12 < py < self.height*.88:
                        annotation = Annotation('landmark', stop['name'], min(1.0, (1.5-distance)/.35), anchor)
                    break
        return FilmFrame(index, elapsed, viewport, annotation)

    def verify(self):
        max_words, max_error, annotated_frames = 0, 0.0, 0
        intervals = []
        previous_key = None
        for index in range(self.count):
            frame = self.frame(index)
            view, annotation = frame.viewport, frame.annotation
            left, top, right, bottom = self.safe_bounds
            if not (view.left >= left-1e-8 and view.top >= top-1e-8 and view.left+view.width <= right+1e-8 and view.top+view.height <= bottom+1e-8):
                raise AssertionError(f'Frame {index} leaves the safe source area')
            words = word_count(annotation.text) if annotation and annotation.opacity > 0 else 0
            max_words = max(max_words, words)
            if words > MAX_WORDS:
                raise AssertionError(f'Frame {index} exceeds seven words')
            if annotation and annotation.source_anchor:
                annotated_frames += 1
                projected = view.project(*annotation.source_anchor)
                recovered = view.unproject(*projected)
                error = math.dist(recovered, annotation.source_anchor)
                max_error = max(max_error, error)
                if error > 1e-8:
                    raise AssertionError('Landmark leader tip does not map to the original source coordinate')
            key = (annotation.kind, annotation.text) if annotation else None
            if key != previous_key:
                intervals.append({'first_frame': index, 'seconds': frame.seconds, 'kind': key[0] if key else None, 'text': key[1] if key else ''})
                previous_key = key
        return {
            'result': 'pass', 'frames_checked': self.count, 'maximum_on_screen_words': max_words,
            'maximum_source_projection_roundtrip_error_pixels': max_error,
            'landmark_annotated_frames': annotated_frames, 'all_viewports_inside_safe_source': True,
            'safe_source_bounds': self.safe_bounds,
            'source_bottom_band_excluded_pixels': self.manifest['height'] - self.safe_bounds[3],
            'annotation_intervals': intervals,
            'no_title_card_corner_labels_hud_or_progress_bar': True,
        }


def annotation_geometry(annotation, viewport, size, name_font):
    width, height = size
    anchor = viewport.project(*annotation.source_anchor)
    bounds = name_font.getbbox(annotation.text)
    text_width = bounds[2] - bounds[0]
    if text_width > width * .70:
        raise ValueError(f'Landmark label does not fit: {annotation.text}')
    label_x = clamp(anchor[0] + width*.075, width*.15, width*.85-text_width)
    label_y = clamp(anchor[1] - height*.13, height*.16, height*.78)
    elbow = (label_x - width*.015, label_y + (bounds[3]-bounds[1])*.5)
    return anchor, elbow, (label_x, label_y)


def render_frame(source, frame, name_font):
    view = frame.viewport
    image = source.transform((view.output_width, view.output_height), Image.Transform.AFFINE,
        (view.scale, 0, view.left, 0, view.scale, view.top), resample=Image.Resampling.BICUBIC)
    annotation = frame.annotation
    if annotation is None or annotation.opacity <= 0:
        return image
    overlay = Image.new('RGBA', image.size)
    draw = ImageDraw.Draw(overlay)
    alpha = round(255 * annotation.opacity)
    ink, paper = (20, 32, 33, alpha), (255, 248, 224, alpha)
    anchor, elbow, label = annotation_geometry(annotation, view, image.size, name_font)
    draw.line((anchor, elbow, (label[0]-3, elbow[1])), fill=ink, width=5)
    draw.line((anchor, elbow, (label[0]-3, elbow[1])), fill=paper, width=2)
    draw.ellipse((anchor[0]-3, anchor[1]-3, anchor[0]+3, anchor[1]+3), fill=paper, outline=ink, width=1)
    draw.text(label, annotation.text, font=name_font, fill=paper, stroke_width=2, stroke_fill=ink)
    return Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')


def credits_text(manifest):
    credit = manifest.get('media_attribution') or manifest.get('credits') or manifest.get('attribution')
    if not credit:
        raise ValueError('Supply source credits in the manifest before exporting media')
    if not isinstance(credit, str):
        credit = '\n'.join(map(str, credit)) if isinstance(credit, list) else json.dumps(credit)
    return credit + '\nKeep these credits with the film and its sharing description.\n'


def export(args):
    manifest = json.loads(args.manifest.read_text())
    if args.output is None:
        slug = manifest.get('city_id') or re.sub(r'[^a-z0-9]+', '-', manifest.get('title', 'city').lower()).strip('-')
        args.output = ROOT/'outputs/media'/f'{slug}-flythrough.mp4'
    if args.output.exists() and (not args.overwrite or args.plan_only):
        raise FileExistsError(f'Output exists; choose a fresh path, or use --overwrite to re-encode: {args.output}')
    plan = FilmPlan(manifest, args.seconds, args.fps, args.width, args.height, args.callouts, args.source_credit_band)
    proof = plan.verify()
    if manifest.get('image_sha256') and digest(args.image) != manifest['image_sha256']:
        raise ValueError('Map pixels do not match the manifest hash')
    source = Image.open(args.image)
    if source.size != (manifest['width'], manifest['height']):
        raise ValueError('Map dimensions do not match the manifest')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    credit = credits_text(manifest)
    args.output.with_suffix('.credits.txt').write_text(credit)
    args.output.with_suffix('.route.json').write_text(json.dumps({
        'seconds': args.seconds, 'fps': args.fps, 'size': [args.width, args.height],
        'route': plan.stops, 'interpolation': 'closed Catmull-Rom centres, sinusoidal zoom, safe bounds clamp',
        'source': str(args.image), 'image_sha256': digest(args.image),
        'safe_source_bounds': plan.safe_bounds, 'callouts': args.callouts,
        'framing': plan.framing,
    }, indent=2) + '\n')
    args.output.with_suffix('.verification.json').write_text(json.dumps(proof, indent=2) + '\n')
    if args.plan_only:
        source.close()
        print(json.dumps(proof, indent=2))
        return proof
    source = source.convert('RGB')
    name_font = font(max(14, round(args.height/36)))
    # Resolve layout failures before opening the encoder or writing a partial movie.
    for index in range(plan.count):
        frame = plan.frame(index)
        if frame.annotation and frame.annotation.source_anchor:
            annotation_geometry(frame.annotation, frame.viewport, (args.width, args.height), name_font)
    temporary = args.output.with_name(args.output.stem + '.rendering.mp4')
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{args.width}x{args.height}', '-r', str(args.fps), '-i', '-', '-an', '-c:v', 'libx264',
        '-preset', 'fast', '-crf', '17', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        '-metadata', f"title={manifest.get('title', 'City')} fly-through", '-metadata', f'comment={credit}', str(temporary)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    contact = []
    sample_indices = {0, plan.count//4, plan.count//2, 3*plan.count//4, plan.count-1}
    try:
        for index in range(plan.count):
            image = render_frame(source, plan.frame(index), name_font)
            if index in sample_indices:
                contact.append(image.resize((640, 360), Image.Resampling.LANCZOS))
            encoder.stdin.write(image.tobytes())
            if index % (args.fps*10) == 0:
                print(f'{index/args.fps:.0f}/{args.seconds:g} seconds', flush=True)
        encoder.stdin.close()
        code = encoder.wait()
        if code:
            raise RuntimeError(f'ffmpeg exited with {code}')
    finally:
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait()
    temporary.replace(args.output)
    sheet = Image.new('RGB', (640*len(contact), 360))
    for index, image in enumerate(contact):
        sheet.paste(image, (index*640, 0))
    sheet.save(args.output.with_suffix('.contact.png'))
    print(args.output, flush=True)
    return proof


def teaser(image, manifest, output, centres, width=800, height=450, hold=2., overwrite=False):
    """Four held native crops use one shared palette, with no dither or text."""
    image, manifest, output = map(Path, (image, manifest, output))
    data = json.loads(manifest.read_text())
    credits = credits_text(data)
    if data.get('image_sha256') and digest(image) != data['image_sha256']:
        raise ValueError('Map pixels do not match the manifest hash')
    if output.suffix.lower() != '.gif' or type(width) is not int or type(height) is not int or min(width, height) < 1:
        raise ValueError('Use a GIF output and positive integer crop dimensions')
    if not math.isfinite(hold) or not .1 <= hold <= 60 or not centres or len(centres) > 12:
        raise ValueError('Use 1..12 centres and a finite hold duration in 0.1..60 seconds')
    if output.exists() and not overwrite:
        raise FileExistsError(f'Output exists: {output}')
    with Image.open(image) as opened:
        source = opened.convert('RGB')
    if source.size != (data['width'], data['height']) or source.width < width or source.height < height:
        raise ValueError('Crop must fit the native map; manifest dimensions must match')
    crops = []
    for centre in centres:
        if len(centre) != 2 or not all(type(v) in (int, float) and math.isfinite(v) for v in centre):
            raise ValueError('Each centre needs two finite source pixel coordinates')
        cx, cy = centre
        if not 0 <= cx < source.width or not 0 <= cy < source.height:
            raise ValueError('Crop centres must lie inside the native map')
        left, top = round(clamp(cx-width//2, 0, source.width-width)), round(clamp(cy-height//2, 0, source.height-height))
        crops.append((left, top, left+width, top+height))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='atlas-teaser-') as temporary:
        temporary = Path(temporary)
        for index, bounds in enumerate(crops):
            source.crop(bounds).save(temporary/f'{index:02d}.png')
        encoded = temporary/'teaser.gif'
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-framerate', f'1/{hold}',
            '-i', str(temporary/'%02d.png'), '-vf',
            'split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];[b][p]paletteuse=dither=none',
            '-loop', '0', str(encoded)], check=True)
        output.write_bytes(encoded.read_bytes())
    source.crop(crops[0]).save(output.with_suffix('.png'))
    output.with_suffix('.credits.txt').write_text(credits)
    record = {'image_sha256': digest(image), 'gif_sha256': digest(output), 'dimensions': [width, height],
              'native_crop_bounds': crops, 'hold_seconds': hold, 'loop': 0, 'maximum_on_screen_words': 0,
              'palette': 'one shared 256-colour palette, no dither'}
    write(output.with_suffix('.teaser.json'), record)
    return record


def tease_main(argv=None):
    parser = argparse.ArgumentParser(description='Native landmark crop GIF, no text or resampling')
    for name in ('image', 'manifest', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--centres', nargs='+', required=True, metavar='X,Y')
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--height', type=int, default=450)
    parser.add_argument('--hold', type=float, default=2.)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    try:
        centres = [list(map(float, value.split(','))) for value in args.centres]
        print(json.dumps(teaser(args.image, args.manifest, args.output, centres, args.width, args.height,
                                args.hold, args.overwrite), indent=2))
        return 0
    except (ValueError, OSError) as error:
        parser.error(str(error))


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, default=ROOT/'examples/dc.png')
    parser.add_argument('--manifest', type=Path, default=ROOT/'examples/dc.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seconds', type=float, default=90)
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--callouts', action='store_true', help='Occasional landmark names with map-anchored leader lines')
    parser.add_argument('--source-credit-band', type=int, help='Bottom source pixels excluded; default zero')
    parser.add_argument('--plan-only', action='store_true', help='Verify every planned frame and write sidecars without encoding')
    parser.add_argument('--overwrite', action='store_true')
    return parser


def main(argv=None):
    parser = argument_parser()
    try:
        export(parser.parse_args(argv))
        return 0
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
