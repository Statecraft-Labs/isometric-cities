"""Registered, neighbour-conditioned local edits. This is not shared-latent sampling."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

import local_style as style
from project import ownership

WINDOW, FEATHER = 1024, 128


def starts(length):
    if length < WINDOW:
        raise ValueError('Every source side must be at least 1024 pixels; no resizing or padding. Style a smaller source as one image with make --style, or widen the recipe frame.')
    count = -(-(length - WINDOW) // 800) + 1 if length > WINDOW else 1
    return [round(i * (length - WINDOW) / max(count - 1, 1)) for i in range(count)]


def composite(canvas, done, raw, x, y):
    """The exact float32/EDT feather used by the selected city studies."""
    covered = done[y:y + WINDOW, x:x + WINDOW]
    alpha = np.ones((WINDOW, WINDOW), np.float32)
    if covered.any():
        distance = distance_transform_edt(covered)
        alpha = np.where(covered, np.clip(1 - distance / FEATHER, 0, 1), 1).astype(np.float32)
        if covered.all():
            alpha[:] = 0
    current = canvas[y:y + WINDOW, x:x + WINDOW].astype(np.float32)
    canvas[y:y + WINDOW, x:x + WINDOW] = np.clip(
        current * (1 - alpha[..., None]) + raw.astype(np.float32) * alpha[..., None] + .5,
        0, 255).astype(np.uint8)
    done[y:y + WINDOW, x:x + WINDOW] = True


def pixels(path, size=None):
    with Image.open(path) as image:
        if image.mode != 'RGB' or image.getexif().get(274, 1) != 1:
            raise ValueError('Use RGB pixels without pending EXIF orientation')
        if size is not None and image.size != tuple(size):
            raise ValueError(f'Wrong native dimensions: {path}')
        return np.array(image)


def prompt_for(caption, preset):
    if not caption.strip() or len(caption) > 4000:
        raise ValueError('Caption must contain 1..4000 characters')
    template = preset['inference'].get('prompt_template')
    if template:
        return template.format(scene=caption)
    trigger = preset['adapter']['trigger']
    return caption if caption.startswith(trigger) else trigger + ' ' + caption


def adoption_plan(folder, identity):
    """Read the frozen experimental method; never trust unbound PNG discovery."""
    plan = json.loads((folder / 'plan.json').read_text())
    expected = {
        'source_sha256': identity['source_sha256'], 'size': identity['size'],
        'box': [0, 0, *identity['size']], 'window': WINDOW, 'feather': FEATHER,
        'windows': identity['windows'], 'prompt': identity['prompt']}
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError('Adopted experiment differs from source, prompt or window geometry')
    preset = identity['preset']
    settings = preset['inference']
    recipe = plan['recipe']
    expected_recipe = {
        'model': settings['queue_route'], 'base_revision': preset['base']['revision'],
        'adapter_sha256': preset['adapter']['sha256'], 'steps': settings['steps'],
        'guidance': settings['guidance'], 'seed_base': settings['seed']}
    if any(recipe.get(key) != value for key, value in expected_recipe.items()) or float(recipe['scale']) != settings['adapter_scale']:
        raise ValueError('Adopted experiment model, adapter or schedule differs')
    return plan


def adopt_window(folder, wd, expected_pixels, identity, index):
    old = folder / 'windows' / wd.name
    record = json.loads((old / 'run.json').read_text())
    source = old / 'input.png'
    if record['returncode'] != 0 or style.digest(source) != record['input_sha256']:
        raise ValueError('Adopted window failed or its conditioned input changed')
    if not np.array_equal(pixels(source, [WINDOW, WINDOW]), expected_pixels):
        raise ValueError('Adopted window does not contain the expected source and finished neighbours')
    argv = record['argv']
    settings, preset = identity['preset']['inference'], identity['preset']
    expected_flags = {'-m': settings['queue_route'], '--steps': str(settings['steps']),
                      '--seed': str(settings['seed'] + index), '--width': str(WINDOW),
                      '--height': str(WINDOW), '--reference-max-dimension': str(WINDOW),
                      '--guidance': str(settings['guidance'])}
    if argv[1] != identity['prompt'] or any(argv[argv.index(flag) + 1] != value for flag, value in expected_flags.items()):
        raise ValueError('Adopted window command differs from the recipe')
    if preset['base']['revision'] not in argv[argv.index('--model') + 1]:
        raise ValueError('Adopted window used a different base revision')
    lora_index = argv.index('--lora')
    if preset['adapter']['revision'] not in argv[lora_index + 1] or float(argv[lora_index + 2]) != settings['adapter_scale']:
        raise ValueError('Adopted window used a different adapter revision or scale')
    outputs = record['outputs']
    if len(outputs) != 1:
        raise ValueError('Adoption requires exactly one hash-bound prediction per window')
    name, expected_sha = next(iter(outputs.items()))
    prediction = old / Path(name).name
    if style.digest(prediction) != expected_sha:
        raise ValueError('Adopted raw prediction changed')
    pixels(prediction, [WINDOW, WINDOW])
    target = wd / 'raw.png'
    shutil.copyfile(prediction, target)
    result = {'path': str(target), 'sha256': style.digest(target),
              'input_sha256': style.digest(wd / 'input.png'),
              'imported_record_sha256': style.digest(old / 'run.json'), 'job': record.get('job')}
    style.write(wd / 'adopted.json', result)
    return result


def paint(source, out, caption, *, preset_path=None, adopt=None, plan_only=False, backend='auto'):
    source, out = Path(source).resolve(), Path(out).resolve()
    preset_path = Path(preset_path or style.DEFAULT_PRESET).resolve()
    preset = json.loads(preset_path.read_text())
    settings = preset['inference']
    if settings.get('window') != WINDOW or settings.get('max_native_reference_area') != WINDOW * WINDOW:
        raise ValueError('Context infill requires a validated native 1024-square preset')
    if settings.get('adapter_scale') not in settings.get('scales_tested', []):
        raise ValueError('Preset adapter scale has not been tested')
    original = pixels(source)
    height, width = original.shape[:2]
    if width * height > 64_000_000:
        raise ValueError('Source exceeds the 64 megapixel preparation limit')
    grid = [[x, y] for y in starts(height) for x in starts(width)]
    identity = {'algorithm': 'context-infill-v1', 'implementation_sha256': {name: style.digest(Path(__file__).with_name(name)) for name in ('context_infill.py', 'local_style.py', 'project.py')},
                'source_sha256': style.digest(source), 'size': [width, height],
                'preset': preset, 'prompt': prompt_for(caption, preset), 'backend': backend,
                'windows': grid, 'window': WINDOW, 'feather': FEATHER,
                'method': 'sequential neighbour-conditioned edits; not shared-latent denoising'}
    adoption = Path(adopt).resolve() if adopt else None
    experimental = adoption_plan(adoption, identity) if adoption else None
    with ownership(out):
        plan_path = out / 'plan.json'
        if plan_path.exists():
            previous = json.loads(plan_path.read_text())
            if previous['identity'] != identity:
                raise ValueError('Source, prompt, model, schedule or composition code changed; use a fresh output')
        else:
            style.write(plan_path, {'identity': identity, 'source': str(source), 'state': 'planned'})
        if plan_only:
            return {'state': 'planned', 'windows': len(grid), 'plan': str(plan_path), 'queued': False}
        canvas, done = original.copy(), np.zeros((height, width), bool)
        for index, (x, y) in enumerate(grid):
            wd = out / 'windows' / f'w{index:02d}-x{x}-y{y}'
            wd.mkdir(parents=True, exist_ok=True)
            expected = canvas[y:y + WINDOW, x:x + WINDOW].copy()
            input_path = wd / 'input.png'
            if input_path.exists() and not np.array_equal(pixels(input_path, [WINDOW, WINDOW]), expected):
                raise ValueError('Window source or finished-neighbour pixels changed; resume refused')
            if not input_path.exists():
                Image.fromarray(expected).save(input_path)
            adopted_path = wd / 'adopted.json'
            if adopted_path.exists():
                result = json.loads(adopted_path.read_text())
                if result['input_sha256'] != style.digest(input_path):
                    raise ValueError('Imported window input changed; resume refused')
            elif adoption:
                result = adopt_window(adoption, wd, expected, identity, index)
            else:
                window_preset = deepcopy(preset)
                window_preset['inference']['seed'] += index
                window_preset_path = wd / 'preset.json'
                style.write(window_preset_path, window_preset)
                prepared = {'image': str(input_path), 'sha256': style.digest(input_path),
                            'conditioning': style.native_dimensions(WINDOW, WINDOW)}
                candidate = style.style_plan(prepared, wd, caption, backend=backend, preset_path=window_preset_path)
                receipt_path = wd / 'receipt.json'
                if receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text())
                    if receipt['source']['sha256'] != prepared['sha256'] or not style.same_style_job(receipt['style_plan'], candidate):
                        raise ValueError('Window receipt identity changed; resume refused')
                    if not receipt.get('job'):
                        # Nothing was submitted, so readiness and model paths may have changed since.
                        receipt['style_plan'] = candidate
                        style.write(receipt_path, receipt)
                else:
                    receipt = {'source': prepared, 'style_plan': candidate, 'state': 'prepared'}
                    style.write(receipt_path, receipt)
                while not style.completed_art(receipt):
                    style.queue_style(receipt, receipt_path, wait=True)
                if not style.completed_art(receipt):
                    raise ValueError('Window has no verified completed art')
                if len(receipt['art']) != 1:
                    raise ValueError('One prediction per window is required')
                result = receipt['art'][0]
            if style.digest(result['path']) != result['sha256']:
                raise ValueError('Raw prediction changed; resume refused')
            raw = pixels(result['path'], [WINDOW, WINDOW])
            composite(canvas, done, raw, x, y)
            Image.fromarray(canvas).save(out / 'map-progress.png')
            print(f'{index + 1}/{len(grid)} windows verified', flush=True)
        target = out / 'map.png'
        if not done.all():
            raise ValueError('Window grid left uncovered source pixels')
        Image.fromarray(canvas).save(target)
        if experimental and experimental.get('map_sha256') and style.digest(target) != experimental['map_sha256']:
            raise ValueError('Replayed map differs from the experiment final image')
        result = {'state': 'art-ready', 'image': str(target), 'sha256': style.digest(target),
                  'windows': len(grid), 'visual_acceptance': 'unreviewed'}
        style.write(plan_path, {'identity': identity, 'source': str(source), **result})
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--caption-file', type=Path, required=True)
    parser.add_argument('--preset', type=Path, default=style.DEFAULT_PRESET)
    parser.add_argument('--adopt', type=Path, help='Replay existing hash-bound experimental windows without inference')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--backend', choices=['auto', 'imagine', 'mflux'], default='auto')
    args = parser.parse_args(argv)
    try:
        result = paint(args.source, args.out, args.caption_file.read_text().strip(), preset_path=args.preset,
                       adopt=args.adopt, plan_only=args.plan_only, backend=args.backend)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, KeyError, IndexError) as error:
        print(json.dumps({'state': 'failed', 'reason': str(error)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
