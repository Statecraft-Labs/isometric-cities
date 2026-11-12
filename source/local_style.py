"""Native image preparation and resumable local styling: the isolated MFlux worker or a compatible job queue."""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from project import ROOT, digest, write
import re
import shutil
import subprocess

from PIL import Image

DEFAULT_PRESET = ROOT / 'model-presets/pixelart-redmond-klein9b.json'
MAX_DIMENSION = 2048
MAX_AREA = 2048 * 2048


def native_dimensions(width, height):
    if any(type(v) is not int or v < 256 or v > MAX_DIMENSION or v % 16 for v in (width, height)):
        raise ValueError('Native style dimensions must be multiples of 16 in 256..2048; choose an exact --crop. No automatic resize.')
    if width * height > MAX_AREA:
        raise ValueError('Reference exceeds the 4,194,304 pixel conditioning limit')
    return {'dimensions': [width, height], 'reference_max_dimension': math.ceil(max(width, height) / 32) * 32,
            'reference_area': width * height, 'maximum_reference_area': MAX_AREA,
            'required_backend_reference_area': max(1024 * 1024, width * height),
            'route': 'klein4b' if width * height <= 1024 * 1024 else 'klein4b-wide'}


def prepare_source(source, out, *, crop=None, manifest=None, allow_large=False):
    """Preserve literal pixels; attach supplied provenance without inferring registration."""
    source, out = Path(source).resolve(), Path(out).resolve()
    with Image.open(source) as image:
        if image.mode != 'RGB' or image.getexif().get(274, 1) != 1:
            raise ValueError('Use native RGB pixels with no pending EXIF orientation transform')
        image.load()
        x, y, width, height = crop or (0, 0, *image.size)
        if any(type(v) is not int for v in (x, y, width, height)) or min(x, y) < 0 or min(width, height) <= 0 or x + width > image.width or y + height > image.height:
            raise ValueError('Crop X Y WIDTH HEIGHT must lie exactly inside the source; it is never clamped')
        try:
            limits = native_dimensions(width, height)
        except ValueError:
            if not allow_large:
                raise
            if max(width, height) > 32768 or width * height > 64_000_000:
                raise ValueError('Source exceeds the 64 megapixel preparation limit')
            # A source render is independent of any model's bounded edit window.
            # Preserve all pixels; an unvalidated large pass is never submitted.
            limits = {'dimensions':[width,height], 'reference_max_dimension':min(2048, math.ceil(max(width,height)/32)*32),
                      'reference_area':width*height, 'maximum_reference_area':MAX_AREA,
                      'required_backend_reference_area':max(1024*1024,width*height),
                      'route':'klein4b-wide', 'native_supported':False}
        metadata = json.loads(Path(manifest).read_text()) if manifest else None
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError('Manifest must be a JSON object')
        if metadata:
            declared = metadata.get('camera', {})
            if not isinstance(declared, dict):
                raise ValueError('Manifest camera must be a JSON object')
            for key, actual in [('width', image.width), ('height', image.height)]:
                expected = metadata.get(key, declared.get(key))
                if expected is not None and expected != actual:
                    raise ValueError(f'Manifest {key} differs from actual source')
            # Comparison manifests bind both files; --source names their source side.
            expected_hash = metadata.get('source_sha256', metadata.get('image_sha256', metadata.get('sha256', metadata.get('deep_zoom', {}).get('image_sha256'))))
            if expected_hash is not None and expected_hash != digest(source):
                raise ValueError('Manifest image SHA-256 differs from actual source')
        camera = deepcopy(metadata.get('camera')) if metadata else None
        if camera and 'pixel_matrix' in camera:
            matrix = camera['pixel_matrix']
            if len(matrix) != 2 or any(len(row) != 4 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in row) for row in matrix):
                raise ValueError('Supplied camera pixel_matrix must be finite 2x4')
            matrix[0][3] -= x
            matrix[1][3] -= y
            camera.update(width=width, height=height)
            camera['source_pixel_box'] = [x, y, x + width, y + height]
            # The source frame describes the parent image; do not leave it as the crop frame.
            if 'frame' in camera:
                camera['parent_frame'] = camera.pop('frame')
        elif camera and (x or y or (width, height) != image.size):
            raise ValueError('A supplied camera needs pixel_matrix for exact crop translation')
        out.mkdir(parents=True, exist_ok=True)
        target = out / 'source.png'
        if target == source:
            raise ValueError('Output source.png must differ from the original source')
        image.crop((x, y, x + width, y + height)).save(target)
        result = {'original': {'path': str(source), 'sha256': digest(source), 'dimensions': list(image.size)},
                  'image': str(target), 'sha256': digest(target), 'crop_xywh': [x, y, width, height],
                  'crop_box': [x, y, x + width, y + height], 'resize': None,
                  'conditioning': limits, 'source_camera': metadata.get('camera') if metadata else None,
                  'camera': camera, 'registration': 'supplied-manifest; inspect its coordinate evidence' if camera else 'unregistered-image-study',
                  'manifest': {'path': str(Path(manifest).resolve()), 'sha256': digest(manifest)} if manifest else None,
                  'source_metadata': metadata}
    write(out / 'source-provenance.json', result)
    return result


def hf_cache():
    return Path(os.environ.get('HF_HUB_CACHE', Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface')) / 'hub')).expanduser()


def model_paths(preset, cache=None):
    """Resolve revision contents, including the cache's literal trailing-space subfolder."""
    cache = Path(cache) if cache else hf_cache()
    base = preset['base']
    snapshot = cache / ('models--' + base['repo'].replace('/', '--')) / 'snapshots' / base['revision']
    candidates = [snapshot, *[p for p in snapshot.iterdir() if p.is_dir()]] if snapshot.exists() else []
    bases = [p for p in candidates if all(any((p / name).glob('*.safetensors')) for name in ('transformer', 'text_encoder', 'vae'))]
    if len(bases) != 1:
        raise ValueError(f'Pinned base cache needs exactly one model directory at {snapshot}; run ./setup.sh --models')
    adapter = preset['adapter']
    path = cache / ('models--' + adapter['repo'].replace('/', '--')) / 'snapshots' / adapter['revision'] / adapter['filename']
    if not path.is_file() or digest(path) != adapter['sha256']:
        raise ValueError(f'Pinned adapter missing or SHA-256 mismatch: {path}; run ./setup.sh --models')
    return str(bases[0]), str(path)


def imagine_path():
    """An explicit ATLAS_IMAGINE, or an installed client that speaks this queue protocol.

    Other programs are also called imagine; one found by name alone is never run.
    """
    explicit = os.environ.get('ATLAS_IMAGINE')
    if explicit:
        executable = shutil.which(explicit)
        if not executable:
            raise ValueError('ATLAS_IMAGINE must name an executable queue client; correct or unset it before retrying')
        return str(Path(executable).resolve())
    for candidate in (shutil.which('imagine'), Path.home() / '.local/bin/imagine'):
        try:
            if candidate and b'[imagine] queued' in Path(candidate).read_bytes():
                return str(candidate)
        except OSError:
            pass
    return None


def style_plan(source, out, caption, scale=None, *, cache=None, backend='auto', preset_path=None):
    path = Path(preset_path) if preset_path else DEFAULT_PRESET
    preset = json.loads(path.read_text())
    inference = preset['inference']
    scale = inference.get('adapter_scale', .65) if scale is None else scale
    scales = inference.get('scales_tested', [.5, .55, .65])
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale not in scales:
        raise ValueError(f'Use one of the preset\'s tested adapter scales: {scales}')
    if type(inference['steps']) is not int or inference['steps'] < 1 or type(inference['seed']) is not int:
        raise ValueError('A style preset needs positive integer steps and an integer seed')
    if type(inference['guidance']) not in (int, float) or not math.isfinite(inference['guidance']):
        raise ValueError('Style guidance must be finite')
    if not caption.strip() or len(caption) > 4000:
        raise ValueError('Caption must contain 1..4000 characters')
    if inference.get('prompt_template'):
        caption = inference['prompt_template'].format(scene=caption)
    elif not caption.startswith(preset['adapter']['trigger']):
        caption = preset['adapter']['trigger'] + ' ' + caption
    ready, problem = True, None
    try:
        base, adapter = model_paths(preset, cache)
    except ValueError as error:
        ready, problem = False, str(error)
        base = f"<HF:{preset['base']['repo']}@{preset['base']['revision']}>"
        adapter = f"<HF:{preset['adapter']['repo']}@{preset['adapter']['revision']}/{preset['adapter']['filename']}>"
    executable = imagine_path()
    backend = ('imagine' if executable else 'mflux') if backend == 'auto' else backend
    if backend == 'imagine' and not executable:
        ready, problem = False, 'No compatible queue client found. Set ATLAS_IMAGINE to one, or use --backend mflux.'
    settings = deepcopy(source['conditioning'])
    width, height = settings['dimensions']
    settings['route'] = inference.get('queue_route', settings['route'])
    if settings['route'] not in ('klein4b', 'klein4b-wide', 'klein4b-base', 'klein9b', 'klein9b-base', 'qwen', 'qwen-best'):
        raise ValueError('Preset needs a supported local edit route; generic klein is FLUX 3')
    if settings['reference_area'] > inference.get('max_native_reference_area', MAX_AREA):
        ready, problem = False, 'Source exceeds this preset\'s tested native conditioning area. Use an exact pilot crop, or atlas.py paint for native context infill.'
    if settings.get('native_supported') is False:
        ready, problem = False, 'Full source preserved. Use an exact native pilot crop, or atlas.py paint for native context infill. Automatic shrinking is disabled.'
    argv = [executable or 'imagine', caption, source['image'], '-m', settings['route'],
            '--steps', str(inference['steps']), '--seed', str(inference['seed']),
            '--width', str(width), '--height', str(height),
            '--reference-max-dimension', str(settings['reference_max_dimension']),
            '--out', str(Path(out) / 'art'), '--no-wait']
    if 'output_resolution' in inference:
        argv += ['--output-resolution', str(inference['output_resolution'])]
    argv += ['--extra', '--model', base, '--guidance', str(inference['guidance']), '--lora', adapter, str(scale)]
    if backend == 'imagine' and max(width, height) < 512:
        ready, problem = False, 'imagine enlarges tiny references. Choose an exact crop at least 512 pixels on its longest edge.'
    native = native_argv(source, out, caption, preset, base, adapter, scale)
    if backend == 'mflux':
        python = ROOT / 'outputs/image-runtime/bin/python'
        argv = [str(python), '-I', str(Path(__file__).resolve()), '--worker', str(Path(out) / 'receipt.json')]
        if not python.is_file():
            ready, problem = False, 'Install the pinned image runtime and weights: ./setup.sh --models'
        if executable:
            ready, problem = False, 'A compatible local queue is installed and owns the GPU here. Use --backend imagine.'
        elif native is None:
            ready, problem = False, 'This preset needs the imagine backend; the isolated native worker currently supports FLUX.2 Klein only.'
    return {'argv': argv, 'backend': backend, 'native_argv': native, 'ready': ready, 'bootstrap_reason': problem, 'preset': preset,
            'preset_sha256': digest(path), 'caption': caption, 'scale': scale, 'conditioning': settings,
            'source_sha256': source['sha256'], 'resized': False,
            'visual_acceptance': 'unreviewed',
            'note': 'Inspect actual backend conditioning dimensions in the queue log before review.'}


def same_style_job(previous, candidate):
    """A resume cannot silently substitute model weights, schedule or a different style."""
    keys = ('caption', 'scale', 'backend', 'preset', 'conditioning')
    return all(previous.get(key) == candidate.get(key) for key in keys)


def queue_style(receipt, path, *, wait=False, run=None):
    """The caller holds the receipt lock. Persist intent before submitting once."""
    if completed_art(receipt):
        export_viewer(receipt, Path(path).parent)
        return
    if receipt['style_plan']['backend'] == 'mflux':
        return queue_native(receipt, path, wait=wait)
    run = run or subprocess.run
    job = receipt.get('job')
    if job and not job.get('id'):
        log_path = Path(path).parent / 'submission.log'
        matches = re.findall(r'\[imagine\] queued ([A-Za-z0-9_-]+)', log_path.read_text() if log_path.exists() else '')
        if len(set(matches)) == 1:
            job.update(id=matches[0], status='queued')
            receipt['state'] = 'queued'
            write(path, receipt)
    if job and not job.get('id'):
        raise ValueError('Submission was interrupted or ambiguous. Inspect submission.log and recover the existing job ID in receipt.json; automatic resubmission is disabled.')
    plan = receipt['style_plan']
    if not job:
        if not plan['ready']:
            raise ValueError(plan['bootstrap_reason'])
        receipt['job'] = {'id': None, 'status': 'submitting', 'argv': plan['argv']}
        write(path, receipt)
        # Direct stdout to disk so the queue ID survives interruption of this process.
        log_path = Path(path).parent / 'submission.log'
        with log_path.open('w') as log:
            result = run(plan['argv'], stdout=log, stderr=subprocess.STDOUT, text=True)
        output = log_path.read_text()
        matches = re.findall(r'\[imagine\] queued ([A-Za-z0-9_-]+)', output)
        receipt['job'].update(returncode=result.returncode, submission_log=str(log_path))
        if len(set(matches)) == 1:
            receipt['job'].update(id=matches[0], status='queued')
            receipt['state'] = 'queued'
        write(path, receipt)
        if not receipt['job'].get('id'):
            raise ValueError(f'No unique queue ID captured. Inspect {log_path}; no job will be resubmitted automatically.')
    if not wait:
        return
    argv = [receipt['job']['argv'][0], 'wait', receipt['job']['id'], '45']
    result = run(argv, capture_output=True, text=True)
    receipt['job']['wait'] = {'argv': argv, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    if result.returncode == 0:
        paths = re.findall(r'!\[[^\]]*\]\(([^\n]+)\)', result.stdout)
        images = []
        for value in paths:
            image_path = Path(value)
            with Image.open(image_path) as image:
                if list(image.size) != plan['conditioning']['dimensions']:
                    raise ValueError('Model result dimensions differ from the recorded native source')
            images.append({'path': str(image_path), 'sha256': digest(image_path)})
        if not images:
            raise ValueError('Queue completed without image paths; inspect wait output before continuing')
        receipt.update(state='art-ready', art=images, visual_acceptance='unreviewed')
        receipt['job']['status'] = 'completed'
    elif result.returncode != 3:
        receipt['job']['status'] = 'failed-or-interrupted'
    write(path, receipt)
    if receipt['state'] == 'art-ready':
        export_viewer(receipt, Path(path).parent)
    if result.returncode not in (0, 3):
        raise ValueError('Existing local job failed or is interrupted; inspect recorded wait output. It was not resubmitted.')


def native_argv(source, out, caption, preset, base, adapter, scale):
    """Arguments to the pinned native CLI, executed only inside the isolated worker."""
    width, height = source['conditioning']['dimensions']
    settings = preset['inference']
    model_type = preset['base'].get('model_type', 'flux2-klein-4b')
    if settings.get('queue_route', '').startswith('qwen'):
        return None
    if model_type not in ('flux2-klein-4b', 'flux2-klein-base-4b', 'flux2-klein-9b', 'flux2-klein-base-9b'):
        raise ValueError('Unsupported FLUX.2 Klein model type in preset')
    return ['mflux-generate-flux2-edit', '--model', base, '--base-model', model_type,
            '--prompt', caption, '--image-paths', source['image'], '--output', str(Path(out) / 'art/raw.png'),
            '--width', str(width), '--height', str(height), '--steps', str(settings['steps']),
            '--guidance', str(settings['guidance']), '--seed', str(settings['seed']),
            '--lora-paths', adapter, '--lora-scales', str(scale)]


def worker_gone(pid):
    """True once the launched worker has exited, whether or not this process is its parent."""
    try:
        if os.waitpid(pid, os.WNOHANG)[0]:
            return True
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        pass
    return False


def queue_native(receipt, path, *, wait=False, launch=None):
    if completed_art(receipt):
        export_viewer(receipt, Path(path).parent)
        return
    import time
    import uuid
    launch = launch or subprocess.Popen
    plan = receipt['style_plan']
    result_path = Path(path).parent / 'native-result.json'
    if receipt.get('job') and not receipt['job'].get('id'):
        raise ValueError('Native job identity is ambiguous. Inspect its receipt and worker log; no replacement was launched.')
    if receipt.get('job') and not receipt['job'].get('pid') and not result_path.exists():
        raise ValueError('Native launch has no recorded worker PID. Inspect its receipt and native-worker.log; automatic relaunch is disabled.')
    if not receipt.get('job'):
        if not plan['ready']:
            raise ValueError(plan['bootstrap_reason'])
        # A durable ID precedes the launch. An interrupted launch needs inspection, never resubmission.
        receipt['job'] = {'id': 'native-' + uuid.uuid4().hex[:12], 'status': 'launching', 'argv': plan['argv']}
        receipt['state'] = 'queued'
        write(path, receipt)
        with (Path(path).parent / 'native-worker.log').open('a') as log:
            process = launch(plan['argv'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt['job'].update(pid=process.pid, status='queued')
        write(path, receipt)
    deadline, pid, gone = time.monotonic() + (45 if wait else 0), receipt['job'].get('pid'), False
    while not result_path.exists():
        gone = bool(pid) and worker_gone(pid)
        if gone or time.monotonic() >= deadline:
            break
        time.sleep(.25)
    if gone and not result_path.exists():
        # A worker killed before its result (restart, kill, memory pressure) would otherwise wait forever.
        receipt['job']['status'] = 'failed'
        write(path, receipt)
        raise ValueError('Native worker exited without a result; inspect native-worker.log. It was not relaunched.')
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get('job_id') != receipt['job']['id']:
            raise ValueError('Native completion belongs to a different job')
        receipt['job']['result'] = result
        if not result.get('ok'):
            receipt['job']['status'] = 'failed'
            write(path, receipt)
            raise ValueError('Native worker failed; inspect native-worker.log. It will not be resubmitted.')
        image = Path(result['image'])
        if digest(image) != result['sha256']:
            raise ValueError('Native result image changed after completion')
        receipt.update(state='art-ready', art=[{'path': str(image), 'sha256': result['sha256']}], visual_acceptance='unreviewed')
        receipt['job']['status'] = 'completed'
        write(path, receipt)
        export_viewer(receipt, Path(path).parent)


def run_native_cli(plan, *, helper=None, main=None):
    """Set the cap only in this process and prove the conditioning dimensions before inference."""
    import sys
    if helper is None:
        from mflux.models.flux2.variants.edit import flux2_klein_edit_helpers as helper
    if main is None:
        from mflux.models.flux2.cli.flux2_edit_generate import main
    width, height = plan['conditioning']['dimensions']
    native_dimensions(width, height)
    helper.MAX_REFERENCE_AREA = plan['conditioning']['required_backend_reference_area']
    actual = list(helper._Flux2KleinEditHelpers.reference_dims(width, height))
    if actual != [width, height]:
        raise ValueError(f'Backend would change conditioning dimensions: {actual}')
    print(f'Native conditioning verified: {width}x{height}, area cap={helper.MAX_REFERENCE_AREA}', flush=True)
    previous = sys.argv
    try:
        sys.argv = plan['native_argv']
        main()
    finally:
        sys.argv = previous
    return actual


def native_worker(path):
    """Portable single-worker process. No model import occurs in normal atlas commands."""
    import fcntl
    import importlib.metadata
    import platform
    path = Path(path).resolve()
    receipt = json.loads(path.read_text())
    result = {'job_id': receipt['job']['id'], 'ok': False, 'validation': 'Local fallback; inspect resulting images before review'}
    try:
        if imagine_path():
            raise ValueError('A compatible local queue is installed. The native worker is disabled while it owns the GPU.')
        if platform.system() != 'Darwin' or platform.machine() != 'arm64':
            raise ValueError('Native local inference requires Apple Silicon macOS')
        expected = dict(line.split('==') for line in (ROOT/'source/image-requirements.txt').read_text().splitlines() if line.strip())
        versions = {name: importlib.metadata.version(name) for name in expected}
        if versions != expected:
            raise ValueError(f'Image runtime differs from pins: {versions}')
        plan = receipt['style_plan']
        if digest(receipt['source']['image']) != plan['source_sha256']:
            raise ValueError('Native source changed before inference')
        base, adapter = model_paths(plan['preset'])
        expected_argv = native_argv(receipt['source'], path.parent, plan['caption'], plan['preset'], base, adapter, plan['scale'])
        if plan['native_argv'] != expected_argv:
            raise ValueError('Native command differs from pinned source/model/settings')
        (path.parent / 'art').mkdir(exist_ok=True)
        target = path.parent / 'art/raw.png'
        if target.exists():
            raise ValueError('Existing native output will not be overwritten')
        os.environ.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1', TRANSFORMERS_OFFLINE='1')
        lock_path = ROOT / 'outputs/native-style-worker.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            actual = run_native_cli(plan)
        with Image.open(target) as image:
            if list(image.size) != plan['conditioning']['dimensions']:
                raise ValueError('Native output dimensions differ from the source')
        result.update(ok=True, image=str(target), sha256=digest(target), versions=versions, actual_conditioning_dimensions=actual)
    except Exception as error:
        result['error'] = str(error)
        raise
    finally:
        write(path.parent / 'native-result.json', result)


def completed_art(receipt):
    if receipt['state'] not in ('art-ready', 'reviewed'):
        return False
    if not receipt.get('art'):
        raise ValueError('Completed receipt has no art')
    for item in receipt['art']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Completed art changed; its review no longer applies')
    if digest(receipt['source']['image']) != receipt['source']['sha256']:
        raise ValueError('Prepared source changed; its review no longer applies')
    return True


def export_viewer(receipt, out):
    """Pilots and final maps use the same offline explorer."""
    from showcase import page
    source = receipt['source']
    metadata = source.get('source_metadata') or {}
    recipe = receipt.get('location', {}).get('recipe') or {}
    credits = [metadata.get('credit_line'), metadata.get('attribution'), metadata.get('credits'),
               recipe.get('photographic', {}).get('attribution'), metadata.get('terrain', {}).get('attribution')]
    credits = ' '.join(dict.fromkeys(str(value) for value in credits if value))
    record = {'title': receipt.get('location', {}).get('query', 'Local city study'),
              'status': 'Visual review: ' + receipt.get('visual_acceptance', 'unreviewed'),
              'credits': credits or 'Source credit not supplied. Resolve before redistribution.',
              'camera': source.get('camera'), 'method': {'source': source, 'style': receipt.get('style_plan'), 'review': receipt.get('review')}}
    (Path(out)/'viewer.html').write_text(page(receipt['art'][0]['path'], source['image'], record))


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3 or sys.argv[1] != '--worker':
        raise SystemExit('Local worker entry only: --worker RECEIPT')
    native_worker(sys.argv[2])
