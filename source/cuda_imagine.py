"""Local CUDA job queue speaking the ATLAS_IMAGINE protocol, for Linux with an NVIDIA GPU.

Submit:  cuda-imagine CAPTION IMAGE -m ROUTE --steps N --seed S --width W --height H
                      --reference-max-dimension R --out DIR --no-wait --extra --model BASE --guidance G --lora PATH SCALE
         prints "[imagine] queued ID" and starts one detached worker.
Wait:    cuda-imagine wait ID SECONDS
         exit 0 and "![art](PATH)" when done, 3 while queued or running, 1 on failure.

The worker holds an exclusive lock, so one model runs at a time. Inference uses diffusers Flux2KleinPipeline in
bfloat16 with the LoRA fused at the requested scale; weights stream from system memory by group offload.
--reference-max-dimension is recorded only: the worker refuses any reference that is not the native size.
A failed job is never relaunched. Torch and diffusers are imported only inside infer().
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
POLL_SECONDS = 2.0
ROUTES = ('klein9b', 'klein4b')
OFFLINE = {'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'TRANSFORMERS_OFFLINE': '1'}


def queue_dir() -> Path:
    return Path(os.environ.get('CUDA_IMAGINE_QUEUE') or ROOT / 'outputs/cuda-imagine/queue')


def job_path(job_id: str) -> Path:
    if not job_id or not job_id.replace('-', '').replace('_', '').isalnum():
        raise SystemExit('Invalid job ID')
    return queue_dir() / 'jobs' / f'{job_id}.json'


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=1) + '\n')
    tmp.replace(path)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def parse(argv: list[str]) -> tuple[argparse.Namespace, argparse.Namespace]:
    parser = argparse.ArgumentParser(prog='cuda-imagine')
    parser.add_argument('caption')
    parser.add_argument('image')
    parser.add_argument('-m', dest='route', required=True)
    for name in ('--steps', '--seed', '--width', '--height', '--reference-max-dimension'):
        parser.add_argument(name, type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--no-wait', action='store_true')
    parser.add_argument('--output-resolution')
    split = argv.index('--extra') if '--extra' in argv else len(argv)
    args = parser.parse_args(argv[:split])
    ext = argparse.ArgumentParser(prog='cuda-imagine --extra')
    ext.add_argument('--model', required=True)
    ext.add_argument('--guidance', type=float, required=True)
    ext.add_argument('--lora', nargs=2, required=True)
    return args, ext.parse_args(argv[split + 1:])


def submit(argv: list[str]) -> int:
    args, extra = parse(argv)
    if args.route not in ROUTES:
        raise SystemExit(f'Unsupported route {args.route}')
    if args.output_resolution:
        raise SystemExit('--output-resolution is not supported; this queue returns native-size images only')
    if not Path(args.image).is_file() or not Path(extra.model).is_dir() or not Path(extra.lora[0]).is_file():
        raise SystemExit('Image, base model directory and LoRA file must exist')
    jobs = queue_dir() / 'jobs'
    jobs.mkdir(parents=True, exist_ok=True)
    job_id = time.strftime('%Y%m%d%H%M%S') + '-' + uuid.uuid4().hex[:8]
    # absolute() keeps the Hugging Face cache filenames readable; resolve() would follow them to blob hashes.
    job = {'id': job_id, 'status': 'queued', 'submitted': time.time(), 'caption': args.caption,
           'image': str(Path(args.image).absolute()), 'image_sha256': sha256(args.image), 'route': args.route,
           'steps': args.steps, 'seed': args.seed, 'width': args.width, 'height': args.height,
           'reference_max_dimension': args.reference_max_dimension, 'out': str(args.out.absolute()),
           'model': str(Path(extra.model).absolute()), 'guidance': extra.guidance,
           'lora': str(Path(extra.lora[0]).absolute()), 'lora_scale': float(extra.lora[1])}
    path = job_path(job_id)
    write_json(path, job)
    env = {**os.environ, 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
    with (jobs / f'{job_id}.log').open('w') as log:
        proc = subprocess.Popen([sys.executable, '-I', str(Path(__file__).resolve()), '--run', job_id],
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    job['pid'] = proc.pid
    write_json(path, job)
    print(f'[imagine] queued {job_id}', flush=True)
    return 0


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # a zombie counts as gone
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except OSError:
        return False


def wait(job_id: str, seconds: float) -> int:
    path = job_path(job_id)
    deadline = time.time() + seconds
    while True:
        job = json.loads(path.read_text())
        if job['status'] == 'completed':
            print(f"[imagine] {job_id} completed in {job.get('seconds', 0):.0f}s")
            print(f"![art]({job['result']})")
            return 0
        if job['status'] == 'failed':
            print(f"[imagine] {job_id} failed: {job.get('error')}")
            return 1
        if not alive(job.get('pid')):
            job.update(status='failed', error='worker exited without a result; see log')
            write_json(path, job)
            print(f'[imagine] {job_id} failed: worker exited without a result')
            return 1
        if time.time() >= deadline:
            print(f"[imagine] {job_id} {job['status']}")
            return 3
        time.sleep(POLL_SECONDS)


def infer(job: dict) -> dict:
    import diffusers
    from diffusers import Flux2KleinPipeline
    from diffusers.hooks import apply_group_offloading
    from PIL import Image
    import torch

    start = time.time()
    pipe = Flux2KleinPipeline.from_pretrained(job['model'], torch_dtype=torch.bfloat16)
    lora = Path(job['lora'])  # offline mode needs the folder and file name passed separately
    pipe.load_lora_weights(str(lora.parent), weight_name=lora.name, adapter_name='style')
    pipe.fuse_lora(lora_scale=job['lora_scale'], adapter_names=['style'])
    pipe.unload_lora_weights()
    cuda, cpu = torch.device('cuda'), torch.device('cpu')
    # Transformer: one block group at a time. Text encoder: one module at a time, because its
    # embedding table and output head (about 1.2 GB each) would otherwise move to the GPU together.
    apply_group_offloading(pipe.transformer, onload_device=cuda, offload_device=cpu,
                           offload_type='block_level', num_blocks_per_group=1)
    apply_group_offloading(pipe.text_encoder, onload_device=cuda, offload_device=cpu, offload_type='leaf_level')
    pipe.vae.to(cuda)
    print(f'loaded in {time.time() - start:.0f}s', flush=True)
    reference = Image.open(job['image']).convert('RGB')
    if list(reference.size) != [job['width'], job['height']]:
        raise ValueError('Reference dimensions differ from the requested native size')
    image = pipe(prompt=job['caption'], image=[reference], width=job['width'], height=job['height'],
                 num_inference_steps=job['steps'], guidance_scale=job['guidance'],
                 generator=torch.Generator('cpu').manual_seed(job['seed'])).images[0]
    if list(image.size) != [job['width'], job['height']]:
        raise ValueError(f'Model returned {image.size}, not the native size')
    out = Path(job['out'])
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'raw.png'
    image.save(target)
    return {'result': str(target), 'runtime': {
        'torch': torch.__version__, 'diffusers': diffusers.__version__, 'device': torch.cuda.get_device_name(0),
        'dtype': 'bfloat16', 'offload': 'group block_level', 'lora': 'fused',
        'peak_vram_gib': round(torch.cuda.max_memory_allocated() / 2**30, 2)}}


def run(job_id: str) -> int:
    path = job_path(job_id)
    os.environ.update(OFFLINE)  # inference never touches the network or a stored token
    with (queue_dir() / 'gpu.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        job = json.loads(path.read_text())
        job.update(status='running', started=time.time())
        write_json(path, job)
        try:
            if sha256(job['image']) != job['image_sha256']:
                raise ValueError('Input image changed after submission')
            job.update(infer(job), status='completed', seconds=time.time() - job['started'])
        except Exception as error:  # recorded, never retried
            traceback.print_exc()
            job.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(path, job)
    return 0 if job['status'] == 'completed' else 1


def main(argv: list[str]) -> int:
    if argv[:1] == ['--run'] and len(argv) == 2:
        return run(argv[1])
    if argv[:1] == ['wait'] and len(argv) in (2, 3):
        return wait(argv[1], float(argv[2]) if len(argv) == 3 else 45)
    return submit(argv)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
