"""The Linux CUDA queue client: protocol, job records and failure handling. No GPU, torch or diffusers needed."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
import cuda_imagine as cuda
import local_style as style

CUDA_PRESET = ROOT / 'model-presets/pixelart-redmond-klein9b-cuda.json'
QUEUED = r'\[imagine\] queued ([A-Za-z0-9_-]+)'  # the pattern queue_style() reads
IMAGE = r'!\[[^\]]*\]\(([^\n]+)\)'


class CudaQueue(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        for env in (patch.dict(os.environ, {'CUDA_IMAGINE_QUEUE': str(self.folder / 'queue')}),
                    patch.object(cuda, 'subprocess', SimpleNamespace(
                        Popen=lambda *a, **k: SimpleNamespace(pid=os.getpid()), STDOUT=subprocess.STDOUT))):
            env.start()
            self.addCleanup(env.stop)
        self.cache = self.folder / 'hub'
        preset = json.loads(CUDA_PRESET.read_text())
        base, adapter = preset['base'], preset['adapter']
        snapshot = self.cache / ('models--' + base['repo'].replace('/', '--')) / 'snapshots' / base['revision']
        for part in ('transformer', 'text_encoder', 'vae'):
            (snapshot / part).mkdir(parents=True)
            (snapshot / part / '0.safetensors').write_bytes(b'fixture')
        lora = self.cache / ('models--' + adapter['repo'].replace('/', '--')) / 'snapshots' / adapter['revision'] / adapter['filename']
        lora.parent.mkdir(parents=True)
        lora.write_bytes(b'adapter fixture')
        adapter['sha256'] = style.digest(lora)
        self.preset = self.folder / 'cuda-preset.json'
        self.preset.write_text(json.dumps(preset))
        Image.new('RGB', (1024, 1024), (90, 120, 80)).save(self.folder / 'source.png')
        self.source = style.prepare_source(self.folder / 'source.png', self.folder / 'out')
        with patch.object(style, 'imagine_path', return_value=str(ROOT / 'source/cuda-imagine')):
            self.plan = style.style_plan(self.source, self.folder / 'out', 'the visible scene', cache=self.cache,
                                         backend='imagine', preset_path=self.preset)

    def call(self, argv):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cuda.main(argv)
        return code, out.getvalue()

    def submit(self, argv=None):
        code, out = self.call(argv or self.plan['argv'][1:])
        self.assertEqual(code, 0)
        ids = re.findall(QUEUED, out)
        self.assertEqual(len(ids), 1)
        return ids[0], json.loads(cuda.job_path(ids[0]).read_text())

    def fake_infer(self, job):
        target = Path(job['out']) / 'raw.png'
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (job['width'], job['height']), (200, 60, 40)).save(target)
        return {'result': str(target), 'runtime': {'device': 'fixture'}}

    def test_atlas_plan_round_trips_into_a_job_record(self):
        self.assertTrue(self.plan['ready'], self.plan['bootstrap_reason'])
        _, job = self.submit()
        base, lora = style.model_paths(json.loads(self.preset.read_text()), self.cache)
        self.assertEqual((job['status'], job['route'], job['steps'], job['seed']), ('queued', 'klein9b', 4, 4601))
        self.assertEqual((job['width'], job['height'], job['guidance'], job['lora_scale']), (1024, 1024, 1.0, .7))
        self.assertTrue(job['caption'].startswith('Pixel Art, PixArFK. '))
        self.assertEqual((job['model'], job['lora']), (base, lora))
        self.assertTrue(job['lora'].endswith('PixelArt_Redmond.safetensors'))
        self.assertEqual(job['image_sha256'], self.source['sha256'])
        self.assertEqual(job['pid'], os.getpid())

    def test_protocol_code_never_imports_torch_or_diffusers(self):
        job_id, _ = self.submit()
        self.assertEqual(self.call(['wait', job_id, '0'])[0], 3)
        script = (f'import sys; sys.path.insert(0, {str(ROOT / "source")!r}); import cuda_imagine; '
                  'print(sorted({"torch", "diffusers"} & set(sys.modules)))')
        result = subprocess.run([sys.executable, '-I', '-c', script], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), '[]')
        self.assertFalse({'torch', 'diffusers'} & set(sys.modules))

    def test_job_ids_cannot_escape_the_queue(self):
        for value in ('../x', 'a/b', '', '.'):
            with self.assertRaises(SystemExit):
                cuda.job_path(value)

    def test_wait_reports_done_pending_failed_and_vanished_workers(self):
        job_id, job = self.submit()
        self.assertEqual(self.call(['wait', job_id, '0'])[0], 3)
        cuda.write_json(cuda.job_path(job_id), {**job, 'status': 'completed', 'result': '/art/raw.png', 'seconds': 5})
        code, out = self.call(['wait', job_id, '0'])
        self.assertEqual((code, re.findall(IMAGE, out)), (0, ['/art/raw.png']))
        cuda.write_json(cuda.job_path(job_id), {**job, 'status': 'failed', 'error': 'boom'})
        self.assertEqual(self.call(['wait', job_id, '0'])[0], 1)
        cuda.write_json(cuda.job_path(job_id), {**job, 'pid': None})
        self.assertEqual(self.call(['wait', job_id, '0'])[0], 1)
        self.assertEqual(json.loads(cuda.job_path(job_id).read_text())['status'], 'failed')

    def test_worker_records_failure_once_and_success_with_runtime(self):
        job_id, _ = self.submit()
        with patch.dict(os.environ), patch.object(cuda, 'infer', side_effect=RuntimeError('no CUDA device')) as infer, \
                redirect_stdout(io.StringIO()), patch('sys.stderr', io.StringIO()):
            self.assertEqual(cuda.run(job_id), 1)
            self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')
        job = json.loads(cuda.job_path(job_id).read_text())
        self.assertEqual((job['status'], job['error'], infer.call_count), ('failed', 'RuntimeError: no CUDA device', 1))
        job_id, _ = self.submit()
        with patch.dict(os.environ), patch.object(cuda, 'infer', side_effect=self.fake_infer):
            self.assertEqual(cuda.run(job_id), 0)
        job = json.loads(cuda.job_path(job_id).read_text())
        self.assertEqual((job['status'], job['runtime']), ('completed', {'device': 'fixture'}))
        self.assertGreaterEqual(job['seconds'], 0)

    def test_changed_input_fails_before_inference(self):
        job_id, _ = self.submit()
        Image.new('RGB', (1024, 1024), (0, 0, 0)).save(self.source['image'])
        with patch.dict(os.environ), patch.object(cuda, 'infer') as infer, patch('sys.stderr', io.StringIO()):
            self.assertEqual(cuda.run(job_id), 1)
        infer.assert_not_called()
        self.assertIn('changed after submission', json.loads(cuda.job_path(job_id).read_text())['error'])

    def test_submit_rejects_bad_requests(self):
        argv = self.plan['argv'][1:]
        route, model, lora = argv.index('-m') + 1, argv.index('--model') + 1, argv.index('--lora') + 1
        cases = {route: 'qwen', 1: str(self.folder / 'missing.png'), model: str(self.folder / 'none'), lora: str(self.folder / 'none.safetensors')}
        for index, value in cases.items():
            with self.subTest(index=index), self.assertRaises(SystemExit):
                self.call([*argv[:index], value, *argv[index + 1:]])
        split = argv.index('--extra')
        with self.assertRaisesRegex(SystemExit, 'output-resolution'):
            self.call([*argv[:split], '--output-resolution', '2048', *argv[split:]])
        self.assertFalse(list((self.folder / 'queue').glob('jobs/*.json')))

    def test_queue_style_reaches_art_ready_through_the_real_client(self):
        receipt = {'state': 'source-ready', 'source': self.source, 'style_plan': self.plan}
        path = self.folder / 'out/receipt.json'

        def run(argv, **kwargs):
            if argv[1] == 'wait':
                code, out = self.call(argv[1:])
                return SimpleNamespace(returncode=code, stdout=out, stderr='')
            with redirect_stdout(kwargs['stdout']):
                code = cuda.main(argv[1:])
            kwargs['stdout'].flush()
            with patch.dict(os.environ), patch.object(cuda, 'infer', side_effect=self.fake_infer):
                cuda.run(re.findall(QUEUED, Path(path.parent / 'submission.log').read_text())[0])
            return SimpleNamespace(returncode=code)

        style.queue_style(receipt, path, wait=True, run=run)
        self.assertEqual((receipt['state'], receipt['visual_acceptance']), ('art-ready', 'unreviewed'))
        art = receipt['art'][0]
        self.assertEqual(Path(art['path']), self.folder / 'out/art/raw.png')
        self.assertEqual(art['sha256'], style.digest(art['path']))

    def test_cuda_preset_is_valid_and_distinct_from_the_mac_preset(self):
        with patch.object(style, 'imagine_path', return_value='imagine'):
            mac = style.style_plan(self.source, self.folder / 'out', 'the visible scene', cache=self.cache, backend='imagine')
        self.assertFalse(style.same_style_job(mac, self.plan))
        self.assertEqual(json.loads(CUDA_PRESET.read_text())['adapter'], json.loads(style.DEFAULT_PRESET.read_text())['adapter'])
        self.assertIn(b'[imagine] queued', (ROOT / 'source/cuda-imagine').read_bytes())
        self.assertTrue(os.access(ROOT / 'source/cuda-imagine', os.X_OK))


if __name__ == '__main__':
    unittest.main()
