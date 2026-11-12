"""Composition, adoption and durable job identity, without loading an image model."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
import context_infill as infill
import local_style as style


class ContextInfill(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.preset = json.loads(style.DEFAULT_PRESET.read_text())
        self.caption = 'A registered city with varied real roof colours.'
        self.source = self.folder / 'source.png'
        y, x = np.indices((1024, 1536))
        self.source_pixels = np.stack([x % 256, y % 256, (x + y) % 256], axis=2).astype(np.uint8)
        Image.fromarray(self.source_pixels).save(self.source)

    def test_grid_covers_exact_unaligned_native_source(self):
        self.assertEqual(infill.starts(1024), [0])
        self.assertEqual(infill.starts(1803), [0, 779])
        self.assertEqual(infill.starts(3456), [0, 608, 1216, 1824, 2432])
        with self.assertRaises(ValueError):
            infill.starts(1023)

    def test_overlap_keeps_old_interior_and_transitions_to_new_prediction(self):
        canvas = np.zeros((1024, 1536, 3), np.uint8)
        done = np.zeros((1024, 1536), bool)
        infill.composite(canvas, done, np.full((1024, 1024, 3), 40, np.uint8), 0, 0)
        infill.composite(canvas, done, np.full((1024, 1024, 3), 200, np.uint8), 512, 0)
        np.testing.assert_array_equal(canvas[:, :896], 40)
        np.testing.assert_array_equal(canvas[:, 1024:], 200)
        self.assertEqual(canvas[500, 1023, 0], 199)
        self.assertTrue(done.all())

    def test_plan_only_never_resolves_or_submits_models_and_binds_identity(self):
        out = self.folder / 'planned'
        with patch.object(style, 'queue_style', side_effect=AssertionError('inference')), patch.object(style, 'model_paths', side_effect=AssertionError('model lookup')):
            result = infill.paint(self.source, out, self.caption, plan_only=True)
            self.assertFalse(result['queued'])
            self.assertEqual(result['windows'], 2)
            self.assertFalse((out / 'windows').exists())
            with self.assertRaisesRegex(ValueError, 'changed'):
                infill.paint(self.source, out, self.caption + ' different', plan_only=True)
        changed = deepcopy(self.preset)
        changed['inference']['seed'] += 1
        path = self.folder / 'changed.json'
        style.write(path, changed)
        with self.assertRaisesRegex(ValueError, 'changed'):
            infill.paint(self.source, out, self.caption, preset_path=path, plan_only=True)

    def experiment(self):
        folder = self.folder / 'old'
        grid = [[0, 0], [512, 0]]
        settings, base, adapter = self.preset['inference'], self.preset['base'], self.preset['adapter']
        prompt = infill.prompt_for(self.caption, self.preset)
        plan = {'source_sha256': style.digest(self.source), 'size': [1536, 1024],
                'box': [0, 0, 1536, 1024], 'window': 1024, 'feather': 128,
                'windows': grid, 'prompt': prompt, 'recipe': {
                    'model': settings['queue_route'], 'base_revision': base['revision'],
                    'adapter_sha256': adapter['sha256'], 'steps': settings['steps'],
                    'guidance': settings['guidance'], 'seed_base': settings['seed'], 'scale': str(settings['adapter_scale'])}}
        canvas, done = self.source_pixels.copy(), np.zeros((1024, 1536), bool)
        for index, (x, y) in enumerate(grid):
            wd = folder / 'windows' / f'w{index:02d}-x{x}-y{y}'
            wd.mkdir(parents=True)
            Image.fromarray(canvas[y:y + 1024, x:x + 1024]).save(wd / 'input.png')
            raw = np.full((1024, 1024, 3), 40 + index * 100, np.uint8)
            Image.fromarray(raw).save(wd / 'raw-klein9b.png')
            argv = ['imagine', prompt, str(wd / 'input.png'), '-m', settings['queue_route'],
                    '--steps', str(settings['steps']), '--seed', str(settings['seed'] + index),
                    '--width', '1024', '--height', '1024', '--reference-max-dimension', '1024',
                    '--model', f'/cached/{base["revision"]}', '--guidance', str(settings['guidance']),
                    '--lora', f'/cached/{adapter["revision"]}/adapter.safetensors', str(settings['adapter_scale'])]
            style.write(wd / 'run.json', {'returncode': 0, 'argv': argv, 'job': f'job-{index}',
                'input_sha256': style.digest(wd / 'input.png'),
                'outputs': {str(wd / 'raw-klein9b.png'): style.digest(wd / 'raw-klein9b.png')}})
            infill.composite(canvas, done, raw, x, y)
        Image.fromarray(canvas).save(folder / 'map.png')
        plan['map_sha256'] = style.digest(folder / 'map.png')
        style.write(folder / 'plan.json', plan)
        return folder

    def test_adoption_replays_exact_pixels_without_inference_and_resumes(self):
        experiment = self.experiment()
        out = self.folder / 'replay'
        with patch.object(style, 'queue_style', side_effect=AssertionError('inference')):
            result = infill.paint(self.source, out, self.caption, adopt=experiment)
            self.assertEqual(result['sha256'], style.digest(experiment / 'map.png'))
            self.assertEqual(infill.paint(self.source, out, self.caption)['sha256'], result['sha256'])
            raw = out / 'windows/w01-x512-y0/raw.png'
            Image.new('RGB', (1024, 1024), 'red').save(raw)
            with self.assertRaisesRegex(ValueError, 'prediction changed'):
                infill.paint(self.source, out, self.caption)

    def test_adoption_rejects_wrong_context_prediction_and_schedule(self):
        experiment = self.experiment()
        old = experiment / 'windows/w01-x512-y0'
        Image.new('RGB', (1024, 1024), 'red').save(old / 'input.png')
        record = json.loads((old / 'run.json').read_text())
        record['input_sha256'] = style.digest(old / 'input.png')
        style.write(old / 'run.json', record)
        with self.assertRaisesRegex(ValueError, 'expected source'):
            infill.paint(self.source, self.folder / 'bad-context', self.caption, adopt=experiment)
        plan = json.loads((experiment / 'plan.json').read_text())
        plan['recipe']['steps'] = 8
        style.write(experiment / 'plan.json', plan)
        with self.assertRaisesRegex(ValueError, 'schedule differs'):
            infill.paint(self.source, self.folder / 'bad-schedule', self.caption, adopt=experiment)

    def test_interrupted_ambiguous_submission_is_never_requeued(self):
        out = self.folder / 'resume'
        # Simulate interruption after durable intent but before a queue ID was captured.
        original_queue = style.queue_style
        def interrupted(receipt, path, **kwargs):
            receipt['job'] = {'id': None, 'status': 'submitting'}
            style.write(path, receipt)
            raise RuntimeError('interrupted')
        with patch.object(style, 'queue_style', side_effect=interrupted):
            with self.assertRaises(RuntimeError):
                infill.paint(self.source, out, self.caption, backend='imagine')
        runner = Mock(side_effect=AssertionError('must not submit'))
        def resumed(receipt, path, **kwargs):
            return original_queue(receipt, path, run=runner, **kwargs)
        with patch.object(style, 'queue_style', side_effect=resumed):
            with self.assertRaisesRegex(ValueError, 'ambiguous'):
                infill.paint(self.source, out, self.caption, backend='imagine')
        runner.assert_not_called()


if __name__ == '__main__':
    unittest.main()
