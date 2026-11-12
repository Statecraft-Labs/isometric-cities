"""Offline fixtures for location resolution, exact pixels and durable local job receipts."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
sys.path.insert(0, str(ROOT))
import atlas
import local_style as style
import make_location as make

DETECT_QUEUE_CLIENT = style.imagine_path


class LocationWorkflow(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        fixture = json.loads(style.DEFAULT_PRESET.read_text())
        fixture['base'].update(model_type='flux2-klein-4b', revision='fixture-base-revision')
        fixture['inference'].update(queue_route='klein4b-wide', seed=3701, adapter_scale=.65, scales_tested=[.55,.65], max_native_reference_area=4194304)
        self.compat_preset = self.folder/'fixture-preset.json'
        self.compat_preset.write_text(json.dumps(fixture))
        # Queue-plan fixtures must not depend on imagine being installed on the host.
        queue = patch.object(style, 'imagine_path', return_value='imagine')
        queue.start()
        self.addCleanup(queue.stop)

    def image(self, width=1536, height=1152):
        # A spatially varying fixture catches off-by-one crop and interpolation errors.
        y, x = np.indices((height, width))
        pixels = np.stack((x % 256, y % 256, (x + y) % 256), axis=2).astype('uint8')
        path = self.folder / 'original.png'
        Image.fromarray(pixels).save(path)
        return path, pixels

    def test_known_places_and_dc_ambiguity_need_no_network(self):
        for query, expected in [('Capitol Hill', 'capitol-hill-dc'), ('Capitol Hill, Washington, DC', 'capitol-hill-dc'),
                                ('capitol-hill', 'capitol-hill-dc'), ('Washington DC', 'capitol-hill-dc'),
                                ('Edinburgh', 'edinburgh-v3'), ('edi', 'edinburgh-v3')]:
            with self.subTest(query=query):
                result = make.resolve_location(query, self.folder, geocoder=Mock(side_effect=AssertionError('network')))
                self.assertEqual(result['recipe']['id'], expected)
                self.assertEqual(result['method'], 'recipe')

    def test_bundled_place_uses_its_material_caption(self):
        source,_=self.image(1024,1024)
        for city in ('dc','edinburgh'):
            location='Washington DC' if city=='dc' else city
            out=self.folder/city
            with patch('sys.stdout',new_callable=io.StringIO):
                self.assertEqual(atlas.main(['make',location,'--source',str(source),'--plan-only','--out',str(out)]),0)
            receipt=json.loads((out/'receipt.json').read_text())
            self.assertEqual(receipt['style_plan']['caption'],(ROOT/f'examples/{city}-caption.txt').read_text().strip())

    def test_coordinates_and_geocoded_starter_are_bounded(self):
        a = make.resolve_location('55.8609,-4.2514', self.folder, offline=True)
        b = make.resolve_location('55.8609,-4.2514', self.folder, offline=True)
        self.assertEqual(a, b)
        recipe = a['recipe']
        self.assertEqual(recipe['origin'], {'lat': 55.8609, 'lon': -4.2514})
        s, w, n, e = recipe['bbox']
        self.assertTrue(s < 55.8609 < n and w < -4.2514 < e)
        self.assertLess(n - s, .04)
        self.assertLess(e - w, .06)
        self.assertNotIn('photographic', recipe)
        self.assertIn('not established', recipe['source_quality']['photographic_coverage'])
        result = make.resolve_location('Glasgow city centre', self.folder, geocoder=lambda *a, **k: {'result': {'lat': '55.8609', 'lon': '-4.2514'}})
        self.assertEqual(result['method'], 'geocoder')
        with self.assertRaises(ValueError):
            make.resolve_location('91,0', self.folder, offline=True)

    def test_geocoder_caches_and_serialises_requests_and_errors(self):
        opener = Mock(return_value=io.StringIO('[{"lat":"55.86","lon":"-4.25","display_name":"Glasgow"}]'))
        sleeps = Mock()
        first = make.geocode('Glasgow city centre', self.folder, opener=opener, clock=lambda: 10, sleep=sleeps)
        self.assertEqual(first, make.geocode('Glasgow city centre', self.folder, offline=True, opener=opener))
        self.assertEqual(opener.call_count, 1)
        request = opener.call_args.args[0]
        self.assertIn('IsometricCities/', request.get_header('User-agent'))
        self.assertIn('limit=1', request.full_url)
        next_opener = Mock(return_value=io.StringIO('[{"lat":"56","lon":"-3"}]'))
        make.geocode('A different place', self.folder, opener=next_opener, clock=lambda: 10.1, sleep=sleeps)
        self.assertAlmostEqual(sleeps.call_args.args[0], .95)
        failed = Mock(side_effect=URLError('fixture offline'))
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, 'Geocoder unavailable'):
                make.geocode('Fail fixture', self.folder, opener=failed, clock=lambda: 15, sleep=sleeps)
        self.assertEqual(failed.call_count, 1)
        with self.assertRaisesRegex(ValueError, 'No cached geocode'):
            make.geocode('uncached', self.folder, offline=True, opener=Mock(side_effect=AssertionError('network')))
        empty = Mock(return_value=io.StringIO('[]'))
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, 'no usable result'):
                make.geocode('empty', self.folder, opener=empty, clock=lambda: 20, sleep=sleeps)
        self.assertEqual(empty.call_count, 1)

    def test_exact_dc_crop_camera_hash_and_native_wide_plan(self):
        path, pixels = self.image()
        manifest = self.folder / 'manifest.json'
        camera = {'width': 1536, 'height': 1152, 'pixel_matrix': [[1, 0, 0, 768], [0, 1, -1, 576]], 'origin_projected_metres': [1, 2]}
        manifest.write_text(json.dumps({'camera': camera, 'sha256': style.digest(path)}))
        source = style.prepare_source(path, self.folder / 'out', crop=[64, 112, 1472, 800], manifest=manifest)
        actual = np.asarray(Image.open(source['image']))
        self.assertTrue(np.array_equal(actual, pixels[112:912, 64:1536]))
        self.assertEqual(source['source_camera'], camera)
        self.assertEqual(source['camera']['pixel_matrix'], [[1, 0, 0, 704], [0, 1, -1, 464]])
        self.assertIsNone(source['resize'])
        plan = style.style_plan(source, self.folder / 'out', 'the visible neighbourhood', cache=self.folder / 'missing', preset_path=self.compat_preset)
        self.assertEqual(plan['conditioning']['route'], 'klein4b-wide')
        self.assertEqual(plan['conditioning']['dimensions'], [1472, 800])
        self.assertEqual(plan['conditioning']['required_backend_reference_area'], 1177600)
        self.assertEqual(plan['argv'][plan['argv'].index('--reference-max-dimension') + 1], '1472')
        self.assertIn('fixture-base-revision', str(plan))
        self.assertEqual(plan['preset']['inference']['seed'], 3701)
        self.assertEqual(plan['scale'], .65)
        manifest.write_text(json.dumps({'camera': camera, 'sha256': '0' * 64}))
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            style.prepare_source(path, self.folder / 'bad', crop=[64, 112, 1472, 800], manifest=manifest)
        for crop in ([64, 112, 1473, 800], [-1, 0, 512, 512], [0, 0, 513, 512]):
            with self.assertRaises(ValueError):
                style.prepare_source(path, self.folder / 'bad', crop=crop)
        self.assertEqual(style.native_dimensions(1024, 1024)['route'], 'klein4b')
        panorama, pixels = self.image(2048, 656)
        prepared = style.prepare_source(panorama, self.folder / 'panorama')
        self.assertTrue(np.array_equal(np.asarray(Image.open(prepared['image'])), pixels))
        self.assertEqual(prepared['conditioning']['dimensions'], [2048, 656])
        self.assertEqual(style.native_dimensions(528, 512)['reference_max_dimension'], 544)
        plan = style.style_plan(prepared, self.folder / 'panorama', 'the joined scene', scale=.55, cache=self.folder / 'missing', preset_path=self.compat_preset)
        self.assertEqual(plan['scale'], .55)
        manifest.write_text(json.dumps({'width':2048,'height':656,'source_sha256':style.digest(panorama),'image_sha256':'0'*64}))
        style.prepare_source(panorama, self.folder / 'comparison-source', manifest=manifest)
        manifest.write_text(json.dumps({'source_sha256':'0'*64,'image_sha256':style.digest(panorama)}))
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            style.prepare_source(panorama, self.folder / 'bad-comparison', manifest=manifest)
        with self.assertRaises(ValueError):
            style.native_dimensions(2080, 512)

    def test_large_source_is_preserved_without_silently_submitting_small_conditioning(self):
        path, pixels = self.image(4096, 3072)
        prepared = style.prepare_source(path, self.folder / 'large', allow_large=True)
        self.assertTrue(np.array_equal(np.asarray(Image.open(prepared['image'])), pixels))
        self.assertEqual(prepared['conditioning']['dimensions'], [4096,3072])
        plan = style.style_plan(prepared, self.folder / 'large', 'The full real neighbourhood.', cache=self.folder / 'none')
        self.assertFalse(plan['ready'])
        self.assertIn('Automatic shrinking is disabled', plan['bootstrap_reason'])
        receipt = {'state':'source-ready','source':prepared,'style_plan':plan}
        with self.assertRaisesRegex(ValueError, 'Full source preserved'):
            style.queue_style(receipt, self.folder/'large/receipt.json', run=Mock(side_effect=AssertionError('inference')))

    def test_explicit_pixel_preset_controls_model_and_resume_identity(self):
        path, _ = self.image(1024, 1024)
        prepared = style.prepare_source(path, self.folder / 'pixel')
        preset = ROOT / 'model-presets/pixelart-redmond-klein9b.json'
        plan = style.style_plan(prepared, self.folder / 'pixel', 'The real Capitol neighbourhood.',
                                preset_path=preset, cache=self.folder / 'missing')
        self.assertEqual(plan['scale'], .7)
        self.assertEqual(plan['conditioning']['route'], 'klein9b')
        default = style.style_plan(prepared, self.folder / 'pixel', 'The real Capitol neighbourhood.', cache=self.folder / 'missing')
        self.assertEqual(default['preset'], plan['preset'])
        self.assertTrue(plan['caption'].startswith('Pixel Art, PixArFK.'))
        self.assertIn('flux2-klein-9b', plan['native_argv'])
        self.assertNotIn('flux2-klein-4b', plan['native_argv'])
        self.assertTrue(style.same_style_job(plan, dict(plan)))
        changed = json.loads(json.dumps(plan))
        changed['preset']['adapter']['revision'] = 'changed'
        self.assertFalse(style.same_style_job(plan, changed))
        changed = json.loads(json.dumps(plan))
        changed['preset']['inference']['steps'] = 8
        self.assertFalse(style.same_style_job(plan, changed))
        with self.assertRaisesRegex(ValueError, 'tested adapter scales'):
            style.style_plan(prepared, self.folder, 'The city.', .55, preset_path=preset)
        wide, _ = self.image(2048, 1024)
        wide_source = style.prepare_source(wide, self.folder / 'wide')
        wide_plan = style.style_plan(wide_source, self.folder / 'wide', 'The city.', preset_path=preset, cache=self.folder / 'missing')
        self.assertFalse(wide_plan['ready'])
        self.assertIn('tested native conditioning area', wide_plan['bootstrap_reason'])

    def test_model_cache_resolves_literal_space_directory_and_checks_adapter_hash(self):
        preset = {'base': {'repo': 'org/base', 'revision': 'pin'},
                  'adapter': {'repo': 'org/adapter', 'revision': 'other', 'filename': 'clay/lora.safetensors', 'sha256': ''}}
        base = self.folder / 'models--org--base/snapshots/pin/ '
        for part in ('transformer', 'text_encoder', 'vae'):
            (base / part).mkdir(parents=True)
            (base / part / '0.safetensors').write_bytes(b'fixture')
        adapter = self.folder / 'models--org--adapter/snapshots/other/clay/lora.safetensors'
        adapter.parent.mkdir(parents=True)
        adapter.write_bytes(b'adapter')
        preset['adapter']['sha256'] = style.digest(adapter)
        self.assertEqual(style.model_paths(preset, self.folder), (str(base), str(adapter)))
        adapter.write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            style.model_paths(preset, self.folder)

    def test_one_submission_resumes_wait_and_never_auto_accepts(self):
        path, _ = self.image(512, 512)
        source = style.prepare_source(path, self.folder / 'out')
        plan = style.style_plan(source, self.folder / 'out', 'the visible scene', cache=self.folder / 'none')
        plan.update(ready=True, backend='imagine', argv=['imagine', 'fixture'])
        receipt = {'state': 'source-ready', 'source': source, 'style_plan': plan}
        receipt_path = self.folder / 'out/receipt.json'
        calls = []
        def run(argv, **kwargs):
            calls.append(argv)
            if len(argv) > 1 and argv[1] == 'wait':
                return SimpleNamespace(returncode=0, stdout=f'![result]({source["image"]})\n', stderr='')
            kwargs['stdout'].write('[imagine] queued fixture-123\n')
            return SimpleNamespace(returncode=0)
        style.queue_style(receipt, receipt_path, run=run)
        self.assertEqual(receipt['state'], 'queued')
        style.queue_style(receipt, receipt_path, run=run)
        self.assertEqual(len(calls), 1)
        style.queue_style(receipt, receipt_path, wait=True, run=run)
        self.assertEqual(calls[1], ['imagine', 'wait', 'fixture-123', '45'])
        self.assertEqual(receipt['state'], 'art-ready')
        self.assertEqual(receipt['visual_acceptance'], 'unreviewed')
        style.queue_style(receipt, receipt_path, wait=True, run=run)
        self.assertEqual(len(calls), 2)
        recovered = {'state': 'source-ready', 'style_plan': plan, 'job': {'id': None}}
        style.queue_style(recovered, receipt_path, run=Mock(side_effect=AssertionError('resubmit')))
        self.assertEqual(recovered['job']['id'], 'fixture-123')
        (receipt_path.parent / 'submission.log').write_text('ambiguous interrupted submission')
        with self.assertRaisesRegex(ValueError, 'automatic resubmission is disabled'):
            style.queue_style({'state': 'source-ready', 'style_plan': plan, 'job': {'id': None}}, receipt_path, run=Mock(side_effect=AssertionError('resubmit')))

    def test_native_worker_override_and_cli_are_process_local_and_mocked(self):
        helper = SimpleNamespace(MAX_REFERENCE_AREA=1, _Flux2KleinEditHelpers=SimpleNamespace(reference_dims=lambda w, h: (w, h)))
        plan = {'conditioning': style.native_dimensions(1472, 800), 'native_argv': ['native-cli', '--fixture']}
        observed = []
        def cli():
            observed.append((list(sys.argv), helper.MAX_REFERENCE_AREA))
        self.assertEqual(style.run_native_cli(plan, helper=helper, main=cli), [1472, 800])
        self.assertEqual(observed, [(['native-cli', '--fixture'], 1177600)])
        helper._Flux2KleinEditHelpers.reference_dims = lambda w, h: (1024, 512)
        with self.assertRaisesRegex(ValueError, 'would change conditioning'):
            style.run_native_cli(plan, helper=helper, main=Mock(side_effect=AssertionError('inference')))
        source = {'image': '/tmp/fixture.png', 'conditioning': style.native_dimensions(1472, 800)}
        preset = json.loads((self.compat_preset).read_text())
        argv = style.native_argv(source, self.folder, 'caption', preset, '/base', '/adapter', .65)
        self.assertIn('flux2-klein-4b', argv)
        self.assertIn('--lora-paths', argv)
        self.assertNotIn('flux3', str(argv))

    def test_native_launch_resume_and_visual_review_are_separate(self):
        original, _ = self.image(512, 512)
        out = self.folder / 'native'
        source = style.prepare_source(original, out)
        receipt_path = out / 'receipt.json'
        receipt = {'state': 'source-ready', 'source': source,
                   'style_plan': {'backend': 'mflux', 'ready': True, 'argv': ['isolated-python', '--worker']}}
        launch = Mock(return_value=SimpleNamespace(pid=123))
        receipt['job'] = {'id': None, 'status': 'launching'}
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            style.queue_native(receipt, receipt_path, wait=True, launch=launch)
        launch.assert_not_called()
        receipt.pop('job')
        with patch.object(style, 'worker_gone', return_value=False):
            style.queue_native(receipt, receipt_path, launch=launch)
            style.queue_native(receipt, receipt_path, launch=launch)
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(receipt['state'], 'queued')
        # A worker that died without a result fails once, visibly, and is never relaunched.
        with patch.object(style, 'worker_gone', return_value=True), self.assertRaisesRegex(ValueError, 'exited without a result'):
            style.queue_native(receipt, receipt_path, wait=True, launch=launch)
        self.assertEqual((launch.call_count, receipt['job']['status']), (1, 'failed'))
        style.write(out / 'native-result.json', {'ok': True, 'job_id': receipt['job']['id'],
                    'image': source['image'], 'sha256': source['sha256']})
        style.queue_native(receipt, receipt_path, launch=launch)
        self.assertEqual(receipt['state'], 'art-ready')
        self.assertEqual(receipt['visual_acceptance'], 'unreviewed')
        viewer = (out / 'viewer.html').read_text()
        self.assertIn('data:image/png;base64,', viewer)
        self.assertIn('Credits and provenance', viewer)
        self.assertNotIn('https://', viewer)
        review = {'reviewer': 'Fixture inspection', 'decision': 'accept',
                  'source_sha256': source['sha256'], 'image_sha256': source['sha256'],
                  'findings': ['Synthetic interface fixture only, no city quality claim.'],
                  'checks': {key: 'pass' for key in ('landmarks', 'materials', 'geometry', 'native_detail')}}
        review_path = out / 'review.json'
        style.write(review_path, review)
        make.check_review(receipt, review_path)
        self.assertEqual(receipt['state'], 'reviewed')
        style.queue_native(receipt, receipt_path, launch=Mock(side_effect=AssertionError('relaunch')))
        self.assertEqual(receipt['state'], 'reviewed')
        self.assertEqual(receipt['visual_acceptance'], 'accept')
        self.assertIn('Visual review: accept', (out / 'viewer.html').read_text())
        receipt['source']['source_metadata'] = {'credits':'Andrew Shiva / Wikipedia, CC BY-SA 4.0, https://commons.wikimedia.org/wiki/File:Scotland-2016-Aerial-Edinburgh_Castle.jpg'}
        style.export_viewer(receipt, out)
        viewer = (out / 'viewer.html').read_text()
        self.assertIn('Andrew Shiva / Wikipedia', viewer)
        self.assertIn('CC BY-SA 4.0', viewer)
        self.assertIn('https://commons.wikimedia.org/wiki/File:Scotland-2016-Aerial-Edinburgh_Castle.jpg', viewer)
        self.assertNotIn('Source credit not supplied', viewer)
        saved_pixels = Path(source['image']).read_bytes()
        Path(source['image']).write_bytes(saved_pixels + b'changed')
        with self.assertRaisesRegex(ValueError, 'art changed'):
            make.check_review(receipt, review_path)
        with self.assertRaisesRegex(ValueError, 'art changed'):
            style.queue_style(receipt, receipt_path, run=Mock(side_effect=AssertionError('resubmit')))
        Path(source['image']).write_bytes(saved_pixels)
        review['checks']['geometry'] = 'fail'
        style.write(review_path, review)
        with self.assertRaisesRegex(ValueError, 'four passing'):
            make.check_review(receipt, review_path)
        review.update(decision='reject', source_sha256='0' * 64)
        style.write(review_path, review)
        with self.assertRaisesRegex(ValueError, 'exact prepared source'):
            make.check_review(receipt, review_path)

    def test_cli_plan_only_is_replayable_and_creates_usable_starter(self):
        output = self.folder / 'starter'
        with patch('sys.stdout', new_callable=io.StringIO):
            self.assertEqual(atlas.main(['make', '55.8609,-4.2514', '--offline', '--plan-only', '--out', str(output)]), 0)
        receipt = json.loads((output / 'receipt.json').read_text())
        self.assertEqual(receipt['state'], 'resolved')
        self.assertEqual(json.loads((output/'recipe.json').read_text())['terrain']['provider'], 'skadi')
        path, _ = self.image(512, 512)
        args = ['make', 'A privately supplied Glasgow image', '--source', str(path), '--plan-only', '--out', str(self.folder / 'image')]
        with patch('sys.stdout', new_callable=io.StringIO):
            self.assertEqual(atlas.main(args), 0)
            self.assertEqual(atlas.main(args), 0)
        image_receipt = json.loads((self.folder / 'image/receipt.json').read_text())
        self.assertEqual(image_receipt['state'], 'source-ready')
        self.assertEqual(image_receipt['source']['registration'], 'unregistered-image-study')

    def test_make_cli_selects_repo_runtime_without_caller_site_packages(self):
        result = subprocess.run([sys.executable, '-S', str(ROOT / 'atlas.py'), 'make',
                                 '55.8609,-4.2514', '--offline', '--plan-only',
                                 '--out', str(self.folder / 'isolated')],
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['state'], 'resolved')

    def test_generated_recipe_edits_are_preserved_and_require_a_new_run(self):
        out=self.folder/'draft'
        args=['make','55.8609,-4.2514','--offline','--plan-only','--out',str(out)]
        with patch('sys.stdout',new_callable=io.StringIO):
            self.assertEqual(atlas.main(args),0)
        edited=json.loads((out/'recipe.json').read_text());edited['landmarks'][0]['elevation_m']=1640
        (out/'recipe.json').write_text(json.dumps(edited))
        with patch('sys.stderr',new_callable=io.StringIO) as err:
            self.assertEqual(atlas.main(args),1)
        self.assertIn('Recipe edits detected',err.getvalue())
        self.assertEqual(json.loads((out/'recipe.json').read_text()),edited)
        copy=self.folder/'edited.json';copy.write_text(json.dumps(edited))
        with patch('sys.stdout',new_callable=io.StringIO):
            self.assertEqual(atlas.main(['make',str(copy),'--offline','--plan-only','--out',str(self.folder/'corrected')]),0)
        receipt=json.loads((self.folder/'corrected/receipt.json').read_text())
        self.assertEqual(receipt['location']['recipe']['landmarks'][0]['elevation_m'],1640)
        edited['landmarks'][0]['elevation_m']=1700;copy.write_text(json.dumps(edited))
        with patch('sys.stderr',new_callable=io.StringIO):
            self.assertEqual(atlas.main(['make',str(copy),'--offline','--plan-only','--out',str(self.folder/'corrected')]),1)

    def test_only_a_client_speaking_the_queue_protocol_is_detected(self):
        bin_dir = self.folder / 'bin'
        bin_dir.mkdir()
        client = bin_dir / 'imagine'
        client.write_text('#!/bin/sh\necho an unrelated program with the same name\n')
        client.chmod(0o755)
        with patch.dict(os.environ, {'PATH': str(bin_dir)}), patch.object(style.Path, 'home', return_value=self.folder):
            os.environ.pop('ATLAS_IMAGINE', None)
            self.assertIsNone(DETECT_QUEUE_CLIENT())
            client.write_text("#!/bin/sh\necho '[imagine] queued fixture-1'\n")
            self.assertEqual(DETECT_QUEUE_CLIENT(), str(client))
            os.environ['ATLAS_IMAGINE'] = str(client)
            self.assertEqual(DETECT_QUEUE_CLIENT(), str(client.resolve()))
            os.environ['ATLAS_IMAGINE'] = str(bin_dir/'missing')
            with self.assertRaisesRegex(ValueError, 'ATLAS_IMAGINE must name an executable'):
                DETECT_QUEUE_CLIENT()

    def test_interrupted_native_launch_without_a_pid_never_waits_or_relaunches(self):
        out = self.folder/'interrupted-native'
        out.mkdir()
        path = out/'receipt.json'
        receipt = {'state': 'source-ready', 'style_plan': {'ready': True, 'argv': ['missing-runtime']}}
        failed_launch = Mock(side_effect=OSError('runtime launch failed'))
        with self.assertRaisesRegex(OSError, 'runtime launch failed'):
            style.queue_native(receipt, path, launch=failed_launch)
        saved = json.loads(path.read_text())
        self.assertTrue(saved['job']['id'])
        self.assertNotIn('pid', saved['job'])
        with self.assertRaisesRegex(ValueError, 'no recorded worker PID'):
            style.queue_native(saved, path, wait=True, launch=failed_launch)
        self.assertEqual(failed_launch.call_count, 1)

    def test_isolated_worker_bootstraps_its_project_imports(self):
        result=subprocess.run([sys.executable,'-I',str(ROOT/'source/local_style.py'),'--worker',str(self.folder/'absent.json')],capture_output=True,text=True,timeout=10)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('FileNotFoundError',result.stderr)
        self.assertNotIn('ModuleNotFoundError',result.stderr)

    def test_unicode_location_cache_does_not_collapse_distinct_places(self):
        opener = Mock(side_effect=[io.StringIO('[{"lat":"35.6","lon":"139.7"}]'), io.StringIO('[{"lat":"35.0","lon":"135.7"}]')])
        make.geocode('東京', self.folder, opener=opener, clock=lambda: 10, sleep=Mock())
        make.geocode('京都', self.folder, opener=opener, clock=lambda: 12, sleep=Mock())
        self.assertEqual(opener.call_count, 2)
        self.assertNotEqual(make.slug('東京'), make.slug('京都'))


if __name__ == '__main__':
    unittest.main()
