"""Media keeps native registration, bounded cameras, actual credits and little text."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

from PIL import Image, ImageChops, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source'))
import media as video


def manifest():
    return {'title': 'Test city', 'width': 1200, 'height': 900, 'credits': ['Actual source credit.'],
            'landmarks': [
                {'id': 'west', 'name': 'West edge', 'x': 1, 'y': 1},
                {'id': 'south', 'name': 'South edge', 'x': 1199, 'y': 899},
                {'id': 'original', 'name': 'Original landmark', 'x': 500, 'y': 400},
                {'id': 'east', 'name': 'East edge', 'x': 1199, 'y': 1}]}


class FilmTests(unittest.TestCase):
    def test_zero_text_default_keeps_actual_credits_in_sidecar(self):
        data = manifest()
        data['media_attribution'] = 'Andrew Shiva / Wikipedia. CC BY-SA 4.0.'
        plan = video.FilmPlan(data, seconds=2, fps=30)
        self.assertEqual(plan.verify()['maximum_on_screen_words'], 0)
        self.assertIn('Andrew Shiva', video.credits_text(data))
        self.assertNotIn('OpenStreetMap', video.credits_text(data))
        data['film'] = {'opening_credit': 'Any title'}
        with self.assertRaisesRegex(ValueError, 'Opening text'):
            video.FilmPlan(data)
        with self.assertRaisesRegex(ValueError, 'Supply source credits'):
            video.credits_text({})

    def test_bundled_native_films_are_bounded_and_loop_continuously(self):
        for city in ('dc', 'edinburgh'):
            data = json.loads((ROOT/f'examples/{city}.json').read_text())
            data['film'].update(view_width_pixels=1920, zoom_fraction=0)
            plan = video.FilmPlan(data, seconds=60, fps=60)
            proof = plan.verify()
            self.assertEqual(proof['frames_checked'], 3600)
            self.assertEqual(proof['maximum_on_screen_words'], 0)
            self.assertTrue(proof['all_viewports_inside_safe_source'])
            self.assertFalse(plan.framing['clamped_to_source'])
            self.assertEqual(plan.frame(0).viewport.scale, 1)
            for field in ('left', 'top', 'width', 'height'):
                self.assertAlmostEqual(getattr(plan.frame(0).viewport, field), getattr(plan.frame(plan.count-1).viewport, field), places=8)

    def test_edge_camera_clamping_keeps_anchor_and_explicit_route(self):
        data = manifest()
        data['film'] = {'route': ['east', 'original', 'outside-preview', 'west']}
        plan = video.FilmPlan(data, seconds=2, fps=30, callouts=True)
        self.assertEqual([point['id'] for point in plan.stops], ['east', 'original', 'west'])
        self.assertEqual((plan.stops[1]['anchor_x'], plan.stops[1]['anchor_y']), (500, 400))
        self.assertNotEqual(plan.stops[-1]['camera_x'], 1)
        self.assertTrue(plan.verify()['all_viewports_inside_safe_source'])
        self.assertEqual(data['landmarks'][0]['id'], 'west')
        data['film']['route'] = ['west', 'west']
        with self.assertRaises(ValueError):
            video.FilmPlan(data)

    def test_actual_source_pixel_and_label_tip_project_to_same_point(self):
        source = Image.new('RGB', (1200, 900))
        ImageDraw.Draw(source).ellipse((496, 396, 504, 404), fill=(255, 0, 0))
        view = video.Viewport(120, 80, 800, 450, 640, 360)
        rendered = video.render_frame(source, video.FilmFrame(0, 0, view, None), video.font(18))
        bbox = rendered.getbbox()
        projected = view.project(500, 400)
        self.assertLessEqual(abs((bbox[0]+bbox[2]-1)/2-projected[0]), 1)
        self.assertLessEqual(abs((bbox[1]+bbox[3]-1)/2-projected[1]), 1)
        annotation = video.Annotation('landmark', 'Original landmark', 1, (500, 400))
        anchor, _, _ = video.annotation_geometry(annotation, view, rendered.size, video.font(18))
        self.assertEqual(anchor, projected)
        labelled = video.render_frame(source, video.FilmFrame(0, 0, view, annotation), video.font(18))
        self.assertGreater(min(labelled.getpixel(tuple(round(v) for v in projected))), 200)

    def test_optional_labels_never_exceed_seven_words_and_roundtrip(self):
        plan = video.FilmPlan(manifest(), seconds=12, fps=30, callouts=True)
        proof = plan.verify()
        self.assertGreater(proof['landmark_annotated_frames'], 0)
        self.assertLessEqual(proof['maximum_on_screen_words'], 7)
        self.assertLessEqual(proof['maximum_source_projection_roundtrip_error_pixels'], 1e-8)
        data = manifest()
        data['landmarks'][2]['name'] = 'one two three four five six seven eight'
        with self.assertRaisesRegex(ValueError, 'seven words'):
            video.FilmPlan(data, callouts=True)

    def test_explicit_width_fits_portrait_and_landscape_sources(self):
        data = manifest()
        data['film'] = {'view_width_pixels': 10000}
        for width, height in ((1920, 1080), (1080, 1920)):
            plan = video.FilmPlan(data, seconds=2, fps=12, width=width, height=height, callouts=True)
            self.assertTrue(plan.framing['clamped_to_source'])
            self.assertTrue(plan.verify()['all_viewports_inside_safe_source'])
            first = plan.frame(0).viewport
            self.assertLessEqual(first.width, 1196+1e-8)
            self.assertLessEqual(first.height, 898+1e-8)

    def test_invalid_values_fail_before_encoding(self):
        for value in (float('nan'), float('inf'), -1, .5, True, None, '1280', 10**1000):
            data = manifest(); data['film'] = {'view_width_pixels': value}
            with self.assertRaisesRegex(ValueError, 'film.view_width_pixels'):
                video.FilmPlan(data)
        for kwargs in ({'seconds': float('nan')}, {'seconds': 0}, {'width': 319}, {'fps': 0}, {'source_credit_band': -1}):
            with self.assertRaises(ValueError):
                video.FilmPlan(manifest(), **kwargs)
        for value in (-.1, .2, True, float('nan')):
            data = manifest(); data['film'] = {'zoom_fraction': value}
            with self.assertRaises(ValueError):
                video.FilmPlan(data)

    def test_plan_cli_writes_actual_source_hash_and_credits_without_movie(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'source.png'; meta = root/'manifest.json'; output = root/'film.mp4'
            Image.new('RGB', (1200, 900), (17, 46, 75)).save(source)
            meta.write_text(json.dumps(manifest()))
            with redirect_stdout(io.StringIO()):
                self.assertEqual(video.main(['--image', str(source), '--manifest', str(meta), '--output', str(output),
                                            '--seconds', '1', '--fps', '2', '--plan-only']), 0)
            self.assertFalse(output.exists())
            self.assertIn('Actual source credit.', output.with_suffix('.credits.txt').read_text())
            self.assertEqual(json.loads(output.with_suffix('.route.json').read_text())['image_sha256'], video.digest(source))
            output.write_bytes(b'Existing movie')
            with self.assertRaises(FileExistsError):
                video.export(video.argument_parser().parse_args(['--image', str(source), '--manifest', str(meta),
                             '--output', str(output), '--plan-only', '--overwrite']))
            data = manifest(); data['image_sha256'] = '0'*64; meta.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'hash'):
                video.export(video.argument_parser().parse_args(['--image', str(source), '--manifest', str(meta), '--plan-only']))

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg required for GIF encoding')
    def test_teaser_is_native_held_crops_with_static_fallback_and_durable_credits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); image = root/'map.png'; meta = root/'map.json'; output = root/'teaser.gif'
            source = Image.new('RGB', (1200, 900), (30, 50, 70)); ImageDraw.Draw(source).rectangle((400, 300, 700, 600), fill=(240, 210, 170))
            source.save(image); meta.write_text(json.dumps(manifest()))
            record = video.teaser(image, meta, output, [[1, 1], [600, 450]], width=400, height=300)
            self.assertEqual(record['native_crop_bounds'], [(0, 0, 400, 300), (400, 300, 800, 600)])
            with Image.open(output) as gif:
                self.assertEqual(gif.n_frames, 2); self.assertEqual(gif.size, (400, 300)); self.assertEqual(gif.info['duration'], 2000)
                self.assertEqual(gif.info['loop'], 0)
            self.assertIsNone(ImageChops.difference(Image.open(output.with_suffix('.png')), source.crop((0, 0, 400, 300))).getbbox())
            self.assertIn('Actual source credit.', output.with_suffix('.credits.txt').read_text())
            with self.assertRaises(FileExistsError):
                video.teaser(image, meta, output, [[1, 1]])
            with self.assertRaisesRegex(ValueError, 'fit the native map'):
                video.teaser(image, meta, root/'other.gif', [[1, 1]], width=2000)


if __name__ == '__main__':
    unittest.main()
