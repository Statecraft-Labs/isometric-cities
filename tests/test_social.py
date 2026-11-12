"""Feed films stay inside the map, share one viewport across the wipe, loop and carry no text."""
import json
from pathlib import Path
import sys
import unittest

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source'))
import social


def manifest():
    return {'title': 'Test city', 'width': 1200, 'height': 900, 'opening_landmark': 'hall',
            'landmarks': [{'id': 'hall', 'x': 500, 'y': 400}, {'id': 'edge', 'x': 1199, 'y': 1}]}


def recipe(**changes):
    return {'size': [160, 200], 'fps': 10, 'loop': True, 'shots': [
        {'seconds': 1, 'path': [{'at': 'hall'}], 'wipe': [.5, 1.02]},
        {'seconds': 2.1, 'path': [{'at': 'hall'}, {'at': 'edge', 'width': 400}, {'at': 'hall'}]},
        {'seconds': 1, 'path': [{'at': 'hall'}], 'wipe': [1.02, .5]}], **changes}


class SocialTests(unittest.TestCase):
    def test_unlabelled_places_need_an_explicit_recipe(self):
        data = {**manifest(), 'landmarks': []}
        with self.assertRaisesRegex(ValueError, 'map-coordinate shots'):
            social.default_recipe(data)
        explicit = recipe(shots=[{'seconds': 1, 'path': [{'at': [600, 450]}]}])
        self.assertEqual(social.SocialPlan(data, explicit).verify()['result'], 'pass')

    def test_every_frame_stays_inside_and_the_last_returns_to_the_first(self):
        plan = social.SocialPlan(manifest(), recipe())
        proof = plan.verify()
        self.assertEqual((proof['frames_checked'], proof['maximum_on_screen_words']), (41, 0))
        self.assertEqual(proof['loop_closure_error'], 0)
        self.assertEqual(plan.frame(0)[1], .5)
        edge = plan.frame(20)[0]
        self.assertAlmostEqual(edge.left + edge.width, 1200)
        self.assertAlmostEqual(edge.top, 0)
        broken = recipe()
        broken['shots'][-1]['wipe'] = [1.02, .4]
        with self.assertRaises(AssertionError):
            social.SocialPlan(manifest(), broken).verify()
        # A fully photographic first frame differs from an all-art last frame.
        broken['shots'][0]['wipe'] = [0, 1.02]
        broken['shots'][-1]['wipe'] = [1.02, 1]
        with self.assertRaises(AssertionError):
            social.SocialPlan(manifest(), broken).verify()

    def test_zoom_needs_an_explicit_square_pixel_limit(self):
        close = recipe()
        close['shots'][1]['path'][1]['width'] = 100
        with self.assertRaisesRegex(ValueError, 'View width'):
            social.SocialPlan(manifest(), close)
        self.assertEqual(social.SocialPlan(manifest(), {**close, 'pixel_zoom': 2}).verify()['result'], 'pass')
        with self.assertRaisesRegex(ValueError, 'Unknown landmark'):
            social.SocialPlan(manifest(), {'shots': [{'seconds': 1, 'path': [{'at': 'missing'}]}]})

    def test_wipe_layers_use_one_transform_and_native_views_are_exact(self):
        art = Image.effect_noise((1200, 900), 60).convert('RGB')
        plan = social.SocialPlan(manifest(), recipe())
        view, wipe = plan.frame(0)
        still = social.render(art, art, view, None)
        self.assertEqual(still.tobytes(), art.crop((420, 300, 580, 500)).tobytes())
        split = social.render(art, art, view, wipe)
        self.assertIsNone(ImageChops.difference(split, still).crop((84, 0, 160, 200)).getbbox())
        self.assertIsNone(ImageChops.difference(split, still).crop((0, 0, 76, 200)).getbbox())
        zoomed = social.sample(art, social.Viewport(420, 300, 80, 100, 160, 200))
        self.assertEqual(zoomed.tobytes(), art.crop((420, 300, 500, 400)).resize((160, 200), Image.Resampling.NEAREST).tobytes())

    def test_bundled_examples_plan_from_their_landmarks(self):
        for city in ('dc', 'edinburgh'):
            data = json.loads((ROOT/f'examples/{city}.json').read_text())
            for chosen in (social.default_recipe(data), json.loads((ROOT/f'examples/{city}-social.json').read_text())):
                proof = social.SocialPlan(data, chosen).verify()
                self.assertGreaterEqual(proof['source_wipe_frames'], 1)
                self.assertLess(proof['loop_closure_error'], 1e-9)

    def test_dc_feed_film_opens_on_finished_art_then_sweeps_the_source(self):
        data = json.loads((ROOT/'examples/dc.json').read_text())
        plan = social.SocialPlan(data, json.loads((ROOT/'examples/dc-social.json').read_text()))
        hook = 3 * plan.fps
        self.assertEqual({plan.frame(index)[1] for index in range(hook)}, {None})
        self.assertEqual(plan.frame(hook)[1], 0)
        self.assertEqual(plan.shots[3]['first'], hook)
        anchors = {point['id']: (point['x'], point['y']) for point in data['landmarks']}
        for second, mark in enumerate(('capitol', 'washington-monument', 'white-house')):
            view = plan.frame(second * plan.fps + plan.fps // 2)[0]
            x, y = view.project(*anchors[mark])
            self.assertTrue(0 < x < plan.width and 0 < y < plan.height, mark)
            self.assertLessEqual(view.width, 720)


if __name__ == '__main__':
    unittest.main()
