"""Material corrections cannot escape native masks or silently use other images."""
import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'))
import material_finish as finish

SOURCE_SETTINGS={'chroma_max':30,'luminance_min':30,'luminance_max':150,'green_excess':6,
 'close':9,'open':7,'feather':3,'protect_green_red':4,'protect_green_blue':10,
 'protect_warm_luminance':112,'protect_warm_red_blue':18,'protect_source_luminance':138,
 'protect_source_kernel':5,'channel_scale':[.985,.99,1.03],'gain':.9}


class MaterialFinish(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.folder=Path(self.temp.name)
        self.pixels=np.full((64,64,3),[142,126,102],np.uint8)
        Image.fromarray(self.pixels).save(self.folder/'art.png');Image.fromarray(self.pixels).save(self.folder/'source.png')
        self.recipe={'input_sha256':finish.digest(self.folder/'art.png'),'source_sha256':finish.digest(self.folder/'source.png'),'regions':[]}

    def run_finish(self):
        p=self.folder/'recipe.json';p.write_text(json.dumps(self.recipe))
        return finish.finish(self.folder/'art.png',self.folder/'source.png',p,self.folder/'out')

    def test_colour_edit_preserves_every_pixel_outside_polygon_and_exclusion(self):
        self.recipe['regions']=[{'name':'rock','polygon':[[10,10],[40,10],[40,40],[10,40]],
            'exclude_polygons':[[[20,20],[30,20],[30,30],[20,30]]],'selection':'warm','target_rgb':[90,94,98]}]
        proof=self.run_finish();actual=np.array(Image.open(self.folder/'out/map.png'))
        mask=finish.polygon_mask((64,64),self.recipe['regions'][0]['polygon'])
        mask&=~finish.polygon_mask((64,64),self.recipe['regions'][0]['exclude_polygons'][0])
        np.testing.assert_array_equal(actual[~mask],self.pixels[~mask])
        self.assertGreater(np.count_nonzero(actual[mask]!=self.pixels[mask]),0)
        self.assertEqual(proof['material_changes_outside_masks'],0)
        self.assertEqual(proof['dimensions'],[64,64])

    def test_identity_change_is_rejected_before_output(self):
        self.recipe['source_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'different source/art'):self.run_finish()
        self.assertFalse((self.folder/'out').exists())

    def test_local_ink_preserves_flat_planes_and_dark_window_interior(self):
        a=np.full((64,64,3),[180,160,130],np.uint8);a[15:40,15:40]=25;a[:,48]=25
        b,_=finish.local_ink(a)
        np.testing.assert_array_equal(b[25,25],a[25,25])
        np.testing.assert_array_equal(b[55,55],a[55,55])
        self.assertTrue(np.all(b[55,48]>a[55,48]))

    def test_gentle_ink_lift_leaves_mid_dark_lines_unchanged(self):
        a=np.full((64,64,3),180,np.uint8);a[:,16]=25;a[:,48]=60
        b,_=finish.local_ink(a,luminance_max=40,strength=.25)
        np.testing.assert_array_equal(b[:,48],a[:,48])
        self.assertGreater(int(b[32,16,0]),25)
        self.assertLess(int(b[32,16,0]),int(finish.local_ink(a)[0][32,16,0]))

    def test_native_runs_preserve_arch_gaps_and_limit_saturation_change(self):
        self.recipe['regions']=[{'name':'visible stone','runs':[[12,10,15],[12,20,25]],
            'saturation':0,'luminance_scale':.7,'feather':0}]
        self.run_finish();actual=np.array(Image.open(self.folder/'out/map.png'))
        np.testing.assert_array_equal(actual[12,17],self.pixels[12,17])
        self.assertEqual(np.count_nonzero(np.any(actual!=self.pixels,axis=2)),10)
        self.assertEqual(len(set(map(int,actual[12,11]))),1)

    def test_partial_roof_tint_keeps_luminance(self):
        self.recipe['regions']=[{'name':'roof','runs':[[12,10,25]],
            'target_rgb':[111,123,114],'strength':.3,'feather':0}]
        self.run_finish();actual=np.array(Image.open(self.folder/'out/map.png'))
        before=float(self.pixels[12,11]@finish.LUMA);after=float(actual[12,11]@finish.LUMA)
        self.assertLess(abs(after-before),1)
        self.assertGreater(float(actual[12,11,1]/actual[12,11,0]),float(self.pixels[12,11,1]/self.pixels[12,11,0]))

    def test_material_transform_uses_the_already_finished_ink(self):
        self.pixels[:]=180;self.pixels[:,16]=25
        Image.fromarray(self.pixels).save(self.folder/'art.png')
        self.recipe.update(input_sha256=finish.digest(self.folder/'art.png'),
            local_material_ink={'luminance_max':40,'strength':.25},
            regions=[{'name':'stone','runs':[[32,16,17]],'saturation':0,'luminance_scale':.7,'feather':0}])
        self.run_finish();actual=np.array(Image.open(self.folder/'out/map.png'))
        np.testing.assert_array_equal(actual[32,16],[28,28,28])

    def test_invalid_native_runs_fail_closed(self):
        for runs in ([[[1,3,8],[1,7,10]]],[[[64,0,2]]],[[[1,0,65]]],[[[1,3,3]]]):
            with self.assertRaises(ValueError):finish.region_mask((64,64),{'runs':runs[0]})

    def patch_fixture(self):
        patch=self.pixels[20:36,10:26].copy();patch[4,5]=[90,130,70]
        path=self.folder/'patch.png';Image.fromarray(patch).save(path)
        return {'name':'reviewed source repair','image':'patch.png','image_sha256':finish.digest(path),
            'position':[10,20],'before_pixels_sha256':hashlib.sha256(self.pixels[20:36,10:26].tobytes()).hexdigest(),
            'runs':[[4,5,6]]}

    def test_reviewed_native_patch_replays_only_inside_its_support(self):
        self.recipe['patches']=[self.patch_fixture()];proof=self.run_finish()
        actual=np.array(Image.open(self.folder/'out/map.png'));expected=self.pixels.copy();expected[24,15]=[90,130,70]
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(proof['patch_changes_outside_masks'],0)
        self.assertEqual(proof['changes'][0]['pixels'],1)

    def test_patch_rejects_changed_asset_location_and_escaped_pixels(self):
        patch=self.patch_fixture();self.recipe['patches']=[patch]
        patch['before_pixels_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'prior pixels'):self.run_finish()
        patch.update(self.patch_fixture());im=np.array(Image.open(self.folder/'patch.png'));im[1,1]=0;Image.fromarray(im).save(self.folder/'patch.png')
        with self.assertRaisesRegex(ValueError,'hash changed'):self.run_finish()
        patch['image_sha256']=finish.digest(self.folder/'patch.png')
        with self.assertRaisesRegex(ValueError,'outside its reviewed support'):self.run_finish()
        self.assertFalse((self.folder/'out').exists())

    def test_final_repair_is_bound_to_finished_pixels_and_keeps_reviewed_colours(self):
        self.recipe['regions']=[{'name':'stone','runs':[[24,15,16]],'saturation':0,'luminance_scale':.7,'feather':0}]
        self.run_finish();finished=np.array(Image.open(self.folder/'out/map.png'))
        patch=self.patch_fixture()
        replacement=finished[20:36,10:26].copy();replacement[4,5]=[210,209,194]
        Image.fromarray(replacement).save(self.folder/'patch.png')
        patch.update(image_sha256=finish.digest(self.folder/'patch.png'),before_pixels_sha256=hashlib.sha256(finished[20:36,10:26].tobytes()).hexdigest())
        self.recipe['post_patches']=[patch]
        recipe=self.folder/'final-recipe.json';recipe.write_text(json.dumps(self.recipe))
        proof=finish.finish(self.folder/'art.png',self.folder/'source.png',recipe,self.folder/'final')
        expected=finished.copy();expected[24,15]=[210,209,194]
        np.testing.assert_array_equal(np.array(Image.open(self.folder/'final/map.png')),expected)
        self.assertEqual(proof['patch_changes_outside_masks'],0)
        patch['before_pixels_sha256']=hashlib.sha256(self.pixels[20:36,10:26].tobytes()).hexdigest()
        recipe.write_text(json.dumps(self.recipe))
        with self.assertRaisesRegex(ValueError,'prior pixels'):
            finish.finish(self.folder/'art.png',self.folder/'source.png',recipe,self.folder/'bad-final')

    def test_source_mask_protects_moss_and_lit_masonry(self):
        self.pixels[:]=[100,90,70]
        self.pixels[15:24,15:24]=[60,95,50]
        self.pixels[30:39,15:24]=[180,155,110]
        src=np.full_like(self.pixels,90);src[30:39,30:39]=200
        Image.fromarray(self.pixels).save(self.folder/'art.png')
        Image.fromarray(src).save(self.folder/'source.png')
        settings=SOURCE_SETTINGS
        self.recipe.update(input_sha256=finish.digest(self.folder/'art.png'),source_sha256=finish.digest(self.folder/'source.png'))
        self.recipe['regions']=[{'name':'source rock','polygon':[[8,8],[54,8],[54,54],[8,54]],'source_mask':settings}]
        proof=self.run_finish();actual=np.array(Image.open(self.folder/'out/map.png'))
        for y,x in [(19,19),(34,19),(34,34),(0,0)]:
            np.testing.assert_array_equal(actual[y,x],self.pixels[y,x])
        self.assertLess(int(actual[45,45].max()-actual[45,45].min()),8)
        self.assertGreater(proof['changes'][0]['pixels'],0)
        self.assertEqual(proof['material_changes_outside_masks'],0)

    def test_source_feather_stays_inside_hull_and_protects_exclusions(self):
        self.pixels[:]=[120,100,80]
        Image.fromarray(self.pixels).save(self.folder/'art.png')
        Image.fromarray(np.full_like(self.pixels,90)).save(self.folder/'source.png')
        self.recipe.update(input_sha256=finish.digest(self.folder/'art.png'),source_sha256=finish.digest(self.folder/'source.png'))
        settings=SOURCE_SETTINGS
        self.recipe['regions']=[{'name':'bounded source mask','polygon':[[12,12],[50,12],[50,50],[12,50]],'exclude_polygons':[[[26,26],[36,26],[36,36],[26,36]]],'source_mask':settings}]
        self.run_finish();actual=np.array(Image.open(self.folder/'out/map.png'))
        region=self.recipe['regions'][0];allowed=finish.polygon_mask((64,64),region['polygon'])
        allowed&=~finish.polygon_mask((64,64),region['exclude_polygons'][0])
        np.testing.assert_array_equal(actual[~allowed],self.pixels[~allowed])
        self.assertGreater(np.count_nonzero(actual[allowed]!=self.pixels[allowed]),0)

    def test_bad_coordinates_and_nonfinite_recipe_fail(self):
        with self.assertRaises(ValueError):finish.polygon_mask((64,64),[[-1,0],[10,0],[0,10]])
        self.recipe['regions']=[{'name':'bad','polygon':[[0,0],[10,0],[0,10]],'target_rgb':[float('nan'),0,0]}]
        with self.assertRaises(ValueError):self.run_finish()


if __name__=='__main__':unittest.main()
