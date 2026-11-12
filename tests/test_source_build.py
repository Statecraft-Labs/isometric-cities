"""Compact source behaviour: explicit geography, holes, mapped parts and height provenance."""
from contextlib import redirect_stdout
from io import StringIO
import json
import gzip
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from shapely.geometry import Polygon,shape

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'))
from geometry import SceneCoordinates,load_features,reconcile_parts,triangles,hgt_tiles
from project import read,write
from source_build import build_source


class CompactSource(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.out=Path(self.temp.name)
        self.scene=SceneCoordinates(-41.28,174.77)
        def ring(x0,y0,x1,y1):
            return [dict(zip(('lon','lat'),map(float,self.scene.lonlat(x,y)))) for x,y in [(x0,y0),(x1,y0),(x1,y1),(x0,y1),(x0,y0)]]
        self.payload={'elements':[
            {'type':'relation','id':1,'tags':{'type':'multipolygon','building':'yes','height':'12','roof:shape':'flat','roof:colour':'#465461'},'members':[
                {'type':'way','ref':10,'role':'outer','geometry':ring(-60,-50,30,40)},
                {'type':'way','ref':11,'role':'inner','geometry':ring(-35,-20,5,20)}]},
            {'type':'way','id':2,'tags':{'building':'yes'},'geometry':ring(50,-50,110,40)},
            {'type':'way','id':3,'tags':{'building:part':'yes','height':'9','roof:colour':'#b6324c'},'geometry':ring(50,-50,80,40)},
            {'type':'way','id':4,'tags':{'building:part':'yes','building:levels':'4'},'geometry':ring(80,-50,110,40)}]}
        self.osm=self.out/'fixture.osm.json';write(self.osm,self.payload)
        self.city={'id':'fixture','title':'Wellington source fixture','origin':{'lat':self.scene.lat,'lon':self.scene.lon},
                   'bbox':[-41.285,174.765,-41.275,174.775],'osm_source':str(self.osm),
                   'terrain':{'provider':'flat','flat_height_m':1640},'camera':{'pixels_per_metre':1,'view_quadrant':'NE','terrain_cell_metres':20},
                   'landmarks':[{'id':'focus','name':'Measured focus','lat':self.scene.lat,'lon':self.scene.lon,'elevation_m':1640}]}

    def test_source_preserves_holes_parts_height_evidence_and_native_camera(self):
        with patch('requests.Session.get',side_effect=AssertionError('network')),redirect_stdout(StringIO()):
            result=build_source(self.city,self.out/'render',offline=True)
        manifest=read(result['manifest']);records=read(self.out/'render/footprints.json')['features']
        self.assertEqual(manifest['camera']['world_origin'],self.city['origin'])
        self.assertEqual(manifest['focus_elevation_m'],1640)
        self.assertEqual(manifest['geometry']['courtyard_holes'],1)
        self.assertEqual(manifest['geometry']['parts']['suppressed_parent_outlines'],['way/2'])
        self.assertEqual({v['osm_id'] for v in records},{'relation/1','way/3','way/4'})
        court=next(v for v in records if v['osm_id']=='relation/1');geometry=shape(court['geometry'])
        self.assertEqual(len(geometry.interiors),1)
        self.assertAlmostEqual(sum(Polygon(t).area for t in triangles(geometry)),geometry.area)
        self.assertEqual(court['height_source'],'osm:height')
        levels=next(v for v in records if v['osm_id']=='way/4');self.assertEqual(levels['height_source'],'inferred:building:levels*3.2m+1m')
        self.assertAlmostEqual(levels['height_m'],13.8)
        pixels=np.array(Image.open(result['image']));self.assertEqual(pixels.shape,(320,512,3))
        self.assertGreater(np.count_nonzero(np.all(pixels==[182,50,76],axis=2)),0)
        self.assertEqual(manifest['terrain_coverage']['unmapped_pixels'],0)
        self.assertIn('ODbL',manifest['credits'])
        before=result['image'].read_bytes()
        with patch('requests.Session.get',side_effect=AssertionError('network')),redirect_stdout(StringIO()):
            again=build_source(self.city,self.out/'render',offline=True)
        self.assertEqual(again['image'].read_bytes(),before)

    def test_missing_dem_uses_explicit_measured_focus_and_is_visible(self):
        self.city['terrain']={'provider':'skadi'}
        with patch('requests.Session.get',side_effect=AssertionError('network')),redirect_stdout(StringIO()):
            result=build_source(self.city,self.out/'missing',offline=True)
        manifest=read(result['manifest'])
        self.assertEqual(manifest['terrain']['mode'],'flat-missing-dem')
        self.assertIn('1640 m',manifest['terrain']['description'])
        focus=manifest['landmarks'][0];self.assertAlmostEqual(focus['x'],256);self.assertAlmostEqual(focus['y'],160)

    def test_input_freeze_rejects_changed_cache_original_and_hgt_before_render(self):
        self.city['terrain']={'provider':'skadi'}
        hgt=gzip.compress(np.full((2,2),1640,dtype='>i2').tobytes())
        response=SimpleNamespace(content=hgt,raise_for_status=lambda:None)
        folder=self.out/'frozen'
        with patch('requests.Session.get',return_value=response) as get,redirect_stdout(StringIO()):
            result=build_source(self.city,folder)
        self.assertEqual(get.call_count,1)
        before=result['image'].read_bytes()
        self.assertTrue((folder/'source-inputs.json').is_file())
        for path in [folder/'cache/osm.json',self.osm,folder/'cache/S42E174.hgt.gz']:
            original=path.read_bytes();path.write_bytes(original+b'changed')
            try:
                with patch('requests.Session.get',side_effect=AssertionError('network')),self.assertRaisesRegex(ValueError,'Source input changed'):
                    build_source(self.city,folder,offline=True)
                self.assertEqual(result['image'].read_bytes(),before)
            finally:path.write_bytes(original)
        with patch('requests.Session.get',side_effect=AssertionError('network')),redirect_stdout(StringIO()):
            build_source(self.city,folder,offline=True)
        self.assertEqual(result['image'].read_bytes(),before)

    def test_input_freeze_precedes_render_and_binds_missing_dem(self):
        self.city['terrain']={'provider':'skadi'};folder=self.out/'interrupted'
        with patch('source_build.Raster',side_effect=RuntimeError('interrupted')),redirect_stdout(StringIO()),self.assertRaisesRegex(RuntimeError,'interrupted'):
            build_source(self.city,folder,offline=True)
        frozen=read(folder/'source-inputs.json');self.assertIn(None,frozen.values())
        missing=next(Path(path) for path,value in frozen.items() if value is None)
        missing.write_bytes(gzip.compress(np.full((2,2),1640,dtype='>i2').tobytes()))
        with self.assertRaisesRegex(ValueError,'Source input changed'):
            build_source(self.city,folder,offline=True)
        self.assertFalse((folder/'source.png').exists())

    def test_late_dem_acquisition_cannot_change_an_existing_source(self):
        self.city['terrain']={'provider':'skadi'};folder=self.out/'late-dem'
        with redirect_stdout(StringIO()):result=build_source(self.city,folder,offline=True)
        before=result['image'].read_bytes()
        def late_fetch(city,offline):
            (folder/'cache/S42E174.hgt.gz').write_bytes(gzip.compress(np.full((2,2),1700,dtype='>i2').tobytes()))
            return {}
        with patch('source_build._fetch',side_effect=late_fetch),self.assertRaisesRegex(ValueError,'changed during acquisition'):
            build_source(self.city,folder)
        self.assertEqual(result['image'].read_bytes(),before)

    def test_partial_parts_keep_only_parent_residual_and_explicit_height(self):
        payload=json.loads(json.dumps(self.payload));payload['elements']=payload['elements'][1:3]
        payload['elements'][0]['tags']['height']='20';write(self.osm,payload)
        features,_=load_features(self.osm,self.scene);rendered,stats=reconcile_parts(features)
        original=next(f for f in features if f.osm_id=='way/2');part=next(f for f in features if f.osm_id=='way/3')
        self.assertEqual(stats['parents_rendered_as_residuals'],[])
        self.assertAlmostEqual(original.geometry.intersection(part.geometry).area/original.geometry.area,.5)
        self.assertEqual(next(f for f in rendered if f.osm_id=='way/2').height,20)
        from shapely.affinity import scale
        part.geometry=scale(part.geometry,xfact=1.5,yfact=1,origin=(50,-50))
        rendered,stats=reconcile_parts(features)
        self.assertEqual(stats['parents_rendered_as_residuals'],['way/2'])
        residual=next(f for f in rendered if '@residual' in f.osm_id)
        self.assertAlmostEqual(residual.geometry.area,original.geometry.area*.25)
        self.assertAlmostEqual(residual.geometry.intersection(part.geometry).area,0)
        self.assertEqual(residual.height,20)
        self.assertEqual(residual.height_source,'osm:height')

    def test_source_identity_missing_origin_and_bounds_fail_without_overwriting(self):
        with redirect_stdout(StringIO()):build_source(self.city,self.out/'identity',offline=True)
        image=(self.out/'identity/source.png').read_bytes();self.city['title']='different'
        with self.assertRaisesRegex(ValueError,'recipe changed'):build_source(self.city,self.out/'identity',offline=True)
        self.assertEqual((self.out/'identity/source.png').read_bytes(),image)
        bad=dict(self.city,origin={'lat':float('nan'),'lon':174.77})
        with self.assertRaises(ValueError),redirect_stdout(StringIO()):build_source(bad,self.out/'bad',offline=True)
        with self.assertRaises(TypeError):SceneCoordinates()
        self.assertEqual(set(hgt_tiles([-41.1,174.9,-40.9,175.1])),{'S42E174','S42E175','S41E174','S41E175'})


if __name__=='__main__':unittest.main()
