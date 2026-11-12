"""Rectangle cameras and actual bounded OSM/terrain rendering in both views."""
from contextlib import redirect_stdout
import gzip
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'))
from geometry import Camera, SceneCoordinates
from project import read,write
from rectangle_frame import inverse_screen_xy,rectangle_camera,rectangle_frame_bounds,source_bbox_for_frame,terrain_coverage
from source_build import build_source


class RectangleTests(unittest.TestCase):
    def test_inverse_projection_and_exact_aspect(self):
        for quadrant,expected in [('NE',[100,100]),('SE',[100,-100])]:
            np.testing.assert_allclose(inverse_screen_xy([[0,0]],100,quadrant),[expected])
            camera=rectangle_camera([-100,-140,100,-15],scale=.8,view_quadrant=quadrant)
            self.assertEqual((camera.width,camera.height),(160,100))
            points=np.array([[-150,80,140],[20,-60,45],[10,20,30]])
            projected=Camera(1,0,0,1,1,view_quadrant=quadrant).project(points)
            np.testing.assert_allclose(inverse_screen_xy(projected[:,:2],points[:,2],quadrant),points[:,:2])
        coverage=terrain_coverage(np.array([[0.,0.],[0.,-np.inf]]))
        self.assertEqual(coverage['unmapped_pixels'],1)
        self.assertFalse(coverage['corners_mapped']['bottom_right'])

    def test_focus_selects_landmarks_and_measured_elevation(self):
        scene=SceneCoordinates(55.951,-3.1885)
        def point(identifier,x,y):
            lon,lat=scene.lonlat(x,y)
            return {'id':identifier,'lon':float(lon),'lat':float(lat)}
        city={'camera':{'view_quadrant':'NE','frame':{'focus_landmarks':['a','b'],'padding_metres':20,'aspect':[16,10]}},
              'landmarks':[point('a',-30,-10),point('b',40,20),point('outside',10000,10000)]}
        class Terrain:
            mode='measured';
            def elevation(self,x,y):return 1640+x*.01+y*.02
        bounds=rectangle_frame_bounds(city,scene,Terrain());camera=rectangle_camera(bounds,scene=scene,view_quadrant='NE')
        self.assertAlmostEqual(camera.width/camera.height,1.6)
        self.assertLess(bounds[2]-bounds[0],250)
        for marker in city['landmarks'][:2]:
            x,y=scene.local_xy(marker['lon'],marker['lat'])
            px,py,_=camera.project([x,y,Terrain().elevation(x,y)])
            self.assertTrue(0<px<camera.width and 0<py<camera.height)

    def test_actual_sources_cover_edges_keep_holes_tall_buildings_and_clip_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);scene=SceneCoordinates(.5,.5)
            dem=root/'N00E000.hgt.gz';dem.write_bytes(gzip.compress(np.array([[30,100],[60,130]],dtype='>i2').tobytes()))
            for quadrant in ['NE','SE']:
                with self.subTest(quadrant=quadrant):
                    sign=1 if quadrant=='NE' else -1
                    def ring(x0,y0,x1,y1):
                        return [dict(zip(('lon','lat'),map(float,scene.lonlat(x,y*sign)))) for x,y in [(x0,y0),(x1,y0),(x1,y1),(x0,y1),(x0,y0)]]
                    def marker(identifier,x,y):
                        lon,lat=scene.lonlat(x,y*sign)
                        return {'id':identifier,'lon':float(lon),'lat':float(lat)}
                    osm=root/f'{quadrant}.json';write(osm,{'elements':[
                        {'type':'relation','id':1,'tags':{'type':'multipolygon','natural':'water'},'members':[
                            {'type':'way','ref':10,'role':'outer','geometry':ring(-20,-20,40,40)},
                            {'type':'way','ref':11,'role':'inner','geometry':ring(-9,-9,9,9)}]},
                        {'type':'relation','id':2,'tags':{'type':'multipolygon','building':'yes','height':'8','roof:shape':'flat'},'members':[
                            {'type':'way','ref':20,'role':'outer','geometry':ring(-65,-20,-25,20)},
                            {'type':'way','ref':21,'role':'inner','geometry':ring(-57,-12,-33,12)}]},
                        {'type':'way','id':3,'tags':{'building':'tower','height':'180','roof:shape':'flat','roof:colour':'#cf2877'},'geometry':ring(80,140,100,160)},
                        {'type':'way','id':4,'tags':{'building':'yes','height':'20'},'geometry':ring(290,-10,310,10)}]})
                    bounds=[-100,-140,100,-15]
                    city={'id':quadrant.lower(),'title':'Rectangle source','origin':{'lat':scene.lat,'lon':scene.lon},
                          'bbox':source_bbox_for_frame(bounds,scene,quadrant,[-20,300],40),'osm_source':str(osm),'dem_sources':[str(dem)],
                          'terrain':{'provider':'skadi'},'camera':{'pixels_per_metre':.8,'view_quadrant':quadrant,'terrain_cell_metres':5,
                          'frame':{'screen_bounds_metres':bounds,'aspect':[16,10],'elevation_range_metres':[-20,300]}},
                          'landmarks':[marker('visible',0,0),marker('offscreen',300,0)]}
                    with patch('requests.Session.get',side_effect=AssertionError('network')),redirect_stdout(StringIO()):
                        result=build_source(city,root/quadrant,offline=True)
                    manifest=read(result['manifest']);image=np.array(Image.open(result['image']))
                    self.assertEqual(image.shape,(100,160,3));self.assertEqual(manifest['terrain']['mode'],'open-dem')
                    coverage=manifest['terrain_coverage'];self.assertEqual(coverage['unmapped_pixels'],0)
                    self.assertTrue(all(coverage['corners_mapped'].values()));self.assertFalse(any(coverage['edge_strips_unmapped'].values()))
                    self.assertGreater(coverage['finite_depth_max']-coverage['finite_depth_min'],1)
                    self.assertEqual([m['id'] for m in manifest['landmarks']],['visible'])
                    records=read(root/quadrant/'footprints.json')['features']
                    self.assertEqual({r['osm_id'] for r in records},{'relation/2','way/3'})
                    self.assertEqual(manifest['geometry']['courtyard_holes'],1)
                    self.assertGreater(np.count_nonzero(np.all(image==[207,40,119],axis=2)),0)
                    matrix=np.array(manifest['camera']['pixel_matrix'])
                    def sample(x,y):
                        px,py=matrix@np.array([x,y*sign,80,1]);return image[int(py),int(px)].tolist()
                    self.assertEqual(sample(0,0),[190,193,170])
                    self.assertEqual(sample(20,0),[91,134,153])
