"""Native surface geometry, source identity and explicit cache failures."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source'))
spec=importlib.util.spec_from_file_location('surface_patch',ROOT/'source/surface_patch.py')
surface=importlib.util.module_from_spec(spec)
spec.loader.exec_module(surface)


def fixture():
    return {'id':'fixture','bbox':[-.001,-.001,.001,.001], 'surface':{
        'crs':'EPSG:3857','resolution_m':1.,'bounds_projected_m':[-20.,-20.,20.,20.],
        'description':'Synthetic measured surface test','colour_evidence':'Fixture tags',
        'attribution':['Fixture'], 'rasters':[],
        'camera':{'centre_projected_m':[0.,0.],'u_range_m':[-30.,30.],
                  'y_range_screen_m':[-30.,30.],'z0_m':0.,'bearing_deg':45.,
                  'elevation_deg':30.,'pixels_per_metre':2.}}}


class SurfaceSource(unittest.TestCase):
    def test_camera_axes_and_height(self):
        u,v=surface.horizontal(np.array([10.,0.]),np.array([0.,10.]),45)
        np.testing.assert_allclose(u,[10/np.sqrt(2),-10/np.sqrt(2)])
        np.testing.assert_allclose(v,[10/np.sqrt(2),10/np.sqrt(2)])
        u,v=surface.horizontal(10.,20.,0)
        self.assertEqual((u,v),(10.,20.))

    def test_matrix_matches_projection_and_crop_translates_offsets_only(self):
        import sys
        sys.path.insert(0, str(ROOT/'source'))
        from local_style import prepare_source
        city=json.loads((ROOT/'cities/edinburgh.json').read_text())
        camera=city['surface']['camera']
        matrix=np.array(surface.pixel_matrix(camera))
        points=np.array([[325123.3,673500.4,130.,1.],[325815.7,673117.5,85.,1.]])
        dx=points[:,0]-camera['centre_projected_m'][0]
        dy=points[:,1]-camera['centre_projected_m'][1]
        u,v=surface.horizontal(dx,dy,camera['bearing_deg'])
        e=np.deg2rad(camera['elevation_deg']);s=camera['pixels_per_metre']
        expected=np.array([(u-camera['u_range_m'][0])*s,
            (-(np.sin(e)*v+np.cos(e)*(points[:,2]-camera['z0_m']))-camera['y_range_screen_m'][0])*s])
        np.testing.assert_allclose(matrix@points.T,expected,atol=1e-8)
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t); yy,xx=np.indices((1280,1536))
            data=np.stack([xx%256,yy%256,(xx+yy)%256],-1).astype(np.uint8)
            Image.fromarray(data).save(folder/'source.png')
            (folder/'source.json').write_text(json.dumps({'camera':{'pixel_matrix':matrix.tolist(),'width':1536,'height':1280}}))
            result=prepare_source(folder/'source.png',folder/'crop',crop=[128,64,1024,1024],manifest=folder/'source.json')
            actual=np.array(result['camera']['pixel_matrix']); shifted=matrix.copy();shifted[:,3]-=[128,64]
            np.testing.assert_array_equal(actual,shifted)
            np.testing.assert_array_equal(np.array(Image.open(result['image'])),data[64:1088,128:1152])
            np.testing.assert_array_equal(result['source_camera']['pixel_matrix'],matrix)

    def test_landmark_projection_matches_source_matrix_at_native_and_half_scale(self):
        from landmark_overlay import Camera
        camera=json.loads((ROOT/'cities/edinburgh.json').read_text())['surface']['camera']
        xyz=np.array([[325100.,673200.,80.],[325700.,673800.,145.]])
        for scale in (1.,.5):
            c=dict(camera,pixels_per_metre=camera['pixels_per_metre']*scale)
            expected=np.c_[xyz,np.ones(len(xyz))]@np.array(surface.pixel_matrix(c)).T
            np.testing.assert_allclose(Camera(c).project(xyz)[:,:2],expected,atol=1e-8)

    def test_landmark_inputs_and_camera_fail_closed_before_drawing(self):
        from landmark_overlay import Overlay
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t); inputs=[]
            for name in ('surface','osm'):
                p=folder/name;p.write_bytes(name.encode());inputs.append(p)
            c=fixture()['surface'];poly={'type':'Polygon','coordinates':[[[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]]]}
            shape_record={'geometry':poly,'base_m':0.,'top_m':10.}
            d={'schema':1,'crs':c['crs'],'camera':c['camera'],
               'input_sha256':{'surface':surface.sha(inputs[0]),'osm':surface.sha(inputs[1]),'horizontal_grid':None},
               'suppression_tolerance_m':0.,'buildings':[],'structures':[dict(shape_record,kind='open_crown',rib_count=8)],
               'material_overrides':[],'references':[],'limits':[]}
            path=folder/'landmarks.json';path.write_text(json.dumps(d));c['landmarks']=str(path)
            L={'dsm':np.ones((40,40)), 'dtm':np.zeros((40,40)), 'bounds':[-20,-20,20,20]}
            overlay=Overlay(c,L,*inputs,None)
            original=np.ones((40,40));result=overlay.heightfield(original,L['dtm'])
            self.assertEqual(np.count_nonzero(result==0),4)
            np.testing.assert_array_equal(original,np.ones((40,40)))
            inputs[0].write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'input hash mismatch'):Overlay(c,L,*inputs,None)
            inputs[0].write_bytes(b'surface')
            changed=copy.deepcopy(c);changed['camera']['bearing_deg']+=1
            with self.assertRaisesRegex(ValueError,'camera mismatch'):Overlay(changed,L,*inputs,None)
            d['structures'][0]['top_m']=float('nan');path.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'finite metres'):Overlay(c,L,*inputs,None)


    def test_crown_drawing_stays_inside_footprint_and_measured_apex(self):
        from landmark_overlay import crown
        from shapely.geometry import shape, Point
        record=next(r for r in json.loads((ROOT/'cities/landmarks/edinburgh.json').read_text())['structures'] if r['kind']=='open_crown')
        class Mesh:
            def __init__(self):self.triangles=[]
            def triangle(self,xyz,*args,**kwargs):self.triangles.append(np.asarray(xyz))
        mesh=Mesh();crown(mesh,record);xyz=np.concatenate(mesh.triangles)
        polygon=shape(record['geometry'])
        self.assertLessEqual(max(Point(p).distance(polygon) for p in xyz[:,:2]),1e-8)
        self.assertEqual(xyz[:,2].max(),record['top_m'])
        self.assertEqual(xyz[:,2].min(),record['base_m'])
        # A six-sided rib's narrowest width is diameter*cos30, at native 3px/m.
        self.assertGreaterEqual(record['drawing']['rib_radius_m']*2*np.cos(np.pi/6)*3,3.)
        self.assertEqual(record['rib_count'],8)

    def test_cited_parapets_and_gun_openings_stay_on_existing_source_geometry(self):
        from landmark_overlay import parapet, embrasures
        from shapely.geometry import shape, Point
        data=json.loads((ROOT/'cities/landmarks/edinburgh.json').read_text())
        class Mesh:
            def __init__(self):self.triangles=[]
            def triangle(self,xyz,*args,**kwargs):self.triangles.append(np.asarray(xyz))
        count=0
        for record in data['buildings']:
            kind='parapet' if 'parapet'in record else 'embrasures' if 'embrasures'in record else None
            if kind is None:continue
            count+=1;mesh=Mesh();(parapet if kind=='parapet' else embrasures)(mesh,record)
            xyz=np.concatenate(mesh.triangles);polygon=shape(record['geometry'])
            self.assertLessEqual(max(Point(p).distance(polygon) for p in xyz[:,:2]),1e-8)
            self.assertLessEqual(xyz[:,2].max(),record['top_m'])
            self.assertTrue(record[kind]['source_refs'])
            self.assertTrue(record[kind]['illustrative'])
            if record['osm_id']=='way/714452595':self.assertEqual(kind,'embrasures')
        self.assertGreaterEqual(count,3)

    def test_scott_mesh_preserves_mapped_support_apex_and_outer_turrets(self):
        from landmark_overlay import gothic_spire, Camera
        from shapely.geometry import shape, Point
        record=next(r for r in json.loads((ROOT/'cities/landmarks/edinburgh.json').read_text())['structures'] if r['kind']=='gothic_spire')
        class Mesh:
            def __init__(self):self.triangles=[]
            def triangle(self,xyz,*args,**kwargs):self.triangles.append(np.asarray(xyz))
        mesh=Mesh();gothic_spire(mesh,shape(record['geometry']),record)
        xyz=np.concatenate(mesh.triangles);polygon=shape(record['geometry'])
        self.assertLessEqual(max(Point(p).distance(polygon) for p in xyz[:,:2]),1e-8)
        self.assertEqual(xyz[:,2].min(),record['base_m'])
        self.assertEqual(xyz[:,2].max(),record['top_m'])
        height=record['top_m']-record['base_m']
        for centre in record['massing']['outer_turret_centres_m']:
            nearby=np.linalg.norm(xyz[:,:2]-centre,axis=1)<1.3
            self.assertLessEqual(xyz[nearby,2].min(),record['base_m']+.01)
            self.assertGreaterEqual(xyz[nearby,2].max(),record['base_m']+.48*height-.01)
        # Intersections of the actual mesh with two height planes retain the
        # broad lower silhouette and upper body, rather than the old needle.
        camera=Camera(json.loads((ROOT/'cities/edinburgh.json').read_text())['surface']['camera'])
        for fraction,minimum_width in ((.22,45),(.60,20)):
            plane=record['base_m']+height*fraction;points=[]
            for tri in mesh.triangles:
                for a,b in zip(tri,np.roll(tri,-1,axis=0)):
                    if (a[2]-plane)*(b[2]-plane)<0:
                        points.append(a+(b-a)*(plane-a[2])/(b[2]-a[2]))
            self.assertGreaterEqual(np.ptp(camera.project(points)[:,0]),minimum_width)

    def test_scott_base_is_solid_above_a_real_pointed_opening(self):
        from landmark_overlay import gothic_spire
        from raster import Raster
        from shapely.geometry import shape
        record=next(r for r in json.loads((ROOT/'cities/landmarks/edinburgh.json').read_text())['structures'] if r['kind']=='gothic_spire')
        centre=np.array(shape(record['geometry']).centroid.coords[0])
        axis=np.array(record['massing']['axis_unit']);axis/=np.linalg.norm(axis)
        across=np.array([-axis[1],axis[0]])
        class Front:
            width,height=256,256
            def project(self,points):
                p=np.asarray(points);xy=p[:,:2]-centre;z=p[:,2]-record['base_m']
                return np.c_[128+(xy@axis)*5,240-z*3,-(xy@across)]
        raster=Raster(Front(),{'background_rgb':(230,240,250)})
        gothic_spire(raster,shape(record['geometry']),record)
        sample=lambda x,z:raster.rgb[int(240-z*3),int(128+x*5)]
        np.testing.assert_array_equal(sample(0,8),[230,240,250])
        for x,z in ((0,17),(-4.2,8),(4.2,8)):
            self.assertLess(sample(x,z).max(),100)

    def test_scott_massing_rejects_unmeasured_or_unbounded_parameters(self):
        from landmark_overlay import validate_data
        original=json.loads((ROOT/'cities/landmarks/edinburgh.json').read_text())
        for mutation in ('axis','turret','width','tiers'):
            d=copy.deepcopy(original)
            m=next(r for r in d['structures'] if r['kind']=='gothic_spire')['massing']
            if mutation=='axis':m['axis_unit']=[0,0]
            if mutation=='turret':m['outer_turret_centres_m'][0]=[0,0]
            if mutation=='width':m['arch_clear_span_m']=1000
            if mutation=='tiers':m['tiers'][2][3]=1.1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):validate_data(d)

    def test_drawing_validation_rejects_oversized_or_uncited_details(self):
        from landmark_overlay import validate_data
        original=json.loads((ROOT/'cities/landmarks/edinburgh.json').read_text())
        for mutation in ('rib','height','edge','citation','colour'):
            d=copy.deepcopy(original)
            crown=next(r for r in d['structures'] if r['kind']=='open_crown')
            wall=next(r for r in d['buildings'] if 'parapet'in r)
            if mutation=='rib':crown['drawing']['rib_radius_m']=float('nan')
            if mutation=='height':wall['parapet']['height_m']=40
            if mutation=='edge':wall['parapet']['edge_indices']=[9999]
            if mutation=='citation':wall['parapet']['source_refs']=[]
            if mutation=='colour':crown['drawing']['rib_rgb']=[-1,500,20]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):validate_data(d)

    def test_validation_rejects_nan_oversized_and_misaligned(self):
        for key,value in [('resolution_m',float('nan')),('bounds_projected_m',[0,0,100000,100000]),('bounds_projected_m',[0,0,20.2,20])]:
            city=fixture(); city['surface'][key]=value
            with self.assertRaises(ValueError): surface.validate(city['surface'])
        city=fixture(); city['surface']['camera']['bearing_deg']=float('nan')
        with self.assertRaises(ValueError): surface.validate(city['surface'])

    def test_real_material_tags_and_fallbacks(self):
        self.assertEqual(surface.roof_from_tags({'roof:material':'copper'}),surface.DEFAULT_PALETTE['copper'])
        self.assertEqual(surface.roof_from_tags({'roof:colour':'#334455'}),(51,68,85))
        self.assertIsNone(surface.roof_from_tags({'roof:colour':'#705544'}))
        self.assertEqual(surface.wall_from_tags({'building:material':'glass'}),surface.DEFAULT_PALETTE['glass'])
        self.assertIsNone(surface.roof_from_tags({}))

    def test_offline_missing_inputs_dont_fetch(self):
        with tempfile.TemporaryDirectory() as t, patch('requests.get',side_effect=AssertionError('network')):
            with self.assertRaisesRegex(ValueError,'Offline DSM/DTM cache missing'):
                surface.acquire(fixture(),Path(t),offline=True)

    def test_render_replays_exact_pixels_and_binds_inputs(self):
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t); dsm=np.zeros((40,40),np.float32)
            dsm[14:26,14:26]=8
            np.savez_compressed(folder/'surface.npz',dsm=dsm,dtm=np.zeros_like(dsm),bounds=[-20,-20,20,20])
            (folder/'osm.json').write_text(json.dumps({'elements':[{'type':'way','id':1,'tags':{'building':'yes','roof:colour':'#334455'},'geometry':[{'lon':-.00005,'lat':-.00005},{'lon':.00005,'lat':-.00005},{'lon':.00005,'lat':.00005},{'lon':-.00005,'lat':.00005},{'lon':-.00005,'lat':-.00005}]}]}))
            city=fixture()
            a=surface.render(city,folder/'surface.npz',folder/'osm.json',None,folder/'a')
            b=surface.render(city,folder/'surface.npz',folder/'osm.json',None,folder/'b')
            self.assertEqual(a['source_sha256'],b['source_sha256'])
            self.assertEqual(a['input_sha256']['surface'],surface.sha(folder/'surface.npz'))
            with Image.open(folder/'a/source.png') as image:
                self.assertEqual(image.size,(120,120))
                self.assertGreater(np.unique(np.asarray(image).reshape(-1,3),axis=0).shape[0],3)
            bad=copy.deepcopy(city);bad['surface']['bounds_projected_m']=[-21,-20,19,20]
            with self.assertRaisesRegex(ValueError,'bounds do not match'):
                surface.render(bad,folder/'surface.npz',folder/'osm.json',None,folder/'bad')
            dsm[0,0]=np.nan
            np.savez_compressed(folder/'surface.npz',dsm=dsm,dtm=np.zeros_like(dsm),bounds=[-20,-20,20,20])
            with self.assertRaisesRegex(ValueError,'finite, complete grid'):
                surface.render(city,folder/'surface.npz',folder/'osm.json',None,folder/'bad')

    def test_default_recipe_and_pinned_grid(self):
        city=json.loads((ROOT/'cities/edinburgh.json').read_text())
        c=surface.validate(city['surface'])['camera']
        self.assertEqual(round((c['u_range_m'][1]-c['u_range_m'][0])*c['pixels_per_metre']),3456)
        self.assertEqual(round((c['y_range_screen_m'][1]-c['y_range_screen_m'][0])*c['pixels_per_metre']),1803)
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t);grid=folder/'wrong.tif';grid.write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                surface.acquire(city,folder,offline=True,grid=grid)


if __name__=='__main__': unittest.main()
