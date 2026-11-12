"""Exercise real geographic texture projection and cached replay on a tiny scene."""

from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
import importlib.util
import json
import tempfile
from threading import Lock
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np
from PIL import Image
from pyproj import Transformer
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "photo_patch", ROOT / "source/photographic_patch.py"
)
photo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(photo)


class GeographicPhotoPatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="photo-patch-")
        cls.root = Path(cls.temp.name)
        cls.previous_root = photo.ROOT
        photo.ROOT = cls.root
        cls.out = cls.root / "run"
        src = cls.out / "source"
        src.mkdir(parents=True)
        (src / "osm.json").write_text(json.dumps({"elements": []}))
        origin = {"lat": 38.8896, "lon": -77.0056}
        t = Transformer.from_crs(4326, 26985, always_xy=True)
        inverse = Transformer.from_crs(26985, 4326, always_xy=True)
        cx, cy = t.transform(origin["lon"], origin["lat"])
        rgb = np.empty((3, 128, 128), np.uint8)
        rgb[:, :, :64] = np.array([30, 125, 150])[:, None, None]
        rgb[:, :, 64:] = np.array([160, 90, 75])[:, None, None]
        transform = from_origin(cx - 64, cy + 64, 1, 1)
        for name, array in [
            ("aerial", rgb),
            ("dtm", np.full((1, 128, 128), 10, np.float32)),
        ]:
            with rasterio.open(
                src / (name + ".tif"),
                "w",
                driver="GTiff",
                width=128,
                height=128,
                count=len(array),
                dtype=array.dtype,
                crs="EPSG:26985",
                transform=transform,
            ) as f:
                f.write(array)
        faces = []
        owners = []
        for oid, center in enumerate([-12, 12], 1):
            a, b, c, d = [
                (center - 4, -4),
                (center + 4, -4),
                (center + 4, 4),
                (center - 4, 4),
            ]
            roof = np.array([[*p, 18] for p in [a, b, c, d]])
            faces.extend([roof[[0, 1, 2]], roof[[0, 2, 3]]])
            owners.extend([oid, oid])
            for p, q in [(a, b), (b, c), (c, d), (d, a)]:
                wall = np.array([[*p, 10], [*q, 10], [*q, 18], [*p, 18]])
                faces.extend([wall[[0, 1, 2]], wall[[0, 2, 3]]])
                owners.extend([oid, oid])
        vertices = np.array(faces)
        lon, lat = inverse.transform(vertices[:, :, 0] + cx, vertices[:, :, 1] + cy)
        np.savez_compressed(
            src / "buildings-mesh.npz",
            lonlatz=np.stack([lon, lat, vertices[:, :, 2]], -1),
            owners=np.array(owners, np.uint64),
        )
        cls.city = {
            "id": "fixture",
            "origin": origin,
            "landmarks": [],
            "photographic": {
                "crs": 26985,
                "source_span_m": 128,
                "terrain_cell_m": 4,
                "camera": {
                    "width": 256,
                    "height": 256,
                    "pixels_per_metre": 3,
                    "azimuth_degrees": 225,
                    "elevation_degrees": 35.264389682754654,
                    "target_local_m": [0, 0, 10],
                },
                "osm_parts_centre": origin,
                "explicit_osm_parts": [],
                "rasters": {
                    "aerial": {"year": 2023, "native_metres": 1, "sample_metres": 1},
                    "dtm": {"year": 2024},
                },
            },
        }
        cls.recipe = cls.root / "fixture.json"
        cls.recipe.write_text(json.dumps(cls.city))
        with redirect_stdout(StringIO()):
            photo.main(["--city", str(cls.recipe), "--out", str(cls.out), "--offline"])
        cls.report = json.loads((cls.out / "manifest.json").read_text())
        cls.matrix = np.array(cls.report["camera"]["pixel_matrix"])
        cls.image = np.asarray(Image.open(cls.out / "source.png"))

    @classmethod
    def tearDownClass(cls):
        photo.ROOT = cls.previous_root
        cls.temp.cleanup()

    def pixel(self, world):
        x, y = self.matrix @ np.array([*world, 1])
        return self.image[int(y), int(x)]

    def test_roofs_sample_their_own_geographic_photo_pixels(self):
        np.testing.assert_array_equal(self.pixel([-12, 0, 18]), [30, 125, 150])
        np.testing.assert_array_equal(self.pixel([12, 0, 18]), [160, 90, 75])
        np.testing.assert_array_equal(self.pixel([-12, -15, 10]), [30, 125, 150])

    def test_vertical_wall_does_not_receive_the_overhead_photo(self):
        wall = self.pixel([-12, -4, 14])
        self.assertFalse(np.array_equal(wall, [30, 125, 150]))
        self.assertGreater(int(wall[0]), int(wall[2]))

    def test_optional_landmark_detail_requires_evidence(self):
        for spec in [
            {"kind": "columns", "count": 36},
            {"kind": "columns", "count": 36, "source_refs": ["https://example.org"]},
        ]:
            with self.assertRaises(ValueError):
                photo.radial_surface_detail(spec, [0, 0], 10, 2, 12)

    def test_surface_detail_does_not_mutate_geometry_or_base_colour(self):
        world = np.array([[10.0, 0.0, 16.0], [9.0, 1.0, 17.0], [8.0, 2.0, 18.0]])
        original = world.copy()
        colour = np.array([230, 228, 216], np.uint8)
        original_colour = colour.copy()
        for kind in ["columns", "ribs"]:
            shader = photo.radial_surface_detail(
                {
                    "kind": kind,
                    "count": 36,
                    "source_refs": ["https://example.org"],
                    "interpretation": "Synthetic callback fixture",
                },
                [0, 0],
                10,
                2,
                12,
            )
            result = shader(world, colour, None, None)
            self.assertEqual(result.shape, (3, 3))
            self.assertEqual(result.dtype, np.uint8)
            np.testing.assert_array_equal(world, original)
            np.testing.assert_array_equal(colour, original_colour)

    def test_camera_has_orthogonal_unit_axes(self):
        camera = self.report["camera"]
        basis = np.array([camera["horizontal"], camera["vertical"], camera["depth"]])
        np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)

    def test_changed_cached_imagery_is_rejected_before_render(self):
        path = self.out / "source/aerial.tif"
        original = path.read_bytes()
        before = (self.out / "source.png").read_bytes()
        try:
            path.write_bytes(original + b"changed")
            with (
                redirect_stdout(StringIO()),
                redirect_stderr(StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                photo.main(
                    ["--city", str(self.recipe), "--out", str(self.out), "--offline"]
                )
            self.assertEqual(error.exception.code, 2)
            self.assertEqual((self.out / "source.png").read_bytes(), before)
        finally:
            path.write_bytes(original)


class ExpandedPhotographicSource(unittest.TestCase):
    def test_aligned_provider_tiles_assemble_exact_native_pixels(self):
        with tempfile.TemporaryDirectory(prefix="photo-mosaic-") as folder:
            out=Path(folder)
            origin={"lat":38.8896,"lon":-77.0056}
            t=Transformer.from_crs(4326,26985,always_xy=True)
            cx,cy=t.transform(origin["lon"],origin["lat"])
            left,top=round(cx-512),round(cy+512)
            city={"origin":origin,"photographic":{"crs":26985,"source_span_m":1024,"rasters":{"aerial":{"url":"https://fixture.invalid/image","sample_metres":2,"max_export_pixels":256}}}}
            data,requests_seen,lock={},[],Lock()
            def get(url,params=None,timeout=None):
                if params is None:
                    return SimpleNamespace(content=data[url],raise_for_status=lambda:None)
                if url == 'https://fixture.invalid/image':
                    return SimpleNamespace(json=lambda:{'maxImageWidth':256,'maxImageHeight':256},raise_for_status=lambda:None)
                box=list(map(float,params["bbox"].split(',')));width,height=map(int,params['size'].split(','))
                self.assertLessEqual(max(width,height),256)
                xx=np.arange(width)+round((box[0]-left)/2)
                yy=np.arange(height)+round((top-box[3])/2)
                x,y=np.meshgrid(xx,yy);array=np.stack([x//2,y//2,(x+3*y)%251]).astype('uint8')
                with rasterio.MemoryFile() as memory:
                    with memory.open(driver='GTiff',width=width,height=height,count=3,dtype='uint8',crs='EPSG:26985',transform=from_origin(box[0],box[3],2,2)) as image:image.write(array)
                    content=memory.read()
                with lock:
                    href=f'https://fixture.invalid/{len(requests_seen)}.tif';requests_seen.append(params);data[href]=content
                return SimpleNamespace(json=lambda:{'href':href},raise_for_status=lambda:None)
            with patch.object(photo.requests,'get',side_effect=get),redirect_stdout(StringIO()):
                photo.acquire_rasters(city,out)
            self.assertEqual(len(requests_seen),4)
            with rasterio.open(out/'aerial.tif') as image:
                self.assertEqual(image.shape,(512,512));self.assertEqual(image.res,(2,2));actual=image.read()
            x,y=np.meshgrid(np.arange(512),np.arange(512))
            np.testing.assert_array_equal(actual,np.stack([x//2,y//2,(x+3*y)%251]).astype('uint8'))
            record=json.loads((out/'aerial-request.json').read_text());self.assertEqual(len(record['tiles']),4)
            with patch.object(photo.requests,'get',side_effect=AssertionError('cache unexpectedly downloaded')),redirect_stdout(StringIO()):
                photo.acquire_rasters(city,out)
            bad=json.loads(json.dumps(city));bad['photographic']['rasters']['aerial']['sample_metres']=3
            with self.assertRaisesRegex(ValueError,'align'):
                photo.acquire_rasters(bad,out)

    def test_mesh_configured_bound_stops_before_extra_requests(self):
        with tempfile.TemporaryDirectory(prefix='photo-node-bound-') as folder:
            out=Path(folder);(out/'i3s').mkdir()
            (out/'layer.json').write_text(json.dumps({'store':{'version':'1.6','defaultGeometrySchema':{'ordering':['position','normal','uv0','color']}}}))
            mbs=[-77.0056,38.8896,20,10]
            (out/'i3s/root.json').write_text(json.dumps({'mbs':mbs,'children':[{'id':'a','mbs':mbs},{'id':'b','mbs':mbs}]}))
            city={'origin':{'lat':38.8896,'lon':-77.0056},'photographic':{'source_span_m':3600,'i3s_layer':'https://fixture.invalid/layer','mesh_max_nodes':1}}
            with patch.object(photo.requests,'get',side_effect=AssertionError('node limit did not stop fetch')),redirect_stdout(StringIO()),self.assertRaisesRegex(ValueError,'bound'):
                photo.acquire_mesh(city,out)

    def test_linear_relief_requires_evidence_and_preserves_world_vertices(self):
        spec={'count':8,'length_metres':34,'along_axis':'y','source_refs':['https://fixture.invalid/architecture'],'interpretation':'Synthetic surface-only test'}
        with self.assertRaises(ValueError):photo.linear_surface_detail(dict(spec,source_refs=[]),[0,0],0,20)
        world=np.array([[0.,0.,8.],[0.,2.,8.],[0.,4.,8.]])
        original=world.copy();colour=np.array([220,219,207],dtype='uint8');before=colour.copy()
        shader=photo.linear_surface_detail(spec,[0,0],0,20);result=shader(world,colour,None,None)
        self.assertEqual(result.shape,(3,3));self.assertEqual(result.dtype,np.uint8)
        np.testing.assert_array_equal(world,original);np.testing.assert_array_equal(colour,before)


class LandmarkRoofGeometry(unittest.TestCase):
    def test_source_part_corrections_require_selected_geometry_and_cited_bounded_values(self):
        recipe={'photographic':{'explicit_osm_parts':['way/1'],
            'part_min_height':{'way/1':{'metres':24.0,'source_refs':['https://fixture.invalid/survey'],'interpretation':'Measured lower bound'}},
            'legacy_part_replacement':{'margin_metres':2,'source_refs':['https://fixture.invalid/survey'],'interpretation':'Measured old shell support'}}}
        photo.validate_part_options(recipe)
        for change in [
            {'part_min_height':{'way/2':recipe['photographic']['part_min_height']['way/1']}},
            {'part_min_height':{'way/1':{'metres':float('nan'),'source_refs':['https://fixture.invalid'],'interpretation':'Invalid'}}},
            {'legacy_part_replacement':{'margin_metres':100000,'source_refs':['https://fixture.invalid'],'interpretation':'Invalid'}},
            {'legacy_part_replacement':{'margin_metres':2,'source_refs':[],'interpretation':'Uncited'}},
        ]:
            with self.subTest(change=change),self.assertRaises(ValueError):
                photo.validate_part_options({'photographic':dict(recipe['photographic'],**change)})

    def test_dome_faces_cover_source_footprint_and_respect_height_profile(self):
        from geometry import Feature
        from roofs import radial_surface
        from shapely.geometry import Point, Polygon
        for shape,profile in [
            (Polygon([(-5,-4),(5,-4),(5,4),(-5,4)]),{}),
            (Point(0,0).buffer(5,quad_segs=16),{'contour':'circle','radius_m':5})]:
            feature=Feature('fixture/dome',shape,{},height=20,min_height=2)
            recipe=dict(profile,rings=[{'radius':1,'z':5},{'radius':.6,'z':14},{'radius':0,'z':20}])
            faces=radial_surface(feature,recipe)
            self.assertGreater(len(faces),0)
            self.assertAlmostEqual(sum(Polygon(face[:,:2]).area for face in faces),shape.area,places=8)
            vertices=np.concatenate(faces)
            self.assertTrue(np.isfinite(vertices).all())
            self.assertAlmostEqual(vertices[:,2].min(),5)
            self.assertAlmostEqual(vertices[:,2].max(),20)
            with self.assertRaisesRegex(ValueError,'exceeds'):
                radial_surface(feature,dict(profile,rings=[{'radius':1,'z':5},{'radius':0,'z':21}]))


class CatalogueSafeRasterAcquisition(unittest.TestCase):
    def test_source_count_splits_below_pixel_cap_and_assembles_exact_native_samples(self):
        service = 'https://fixture.invalid/terrain'
        recipe = {'origin': {'lat': 0, 'lon': 0}, 'photographic': {
            'crs': 3857, 'source_span_m': 64,
            'rasters': {'dtm': {'url': service, 'sample_metres': 1, 'max_export_pixels': 256}}}}
        expected = np.arange(64*64, dtype=np.float32).reshape(64,64) + 1000
        downloads, export_sizes = {}, []

        class Response:
            def __init__(self, payload=None, content=b''):
                self.payload, self.content = payload, content
            def raise_for_status(self): pass
            def json(self): return self.payload

        def request(url, params=None, timeout=None):
            if url == service:
                return Response({'maxMosaicImageCount': 1, 'maxImageWidth': 256, 'maxImageHeight': 256})
            if url.endswith('/query'):
                left,bottom,right,top = map(float,params['geometry'].split(','))
                count = int(np.ceil((right+32)/16)-np.floor((left+32)/16)) * int(np.ceil((top+32)/16)-np.floor((bottom+32)/16))
                return Response({'count': count})
            if url.endswith('/exportImage'):
                left,bottom,right,top = map(float,params['bbox'].split(','))
                width,height = map(int,params['size'].split(','))
                x,y = int(left+32),int(32-top)
                export_sizes.append((width,height))
                with rasterio.io.MemoryFile() as memory:
                    with memory.open(driver='GTiff',width=width,height=height,count=1,dtype='float32',crs=3857,transform=from_origin(left,top,1,1)) as image:
                        image.write(expected[y:y+height,x:x+width],1)
                    data = memory.read()
                href = f'memory://{x}-{y}'
                downloads[href] = data
                return Response({'href': href, 'width': width, 'height': height})
            return Response(content=downloads[url])

        with tempfile.TemporaryDirectory(prefix='catalogue-raster-') as folder:
            out = Path(folder)
            with patch.object(photo.requests,'get',side_effect=request), redirect_stdout(StringIO()):
                photo.acquire_rasters(recipe,out)
            with rasterio.open(out/'dtm.tif') as image:
                np.testing.assert_array_equal(image.read(1), expected)
                self.assertEqual(image.transform,from_origin(-32,32,1,1))
            self.assertEqual(len(export_sizes),16)
            self.assertTrue(all(w <= 16 and h <= 16 for w,h in export_sizes))
            photo.validate_raster_coverage(recipe,out)
            shifted = dict(recipe,origin={'lat':0,'lon':.001})
            with self.assertRaisesRegex(ValueError,'different geographic acquisition grid'):
                photo.validate_raster_coverage(shifted,out)
            proof_path=out/'dtm-coverage.json'
            proof=json.loads(proof_path.read_text())
            proof['exports'][0]['source_count']=2
            proof_path.write_text(json.dumps(proof))
            with self.assertRaisesRegex(ValueError,'mosaic-source limit'):
                photo.validate_raster_coverage(recipe,out)
            proof_path.rename(out/'retained-unsafe-proof.json')
            with self.assertRaisesRegex(ValueError,'fresh --out'):
                photo.validate_raster_coverage(recipe,out)

    def test_unavailable_catalogue_and_unbounded_plan_fail_before_export(self):
        def response(payload):
            return SimpleNamespace(raise_for_status=lambda:None,json=lambda:payload)
        for count in [None,0,True,100]:
            with self.subTest(count=count), patch.object(photo.requests,'get',side_effect=lambda url,**kwargs: response({'count':count}) if url.endswith('/query') else response({'maxMosaicImageCount':1})):
                with self.assertRaises(ValueError):
                    photo.raster_export_plan('https://fixture.invalid',[-32,-32,32,32],64,1,256,3857)
