"""Physical camera parity and migration, independent of matching old pictures."""
import sys
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import MultiPoint

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from geometry import Camera
from raster import Raster
from rectangle_frame import source_polygon_for_frame


class CameraTests(unittest.TestCase):
    def test_bases_follow_world_up_and_near_depth(self):
        for view in ['NE','SE']:
            camera=Camera(1,0,0,1,1,view_quadrant=view)
            forward=-camera.depth_vector/np.linalg.norm(camera.depth_vector)
            right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
            down=np.cross(forward,right)
            np.testing.assert_allclose(camera.horizontal_vector/np.linalg.norm(camera.horizontal_vector),right)
            np.testing.assert_allclose(camera.vertical_vector/np.linalg.norm(camera.vertical_vector),down)

    def test_real_cube_face_order_has_physical_parity(self):
        faces={
            'east':([[1,-1,0],[1,1,0],[1,1,2],[1,-1,2]],(200,50,50)),
            'north':([[-1,1,0],[1,1,0],[1,1,2],[-1,1,2]],(50,100,200)),
            'west':([[-1,-1,0],[-1,1,0],[-1,1,2],[-1,-1,2]],(50,150,50)),
            'south':([[-1,-1,0],[1,-1,0],[1,-1,2],[-1,-1,2]],(200,150,50)),
            'top':([[-1,-1,2],[1,-1,2],[1,1,2],[-1,1,2]],(180,180,180))}
        for view,visible,left,right in [('NE',{'east','north','top'},'east','north'),
                                       ('SE',{'east','south','top'},'south','east')]:
            images=[]
            for reverse in [False,True]:
                raster=Raster(Camera(30,80,100,160,150,1,view_quadrant=view))
                for xyz,color in list(faces.values())[::(-1 if reverse else 1)]:raster.quad(xyz,color)
                images.append(raster.rgb)
            np.testing.assert_array_equal(*images)
            positions={}
            for name,(_,color) in faces.items():
                ys,xs=np.where(np.all(images[0]==color,axis=2))
                if len(xs):positions[name]=xs.mean()
            self.assertEqual(set(positions),visible)
            self.assertLess(positions[left],positions[right])

    def test_reflected_legacy_frame_retains_world_coverage(self):
        legacy=[-240,-160,400,240]
        corrected=[-legacy[2],legacy[1],-legacy[0],legacy[3]]
        a=np.sqrt(3)/2
        legacy_matrix=np.array([[a,-a],[.5,.5]])
        corners=np.array([[legacy[0],legacy[1]],[legacy[2],legacy[1]],
                          [legacy[2],legacy[3]],[legacy[0],legacy[3]]])
        points=np.concatenate([(corners+np.array([0,z]))@np.linalg.inv(legacy_matrix).T for z in [-20,300]])
        expected=MultiPoint(points).convex_hull.buffer(120,join_style='mitre')
        actual=source_polygon_for_frame(corrected,'NE',[-20,300],120)
        self.assertLess(expected.symmetric_difference(actual).area,1e-7)


if __name__=='__main__':unittest.main()
