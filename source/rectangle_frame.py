"""Screen-space frames and conservative world context for an isometric camera."""
from __future__ import annotations

import math
import numpy as np
from geometry import Camera
from shapely.geometry import MultiPoint


def expand_aspect(bounds, aspect=(16,10)):
    left,top,right,bottom=map(float,bounds)
    width,height=right-left,bottom-top
    width=max(width,height*aspect[0]/aspect[1])
    height=width*aspect[1]/aspect[0]
    x,y=(left+right)/2,(top+bottom)/2
    return (x-width/2,y-height/2,x+width/2,y+height/2)


def rectangle_frame_bounds(city, scene, terrain=None):
    spec=city['camera']['frame']
    aspect=spec.get('aspect',[16,10])
    if 'screen_bounds_metres' in spec:
        return expand_aspect(spec['screen_bounds_metres'],aspect)
    wanted=set(spec.get('focus_landmarks',[p['id'] for p in city.get('landmarks',[])]))
    points=[]
    for place in city.get('landmarks',[]):
        if place['id'] not in wanted:
            continue
        x,y=scene.local_xy(place['lon'],place['lat'])
        z=place.get('elevation_m')
        if z is None:
            z=float(terrain.elevation(x,y)) if terrain is not None and 'missing' not in terrain.mode else spec.get('focus_elevation_metres',0)
        points.append([x,y,z])
    if not points:
        raise ValueError('Rectangle focus needs at least one selected landmark or explicit screen bounds')
    camera=Camera(1,0,0,0,0,scene=scene,view_quadrant=city['camera']['view_quadrant'])
    projected=camera.project(points)
    padding=spec.get('padding_metres',250)
    left,top=projected[:,:2].min(axis=0)-padding
    right,bottom=projected[:,:2].max(axis=0)+padding
    return expand_aspect((left,top,right,bottom),aspect)


def inverse_screen_xy(screen_xy, elevation, view_quadrant='NE'):
    camera=Camera(1,0,0,0,0,view_quadrant=view_quadrant)
    screen=np.asarray(screen_xy,dtype=float)
    level=np.broadcast_to(np.asarray(elevation,dtype=float),screen.shape[:-1])
    ground=screen+level[...,None]*np.array([0,1])
    matrix=np.stack([camera.horizontal_vector[:2],camera.vertical_vector[:2]])
    return ground@np.linalg.inv(matrix).T


def source_polygon_for_frame(bounds, view_quadrant='NE', elevation_range=(-20,300), context_margin=120):
    left,top,right,bottom=bounds
    corners=np.array([[left,top],[right,top],[right,bottom],[left,bottom]])
    points=np.concatenate([inverse_screen_xy(corners,z,view_quadrant) for z in elevation_range])
    return MultiPoint(points).convex_hull.buffer(context_margin,join_style='mitre')


def source_bbox_for_frame(bounds, scene, view_quadrant='NE', elevation_range=(-20,300), context_margin=120):
    x0,y0,x1,y1=source_polygon_for_frame(bounds,view_quadrant,elevation_range,context_margin).bounds
    west,south=scene.lonlat(x0,y0)
    east,north=scene.lonlat(x1,y1)
    return [float(south),float(west),float(north),float(east)]


def terrain_coverage(depth, strip_width=4):
    finite=np.isfinite(depth)
    height,width=finite.shape
    strip=min(strip_width,height,width)
    edges={'top':finite[:strip,:],'bottom':finite[-strip:,:],
           'left':finite[:,:strip],'right':finite[:,-strip:]}
    corners={'top_left':bool(finite[0,0]),'top_right':bool(finite[0,-1]),
             'bottom_left':bool(finite[-1,0]),'bottom_right':bool(finite[-1,-1])}
    count=int(np.count_nonzero(finite))
    return {'checked_before_overlays_and_attribution':True,'native_pixels':int(finite.size),
            'finite_depth_pixels':count,'unmapped_pixels':int(finite.size-count),
            'corners_mapped':corners,'edge_strip_width_pixels':strip,
            'edge_strips_unmapped':{name:int(edge.size-np.count_nonzero(edge)) for name,edge in edges.items()},
            'finite_depth_min':float(np.min(depth[finite])) if count else None,
            'finite_depth_max':float(np.max(depth[finite])) if count else None}


def rectangle_camera(bounds, scale=1.0, scene=None, view_quadrant='SE', aspect=(16,10)):
    left,top,right,bottom=expand_aspect(bounds,aspect)
    divisor=math.gcd(*aspect)
    unit_width,unit_height=aspect[0]//divisor,aspect[1]//divisor
    steps=math.ceil(max((right-left)*scale/unit_width,(bottom-top)*scale/unit_height))
    width,height=steps*unit_width,steps*unit_height
    centre_x,centre_y=(left+right)/2,(top+bottom)/2
    left,top=centre_x-width/(2*scale),centre_y-height/(2*scale)
    actual=(left,top,left+width/scale,top+height/scale)
    return Camera(scale,-left*scale,-top*scale,width,height,1,scene,view_quadrant,
                  {'mode':'rectangle','aspect':list(aspect),'screen_bounds_metres':list(actual)})
