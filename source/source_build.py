"""Acquire and render one registered source; independent of styling and the main CLI."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageColor
from shapely import orient_polygons
from shapely.geometry import LineString, Point, mapping
from shapely.ops import split

from geometry import Camera, SceneCoordinates, Terrain, hgt_tiles, load_features, parse_metres, polygons, reconcile_parts, triangles
from project import ROOT, digest, interpreter, ownership, read, recipe_path, write
from raster import Raster, tint
from rectangle_frame import rectangle_camera, rectangle_frame_bounds, source_polygon_for_frame, terrain_coverage


def _fetch(city, offline):
    from fetch_osm import fetch
    return fetch(city,offline=offline)


def _input_hashes(city, cache):
    paths={cache/'osm.json'}
    if city['terrain']['provider']=='skadi':
        paths.update(cache/f'{name}.hgt.gz' for name in hgt_tiles(city['bbox']))
    for value in [city.get('osm_source'),*city.get('dem_sources',[])]:
        if value:
            path=Path(value).expanduser()
            paths.add(path if path.is_absolute() else ROOT/path)
    return {str(path.absolute()):digest(path) if path.is_file() else None for path in sorted(paths)}


def _colour(tags, key, default):
    try:
        return ImageColor.getrgb(tags.get(key,''))[:3]
    except (ValueError,AttributeError):
        return default


def _roof(feature, base):
    shape=orient_polygons(feature.geometry)
    rise=feature.roof_height if feature.roof_shape in {'gabled','hipped'} else 0
    if not rise:
        return [shape],lambda pts:np.full(len(pts),base+feature.height)
    corners=np.array(shape.minimum_rotated_rectangle.exterior.coords)[:4]
    edges=np.roll(corners,-1,axis=0)-corners
    axis=edges[np.argmax(np.linalg.norm(edges,axis=1))];axis/=np.linalg.norm(axis)
    cross=np.array([-axis[1],axis[0]]);centre=np.array(shape.centroid.coords[0])
    extent=(corners-centre)@cross;middle=(extent.min()+extent.max())/2
    centre+=cross*middle;half=(extent.max()-extent.min())/2
    ridge=LineString([centre-axis*1e6,centre+axis*1e6])
    pieces=list(split(shape,ridge).geoms)
    def height(points):
        distance=np.abs((np.asarray(points)-centre)@cross)
        return base+feature.height-rise+rise*np.clip(1-distance/max(half,1e-9),0,1)
    return pieces,height


def _building(raster, terrain, feature):
    base=float(terrain.elevation(feature.geometry.centroid.x,feature.geometry.centroid.y))
    wall=_colour(feature.tags,'building:colour',(174,166,150))
    material=feature.tags.get('roof:material','')
    defaults={'copper':(106,148,135),'glass':(137,165,174),'metal':(158,163,164),'grass':(100,126,72)}
    roof=_colour(feature.tags,'roof:colour',defaults.get(material,(111,116,125)))
    pieces,height=_roof(feature,base)
    for piece in pieces:
        for ring in [piece.exterior,*piece.interiors]:
            coords=np.array(ring.coords)[:,:2]
            for a,b in zip(coords[:-1],coords[1:]):
                if feature.geometry.boundary.distance(Point((a+b)/2))>1e-7:continue
                normal=np.array([b[1]-a[1],a[0]-b[0]]);length=np.linalg.norm(normal)
                if length<1e-9:continue
                normal/=length;shade=.73+.18*float(normal@np.array([-.45,-.89]))
                z=height([a,b]);low=base+feature.min_height
                raster.quad([[*a,low],[*b,low],[*b,z[1]],[*a,z[0]]],tint(wall,shade))
        for tri in triangles(piece):
            raster.triangle(np.column_stack([tri,height(tri)]),roof,bias=.03)
    return {'osm_id':feature.osm_id,'height_m':feature.height,'min_height_m':feature.min_height,
            'height_source':feature.height_source,'roof_shape':feature.roof_shape,'roof_source':feature.roof_source,
            'geometry':mapping(feature.geometry),'material_tags':{k:v for k,v in feature.tags.items() if k.startswith(('roof:','building:'))}}


def _validate(city):
    scene=SceneCoordinates(**city['origin'])
    if not isinstance(city.get('id'),str) or not isinstance(city.get('title'),str):raise ValueError('Source recipe needs an id and title')
    bbox=city['bbox']
    if len(bbox)!=4 or not np.isfinite(bbox).all() or not (-85<=bbox[0]<bbox[2]<=85 and -180<=bbox[1]<bbox[3]<=180):
        raise ValueError('Source bbox requires finite [south,west,north,east] within geographic bounds')
    if bbox[2]-bbox[0]>.25 or bbox[3]-bbox[1]>.5:
        raise ValueError('OSM starter is bounded to one neighbourhood; provide a compact bbox')
    camera=city.get('camera',{})
    if camera.get('view_quadrant','SE') not in {'NE','SE'}:
        raise ValueError('Source view must be NE or SE')
    scale=camera.get('pixels_per_metre',2)
    if not isinstance(scale,(int,float)) or not math.isfinite(scale) or not .1<=scale<=10:
        raise ValueError('Source pixels_per_metre must be .1..10')
    terrain=city.get('terrain',{'provider':'skadi'})
    if not math.isfinite(terrain.get('flat_height_m',0)):raise ValueError('Flat elevation must be finite')
    if terrain.get('provider') not in {'skadi','flat'}:
        raise ValueError('The OSM starter supports Skadi or configured flat terrain; use a surface recipe for DSM data')
    return scene


def osm_source(city,out,*,offline=False):
    city=deepcopy(city);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    scene=_validate(city);camera_spec=city.get('camera',{});view=camera_spec.get('view_quadrant','SE')
    city.setdefault('terrain',{'provider':'skadi'});cache=Path(city.get('data_dir',out/'cache'))
    if not cache.is_absolute():cache=ROOT/cache
    city['data_dir']=str(cache)
    frozen=out/'source-inputs.json'
    if frozen.exists():
        for name,expected in read(frozen).items():
            path=Path(name)
            if (digest(path) if path.is_file() else None)!=expected:
                raise ValueError(f'Source input changed: {path}. Use a new output directory')
    provenance=_fetch(city,offline)
    inputs=_input_hashes(city,cache)
    if frozen.exists() and read(frozen)!=inputs:
        raise ValueError('Source input changed during acquisition. Use a new output directory')
    if not frozen.exists():write(frozen,inputs)
    focus=city.get('landmarks',[{'lat':scene.lat,'lon':scene.lon}])[0] if city.get('landmarks') else {'lat':scene.lat,'lon':scene.lon}
    frame=camera_spec.get('frame',{})
    fallback=city['terrain'].get('flat_height_m',focus.get('elevation_m',frame.get('focus_elevation_metres',0)))
    terrain=Terrain(cache,scene,city['bbox'],provider=city['terrain']['provider'],flat_height=fallback)
    features,parsed=load_features(cache/'osm.json',scene)
    buildings=[f for f in features if f.height>0]
    buildings,parts=reconcile_parts(buildings)
    fx,fy=scene.local_xy(focus['lon'],focus['lat'])
    z=focus.get('elevation_m',frame.get('focus_elevation_metres'))
    if z is None:z=float(terrain.elevation(fx,fy))
    if not math.isfinite(z):raise ValueError('Source focus elevation must be finite')
    centre=Camera(1,0,0,0,0,scene=scene,view_quadrant=view).project([fx,fy,z])
    bounds=rectangle_frame_bounds(city,scene,terrain) if frame else [centre[0]-256,centre[1]-160,centre[0]+256,centre[1]+160]
    aspect=frame.get('aspect',[16,10]);scale=camera_spec.get('pixels_per_metre',2)
    if len(bounds)!=4 or not np.isfinite(bounds).all() or not(bounds[0]<bounds[2] and bounds[1]<bounds[3]):raise ValueError('Source screen bounds must be finite increasing values')
    if len(aspect)!=2 or any(type(v)is not int or v<=0 for v in aspect):raise ValueError('Source aspect requires two positive integers')
    camera=rectangle_camera(bounds,scale=scale,scene=scene,view_quadrant=view,aspect=aspect)
    if max(camera.width,camera.height)>8192 or camera.width*camera.height>32_000_000:
        raise ValueError('Source camera exceeds 8192 pixels per side or 32 megapixels')
    raster=Raster(camera,palette={'background_rgb':(219,220,209)})
    envelope=frame.get('elevation_range_metres',[z-250,z+300])
    if len(envelope)!=2 or not np.isfinite(envelope).all() or envelope[0]>=envelope[1]:raise ValueError('Elevation envelope must be finite increasing values')
    support=source_polygon_for_frame(camera.frame['screen_bounds_metres'],view,envelope,context_margin=40)
    x0,y0,x1,y1=support.bounds;cell=max(float(camera_spec.get('terrain_cell_metres',8)),max(x1-x0,y1-y0)/128)
    if not math.isfinite(cell) or cell<=0:raise ValueError('Terrain mesh cell must be finite and positive')
    xs=np.linspace(x0,x1,max(2,math.ceil((x1-x0)/cell)+1));ys=np.linspace(y0,y1,max(2,math.ceil((y1-y0)/cell)+1))
    xx,yy=np.meshgrid(xs,ys);zz=terrain.elevation(xx,yy);vertices=np.stack([xx,yy,zz],-1)
    for row in range(len(ys)-1):
        for col in range(len(xs)-1):
            quad=vertices[row:row+2,col:col+2].reshape(4,3)[[0,1,3,2]]
            if not camera.prism_visible((quad[:,0].min(),quad[:,1].min(),quad[:,0].max(),quad[:,1].max()),quad[:,2].min(),quad[:,2].max()):continue
            raster.quad(quad,(190,193,170))
    ground_coverage=terrain_coverage(raster.depth)
    for f in features:
        shape=f.geometry
        if f.height>0:continue
        if f.tags.get('highway'):
            width=parse_metres(f.tags.get('width')) or (2 if f.tags['highway'] in {'footway','path','steps'} else 8)
            shape=shape.buffer(min(30,max(1,width))/2);colour=(119,121,120)
        elif f.tags.get('natural')=='water' or 'water' in f.tags:colour=(91,134,153)
        elif 'leisure' in f.tags or f.tags.get('landuse') in {'grass','forest','meadow'} or f.tags.get('natural') in {'wood','grassland','scrub'}:colour=(120,146,91)
        else:continue
        for polygon in polygons(shape.intersection(support)):
            for tri in triangles(polygon):raster.triangle(np.column_stack([tri,terrain.elevation(tri[:,0],tri[:,1])+.2]),colour,bias=.05)
    records=[_building(raster,terrain,b) for b in buildings if camera.prism_visible(b.geometry.bounds,z-300,z+b.height+300)]
    image=out/'source.png';Image.fromarray(raster.rgb).save(image)
    landmarks=[]
    for point in city.get('landmarks',[]):
        x,y=scene.local_xy(point['lon'],point['lat']);height=point.get('elevation_m',float(terrain.elevation(x,y)));px,py,_=camera.project([x,y,height])
        if 0<=px<camera.width and 0<=py<camera.height:landmarks.append(dict(point,x=float(px),y=float(py)))
    credits='© OpenStreetMap contributors, ODbL. '+terrain.attribution
    manifest={'city_id':city['id'],'title':city['title'],'width':camera.width,'height':camera.height,'source_sha256':digest(image),
              'camera':camera.metadata(),'landmarks':landmarks,'source_quality':'OSM footprint/height source with terrain and material-tag fallbacks; no photographic coverage',
              'terrain':{'mode':terrain.mode,'description':terrain.description,'attribution':terrain.attribution,'missing_tiles':terrain.missing_tiles,'missing_samples':terrain.missing_samples},
              'focus_elevation_m':z,'geometry':{'building_components':len(records),'courtyard_holes':sum(len(b.geometry.interiors) for b in buildings),'parts':parts,'parser':parsed},
              'terrain_coverage':ground_coverage,'credits':credits,'attribution':credits,'acquisition':provenance,
              'limitations':['Facades are untextured. Untagged heights and roofs are recorded defaults.','Roads are draped to terrain; bridges and complex roof forms remain approximations.']}
    write(out/'source.json',manifest);write(out/'footprints.json',{'coordinates':'local metres east/north','attribution':'© OpenStreetMap contributors, ODbL','features':records})
    return {'image':image,'manifest':out/'source.json','quality':manifest['source_quality']}


def build_source(city,out,*,offline=False):
    out=Path(out).resolve()
    with ownership(out):
        recipe=out/'recipe.json'
        if recipe.exists() and read(recipe)!=city:raise ValueError('Source recipe changed; use a new output directory')
        write(recipe,city)
        kind='photo' if city.get('photographic') else 'surface' if city.get('surface') else 'osm'
        if kind=='osm':
            result=osm_source(city,out,offline=offline)
            result['command']=[str(interpreter()),str(Path(__file__).resolve()),'--city',str(recipe),'--out',str(out),*(['--offline'] if offline else [])]
            return result
        script=ROOT/'source'/('photographic_patch.py' if kind=='photo' else 'surface_patch.py')
        command=[str(interpreter()),str(script),'--city',str(recipe),'--out',str(out),*(['--offline'] if offline else [])]
        with (out/'source.log').open('a') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        return {'image':out/'source.png','manifest':out/('manifest.json' if kind=='photo' else 'source.json'),'command':command,
                'quality':'Registered photographic mesh source' if kind=='photo' else 'Measured DSM/DTM source with OSM material evidence'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--city',required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--offline',action='store_true');args=parser.parse_args()
    print(json.dumps(build_source(read(recipe_path(args.city)),args.out,offline=args.offline),default=str,indent=2))


if __name__=='__main__':main()
