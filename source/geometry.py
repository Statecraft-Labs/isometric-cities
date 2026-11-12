"""OSM geometry, height provenance, elevation, and the shared camera model."""
from __future__ import annotations

import gzip
import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from shapely import make_valid, constrained_delaunay_triangles
from shapely.geometry import LineString, Polygon, box
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree

METRES_LAT = 111_320.0


@dataclass(frozen=True)
class SceneCoordinates:
    lat: float
    lon: float

    def __post_init__(self):
        if not (math.isfinite(self.lat) and math.isfinite(self.lon) and -85 <= self.lat <= 85 and -180 <= self.lon <= 180):
            raise ValueError("Scene origin needs finite geographic lat/lon")

    @property
    def metres_lon(self):
        return METRES_LAT * math.cos(math.radians(self.lat))

    def local_xy(self, lon, lat):
        return (np.asarray(lon)-self.lon)*self.metres_lon, (np.asarray(lat)-self.lat)*METRES_LAT

    def lonlat(self, x, y):
        return np.asarray(x)/self.metres_lon+self.lon, np.asarray(y)/METRES_LAT+self.lat

    def bounds_polygon(self, bbox):
        south, west, north, east = bbox
        return box(*self.local_xy(west, south), *self.local_xy(east, north))


def polygons(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [p for g in geometry.geoms for p in polygons(g)]
    return []


def triangles(polygon):
    return [np.asarray(t.exterior.coords)[:3, :2] for t in constrained_delaunay_triangles(polygon).geoms]


def parse_metres(value):
    if value is None:
        return None
    match = re.search(r"-?\d+(?:[.,]\d+)?", str(value))
    if not match:
        return None
    number = float(match.group().replace(",", "."))
    return number * 0.3048 if "ft" in str(value) or "'" in str(value) else number


@dataclass
class Feature:
    osm_id: str
    geometry: object
    tags: dict
    height: float = 0.0
    min_height: float = 0.0
    height_source: str = ""
    roof_height: float = 0.0
    roof_shape: str = "flat"
    roof_source: str = ""
    repaired: bool = False
    height_evidence: dict = field(default_factory=dict)


def building_height(feature):
    t = feature.tags
    tagged = parse_metres(t.get("height") or t.get("building:height"))
    levels = parse_metres(t.get("building:levels"))
    kind = t.get("building:part", t.get("building", "yes"))
    if tagged is not None and tagged > 0:
        height, source = tagged, "osm:height"
    elif levels and levels > 0:
        height, source = levels * 3.2 + 1.0, "inferred:building:levels*3.2m+1m"
    else:
        h = {"garage": 3.5, "garages": 3.5, "shed": 3.0, "greenhouse": 3.0,
             "terrace": 14.0, "house": 10.0, "residential": 15.0, "apartments": 17.0,
             "church": 20.0, "cathedral": 24.0, "tower": 22.0, "industrial": 9.0,
             "retail": 12.0, "commercial": 17.0, "office": 18.0, "train_station": 14.0}
        height = h.get(kind, 15.0 if feature.geometry.area > 80 else 7.5)
        source = "inferred:building-type-default"
    feature.height = height
    feature.height_source = source
    minimum = parse_metres(t.get("min_height"))
    if minimum is None:
        minimum = (parse_metres(t.get("building:min_level")) or 0) * 3.2
    feature.min_height = max(0, minimum)
    roof_tag = t.get("roof:shape")
    flat_types = {"industrial", "retail", "commercial", "train_station", "roof", "parking", "greenhouse"}
    feature.roof_shape = roof_tag or ("flat" if kind in flat_types or feature.geometry.area > 3500 else "gabled")
    rectangle = feature.geometry.minimum_rotated_rectangle
    coords = np.asarray(rectangle.exterior.coords)[:4]
    short = min(np.linalg.norm(coords[(i+1)%4] - coords[i]) for i in range(4))
    rh = parse_metres(t.get("roof:height"))
    feature.roof_height = rh if rh is not None else min(4.0, max(0.8, short * 0.22))
    if feature.roof_shape in {"flat", "terrace"}:
        feature.roof_height = 0
    feature.roof_source = "osm:roof:shape" if roof_tag else "inferred:roof-type-default"
    feature.roof_height = min(feature.roof_height, max(0,height-feature.min_height), height if rh is not None else height*0.35)
    return feature


def load_features(path, scene):
    local_xy = scene.local_xy
    data = json.loads(Path(path).read_text())
    elements = data["elements"]
    used = set()
    features = []
    repairs = []
    failures = []
    relation_members = set()
    for element in elements:
        if element["type"] == "relation" and element.get("tags", {}).get("type") == "multipolygon":
            relation_members.update(m["ref"] for m in element.get("members", []) if m["type"] == "way")
    for element in elements:
        t = element.get("tags", {})
        if not t:
            continue
        key = f'{element["type"]}/{element["id"]}'
        geom = None
        try:
            if element["type"] == "node":
                if "lon" in element and t.get("natural") == "tree":
                    from shapely.geometry import Point
                    geom = Point(*local_xy(element["lon"], element["lat"]))
                else:
                    continue
            elif element["type"] == "way":
                points = [local_xy(v["lon"], v["lat"]) for v in element.get("geometry", []) if v]
                if len(points) < 2:
                    continue
                closed = len(points) >= 4 and points[0] == points[-1]
                area = closed and ("building" in t or "building:part" in t or "landuse" in t or "leisure" in t or "water" in t or t.get("natural") in {"wood", "water", "grassland", "scrub", "bare_rock"} or t.get("area") == "yes" or t.get("historic") in {"castle", "monument"})
                if area:
                    if element["id"] in relation_members and ("building" in t or "landuse" in t or "leisure" in t):
                        continue
                    geom = Polygon(points)
                else:
                    geom = LineString(points)
            elif t.get("type") == "multipolygon":
                rings = {"outer": [], "inner": []}
                for member in element.get("members", []):
                    points = [local_xy(v["lon"], v["lat"]) for v in member.get("geometry", []) if v]
                    if len(points) > 1:
                        rings["inner" if member.get("role") == "inner" else "outer"].append(LineString(points))
                outer = unary_union(list(polygonize(unary_union(rings["outer"]))))
                inner = unary_union(list(polygonize(unary_union(rings["inner"]))))
                geom = outer.difference(inner)
            if geom is None or geom.is_empty:
                continue
            repaired = not geom.is_valid
            if repaired:
                geom = make_valid(geom)
                repairs.append(key)
            if key in used:
                continue
            used.add(key)
            feature = Feature(key, geom, t, repaired=repaired)
            if "building" in t or "building:part" in t or (t.get("historic") == "monument" and polygons(geom)):
                for i, polygon in enumerate(polygons(geom)):
                    if polygon.area > 0.25:
                        part = Feature(key if i == 0 else f"{key}#{i}", polygon, t, repaired=repaired)
                        features.append(building_height(part))
            else:
                features.append(feature)
        except (ValueError, TypeError) as error:
            failures.append({"osm_id": key, "reason": str(error)})
    return features, {"osm_elements": len(elements), "repaired_geometries": repairs, "parse_failures": failures}


class Terrain:
    def __init__(self, folder, scene, bbox, provider="skadi", flat_height=0):
        self.scene = scene
        self.flat_height = float(flat_height)
        self.tiles = []
        self.missing_samples = 0
        if provider == "skadi":
            required = set(hgt_tiles(bbox))
            for name in sorted(required):
                path = Path(folder)/f"{name}.hgt.gz"
                if not path.exists():
                    continue
                values = np.frombuffer(gzip.decompress(path.read_bytes()), dtype=">i2")
                size = math.isqrt(len(values))
                if size < 2 or size*size != len(values):
                    raise ValueError(f"Invalid HGT dimensions: {path}")
                lat = int(name[1:3])*(1 if name[0]=="N" else -1)
                lon = int(name[4:7])*(1 if name[3]=="E" else -1)
                self.tiles.append((name,lat,lon,values.reshape(size,size).astype(np.float32)))
            self.missing_tiles = sorted(required-{t[0] for t in self.tiles})
            self.mode = "open-dem" if self.tiles and not self.missing_tiles else "partial-missing-dem" if self.tiles else "flat-missing-dem"
            self.description = "Mapzen Skadi HGT tiles, bilinear sampling, no vertical exaggeration. Mixed-source DEM; buildings and canopy can bias elevations."
            if self.missing_tiles:
                self.description += f" Missing tiles {', '.join(self.missing_tiles)} use visible flat fallback at {flat_height:g} m."
            self.attribution = "Mapzen Terrain Tiles. USGS SRTM/GMTED2010, Copernicus EU-DEM, UK © Environment Agency 2015; exact per-pixel source not identified."
        elif provider == "flat":
            self.missing_tiles = []
            self.mode = "flat-configured"
            self.description = f"Configured flat terrain at {flat_height:g} m. No measured terrain."
            self.attribution = "No elevation dataset."
        else:
            raise ValueError(f"Unknown terrain provider: {provider}")

    def elevation(self, x, y):
        x,y = np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(y,dtype=float))
        lon,lat = self.scene.lonlat(x,y)
        result = np.full(x.shape,np.nan)
        for _,south,west,dem in self.tiles:
            mask = (lon>=west-1e-10)&(lon<=west+1+1e-10)&(lat>=south-1e-10)&(lat<=south+1+1e-10)&~np.isfinite(result)
            if not np.any(mask):
                continue
            size = dem.shape[0]
            px = np.clip((lon[mask]-west)*(size-1),0,size-1-1e-8)
            py = np.clip((south+1-lat[mask])*(size-1),0,size-1-1e-8)
            ix,iy = px.astype(int),py.astype(int)
            fx,fy = px-ix,py-iy
            samples = np.stack([dem[iy,ix],dem[iy,ix+1],dem[iy+1,ix],dem[iy+1,ix+1]])
            weights = np.stack([(1-fx)*(1-fy),fx*(1-fy),(1-fx)*fy,fx*fy])
            valid = samples != -32768
            weight = np.sum(weights*valid,axis=0)
            result[mask] = np.divide(np.sum(np.where(valid,samples,0)*weights,axis=0),weight,out=np.full_like(weight,np.nan),where=weight>0)
        missing = ~np.isfinite(result)
        self.missing_samples += int(np.count_nonzero(missing))
        return np.where(missing,self.flat_height,result)


@dataclass
class Camera:
    scale: float
    offset_x: float
    offset_y: float
    width: int
    height: int
    upscale: int = 2
    scene: SceneCoordinates | None = None
    view_quadrant: str = 'SE'
    frame: dict = field(default_factory=lambda: {'mode':'fit'})

    def __post_init__(self):
        if self.view_quadrant not in {'NE', 'SE'} or not np.isfinite([self.scale,self.offset_x,self.offset_y]).all() or self.scale <= 0:
            raise ValueError('Camera needs a positive finite scale and NE or SE view')

    @property
    def horizontal_vector(self):
        a=math.sqrt(3)/2
        return np.array([-a,a,0]) if self.view_quadrant=='NE' else np.array([a,a,0])

    @property
    def vertical_vector(self):
        return np.array([.5, .5 if self.view_quadrant=='NE' else -.5, -1])

    @property
    def depth_vector(self):
        return np.array([1,1 if self.view_quadrant=='NE' else -1,1])

    def prism_visible(self, bounds, low, high, margin_pixels=0):
        x0,y0,x1,y1=bounds
        points=np.array([[x,y,z] for x in (x0,x1) for y in (y0,y1) for z in (low,high)])
        screen=self.project(points)[:,:2]
        return bool(screen[:,0].max()>=-margin_pixels and screen[:,0].min()<self.width+margin_pixels
                    and screen[:,1].max()>=-margin_pixels and screen[:,1].min()<self.height+margin_pixels)

    def project(self, xyz):
        a = np.asarray(xyz)
        return np.stack((a@self.horizontal_vector*self.scale+self.offset_x,
                         a@self.vertical_vector*self.scale+self.offset_y,
                         a@self.depth_vector),axis=-1)

    def metadata(self):
        s = self.scale*self.upscale
        return {"type": "orthographic-isometric", "width": self.width*self.upscale, "height": self.height*self.upscale, "world_origin": {"lat": self.scene.lat, "lon": self.scene.lon} if self.scene else None,
                "world_units": "metres east, north, elevation", "metres_per_degree": {"longitude": self.scene.metres_lon, "latitude": METRES_LAT} if self.scene else None,
                "pixel_matrix": [[*(self.horizontal_vector*s),self.offset_x*self.upscale],[*(self.vertical_vector*s),self.offset_y*self.upscale]],
                "depth_vector": self.depth_vector.tolist(), "view_quadrant": self.view_quadrant, "native_pixels_per_metre": self.scale, "nearest_neighbour_upscale": self.upscale,
                "frame":self.frame,"vertical_exaggeration": 1.0,
                "projection_note": "Local equirectangular metric approximation about the fixed configured city origin. Shared by every feature and marker."}


def reconcile_parts(buildings):
    parts=[b for b in buildings if "building:part" in b.tags]
    tree=STRtree([p.geometry for p in parts])
    rendered=[]
    suppressed=[]
    residuals=[]
    for building in buildings:
        if "building:part" in building.tags:
            rendered.append(building)
            continue
        candidates=[parts[i] for i in tree.query(building.geometry,predicate="intersects")]
        candidates=[p for p in candidates if p.geometry.intersection(building.geometry).area/max(.01,p.geometry.area)>.8]
        if not candidates:
            rendered.append(building)
            continue
        union=unary_union([p.geometry for p in candidates])
        coverage=union.intersection(building.geometry).area/building.geometry.area
        if coverage>.98:
            suppressed.append(building.osm_id)
        elif coverage>.6:
            residual=building.geometry.difference(union)
            for i,polygon in enumerate(polygons(residual)):
                if polygon.area>1:
                    height=building.height if building.height_source=="osm:height" else min(building.height,float(np.median([p.height for p in candidates])))
                    rendered.append(replace(building,osm_id=f"{building.osm_id}@residual{i}",geometry=polygon,height=height,roof_height=0,roof_shape="flat",height_source=building.height_source if height==building.height else "inferred:residual-outline-part-median"))
            residuals.append(building.osm_id)
        else:
            rendered.append(building)
    return rendered,{"suppressed_parent_outlines":suppressed,"parents_rendered_as_residuals":residuals}


def hgt_tiles(bbox):
    south, west, north, east = bbox
    for lat in range(math.floor(south), math.ceil(north)):
        for lon in range(math.floor(west), math.ceil(east)):
            yield f'{"N" if lat>=0 else "S"}{abs(lat):02d}{"E" if lon>=0 else "W"}{abs(lon):03d}'
