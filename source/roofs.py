"""Geometric roof surfaces inside their source footprint. No facade or art dependencies."""
import numpy as np
from shapely import orient_polygons
from shapely.affinity import scale
from shapely.geometry import Point, Polygon
from shapely.ops import polygonize, unary_union
from geometry import triangles

def _ring(shape, radius, centre, profile):
    if radius <= 1e-9:
        return Point(centre)
    if profile.get('contour') == 'circle':
        return Point(centre).buffer(profile['radius_m']*radius,quad_segs=16)
    return scale(Polygon(shape.exterior), radius, radius, origin=tuple(centre))

def radial_surface(feature, profile):
    shape = orient_polygons(feature.geometry)
    if shape.geom_type != 'Polygon':
        raise ValueError('A radial roof needs one polygon')
    rings = profile['rings']
    if len(rings) < 2 or abs(rings[0]['radius']-1) > 1e-9 or rings[-1]['radius'] != 0:
        raise ValueError('A radial profile must run from the original boundary to its centre')
    radii = np.array([r['radius'] for r in rings], float)
    heights = np.array([r['z'] for r in rings], float)
    if not np.all(np.isfinite([radii, heights])) or np.any(np.diff(radii) > 0) or np.any(np.diff(heights) < 0):
        raise ValueError('Radial profile radii must decrease while heights increase')
    if heights.min() < feature.min_height or heights.max() > feature.height+1e-8:
        raise ValueError(f'Roof profile exceeds selected total for {feature.osm_id}')
    centre = np.array(profile.get('centre_xy', [shape.centroid.x, shape.centroid.y]), float)
    if not shape.contains(Point(centre)):
        raise ValueError('Radial profile centre must be inside its source footprint')
    circular = profile.get('contour') == 'circle'
    faces = []
    ring_shapes = []
    for radius in radii:
        ring = Polygon(shape.exterior) if not ring_shapes else _ring(shape, radius, centre, profile)
        if ring_shapes and ring.geom_type != 'Point':
            ring = ring.intersection(ring_shapes[-1])
        ring_shapes.append(ring)
    partitions = list(polygonize(unary_union([shape.boundary, *[ring.boundary for ring in ring_shapes if ring.geom_type != 'Point']])))
    for index, (outer, inner, low, high) in enumerate(zip(ring_shapes[:-1], ring_shapes[1:], heights[:-1], heights[1:])):
        if radii[index] == radii[index+1]:
            continue
        band = shape.intersection(outer.difference(inner))
        for part in partitions:
            if not band.covers(part.representative_point()):
                continue
            for tri in triangles(part):
                pieces = [tri]
                if inner.geom_type == 'Point' and Polygon(tri).covers(inner):
                    pieces = [np.array([a,b,centre]) for a,b in zip(tri,np.roll(tri,-1,axis=0))
                              if Polygon([a,b,centre]).area > 1e-12]
                for piece in pieces:
                    d_outer = np.array([outer.boundary.distance(Point(p)) for p in piece])
                    d_inner = np.array([inner.boundary.distance(Point(p)) if inner.geom_type != 'Point' else inner.distance(Point(p)) for p in piece])
                    if circular:
                        distances = np.linalg.norm(piece-centre,axis=1)/profile['radius_m']
                        z = np.interp(distances,[radii[index+1],radii[index]],[high,low])
                    else:
                        z = low+(high-low)*d_outer/np.maximum(d_outer+d_inner, 1e-12)
                    if inner.geom_type == 'Point':
                        z[d_inner < 1e-7] = high
                    faces.append(np.column_stack([piece, z]))
    return faces
