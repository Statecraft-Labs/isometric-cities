"""Optional source-bound landmark meshes for a measured surface render.

Landmark data is authored/measured once, input-hash bound, and independent of city.
Plain structural source geometry only; fine ornament remains illustrative.
"""

from pathlib import Path
import json, math, hashlib
import numpy as np
from scipy import ndimage as ndi
from shapely import contains_xy, orient_polygons
from shapely.geometry import shape, LineString
from shapely.ops import split
from geometry import Feature, triangles, polygons
from raster import Raster, tint
from roofs import radial_surface
import surface_patch as surface

LIGHT = np.array([-0.62, -0.30, 0.72])
LIGHT /= np.linalg.norm(LIGHT)


class Camera:
    def __init__(self, camera):
        self.c = camera
        self.width = round(np.diff(camera["u_range_m"])[0] * camera["pixels_per_metre"])
        self.height = round(
            np.diff(camera["y_range_screen_m"])[0] * camera["pixels_per_metre"]
        )

    def project(self, xyz):
        a = np.asarray(xyz, float)
        c = self.c
        east = a[..., 0] - c["centre_projected_m"][0]
        north = a[..., 1] - c["centre_projected_m"][1]
        u, v = surface.horizontal(east, north, c["bearing_deg"])
        se = (
            0.5
            if c["elevation_deg"] == 30
            else math.sin(math.radians(c["elevation_deg"]))
        )
        ce = (
            math.sqrt(3) / 2
            if c["elevation_deg"] == 30
            else math.cos(math.radians(c["elevation_deg"]))
        )
        s = c["pixels_per_metre"]
        return np.stack(
            (
                (u - c["u_range_m"][0]) * s,
                (-(se * v + ce * (a[..., 2] - c["z0_m"])) - c["y_range_screen_m"][0])
                * s,
                -(ce * v - se * a[..., 2]),
            ),
            axis=-1,
        )


def face(r, xyz, colour):
    p = np.asarray(xyz, float)
    normal = np.cross(p[1] - p[0], p[2] - p[0])
    normal /= max(np.linalg.norm(normal), 1e-9)
    if normal[2] < 0:
        normal = -normal
    shade = 0.40 + 0.75 * max(0, float(normal @ LIGHT))
    r.triangle(p, tint(colour, shade), bias=0.015)


def quad(r, xyz, colour):
    for ids in ([0, 1, 2], [0, 2, 3]):
        face(r, np.asarray(xyz)[ids], colour)


def beam(r, a, b, radius, colour, sides=6):
    a, b = np.array(a, float), np.array(b, float)
    direction = b - a
    direction /= max(np.linalg.norm(direction), 1e-9)
    axis = np.cross(direction, [0, 0, 1])
    if np.linalg.norm(axis) < 0.01:
        axis = np.array([1.0, 0, 0])
    axis /= np.linalg.norm(axis)
    other = np.cross(direction, axis)
    dirs = (
        np.array(
            [
                math.cos(t) * axis + math.sin(t) * other
                for t in np.arange(sides) * 2 * math.pi / sides
            ]
        )
        * radius
    )
    for i in range(sides):
        j = (i + 1) % sides
        quad(r, [a + dirs[i], a + dirs[j], b + dirs[j], b + dirs[i]], colour)


def render_building(r, f, record, DTM, W, N, RES):
    p = f.geometry
    z = record
    top, eave = z["top_m"], z["eave_m"]
    roofcolour = surface.roof_from_tags(f.tags) or (112, 116, 126)
    wall = surface.wall_from_tags(f.tags) or (150, 140, 120)
    kind = z["shape"]
    centre = np.array(p.centroid.coords[0])
    rect = np.array(p.minimum_rotated_rectangle.exterior.coords)[:4]
    edges = np.roll(rect, -1, axis=0) - rect
    axis = edges[np.argmax(np.linalg.norm(edges, axis=1))]
    axis /= np.linalg.norm(axis)
    if f.tags.get("roof:orientation") == "across":
        axis = np.array([-axis[1], axis[0]])
    cross = np.array([-axis[1], axis[0]])
    extent = (rect - centre) @ cross
    centre += cross * (extent.min() + extent.max()) / 2
    half = (extent.max() - extent.min()) / 2

    def roofheight(points):
        points = np.asarray(points)
        return (
            np.full(len(points), top)
            if kind in ("flat", "terrace")
            else eave
            + (top - eave)
            * np.clip(1 - np.abs((points - centre) @ cross) / max(half, 0.1), 0, 1)
        )

    if kind in ("hipped", "mansard", "dome", "pyramidal"):
        fake = Feature(f.osm_id, p, f.tags, height=top - eave, roof_height=top - eave)
        plateau = 0.40 if kind == "mansard" else 0
        if kind == "dome":
            rings = [
                {"radius": math.cos(t), "z": math.sin(t) * (top - eave)}
                for t in np.linspace(0, math.pi / 2, 9)
            ]
            rings[-1]["radius"] = 0
        else:
            rings = [{"radius": 1, "z": 0}, {"radius": plateau, "z": top - eave}] + (
                [{"radius": 0, "z": top - eave}] if plateau else []
            )
        roofs = [
            tri + np.array([0, 0, eave])
            for tri in radial_surface(fake, {"rings": rings})
        ]

        def wallheight(points):
            return np.full(len(points), eave)

    else:
        ridge = LineString([centre - axis * 1e5, centre + axis * 1e5])
        pieces = list(split(p, ridge).geoms) if kind not in ("flat", "terrace") else [p]
        roofs = [
            np.column_stack((tri, roofheight(tri)))
            for piece in pieces
            for tri in triangles(piece)
        ]
        wallheight = roofheight
    for ring in [p.exterior, *p.interiors]:
        coords = np.array(ring.coords)[:, :2]
        for a, b in zip(coords[:-1], coords[1:]):
            zedge = wallheight([a, b])
            j = np.clip(
                ((np.array([a, b])[:, 0] - W) / RES).astype(int), 0, DTM.shape[1] - 1
            )
            i = np.clip(
                ((N - np.array([a, b])[:, 1]) / RES).astype(int), 0, DTM.shape[0] - 1
            )
            low = DTM[i, j]
            quad(r, [[*a, low[0]], [*b, low[1]], [*b, zedge[1]], [*a, zedge[0]]], wall)
    for tri in roofs:
        face(r, tri, roofcolour)
    return roofs


def crown(r, record):
    p = shape(record["geometry"])
    base = record["base_m"]
    top = record["top_m"]
    eave = record["crown_base_m"]
    rise = top - eave
    detail = record.get("drawing", {})
    stone = tuple(detail.get("body_rgb", (116, 111, 100)))
    rib_stone = tuple(detail.get("rib_rgb", stone))
    radius = detail.get("rib_radius_m", 0.22)
    corner = np.array(orient_polygons(p.minimum_rotated_rectangle).exterior.coords)[:4]
    centre = np.array(p.centroid.coords[0])
    body_corner = centre + (corner - centre) * 0.94
    corner = centre + (corner - centre) * detail.get("anchor_scale", 0.94)
    for a, b in zip(body_corner, np.roll(body_corner, -1, axis=0)):
        quad(r, [[*a, base], [*b, base], [*b, eave], [*a, eave]], stone)
    join = eave + rise * detail.get("join_fraction", 0.48)
    anchors = np.vstack((corner, (corner + np.roll(corner, -1, axis=0)) / 2))
    for a in anchors:
        c = np.array(
            [
                [*a, eave],
                [*a, eave + rise * 0.28],
                [*(centre + (a - centre) * 0.55), join - rise * 0.02],
                [*centre, join],
            ]
        )
        ts = np.linspace(0, 1, 12)[:, None]
        pts = (
            (1 - ts) ** 3 * c[0]
            + 3 * (1 - ts) ** 2 * ts * c[1]
            + 3 * (1 - ts) * ts**2 * c[2]
            + ts**3 * c[3]
        )
        for lo, hi in zip(pts[:-1], pts[1:]):
            beam(r, lo, hi, radius, rib_stone)
    for a in corner:
        beam(
            r,
            [*a, eave],
            [*a, eave + rise * 0.30],
            detail.get("pinnacle_radius_m", 0.22),
            rib_stone,
        )
        beam(
            r,
            [*a, eave + rise * 0.30],
            [*a, eave + rise * 0.35],
            detail.get("pinnacle_tip_radius_m", 0.08),
            rib_stone,
        )
    beam(
        r,
        [*centre, join],
        [*centre, eave + rise * detail.get("finial_fraction", 0.75)],
        detail.get("finial_radius_m", 0.25),
        rib_stone,
    )
    beam(
        r,
        [*centre, eave + rise * detail.get("finial_fraction", 0.75)],
        [*centre, top],
        detail.get("tip_radius_m", 0.13),
        rib_stone,
    )


def parapet(r, record):
    """Illustrated coped merlons on cited existing eaves, within the roof envelope."""
    d = record["parapet"]
    p = shape(record["geometry"])
    coords = np.array(p.exterior.coords)
    low = record["eave_m"]
    high = min(record["top_m"], low + d["height_m"])
    colour = tuple(d["rgb"])

    def solid(polygon, top):
        for q in polygons(polygon):
            ring = np.array(q.exterior.coords)
            for a, b in zip(ring[:-1], ring[1:]):
                quad(r, [[*a, low], [*b, low], [*b, top], [*a, top]], colour)
            for t in triangles(q):
                face(r, np.c_[t, np.full(3, top)], colour)

    for index in d["edge_indices"]:
        line = LineString(coords[index : index + 2])
        solid(
            p.intersection(line.buffer(d["depth_m"], cap_style=2)),
            low + (high - low) * 0.25,
        )
        pitch = d["pitch_m"]
        for distance in np.arange(pitch * 0.25, line.length, pitch):
            lo = max(0, distance - pitch * d["duty"] / 2)
            hi = min(line.length, distance + pitch * d["duty"] / 2)
            tooth = LineString([line.interpolate(lo), line.interpolate(hi)])
            solid(p.intersection(tooth.buffer(d["depth_m"], cap_style=2)), high)


def embrasures(r, record):
    """Depth-tested surface illustration of cited arched gun openings, not new voids."""
    d = record["embrasures"]
    coords = np.array(shape(record["geometry"]).exterior.coords)
    for indices in d["edge_paths"]:
        line = LineString(coords[indices])
        pitch = d["pitch_m"]
        vertices = np.array(line.coords)
        lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
        cumulative = np.cumsum(lengths)
        selected = set()
        for distance in np.arange(pitch * 0.5, line.length, pitch):
            index = int(np.searchsorted(cumulative, distance))
            if index in selected:
                continue
            selected.add(index)
            width = min(d["width_m"], lengths[index] * 0.82)
            if width < 0.8:
                continue
            direction = (vertices[index + 1] - vertices[index]) / lengths[index]
            middle = (vertices[index + 1] + vertices[index]) / 2
            left, right = middle - direction * width / 2, middle + direction * width / 2
            mid = (left + right) / 2
            top = record["top_m"] - d["coping_m"]
            bottom = top - d["height_m"]
            outline = [
                [*left, bottom],
                [*right, bottom],
                [*right, top - d["height_m"] * 0.25],
            ]
            for t in np.linspace(0, math.pi, 7)[1:]:
                xy = mid + (right - left) * 0.5 * math.cos(t)
                outline.append(
                    [
                        *xy,
                        top - d["height_m"] * 0.25 + d["height_m"] * 0.25 * math.sin(t),
                    ]
                )
            centre = np.mean(outline, axis=0)
            for a, b in zip(outline, outline[1:] + outline[:1]):
                r.triangle(np.array([centre, a, b]), tuple(d["rgb"]), bias=0.07)


def gothic_spire(r, feature, record):
    """Measured envelope with photo-informed arched massing, turrets and galleries.

    Plan dimensions/lobes come from the mapped footprint. Vertical proportions
    and simplified ornament are explicit illustrative source geometry, not a survey.
    """
    p = feature.geometry if hasattr(feature, "geometry") else feature
    centre = np.array(p.centroid.coords[0])
    d = record["massing"]
    axis = np.array(d["axis_unit"], float)
    axis /= np.linalg.norm(axis)
    across = np.array([-axis[1], axis[0]])
    base, top = record["base_m"], record["top_m"]
    height = top - base
    stone = (64, 61, 57)

    def z(fraction):
        return base + height * fraction

    def point(a, b):
        return centre + axis * a + across * b

    def prism(plan, low, high, colour=stone):
        # Clipping the base piers and turret lobes conserves the mapped envelope.
        plan = plan.intersection(p)
        for polygon in polygons(plan):
            ring = np.array(polygon.exterior.coords)[:-1]
            for a, b in zip(ring, np.roll(ring, -1, axis=0)):
                quad(r, [[*a, low], [*b, low], [*b, high], [*a, high]], colour)
            for tri in triangles(polygon):
                face(r, np.column_stack((tri, np.full(3, high))), colour)

    def square(xy, width):
        from shapely.geometry import Polygon
        return Polygon([xy + axis * width * a + across * width * b
                        for a, b in ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5))])

    def octagon(xy, radius):
        from shapely.geometry import Polygon
        return Polygon([xy + axis * radius * math.cos(t) + across * radius * math.sin(t)
                        for t in np.arange(8) * math.pi / 4])

    def profile(xy, rings, colour=stone, chamfer=False):
        # Chamfered-square galleries retain their broad body between the ledges.
        directions = (np.array([[-1, -.7], [-.7, -1], [.7, -1], [1, -.7],
                                [1, .7], [.7, 1], [-.7, 1], [-1, .7]])
                      if chamfer else np.c_[np.cos(np.arange(8)*math.pi/4),
                                            np.sin(np.arange(8)*math.pi/4)])
        vertices = [np.array([[*(xy + axis*a*radius + across*b*radius), high]
                              for a, b in directions]) for radius, high in rings]
        for lower, upper in zip(vertices, vertices[1:]):
            for i in range(8):
                j = (i+1) % 8
                quad(r, [lower[i], lower[j], upper[j], upper[i]], colour)
        # Gallery tops close the solid body; narrow ledges remain visible.
        for i in range(1, 7):
            face(r, vertices[-1][[0, i, i+1]], colour)

    def pinnacle(xy, low, high, radius):
        profile(xy, [(radius, low), (radius*.80, low+(high-low)*.25),
                     (radius, low+(high-low)*.28), (radius*.62, low+(high-low)*.48),
                     (radius*.75, low+(high-low)*.51), (.035, high)])

    # The former light terrace was read as a white kiosk. Draw only the existing
    # mapped support in dark setts, depth tested against the unchanged terrain.
    for tri in triangles(p):
        face(r, np.column_stack((tri, np.full(3, base+.08))), (128, 122, 114))

    half = d["body_half_width_m"]
    corners = [point(a*half, b*half) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    for xy in corners:
        prism(square(xy, d["pier_width_m"]), base, z(.33))
        pinnacle(xy, z(.33), z(.45), .57)

    # A thick wall surrounds each real pointed opening. Its spandrel is solid,
    # unlike the previous open gantry. Intrados faces expose genuine arch depth.
    span = d["arch_clear_span_m"] / 2
    wall_top, spring, apex = z(.33), z(.12), z(.22)
    curve_x = np.linspace(-span, span, 15)
    curve_z = spring + (apex-spring)*(1-np.abs(curve_x/span))**.72
    for a, b in zip(corners, corners[1:]+corners[:1]):
        midpoint = (a+b)/2
        tangent = (b-a)/np.linalg.norm(b-a)
        inward = centre-midpoint
        inward /= np.linalg.norm(inward)
        xs = np.r_[-half, curve_x, half]
        bottoms = np.r_[base, curve_z, base]
        for depth in (0., d["arch_wall_depth_m"]):
            for x0, x1, h0, h1 in zip(xs[:-1], xs[1:], bottoms[:-1], bottoms[1:]):
                # Vertical springing edges must not become sloped side wedges.
                if x0 < -span: h0 = h1 = base
                if x1 > span: h0 = h1 = base
                q0, q1 = midpoint+tangent*x0+inward*depth, midpoint+tangent*x1+inward*depth
                quad(r, [[*q0, h0], [*q1, h1], [*q1, wall_top], [*q0, wall_top]], stone)
        path = [(curve_x[0], base), *zip(curve_x, curve_z), (curve_x[-1], base)]
        for (x0, h0), (x1, h1) in zip(path, path[1:]):
            q0, q1 = midpoint+tangent*x0, midpoint+tangent*x1
            step = inward*d["arch_wall_depth_m"]
            quad(r, [[*q0, h0], [*q1, h1], [*(q1+step), h1], [*(q0+step), h0]], stone)

    for turret in d["outer_turret_centres_m"]:
        xy = np.array(turret)
        prism(octagon(xy, d["outer_turret_width_m"]/2), base, z(.40))
        pinnacle(xy, z(.40), z(.48), .70)
        corner = min(corners, key=lambda a: np.linalg.norm(a-xy))
        # An upward-curving flying buttress, from the real outer turret lobe.
        fractions = np.linspace(0, 1, 10)
        pts = [np.r_[xy*(1-t)+corner*t, z(.25+.08*(1-(1-t)**2))] for t in fractions]
        for lo, hi in zip(pts[:-1], pts[1:]):
            beam(r, lo, hi, .40, stone)

    for width, low, high, tip in d["tiers"]:
        # Broad bodies, with thin source-lit gallery cornices rather than a cone.
        profile(centre, [(width, z(low)), (width*.98, z(high))], chamfer=True)
        gallery = width*.98 + .35
        profile(centre, [(gallery, z(high)-.50), (gallery, z(high)),
                         (width*.98, z(high))], (94, 87, 76), chamfer=True)
        # One long Gothic recess per face follows the council/BGS photographs.
        # This is simplified source tracery, not a repeating invented window grid.
        lowz, highz = z(low)+1.5, z(high)-1.1
        opening_half = width*.26
        for normal, tangent in ((axis, across), (across, -axis), (-axis, -across), (-across, axis)):
            def wallpoint(offset, zz):
                fraction = (zz-z(low))/(z(high)-z(low))
                return np.r_[centre + normal*(width*(1-.02*fraction)+.045) + tangent*offset, zz]
            outline = [wallpoint(-opening_half, lowz), wallpoint(opening_half, lowz),
                       wallpoint(opening_half, lowz+(highz-lowz)*.65), wallpoint(0, highz),
                       wallpoint(-opening_half, lowz+(highz-lowz)*.65)]
            for i in range(1, 4):
                face(r, np.array(outline)[[0, i, i+1]], (21, 21, 20))
            for left, right in zip(outline, outline[1:]+outline[:1]):
                beam(r, left, right, .16, (91, 86, 77), sides=4)
            beam(r, wallpoint(0, lowz), wallpoint(0, highz-.65), .11, (91, 86, 77), sides=4)
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            pinnacle(point(a*width*.88, b*width*.88), z(high)-1.1, z(tip), min(.46, width*.20))
    profile(centre, [(.94, z(.88)), (.94, z(.885)), (.80, z(.89)), (.04, top)], chamfer=True)

    # Cited seated Carrara statue stays a small pale mass inside the open base.
    prism(square(centre, 2.7), base+.15, base+.9, (184, 183, 173))
    prism(square(centre, 1.5), base+.9, base+3.9, (222, 222, 213))
    beam(r, [*centre, base+3.8], [*centre, base+4.8], .49, (222, 222, 213))
    return record


def validate_detail(record, polygon):
    """Keep illustrative details small, cited and inside their existing envelope."""
    for key in ("drawing", "parapet", "embrasures"):
        d = record.get(key)
        if not d:
            continue
        for name, value in d.items():
            if name.endswith("_rgb") or name == "rgb":
                if len(value) != 3 or any(
                    not isinstance(v, int) or not 0 <= v <= 255 for v in value
                ):
                    raise ValueError("Detail colours need three bytes")
        if key == "drawing":
            for name in (
                "rib_radius_m",
                "pinnacle_radius_m",
                "pinnacle_tip_radius_m",
                "finial_radius_m",
                "tip_radius_m",
            ):
                if name in d and (
                    not math.isfinite(d[name]) or not 0.05 <= d[name] <= 1.5
                ):
                    raise ValueError("Crown drawing radii must be .05..1.5 metres")
            if (
                not 0.3 <= d.get("anchor_scale", 0.94) <= 1
                or not 0.3
                <= d.get("join_fraction", 0.48)
                < d.get("finial_fraction", 0.75)
                < 1
            ):
                raise ValueError("Crown drawing fractions must preserve its envelope")
        else:
            if not d.get("source_refs") or not d.get("illustrative"):
                raise ValueError(
                    "Architectural detail needs a citation and approximation note"
                )
            for name, limits in {
                "height_m": (0.2, 3),
                "pitch_m": (1, 30),
                "depth_m": (0.1, 2),
                "width_m": (0.5, 4),
                "coping_m": (0.1, 2),
            }.items():
                if name in d and (
                    not math.isfinite(d[name]) or not limits[0] <= d[name] <= limits[1]
                ):
                    raise ValueError(
                        "Architectural detail dimensions exceed drawing limits"
                    )
            paths = [d["edge_indices"]] if key == "parapet" else d["edge_paths"]
            maximum = len(polygon.exterior.coords) - (2 if key == "parapet" else 1)
            if not paths or any(
                not path
                or any(type(v) is not int or not 0 <= v <= maximum for v in path)
                for path in paths
            ):
                raise ValueError("Detail edges must refer to the existing footprint")
            if key == "parapet" and (
                not 0.2 <= d["duty"] <= 0.8
                or record["eave_m"] + d["height_m"] > record["top_m"]
            ):
                raise ValueError("Parapet must stay inside the measured roof envelope")


def validate_data(data):
    """Bound optional author-supplied geometry before allocating masks or drawing."""
    records = data["buildings"] + data["structures"] + data["material_overrides"]
    if len(records) > 2000 or not 0 <= data["suppression_tolerance_m"] <= 20:
        raise ValueError("Landmark geometry exceeds source limits")
    if any(
        record.get("kind") not in ("open_crown", "gothic_spire")
        for record in data["structures"]
    ):
        raise ValueError("Unsupported landmark structure kind")
    for record in data["structures"]:
        if record["kind"] == "open_crown" and record.get("rib_count") != 8:
            raise ValueError("The open crown template requires eight ribs")
        if record["kind"] == "gothic_spire" and (
            record.get("pier_count") != 4 or record.get("tier_count") != 5
        ):
            raise ValueError("The Gothic template requires four piers and five tiers")
        if record["kind"] == "gothic_spire":
            d = record.get("massing", {})
            axis = np.asarray(d.get("axis_unit", []), float)
            lobes = np.asarray(d.get("outer_turret_centres_m", []), float)
            tiers = np.asarray(d.get("tiers", []), float)
            if (axis.shape != (2,) or not np.isfinite(axis).all()
                    or not .99 <= np.linalg.norm(axis) <= 1.01
                    or lobes.shape != (4, 2) or not np.isfinite(lobes).all()
                    or tiers.shape != (4, 4) or not np.isfinite(tiers).all()):
                raise ValueError("Gothic massing needs a unit axis, four mapped turrets and four lower tiers")
            polygon = shape(record["geometry"])
            from shapely.geometry import Point
            if any(not polygon.covers(Point(xy)) for xy in lobes):
                raise ValueError("Gothic turrets must lie in the measured footprint")
            for name, limits in {"body_half_width_m": (2, 6), "pier_width_m": (1, 3),
                                 "arch_clear_span_m": (2, 5), "arch_wall_depth_m": (.2, 1.5),
                                 "outer_turret_width_m": (1, 3)}.items():
                value = d.get(name, float("nan"))
                if not math.isfinite(value) or not limits[0] <= value <= limits[1]:
                    raise ValueError("Gothic dimensions exceed the source drawing limits")
            if (not (tiers[:, 0] > 0).all() or not (np.diff(tiers[:, 0]) < 0).all()
                    or tiers[0, 0] > d["body_half_width_m"]
                    or not (tiers[:, 1] < tiers[:, 2]).all()
                    or not (tiers[:, 2] < tiers[:, 3]).all()
                    or not (tiers[:, 1] >= .25).all() or not (tiers[:, 3] < 1).all()
                    or not np.array_equal(tiers[1:, 1], tiers[:-1, 2])):
                raise ValueError("Gothic tiers must narrow and preserve the measured apex")
    if any(
        record["shape"]
        not in ("flat", "terrace", "gabled", "hipped", "mansard", "dome", "pyramidal")
        for record in data["buildings"]
    ):
        raise ValueError("Unsupported measured roof shape")
    for record in records:
        polygon = shape(record["geometry"])
        if (
            polygon.geom_type != "Polygon"
            or not polygon.is_valid
            or polygon.is_empty
            or len(polygon.exterior.coords) > 10000
            or not np.isfinite(polygon.bounds).all()
        ):
            raise ValueError("Landmark geometry needs a finite valid bounded polygon")
        validate_detail(record, polygon)
        for key in (
            "base_m",
            "top_m",
            "eave_m",
            "crown_base_m",
            "absolute_colour_threshold_m",
        ):
            if key in record and (
                not math.isfinite(record[key]) or abs(record[key]) > 10000
            ):
                raise ValueError("Landmark heights must be finite metres")
        if "eave_m" in record and record["eave_m"] > record["top_m"]:
            raise ValueError("Landmark eaves cannot exceed the roof")
        if "base_m" in record and record["base_m"] >= record["top_m"]:
            raise ValueError("Landmark top must exceed its base")
        if (
            "crown_base_m" in record
            and not record["base_m"] < record["crown_base_m"] < record["top_m"]
        ):
            raise ValueError("Crown must rise above its body")
        if "rgb" in record and (
            len(record["rgb"]) != 3
            or any(not isinstance(v, int) or not 0 <= v <= 255 for v in record["rgb"])
        ):
            raise ValueError("Landmark colours need three bytes")
    return data


class Overlay:
    def __init__(self, spec, L, surface_file, osm, grid, *, scale=None):
        path = Path(spec["landmarks"])
        path = path if path.is_absolute() else surface.ROOT / path
        if path.stat().st_size > 2_000_000:
            raise ValueError("Landmark data exceeds 2 MB")
        self.data = validate_data(json.loads(path.read_text()))
        d = self.data
        if d["schema"] != 1 or spec["crs"] != d["crs"] or spec["camera"] != d["camera"]:
            raise ValueError("Landmark source recipe/camera mismatch")
        for key, p in [
            ("surface", surface_file),
            ("osm", osm),
            ("horizontal_grid", grid),
        ]:
            actual = hashlib.sha256(Path(p).read_bytes()).hexdigest() if p else None
            if actual != d["input_sha256"][key]:
                raise ValueError(
                    f"Landmark {key} input hash mismatch; remeasure before rendering"
                )
        self.DSM = L["dsm"]
        self.DTM = L["dtm"]
        self.W, _, _, self.N = L["bounds"]
        self.res = spec["resolution_m"]
        self.camera = Camera(
            {
                **spec["camera"],
                "pixels_per_metre": (
                    scale if scale is not None else spec["camera"]["pixels_per_metre"]
                ),
            }
        )
        self.suppress = np.zeros(self.DSM.shape, bool)
        for record in d["buildings"] + d["structures"]:
            self.suppress |= self.mask(record)
        iterations = round(d["suppression_tolerance_m"] / self.res)
        if iterations:
            self.suppress = ndi.binary_dilation(self.suppress, iterations=iterations)
        self.materials_list = []
        for record in d["material_overrides"]:
            mask = self.mask(record)
            if "absolute_colour_threshold_m" in record:
                mask &= self.DSM > record["absolute_colour_threshold_m"]
            self.materials_list.append((mask, np.array(record["rgb"])))
        self.provenance = {
            "recipe_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "input_sha256": d["input_sha256"],
            "references": d["references"],
            "limits": d["limits"],
            "structural_components": len(d["buildings"]),
            "suppression_tolerance_m": d["suppression_tolerance_m"],
        }

    def mask(self, record):
        p = shape(record["geometry"])
        x0, y0, x1, y1 = p.bounds
        j0 = max(0, int((x0 - self.W) / self.res))
        j1 = min(self.DSM.shape[1], math.ceil((x1 - self.W) / self.res))
        i0 = max(0, int((self.N - y1) / self.res))
        i1 = min(self.DSM.shape[0], math.ceil((self.N - y0) / self.res))
        ii, jj = np.mgrid[i0:i1, j0:j1]
        mask = np.zeros(self.DSM.shape, bool)
        mask[i0:i1, j0:j1] = contains_xy(
            p, self.W + (jj + 0.5) * self.res, self.N - (ii + 0.5) * self.res
        )
        return mask

    def heightfield(self, H, dtm):
        return np.where(self.suppress, dtm, H)

    def materials(self, alb, walla):
        for mask, rgb in self.materials_list:
            alb[mask] = rgb
            walla[mask] = rgb

    def geometry(self, img, zb):
        r = Raster(self.camera)
        r.rgb = img
        r.depth = -zb.astype(np.float32)
        for record in self.data["buildings"]:
            feature = Feature(
                record["osm_id"], shape(record["geometry"]), record.get("tags", {})
            )
            render_building(r, feature, record, self.DTM, self.W, self.N, self.res)
        for record in self.data["buildings"]:
            if record.get("parapet"):
                parapet(r, record)
            if record.get("embrasures"):
                embrasures(r, record)
        for record in self.data["structures"]:
            if record["kind"] == "open_crown":
                crown(r, record)
            else:
                gothic_spire(r, shape(record["geometry"]), record)
        return r.rgb, -r.depth
