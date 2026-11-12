"""Source-bound geometry on the shared world camera and depth buffer.

Surveyed dimensions belong to recipes. Small mouldings and cloth are illustrative.
"""

import math
import copy
import numpy as np
from pyproj import CRS, Transformer

SUN = np.array([-1.0, -2.0, 3.0])
SUN /= np.linalg.norm(SUN)


def _object(value, allowed, required, path):
    if (
        not isinstance(value, dict)
        or set(value) - set(allowed)
        or set(required) - set(value)
    ):
        raise ValueError(f"{path}: unexpected or missing fields")


def _number(value, path, low=-10000, high=10000):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not low <= value <= high
        or not math.isfinite(value)
    ):
        raise ValueError(f"{path}: finite number in [{low}, {high}] required")
    return value


def _integer(value, path, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{path}: integer in [{low}, {high}] required")
    return value


def _array(value, path, minimum, maximum):
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError(f"{path}: list of {minimum}..{maximum} entries required")
    return value


def _rgb(value, path):
    for channel in _array(value, path, 3, 3):
        _integer(channel, path, 0, 255)


def _interval(value, path, low=-10000, high=10000):
    _array(value, path, 2, 2)
    a, b = [_number(v, path, low, high) for v in value]
    if a >= b:
        raise ValueError(f"{path}: increasing interval required")


def _box(value, path):
    _array(value, path, 4, 4)
    for v in value:
        _number(v, path)
    if not 0 < value[2] - value[0] <= 2000 or not 0 < value[3] - value[1] <= 2000:
        raise ValueError(f"{path}: positive box dimensions at most 2000 m required")


def _anchor(value, path):
    _object(value, ("lat", "lon"), ("lat", "lon"), path)
    _number(value["lat"], path + ".lat", -90, 90)
    _number(value["lon"], path + ".lon", -180, 180)


def _validate(city, config):
    """Bound recipe work before any raster allocation or source geometry is changed."""
    _object(config, ("structures", "facades"), (), "landmark_geometry")
    structures = _array(config.get("structures", []), "structures", 0, 32)
    facades = _array(config.get("facades", []), "facades", 0, 64)
    if not structures and not facades:
        return
    _anchor(city.get("origin"), "origin")
    try:
        crs = CRS.from_user_input(city["photographic"]["crs"])
        if not crs.is_projected or any(
            a.unit_name not in ("metre", "meter") for a in crs.axis_info
        ):
            raise ValueError("projected metre CRS required")
        project = Transformer.from_crs(4326, crs, always_xy=True)
        origin = project.transform(
            city["origin"]["lon"], city["origin"]["lat"], errcheck=True
        )
        if not np.isfinite(origin).all():
            raise ValueError("nonfinite projected origin")
    except Exception as exc:
        raise ValueError(
            "landmark_geometry: valid projected metre CRS required"
        ) from exc
    ids, owners, work = set(), set(), 0
    common = {"id", "source_refs", "interpretation"}
    structure_fields = {
        "tapered_obelisk": {
            "anchor",
            "yaw_degrees",
            "height_m",
            "shaft_height_m",
            "base_width_m",
            "upper_width_m",
            "bands",
            "windows",
            "flags",
        },
        "arc_colonnade": {
            "anchor",
            "yaw_degrees",
            "offset_m",
            "radius_m",
            "columns",
            "base_z_m",
            "top_z_m",
            "decks",
            "balustrade",
        },
    }
    window_fields = {
        "anchor",
        "support_box_m",
        "window_phase_m",
        "window_spacing_m",
        "window_width_m",
        "window_z_bands_m",
        "mullion_phase_m",
    }
    for index, spec in enumerate(structures + facades):
        path = f"landmark_geometry[{index}]"
        is_structure = index < len(structures)
        if not isinstance(spec, dict):
            raise ValueError(f"{path}: object required")
        kind = spec.get("kind")
        if is_structure:
            if not isinstance(kind, str) or kind not in structure_fields:
                raise ValueError(f"{path}: unsupported structure kind")
            required = common | {"kind"} | structure_fields[kind]
            allowed = required | {"replace_owners"}
        else:
            required = common | {"owners", "wall_rgb"}
            allowed = required | window_fields | {"opening", "finish"}
            if "finish" in spec and spec["finish"] != "plain":
                raise ValueError(f"{path}: only the plain facade finish is supported")
            if "finish" in spec and "window_z_bands_m" in spec:
                raise ValueError(f"{path}: plain finish cannot also place facade windows")
            if "window_z_bands_m" in spec:
                required |= window_fields
            elif set(spec) & (window_fields - {"anchor"}):
                raise ValueError(f"{path}: partial facade window configuration")
            if "opening" in spec:
                required.add("anchor")
        _object(spec, allowed, required, path)
        identifier = spec["id"]
        if (
            not isinstance(identifier, str)
            or not identifier.strip()
            or len(identifier) > 80
            or identifier in ids
        ):
            raise ValueError(
                f"{path}: unique nonempty id at most 80 characters required"
            )
        ids.add(identifier)
        refs = _array(spec["source_refs"], path + ".source_refs", 1, 16)
        if any(
            not isinstance(v, str)
            or len(v) > 2048
            or not v.startswith(("https://", "http://"))
            for v in refs
        ):
            raise ValueError(f"{path}: HTTP source citations required")
        if (
            not isinstance(spec["interpretation"], str)
            or not spec["interpretation"].strip()
            or len(spec["interpretation"]) > 4000
        ):
            raise ValueError(f"{path}: bounded interpretation required")
        for owner in _array(
            spec.get("replace_owners", []) if is_structure else spec["owners"],
            path + ".owners",
            0 if is_structure else 1,
            128,
        ):
            _integer(owner, path + ".owners", 0, 2**63 - 1)
            if owner in owners:
                raise ValueError(f"{path}: source owner assigned more than once")
            owners.add(owner)
        if len(owners) > 1024:
            raise ValueError("landmark_geometry: at most 1024 source owners")
        if "anchor" in spec:
            _anchor(spec["anchor"], path + ".anchor")
            try:
                point = project.transform(
                    spec["anchor"]["lon"], spec["anchor"]["lat"], errcheck=True
                )
            except Exception as exc:
                raise ValueError(f"{path}: anchor projection failed") from exc
            if (
                not np.isfinite(point).all()
                or np.linalg.norm(np.array(point) - origin) > 100000
            ):
                raise ValueError(
                    f"{path}: projected anchor must be within 100 km of origin"
                )
        if is_structure:
            _number(spec["yaw_degrees"], path + ".yaw_degrees", -360, 360)
        if kind == "tapered_obelisk":
            shaft = _number(spec["shaft_height_m"], path + ".shaft_height_m", 0.1, 1000)
            top = _number(spec["height_m"], path + ".height_m", 0.1, 2000)
            base = _number(spec["base_width_m"], path + ".base_width_m", 0.1, 2000)
            upper = _number(spec["upper_width_m"], path + ".upper_width_m", 0.1, base)
            if top <= shaft:
                raise ValueError(f"{path}: pyramidion must rise above shaft")
            end = 0
            for band in _array(spec["bands"], path + ".bands", 1, 16):
                _array(band, path + ".band", 3, 3)
                _interval(band[:2], path + ".band", 0, shaft)
                _rgb(band[2], path + ".band.rgb")
                if abs(band[0] - end) > 1e-8:
                    raise ValueError(
                        f"{path}: material bands must cover shaft continuously"
                    )
                end = band[1]
            if abs(end - shaft) > 1e-8:
                raise ValueError(f"{path}: material bands must reach shaft top")
            windows = spec["windows"]
            _object(
                windows,
                ("z_band", "centres_m", "half_width_m"),
                ("z_band", "centres_m", "half_width_m"),
                path + ".windows",
            )
            _interval(windows["z_band"], path + ".windows.z_band", 0, shaft)
            width = _number(
                windows["half_width_m"], path + ".windows.half_width_m", 0.01, upper / 2
            )
            centres = _array(windows["centres_m"], path + ".windows.centres_m", 1, 16)
            for centre in centres:
                _number(
                    centre,
                    path + ".windows.centre",
                    -upper / 2 + width,
                    upper / 2 - width,
                )
            if any(a >= b for a, b in zip(centres, centres[1:])):
                raise ValueError(
                    f"{path}: opening centres must be unique and increasing"
                )
            flags = spec["flags"]
            _object(
                flags,
                ("count", "radius_m", "height_m"),
                ("count", "radius_m", "height_m"),
                path + ".flags",
            )
            count = _integer(flags["count"], path + ".flags.count", 0, 128)
            _number(flags["radius_m"], path + ".flags.radius_m", 1, 2000)
            _number(flags["height_m"], path + ".flags.height_m", 2, 50)
            work += count * 40 + len(spec["bands"]) * 8 + 4
        elif kind == "arc_colonnade":
            _array(spec["offset_m"], path + ".offset_m", 2, 2)
            for v in spec["offset_m"]:
                _number(v, path + ".offset_m")
            _number(spec["radius_m"], path + ".radius_m", 1, 2000)
            columns = _integer(spec["columns"], path + ".columns", 2, 128)
            bottom = _number(spec["base_z_m"], path + ".base_z_m", -1000, 10000)
            top = _number(spec["top_z_m"], path + ".top_z_m", -1000, 10000)
            if not 2 < top - bottom <= 200:
                raise ValueError(
                    f"{path}: column rise must be over 2 m and at most 200 m"
                )
            decks = _array(spec["decks"], path + ".decks", 1, 8)
            for deck in decks:
                _array(deck, path + ".deck", 3, 3)
                z = _number(deck[0], path + ".deck.z", bottom, top)
                thickness = _number(deck[1], path + ".deck.thickness", 0.01, 20)
                if z + thickness > top + 1e-8:
                    raise ValueError(f"{path}: deck exceeds column top")
                _rgb(deck[2], path + ".deck.rgb")
            rail = spec["balustrade"]
            _object(
                rail,
                ("count", "bottom_z_m", "top_z_m"),
                ("count", "bottom_z_m", "top_z_m"),
                path + ".balustrade",
            )
            balusters = _integer(rail["count"], path + ".balustrade.count", 0, 256)
            _interval(
                [rail["bottom_z_m"], rail["top_z_m"]],
                path + ".balustrade.z",
                bottom,
                top,
            )
            work += columns * 150 + balusters * 40 + len(decks) * 100
        else:
            _rgb(spec["wall_rgb"], path + ".wall_rgb")
            if "window_z_bands_m" in spec:
                _box(spec["support_box_m"], path + ".support_box_m")
                spacing = _number(
                    spec["window_spacing_m"], path + ".window_spacing_m", 0.1, 100
                )
                _number(spec["window_phase_m"], path + ".window_phase_m")
                _number(spec["mullion_phase_m"], path + ".mullion_phase_m", 0, spacing)
                _interval(spec["window_width_m"], path + ".window_width_m", 0, spacing)
                end = -1000
                for band in _array(
                    spec["window_z_bands_m"], path + ".window_z_bands_m", 1, 16
                ):
                    _interval(band, path + ".window_z_bands_m", -1000, 10000)
                    if band[0] < end:
                        raise ValueError(
                            f"{path}: window bands overlap or are unordered"
                        )
                    end = band[1]
            if "opening" in spec:
                opening = spec["opening"]
                _object(
                    opening,
                    ("box_m", "keep_below_z_m"),
                    ("box_m", "keep_below_z_m"),
                    path + ".opening",
                )
                _box(opening["box_m"], path + ".opening.box_m")
                _number(
                    opening["keep_below_z_m"],
                    path + ".opening.keep_below_z_m",
                    -1000,
                    10000,
                )
        if work > 100000:
            raise ValueError("landmark_geometry: generated triangle budget exceeded")


def cylinder(raster, centre, radius, z0, z1, colour, segments=16):
    angles = np.arange(segments + 1) * 2 * np.pi / segments
    ring = np.column_stack(
        [centre[0] + radius * np.cos(angles), centre[1] + radius * np.sin(angles)]
    )
    for a, b in zip(ring[:-1], ring[1:]):
        normal = np.array([b[1] - a[1], a[0] - b[0], 0])
        normal /= max(np.linalg.norm(normal), 1e-9)
        light = 0.73 + 0.27 * abs(normal @ SUN)
        rgb = tuple(int(v * light) for v in colour)
        raster.quad([[*a, z0], [*b, z0], [*b, z1], [*a, z1]], rgb, bias=0.12)
    for a, b in zip(ring[:-1], ring[1:]):
        raster.triangle([[*centre, z1], [*a, z1], [*b, z1]], colour, bias=0.13)


def quad(raster, points, colour, bias=0.1):
    p = np.asarray(points)
    normal = np.cross(p[1] - p[0], p[2] - p[0])
    normal /= max(np.linalg.norm(normal), 1e-9)
    light = 0.76 + 0.24 * abs(normal @ SUN)
    raster.quad(p, tuple(int(v * light) for v in colour), bias=bias)


def draw_obelisk(raster, city, dtm, spec):
    t = Transformer.from_crs(4326, city["photographic"]["crs"], always_xy=True)
    ox, oy = t.transform(city["origin"]["lon"], city["origin"]["lat"])
    mx, my = t.transform(spec["anchor"]["lon"], spec["anchor"]["lat"])
    centre = np.array([mx - ox, my - oy])
    base = float(dtm(mx, my))
    shaft = spec["shaft_height_m"]
    top = spec["height_m"]
    base_width = spec["base_width_m"]
    top_width = spec["upper_width_m"]
    # Follow the recipe's mapped orientation.
    rotation = math.radians(spec["yaw_degrees"])
    E = np.array([math.cos(rotation), math.sin(rotation)])
    N = np.array([-math.sin(rotation), math.cos(rotation)])

    def ring(z):
        w = base_width + (top_width - base_width) * z / shaft
        return np.array(
            [
                centre + (-E - N) * w / 2,
                centre + (E - N) * w / 2,
                centre + (E + N) * w / 2,
                centre + (-E + N) * w / 2,
            ]
        )

    bands = spec["bands"]

    def windows(world, color, pixels, edge):
        result = np.tile(color, (len(world), 1))
        height = world[:, 2] - base
        # Openings are restricted to their source-bound vertical band.
        relative = (world[:, :2] - centre) @ E
        if np.ptp(relative) < 1:
            relative = (world[:, :2] - centre) @ N
        inside = (
            (height > spec["windows"]["z_band"][0])
            & (height < spec["windows"]["z_band"][1])
            & np.any(
                abs(relative[:, None] - np.array(spec["windows"]["centres_m"]))
                < spec["windows"]["half_width_m"],
                axis=1,
            )
        )
        result[inside] = (55, 60, 57)
        return result

    for z0, z1, colour in bands:
        low, high = ring(z0), ring(z1)
        for i in range(4):
            j = (i + 1) % 4
            p = np.array(
                [
                    [*low[i], base + z0],
                    [*low[j], base + z0],
                    [*high[j], base + z1],
                    [*high[i], base + z1],
                ]
            )
            normal = np.cross(p[1] - p[0], p[2] - p[0])
            normal /= np.linalg.norm(normal)
            light = 0.73 + 0.27 * abs(normal @ SUN)
            rgb = tuple(int(v * light) for v in colour)
            raster.triangle(p[[0, 1, 2]], rgb, bias=0.1, shader=windows)
            raster.triangle(p[[0, 2, 3]], rgb, bias=0.1, shader=windows)
    upper = ring(shaft)
    for i in range(4):
        p = np.array(
            [
                [*upper[i], base + shaft],
                [*upper[(i + 1) % 4], base + shaft],
                [*centre, base + top],
            ]
        )
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        normal /= np.linalg.norm(normal)
        light = 0.75 + 0.25 * abs(normal @ SUN)
        raster.triangle(p, tuple(int(v * light) for v in (237, 234, 220)), bias=0.11)
    # Pole count and dimensions come from the recipe; waving cloth is illustrative.
    for a in np.arange(spec["flags"]["count"]) * 2 * np.pi / spec["flags"]["count"]:
        c = centre + spec["flags"]["radius_m"] * np.array([math.cos(a), math.sin(a)])
        ground = float(dtm(c[0] + ox, c[1] + oy))
        cylinder(
            raster,
            c,
            0.10,
            ground,
            ground + spec["flags"]["height_m"],
            (180, 184, 180),
            segments=6,
        )
        direction = np.array([1.0, 0.25])
        direction /= np.linalg.norm(direction)
        perp = np.array([-direction[1], direction[0]])
        for stripe in range(7):
            z0 = ground + spec["flags"]["height_m"] - stripe * 0.22
            z1 = z0 - 0.22
            tip = c + direction * 2.2 + perp * (0.12 * math.sin(a + stripe * 0.45))
            quad(
                raster,
                [[*c, z0], [*tip, z0 - 0.05], [*tip, z1 - 0.05], [*c, z1]],
                (175, 54, 53) if stripe % 2 == 0 else (239, 235, 218),
                bias=0.14,
            )
        tip = c + direction * 0.9
        quad(
            raster,
            [
                [*c, ground + spec["flags"]["height_m"]],
                [*tip, ground + spec["flags"]["height_m"]],
                [*tip, ground + spec["flags"]["height_m"] - 0.77],
                [*c, ground + spec["flags"]["height_m"] - 0.77],
            ],
            (48, 68, 102),
            bias=0.15,
        )
    vertices = np.concatenate(
        [
            np.column_stack([ring(0), np.full(4, base)]),
            np.column_stack([upper, np.full(4, base + shaft)]),
            np.array([[*centre, base + top]]),
        ]
    )
    return {
        "id": spec["id"],
        "source_refs": spec["source_refs"],
        "height_m": top,
        "shaft_height_m": shaft,
        "base_width_m": base_width,
        "upper_width_m": top_width,
        "marble_change_height_m": bands[0][1],
        "observation_windows": len(spec["windows"]["centres_m"]) * 4,
        "flags": spec["flags"]["count"],
        "vertices": vertices.tolist(),
        "limits": "Mapped position and official dimensions; marble RGB, flag angular phase/cloth and small opening dimensions are illustrative.",
    }


def draw_arc_colonnade(raster, city, dtm, spec):
    t = Transformer.from_crs(4326, city["photographic"]["crs"], always_xy=True)
    ox, oy = t.transform(city["origin"]["lon"], city["origin"]["lat"])
    wx, wy = t.transform(spec["anchor"]["lon"], spec["anchor"]["lat"])
    anchor = np.array([wx - ox, wy - oy])
    theta = math.radians(spec["yaw_degrees"])
    E = np.array([math.cos(theta), math.sin(theta)])
    N = np.array([-math.sin(theta), math.cos(theta)])
    # The profile follows the actual I3S lower portico footprint, with open shafts above its podium.
    centre = anchor + np.array(spec["offset_m"])
    radius = spec["radius_m"]
    podium = spec["base_z_m"]
    top = spec["top_z_m"]
    positions = [
        centre + radius * (math.cos(a) * E + math.sin(a) * N)
        for a in np.linspace(math.pi, 2 * math.pi, spec["columns"])
    ]
    for c in positions:
        cylinder(raster, c, 0.42, podium + 0.2, top - 1.1, (244, 243, 233))
        cylinder(raster, c, 0.62, podium, podium + 0.3, (245, 244, 235), segments=12)
        cylinder(raster, c, 0.65, top - 1.4, top - 1.0, (245, 244, 235), segments=12)
    angles = np.linspace(math.pi, 2 * math.pi, 33)
    arc = np.array(
        [centre + radius * (math.cos(a) * E + math.sin(a) * N) for a in angles]
    )
    # Entablature and balcony are depth-tested physical surfaces.
    for z, thick, colour in spec["decks"]:
        for a, b in zip(arc[:-1], arc[1:]):
            quad(
                raster,
                [[*a, z], [*b, z], [*b, z + thick], [*a, z + thick]],
                colour,
                bias=0.18,
            )
            raster.triangle(
                [[*centre, z + thick], [*a, z + thick], [*b, z + thick]],
                colour,
                bias=0.18,
            )
    # Slender repeated balusters are explicitly illustrative; the historic balcony itself is documented.
    for a in np.linspace(math.pi, 2 * math.pi, spec["balustrade"]["count"]):
        c = centre + (radius - 0.25) * (math.cos(a) * E + math.sin(a) * N)
        cylinder(
            raster,
            c,
            0.095,
            spec["balustrade"]["bottom_z_m"],
            spec["balustrade"]["top_z_m"],
            (239, 238, 228),
            segments=6,
        )
    for a, b in zip(arc[:-1], arc[1:]):
        quad(
            raster,
            [
                [*a, spec["balustrade"]["top_z_m"] - 0.10],
                [*b, spec["balustrade"]["top_z_m"] - 0.10],
                [*b, spec["balustrade"]["top_z_m"] + 0.10],
                [*a, spec["balustrade"]["top_z_m"] + 0.10],
            ],
            (240, 239, 231),
            bias=0.2,
        )
    return {
        "id": spec["id"],
        "source_refs": spec["source_refs"],
        "column_count": spec["columns"],
        "portico_width_m": radius * 2,
        "legacy_profile_elevations_m": [podium, top],
        "limits": "Count, width and balcony are source-documented. Column section, exact angular phase, mouldings, balustrade spacing and local elevations follow the simplified legacy mesh and are illustrative.",
    }


def anchor_xy(city, spec):
    t = Transformer.from_crs(4326, city["photographic"]["crs"], always_xy=True)
    ox, oy = t.transform(city["origin"]["lon"], city["origin"]["lat"])
    wx, wy = t.transform(spec["anchor"]["lon"], spec["anchor"]["lat"])
    return np.array([wx - ox, wy - oy])


def clip_wall(tri, anchor, spec):
    p = np.asarray(tri)
    mid = p.mean(0)
    xy = mid[:2] - anchor
    box = spec["opening"]["box_m"]
    limit = spec["opening"]["keep_below_z_m"]
    if not (box[0] < xy[0] < box[2] and box[1] < xy[1] < box[3]):
        return [p]
    # Open the legacy solid portico above its original raised podium.
    result = []
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        ain, bin_ = a[2] <= limit, b[2] <= limit
        if ain:
            result.append(a)
        if ain != bin_:
            result.append(a + (b - a) * (limit - a[2]) / (b[2] - a[2]))
    return (
        [
            np.array([result[0], result[i], result[i + 1]])
            for i in range(1, len(result) - 1)
        ]
        if len(result) >= 3
        else []
    )


def facade_shader(normal, base, anchor, default, spec):
    along = np.array([-normal[1], normal[0]], dtype=float)
    length = np.linalg.norm(along)
    if length < 0.5:
        return default
    along /= length
    light = 0.80 + 0.20 * abs(normal @ SUN)
    colour = np.array(spec["wall_rgb"]) * light

    def shade(world, color, pixels, edge):
        local = world[:, :2] - anchor
        box = spec["support_box_m"]
        main = (
            (local[:, 0] > box[0])
            & (local[:, 0] < box[2])
            & (local[:, 1] > box[1])
            & (local[:, 1] < box[3])
        )
        result = default(world, color, pixels, edge)
        if not main.any():
            return result
        u = local @ along
        z = world[:, 2]
        # Window placement is illustrative. The visible floor hierarchy follows the source elevation bands.
        phase = np.mod(u + spec["window_phase_m"], spec["window_spacing_m"])
        narrow = (phase > spec["window_width_m"][0]) & (
            phase < spec["window_width_m"][1]
        )
        bands = np.asarray(spec["window_z_bands_m"])
        window = (
            narrow
            & np.any((z[:, None] > bands[:, 0]) & (z[:, None] < bands[:, 1]), axis=1)
            & (edge > 0.35)
        )
        result[main] = colour.astype(np.uint8)
        result[main & window] = (49, 65, 67)
        # Small pale mullions and horizontal sill highlights keep the source readable without masonry hatching.
        mullion = window & (abs(phase - spec["mullion_phase_m"]) < 0.07)
        result[main & mullion] = (191, 198, 190)
        sill = (
            main
            & narrow
            & np.any(
                (z[:, None] > bands[:-1, 1]) & (z[:, None] < bands[:-1, 1] + 0.2),
                axis=1,
            )
        )
        result[sill] = np.clip(colour * 1.05, 0, 255).astype(np.uint8)
        return result

    return shade


class LandmarkLayer:
    def __init__(self, city):
        self.city = copy.deepcopy(city)
        config = self.city["photographic"].get("landmark_geometry", {})
        _validate(self.city, config)
        self.structures = config.get("structures", [])
        self.rules = config.get("facades", [])
        self.replaced = {
            oid for spec in self.structures for oid in spec.get("replace_owners", [])
        }
        for spec in self.structures + self.rules:
            if not spec.get("source_refs") or not spec.get("interpretation"):
                raise ValueError(
                    "Landmark source geometry needs citations and an interpretation"
                )
        self.by_owner = {oid: spec for spec in self.rules for oid in spec["owners"]}

    def validate_owners(self, owners):
        if not self.replaced and not self.by_owner:
            return
        owners = np.asarray(owners)
        if owners.ndim != 1 or owners.dtype.kind not in "iu" or (owners < 0).any():
            raise ValueError("Landmark provider owners must be nonnegative integer IDs")
        missing = (self.replaced | set(self.by_owner)) - set(map(int, owners))
        if missing:
            raise ValueError(f"Landmark provider features missing: {sorted(missing)}")

    def wall_colour(self, owner, default):
        spec = self.by_owner.get(int(owner), {})
        return default if "window_z_bands_m" in spec else spec.get("wall_rgb", default)

    def shader(self, owner, normal, base, default):
        spec = self.by_owner.get(int(owner))
        if spec and spec.get("finish") == "plain":
            return lambda world, color, pixels, edge: np.tile(np.asarray(color, np.uint8), (len(world), 1))
        if spec and "window_z_bands_m" in spec:
            return facade_shader(
                normal, base, anchor_xy(self.city, spec), default, spec
            )
        return default

    def fragments(self, owner, tri):
        spec = self.by_owner.get(int(owner))
        if spec and spec.get("opening"):
            return clip_wall(tri, anchor_xy(self.city, spec), spec)
        return [tri]

    def draw(self, raster, ground):
        kinds = {"tapered_obelisk": draw_obelisk, "arc_colonnade": draw_arc_colonnade}

        def checked_ground(x, y):
            return _number(float(ground(x, y)), "landmark terrain height", -1000, 10000)

        return [
            kinds[spec["kind"]](raster, self.city, checked_ground, spec)
            for spec in self.structures
        ]
