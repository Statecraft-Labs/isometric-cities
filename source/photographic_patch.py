#!/usr/bin/env python3
"""Render a bounded photographic patch from ArcGIS aerial, DTM and I3S sources.

The city recipe owns geography, providers, camera and explicit OSM part selection.
Aerial pixels shade roofs and terrain. Vertical facades remain illustrative.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse, hashlib, json, math, time
import requests
import numpy as np
import rasterio
from scipy.ndimage import map_coordinates
from pyproj import Transformer
from shapely import contains_xy, transform as transform_geom
from shapely.ops import unary_union
from shapely.geometry import Polygon
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
from roofs import radial_surface
from source_landmarks import LandmarkLayer
from geometry import triangles, load_features, SceneCoordinates
from raster import Raster


def osm_input(city, out):
    owned = Path(out) / "source/osm.json"
    if owned.is_file():
        return owned
    supplied = city.get("osm_source")
    if supplied:
        path = Path(supplied).expanduser()
        return path if path.is_absolute() else ROOT / path
    raise ValueError(f"Offline photographic source requires {owned} or an explicit osm_source")


def raster_export_plan(service, bbox, size, pixel_size, limit, crs):
    """Pixel bounds alone do not bound an ImageServer's underlying mosaic sources."""
    response = requests.get(service, params={"f": "json"}, timeout=45)
    response.raise_for_status()
    metadata = response.json()
    if "error" in metadata:
        raise ValueError(f"Raster service metadata failed: {metadata['error']}")
    mosaic_limit = metadata.get("maxMosaicImageCount")
    if mosaic_limit is not None and (type(mosaic_limit) is not int or mosaic_limit < 1):
        raise ValueError("Raster provider has an invalid mosaic-source limit")
    pending = [(x, y, min(limit, size-x), min(limit, size-y))
               for y in range(0, size, limit) for x in range(0, size, limit)]
    result = []
    while pending:
        x, y, width, height = pending.pop(0)
        box = [bbox[0]+x*pixel_size, bbox[3]-(y+height)*pixel_size,
               bbox[0]+(x+width)*pixel_size, bbox[3]-y*pixel_size]
        count = None
        if mosaic_limit is not None:
            response = requests.get(service + "/query", params={
                "f": "json", "where": "1=1", "geometry": ",".join(map(str, box)),
                "geometryType": "esriGeometryEnvelope", "inSR": crs,
                "spatialRel": "esriSpatialRelIntersects", "returnCountOnly": "true",
            }, timeout=45)
            response.raise_for_status()
            count = response.json().get("count")
            if type(count) is not int or count < 1:
                raise ValueError("Raster catalogue coverage is unavailable or could not be verified")
        too_large = (width > metadata.get("maxImageWidth", limit) or
                     height > metadata.get("maxImageHeight", limit) or
                     (count is not None and count > mosaic_limit))
        if too_large:
            if max(width, height) < 2 or len(result) + len(pending) + 2 > 64:
                raise ValueError("Raster source limits cannot be satisfied within 64 native exports")
            if width >= height:
                half = width // 2
                pending[:0] = [(x, y, half, height), (x+half, y, width-half, height)]
            else:
                half = height // 2
                pending[:0] = [(x, y, width, half), (x, y+half, width, height-half)]
        else:
            result.append({"window": [x,y,width,height], "source_count": count})
        if len(result) + len(pending) > 64:
            raise ValueError("A raster capture permits at most 64 aligned source exports")
    return result, metadata


def validate_raster_coverage(city, source):
    """Old shape-valid exports may silently omit catalogue tiles; never trust those offline."""
    for name, spec in city["photographic"].get("rasters", {}).items():
        if not spec.get("url"):
            continue
        proof_path = source / (name + "-coverage.json")
        if not proof_path.is_file():
            raise ValueError(f"{name}: cached catalogue coverage is unverified. Capture online into a fresh --out directory; do not reuse this raster.")
        proof = json.loads(proof_path.read_text())
        path = source / (name + ".tif")
        if proof.get("service") != spec["url"] or proof.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"{name}: raster coverage receipt does not match the cached input")
        with rasterio.open(path) as image:
            size = image.width
            if image.height != size or proof.get("dimensions") != [size,size]:
                raise ValueError(f"{name}: raster coverage dimensions do not match")
            projected = Transformer.from_crs(4326, city["photographic"]["crs"], always_xy=True)
            cx,cy = projected.transform(city["origin"]["lon"],city["origin"]["lat"])
            span = city["photographic"]["source_span_m"] / 2
            box = [round(cx-span),round(cy-span),round(cx+span),round(cy+span)]
            resolution = spec["sample_metres"]
            expected = rasterio.transform.from_origin(box[0],box[3],resolution,resolution)
            if (size != round((box[2]-box[0])/resolution) or
                image.crs != rasterio.crs.CRS.from_user_input(city["photographic"]["crs"]) or
                not image.transform.almost_equals(expected,precision=1e-7)):
                raise ValueError(f"{name}: coverage belongs to a different geographic acquisition grid")
        limit = proof.get("max_mosaic_image_count")
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError(f"{name}: invalid cached mosaic limit")
        exports = proof.get("exports")
        if not isinstance(exports, list) or not 1 <= len(exports) <= 64:
            raise ValueError(f"{name}: invalid cached export plan")
        area, windows = 0, []
        for export in exports:
            window = export.get("window", [])
            if len(window) != 4 or any(type(n) is not int for n in window):
                raise ValueError(f"{name}: malformed cached native window")
            x,y,w,h = window
            if min(x,y) < 0 or min(w,h) < 1 or x+w > size or y+h > size:
                raise ValueError(f"{name}: cached native window escapes the acquisition grid")
            if any(max(x,a) < min(x+w,a+c) and max(y,b) < min(y+h,b+d) for a,b,c,d in windows):
                raise ValueError(f"{name}: cached native export windows overlap")
            count = export.get("source_count")
            if limit is not None and (type(count) is not int or not 1 <= count <= limit):
                raise ValueError(f"{name}: cached export exceeds the provider mosaic-source limit; use a fresh --out capture")
            windows.append(window); area += w*h
        if area != size*size:
            raise ValueError(f"{name}: cached native export windows leave gaps")


def acquire_rasters(city, out):
    t = Transformer.from_crs(4326, city["photographic"]["crs"], always_xy=True)
    cx, cy = t.transform(city["origin"]["lon"], city["origin"]["lat"])
    span = city["photographic"]["source_span_m"] / 2
    bbox = [round(cx - span), round(cy - span), round(cx + span), round(cy + span)]
    (out / "bounds.json").write_text(
        json.dumps(
            {
                "crs": city["photographic"]["crs"],
                "bbox": bbox,
                "centre": [city["origin"]["lon"], city["origin"]["lat"]],
            }
        )
    )
    def get(item):
        name, spec = item
        path = out / (name + ".tif")
        resolution = spec["sample_metres"]
        if not isinstance(resolution, (int, float)) or not math.isfinite(resolution) or resolution <= 0:
            raise ValueError("Raster sample_metres must be finite and positive")
        size = round((bbox[2]-bbox[0]) / resolution)
        limit = spec.get("max_export_pixels", 4096)
        if type(limit) is not int or not 256 <= limit <= 8192 or not 1 <= size or size * size > 100_000_000:
            raise ValueError("Raster requests need a 256..8192 export bound and at most 100 megapixels total")
        pixel_size = (bbox[2] - bbox[0]) / size
        if abs(pixel_size-resolution) > 1e-9:
            raise ValueError("Acquisition bounds must align with sample_metres; no implicit rescaling")
        transform = rasterio.transform.from_origin(bbox[0], bbox[3], pixel_size, pixel_size)
        if path.exists():
            with rasterio.open(path) as cached:
                if cached.width != size or cached.height != size or not cached.transform.almost_equals(transform,precision=1e-7) or cached.crs != rasterio.crs.CRS.from_user_input(city["photographic"]["crs"]):
                    raise ValueError("Cached raster does not match this acquisition grid")
            validate_raster_coverage(dict(city,photographic=dict(city["photographic"],rasters={name:spec})), out)
            print("cached", name, flush=True)
            return
        exports, service_metadata = raster_export_plan(spec["url"],bbox,size,pixel_size,limit,city["photographic"]["crs"])
        chunks = [tuple(export["window"]) for export in exports]
        tiled = len(chunks) > 1
        folder = out / (name + "-tiles")
        if tiled:
            folder.mkdir(exist_ok=True)

        def fetch(chunk):
            x, y, width, height = chunk
            target = folder / f"{x}_{y}.tif" if tiled else path
            box = [bbox[0]+x*pixel_size, bbox[3]-(y+height)*pixel_size,
                   bbox[0]+(x+width)*pixel_size, bbox[3]-y*pixel_size]
            expected = rasterio.transform.from_origin(box[0], box[3], pixel_size, pixel_size)
            def validate():
                with rasterio.open(target) as image:
                    if (image.width, image.height) != (width, height) or image.crs != rasterio.crs.CRS.from_user_input(city["photographic"]["crs"]) or not image.transform.almost_equals(expected, precision=1e-7):
                        raise ValueError("Provider raster does not match the requested native grid")
                    if not image.read_masks().all() or (name != "aerial" and not np.isfinite(image.read()).all()):
                        raise ValueError("Provider raster contains unavailable samples; review source coverage before rendering")
            if target.exists():
                validate()
                return chunk, target
            params = {"f":"json", "bbox":",".join(map(str,box)),
                      "bboxSR":city["photographic"]["crs"], "imageSR":city["photographic"]["crs"],
                      "size":f"{width},{height}", "format":"tiff", "interpolation":"RSP_BilinearInterpolation",
                      "noDataInterpretation":"esriNoDataMatchAny"}
            if name == "aerial":
                params.update(pixelType="U8", bandIds="0,1,2")
            else:
                params.update(pixelType="F32", renderingRule=json.dumps({"rasterFunction":"None"}))
            response = requests.get(spec["url"] + "/exportImage", params=params, timeout=120)
            response.raise_for_status()
            payload = response.json()
            if "href" not in payload:
                raise RuntimeError(payload)
            data = requests.get(payload["href"], timeout=120)
            data.raise_for_status()
            target.write_bytes(data.content)
            validate()
            record = {"service":spec["url"], "request":params, "response":payload,
                      "sha256":hashlib.sha256(data.content).hexdigest(), "bytes":len(data.content)}
            record_path = target.with_suffix(".json") if tiled else out / (name + "-request.json")
            record_path.write_text(json.dumps(record,indent=2)+"\n")
            return chunk, target

        with ThreadPoolExecutor(max_workers=2) as pool:
            parts = list(pool.map(fetch, chunks))
        if tiled:
            with rasterio.open(parts[0][1]) as first:
                profile = first.profile.copy()
            profile.update(width=size, height=size, transform=transform, compress="deflate")
            temporary = out / (name + ".assembling.tif")
            with rasterio.open(temporary, "w", **profile) as whole:
                for (x,y,width,height), tile in parts:
                    with rasterio.open(tile) as image:
                        whole.write(image.read(), window=rasterio.windows.Window(x,y,width,height))
            temporary.replace(path)
            record = {"service":spec["url"], "method":"Aligned native raster windows; no resampling during assembly",
                      "bbox":bbox, "dimensions":[size,size], "pixel_size_metres":pixel_size,
                      "tiles":[str(p.relative_to(out)) for _,p in parts],
                      "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
            (out / (name+"-request.json")).write_text(json.dumps(record,indent=2)+"\n")
        (out / (name+"-coverage.json")).write_text(json.dumps({
            "service":spec["url"], "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "dimensions":[size,size], "max_mosaic_image_count":service_metadata.get("maxMosaicImageCount"),
            "service_metadata":service_metadata, "exports":exports,
        },indent=2)+"\n")
        print("fetched", name, size, "square pixels in", len(parts), "aligned request(s)", flush=True)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(get, city["photographic"]["rasters"].items()))


def acquire_mesh(city, out):
    nodes = out / "i3s"
    nodes.mkdir(exist_ok=True)
    base = city["photographic"]["i3s_layer"].rstrip("/") + "/nodes/"
    layer_record = out / "layer.json"
    if not layer_record.exists():
        response = requests.get(
            city["photographic"]["i3s_layer"], params={"f": "json"}, timeout=45
        )
        response.raise_for_status()
        layer_record.write_text(response.text)
    layer = json.loads(layer_record.read_text())
    store = layer.get("store", {})
    schema = store.get("defaultGeometrySchema", {})
    if store.get("version") != "1.6" or schema.get("ordering") != [
        "position",
        "normal",
        "uv0",
        "color",
    ]:
        raise ValueError(
            "This capture supports the documented uncompressed I3S 1.6 geometry schema only."
        )

    lon0, lat0 = city["origin"]["lon"], city["origin"]["lat"]
    span = city["photographic"]["source_span_m"] / 2
    metres_lon = 111320 * math.cos(math.radians(lat0))

    # Full-resolution leaves only. No parent and child geometry is rendered together.
    def relevant(m):
        x = (m[0] - lon0) * metres_lon
        y = (m[1] - lat0) * 111320
        r = m[3]
        return np.hypot(max(abs(x) - (span + 50), 0), max(abs(y) - (span + 50), 0)) <= r

    max_nodes = city["photographic"].get("mesh_max_nodes", 180)
    if type(max_nodes) is not int or not 1 <= max_nodes <= 2000:
        raise ValueError("mesh_max_nodes must be an integer in 1..2000")
    records = []
    front = city["photographic"].get("root_nodes", ["root"]).copy()
    leaves = []
    while front:
        front = list(dict.fromkeys(front))
        if len(records) + len(front) > max_nodes:
            raise ValueError("Configured I3S node bound exceeded")

        def get(n):
            p = nodes / (n + ".json")
            if not p.exists():
                r = requests.get(base + n, params={"f": "json"}, timeout=45)
                r.raise_for_status()
                p.write_text(r.text)
            j = json.loads(p.read_text())
            if "mbs" not in j:
                raise ValueError(j)
            return n, j

        with ThreadPoolExecutor(max_workers=6) as e:
            batch = list(e.map(get, front))
        front = []
        for n, j in batch:
            records.append(n)
            children = [
                c["id"] for c in (j.get("children") or []) if relevant(c["mbs"])
            ]
            if j.get("children"):
                front.extend(children)
            elif j.get("geometryData") and relevant(j["mbs"]):
                leaves.append(n)
        print(
            "nodes", len(records), "next", len(front), "leaves", len(leaves), flush=True
        )

    def getgeom(n):
        p = nodes / (n + ".bin")
        if not p.exists():
            r = requests.get(base + n + "/geometries/0", timeout=60)
            r.raise_for_status()
            p.write_bytes(r.content)
        b = p.read_bytes()
        vc, fc = np.frombuffer(b, "<u4", count=2)
        if len(b) < 8 + int(vc) * 36 + int(fc) * 16:
            raise ValueError("Unexpected geometry schema")
        v = (
            np.frombuffer(b, "<f4", offset=8, count=int(vc) * 3)
            .reshape(-1, 3)
            .astype(float)
        )
        j = json.loads((nodes / (n + ".json")).read_text())
        v += j["mbs"][:3]
        # Feature IDs and inclusive triangle ranges follow position, normal, uv and color arrays.
        offset = 8 + int(vc) * 36
        ids = np.frombuffer(b, "<u8", offset=offset, count=int(fc))
        ranges = np.frombuffer(
            b, "<u4", offset=offset + int(fc) * 8, count=int(fc) * 2
        ).reshape(-1, 2)
        owners = np.zeros(int(vc) // 3, np.uint64)
        for fid, (a, z) in zip(ids, ranges):
            owners[a : z + 1] = fid
        return v.reshape(-1, 3, 3), owners

    with ThreadPoolExecutor(max_workers=6) as e:
        chunks = list(e.map(getgeom, leaves))
    xyz = np.concatenate([c[0] for c in chunks])
    ids = np.concatenate([c[1] for c in chunks])
    centre = xyz.mean(axis=1)
    mask = (abs((centre[:, 0] - lon0) * metres_lon) < (span + 100)) & (
        abs((centre[:, 1] - lat0) * 111320) < (span + 100)
    )
    np.savez_compressed(out / "buildings-mesh.npz", lonlatz=xyz[mask], owners=ids[mask])
    (out / "mesh-acquisition.json").write_text(
        json.dumps(
            {
                "service": base,
                "nodes": records,
                "node_limit": max_nodes,
                "leaves": leaves,
                "triangles": int(mask.sum()),
                "unique_features": int(len(np.unique(ids[mask]))),
                "position_encoding": "Float32 offset from node MBS in layer coordinates; legacy I3S 1.6",
                "vertical_datum": city["photographic"]["mesh_vertical_note"],
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "saved",
        int(mask.sum()),
        "triangles",
        len(np.unique(ids[mask])),
        "buildings",
        flush=True,
    )


def radial_surface_detail(spec, centre, base, low, high):
    """Draw optional, evidence-labelled surface relief without moving geometry."""
    if not spec:
        return None
    if not spec.get("source_refs") or not spec.get("interpretation"):
        raise ValueError(
            "Landmark surface detail needs source references and an interpretation note."
        )
    count = spec.get("count")
    if type(count) is not int or count < 1:
        raise ValueError("Radial surface detail needs a positive integer count.")
    kind = spec.get("kind")
    if kind not in {"columns", "ribs"}:
        raise ValueError(f"Unsupported radial surface detail: {kind}")
    centre = np.asarray(centre)

    def shade(world, color, pixels, edge):
        relative = world[:, :2] - centre
        phase = (
            np.arctan2(relative[:, 1], relative[:, 0]) * count / (2 * np.pi) + 0.5
        ) % 1 - 0.5
        if kind == "ribs":
            radius = np.linalg.norm(relative, axis=1)
            rib = (abs(phase) < spec.get("half_width", 0.045)) & (radius > 1.0)
            factor = np.where(rib, 1.0 - spec.get("darkening", 0.16), 1.0)
            return np.clip(np.asarray(color)[None, :] * factor[:, None], 0, 255).astype(
                np.uint8
            )
        height = world[:, 2] - base
        shaft = (height > low + 0.7) & (height < high - 0.75)
        half_width = spec.get("column_fraction", 0.4) / 2
        column = abs(phase) < half_width
        relief = np.sqrt(np.maximum(0, 1 - (phase / half_width) ** 2))
        lighting = (0.76 + 0.24 * relief)[:, None]
        result = np.tile(np.asarray(color, float), (len(world), 1))
        result[shaft & ~column] = spec.get("recess_rgb", [92, 100, 98])
        selected = shaft & column
        result[selected] *= lighting[selected]
        return np.clip(result, 0, 255).astype(np.uint8)

    return shade


def linear_surface_detail(spec, centre, base, top):
    """Evidence-labelled column rhythm on an existing planar facade; no geometry edits."""
    if not spec.get("source_refs") or not spec.get("interpretation"):
        raise ValueError("Linear facade detail needs source references and an interpretation")
    count, length = spec.get("count"), spec.get("length_metres")
    if type(count) is not int or count < 1 or type(length) not in (int,float) or not math.isfinite(length) or length <= 0:
        raise ValueError("Linear facade detail needs positive count and length_metres")
    axis = {"x":0,"y":1}.get(spec.get("along_axis"))
    if axis is None:
        raise ValueError("Linear facade detail along_axis must be x or y")
    half_width = spec.get("column_fraction", .44)/2
    if not 0 < half_width < .5:
        raise ValueError("Linear column_fraction must be in 0..1")

    def shade(world, color, pixels, edge):
        relative = world[:,axis]-centre[axis]
        height = world[:,2]-base
        phase = ((relative+length/2)/(length/count))%1
        inside = abs(relative)<length/2
        shaft = inside & (height>spec.get("base_band_metres",2.2)) & (height<top-base-spec.get("top_band_metres",4))
        column = shaft & (abs(phase-.5)<half_width)
        light = np.sqrt(np.clip(1-((phase-.5)/half_width)**2,0,1))
        result = np.tile(np.asarray(color,float),(len(world),1))
        result[shaft & ~column] = spec.get("recess_rgb",[96,101,100])
        result[column] *= (.78+.25*light[column,None])
        steps = inside & (height<=spec.get("base_band_metres",2.2))
        result[steps] *= (.92+.08*((height[steps]*2.5)%1>.5))[:,None]
        return np.clip(result,0,255).astype(np.uint8)
    return shade


def validate_part_options(city):
    photographic = city["photographic"]
    corrections = photographic.get("part_min_height", {})
    if not isinstance(corrections, dict) or len(corrections) > 64 or set(corrections) - set(photographic.get("explicit_osm_parts", [])):
        raise ValueError("Part corrections must bind selected mapped source parts")
    def evidence(spec, fields):
        if (not isinstance(spec, dict) or set(spec) != set(fields) | {"source_refs", "interpretation"} or
            not isinstance(spec["source_refs"], list) or not 1 <= len(spec["source_refs"]) <= 16 or
            any(not isinstance(url,str) or not url.startswith(("https://","http://")) for url in spec["source_refs"]) or
            not isinstance(spec["interpretation"],str) or not spec["interpretation"].strip()):
            raise ValueError("Source part correction needs explicit fields, citations and interpretation")
    for correction in corrections.values():
        evidence(correction, {"metres"})
        value = correction["metres"]
        if type(value) not in (int,float) or not 0 <= value <= 2000 or not math.isfinite(value):
            raise ValueError("Source part lower bound must be finite and bounded")
    replacement = photographic.get("legacy_part_replacement")
    if replacement is not None:
        evidence(replacement, {"margin_metres"})
        value = replacement["margin_metres"]
        if type(value) not in (int,float) or not 0 <= value <= 25 or not math.isfinite(value):
            raise ValueError("Source replacement margin must be finite and bounded")


def render(city, out):
    OUT = out
    SRC = out / "source"
    start = time.time()
    landmark_layer = LandmarkLayer(city)
    validate_part_options(city)
    scene = SceneCoordinates(**city["origin"])
    project = Transformer.from_crs(4326, city["photographic"]["crs"], always_xy=True)
    cx, cy = project.transform(scene.lon, scene.lat)

    class Camera:
        scale = city["photographic"]["camera"]["pixels_per_metre"]
        width = city["photographic"]["camera"]["width"]
        height = city["photographic"]["camera"]["height"]
        upscale = 1
        # One camera basis is shared by ground, roofs, OSM parts and markers.
        az, el = np.radians(
            [
                city["photographic"]["camera"]["azimuth_degrees"],
                city["photographic"]["camera"]["elevation_degrees"],
            ]
        )
        horizontal_vector = np.array([-np.cos(az), np.sin(az), 0.0])
        vertical_vector = np.array(
            [np.sin(az) * np.sin(el), np.cos(az) * np.sin(el), -np.cos(el)]
        )
        depth_vector = np.array(
            [np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)]
        )
        target = np.array(city["photographic"]["camera"]["target_local_m"])

        def project(self, xyz):
            v = np.asarray(xyz) - self.target
            return np.stack(
                [
                    v @ self.horizontal_vector * self.scale + self.width / 2,
                    v @ self.vertical_vector * self.scale + self.height / 2,
                    v @ self.depth_vector,
                ],
                axis=-1,
            )

    camera = Camera()
    raster = Raster(
        camera, palette={"background_rgb": (225, 227, 217)}
    )

    class Sampler:
        def __init__(self, path):
            with rasterio.open(path) as f:
                self.a = f.read()
                self.tr = f.transform
                self.crs = str(f.crs)

        def at(self, x, y):
            c = (np.asarray(x) - self.tr.c) / self.tr.a - 0.5
            r = (np.asarray(y) - self.tr.f) / self.tr.e - 0.5
            coords = [np.asarray(r).ravel(), np.asarray(c).ravel()]
            z = np.stack(
                [map_coordinates(a, coords, order=1, mode="nearest") for a in self.a],
                -1,
            )
            return z.reshape(
                np.broadcast_shapes(np.shape(x), np.shape(y)) + (len(self.a),)
            )

    aerial = Sampler(SRC / "aerial.tif")
    dtm = Sampler(SRC / "dtm.tif")

    def photo(world, color, pixels, edge):
        return aerial.at(world[:, 0] + cx, world[:, 1] + cy).astype(np.uint8)

    # Ground and roofs share the same world-coordinate photograph, projection and lattice.
    span = city["photographic"]["source_span_m"] / 2
    cell = city["photographic"]["terrain_cell_m"]
    xs = np.arange(-span, span + cell, cell)
    ys = np.arange(-span, span + cell, cell)
    xx, yy = np.meshgrid(xs, ys)
    zz = dtm.at(xx + cx, yy + cy)[..., 0]
    vertices = np.stack([xx, yy, zz], -1)
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            xyz = vertices[j : j + 2, i : i + 2].reshape(4, 3)[[0, 1, 3, 2]]
            p = camera.project(xyz)
            if (
                p[:, 0].max() < 0
                or p[:, 0].min() > camera.width
                or p[:, 1].max() < 0
                or p[:, 1].min() > camera.height
            ):
                continue
            raster.triangle(xyz[[0, 1, 2]], (120, 120, 120), shader=photo)
            raster.triangle(xyz[[0, 2, 3]], (120, 120, 120), shader=photo)
    Image.fromarray(raster.rgb).save(OUT / "ground-aerial.png")
    mesh = np.load(SRC / "buildings-mesh.npz")
    geo = mesh["lonlatz"]
    owners = mesh["owners"]
    landmark_layer.validate_owners(owners)
    x, y = project.transform(geo[:, :, 0], geo[:, :, 1])
    xyz = np.stack(
        [
            x - cx,
            y - cy,
            geo[:, :, 2] + city["photographic"].get("mesh_vertical_offset_m", 0),
        ],
        -1,
    )
    # Repeated features in the provider contain duplicated triangles. Preserve just one geometric copy.
    rounded = np.round(xyz, 2)
    order = np.lexsort((rounded[:, :, 2], rounded[:, :, 1], rounded[:, :, 0]), axis=1)
    keys = np.take_along_axis(rounded, order[:, :, None], axis=1).reshape(-1, 9)
    _, unique = np.unique(keys, axis=0, return_index=True)
    xyz = xyz[unique]
    owners = owners[unique]
    # The legacy layer lacks a declared vertical CRS. Its lowest vertices agree with NAVD88 ground without any shift.
    residual = []
    for oid in np.unique(owners):
        v = xyz[owners == oid].reshape(-1, 3)
        lo = v[v[:, 2] < v[:, 2].min() + 0.1]
        g = dtm.at(lo[:, 0] + cx, lo[:, 1] + cy)[:, 0]
        residual.append(float(np.median(g - lo[:, 2])))
    print(
        "ground rendered",
        round(time.time() - start, 1),
        "seconds; mesh triangles",
        len(xyz),
        flush=True,
    )
    # OSM supplies footprint references and names. The city's roof mesh supplies the visible 3D surfaces.
    features, osm_report = load_features(
        osm_input(city, out), scene
    )
    buildings = [
        f for f in features if "building" in f.tags or "building:part" in f.tags
    ]
    landmarks = []
    for point in city["landmarks"]:
        name, lat, lon = point["name"], point["lat"], point["lon"]
        e, n = project.transform(lon, lat)
        z = float(dtm.at(e, n)[0])
        sx, sy, _ = camera.project([e - cx, n - cy, z])
        landmarks.append(
            {"id":point.get("id"), "name": name, "lat": lat, "lon": lon, "x": float(sx), "y": float(sy)}
        )
    SUN = np.array([-1, -2, 3.0])
    SUN /= np.linalg.norm(SUN)

    class Wall:
        def __init__(self, n, base):
            self.n = n
            self.along = np.array([-n[1], n[0], 0])
            self.along /= max(np.linalg.norm(self.along), 1e-9)
            self.base = base

        def __call__(self, world, color, pixels, edge):
            u = world @ self.along
            z = world[:, 2]
            a = u / 3.1
            b = (z - self.base) / 4.2
            fu = a - np.floor(a)
            fv = b - np.floor(b)
            window = (
                (fu > 0.30)
                & (fu < 0.69)
                & (fv > 0.27)
                & (fv < 0.74)
                & (z - self.base > 5)
                & (edge > 0.55)
            )
            v = np.tile(color, (len(world), 1)).astype(np.int16)
            v[window] = (59, 73, 75)
            course = np.mod(z * 2.7, 1) < 0.05
            v[course & ~window] -= 5
            return np.clip(v, 0, 255).astype(np.uint8)

    # Explicit OSM dome parts replace the coarse legacy dome, not the photographic roof wings.
    from dataclasses import replace

    centre = city["photographic"]["osm_parts_centre"]
    capitol_x, capitol_y = project.transform(centre["lon"], centre["lat"])
    dome_parts = []
    for f in buildings:
        lon, lat = scene.lonlat(f.geometry.centroid.x, f.geometry.centroid.y)
        e, n = project.transform(lon, lat)
        if (
            f.tags.get("building:part")
            and f.osm_id in city["photographic"]["explicit_osm_parts"]
        ):

            def convert(coords):
                lon, lat = scene.lonlat(coords[:, 0], coords[:, 1])
                e, n = project.transform(lon, lat)
                return np.column_stack([e - cx, n - cy])

            adjustment = city["photographic"].get("part_min_height", {}).get(f.osm_id)
            if adjustment is not None:
                if (set(adjustment) != {"metres", "source_refs", "interpretation"} or
                    type(adjustment["metres"]) not in (int,float) or
                    not math.isfinite(adjustment["metres"]) or
                    not f.min_height <= adjustment["metres"] < f.height or
                    not adjustment["source_refs"] or not adjustment["interpretation"]):
                    raise ValueError("Mapped part lower-bound correction needs bounded dimensions and source evidence")
                f = replace(f,min_height=adjustment["metres"])
            dome_parts.append(replace(f, geometry=transform_geom(f.geometry, convert)))
    requested_parts = set(city["photographic"]["explicit_osm_parts"])
    found_parts = {f.osm_id for f in dome_parts}
    if missing_parts := requested_parts - found_parts:
        raise ValueError(
            f"Mapped landmark parts are missing: {sorted(missing_parts)}. Review this source before styling."
        )
    replacement = city["photographic"].get("legacy_part_replacement")
    margin = .5
    if replacement is not None:
        if (set(replacement) != {"margin_metres", "source_refs", "interpretation"} or
            type(replacement["margin_metres"]) not in (int,float) or
            not math.isfinite(replacement["margin_metres"]) or
            not 0 <= replacement["margin_metres"] <= 25 or
            not replacement["source_refs"] or not replacement["interpretation"]):
            raise ValueError("Legacy replacement support needs a bounded mapped margin and source evidence")
        margin = replacement["margin_metres"]
    dome_union = unary_union([f.geometry for f in dome_parts]).buffer(margin)
    base_dome = float(dtm.at(capitol_x, capitol_y)[0])
    centres = xyz.mean(axis=1)
    remove = contains_xy(dome_union, centres[:, 0], centres[:, 1]) & (
        centres[:, 2] > base_dome + 24
    )
    xyz = xyz[~remove]
    owners = owners[~remove]
    linear_details = []
    for osm_id, spec in city["photographic"].get("linear_facade_details", {}).items():
        matches = [f for f in buildings if f.osm_id == osm_id]
        if len(matches) != 1:
            raise ValueError(f"Linear facade footprint missing or ambiguous: {osm_id}")
        def projected(coords):
            lon,lat=scene.lonlat(coords[:,0],coords[:,1]);e,n=project.transform(lon,lat)
            return np.column_stack([e-cx,n-cy])
        poly = transform_geom(matches[0].geometry, projected)
        support = poly.buffer(spec.get("support_margin_metres",4))
        middle = np.array([poly.centroid.x,poly.centroid.y])
        selected = contains_xy(support,xyz.mean(1)[:,0],xyz.mean(1)[:,1])
        if not selected.any():
            raise ValueError(f"No source mesh supports linear facade detail: {osm_id}")
        top = float(xyz[selected][:,:,2].max())
        linear_surface_detail(spec,middle,0,top)  # Validate before any wall is drawn.
        if spec.get("face_side") not in {"min","max"}:
            raise ValueError("Linear facade face_side must be min or max")
        linear_details.append((spec,support,middle,top))
    roof_count = wall_count = linear_count = 0
    for k, (tri, oid) in enumerate(zip(xyz, owners)):
        if int(oid) in landmark_layer.replaced:
            continue
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        area = np.linalg.norm(normal)
        if area < 1e-5:
            continue
        normal /= area
        if normal[2] < 0:
            normal = -normal
        if abs(normal[2]) > 0.22:
            raster.triangle(tri, (160, 160, 160), bias=0.02, shader=photo)
            roof_count += 1
        else:
            # Photographic nadir pixels never stretch down vertical walls. These facades are illustrative.
            shade = 0.70 + 0.28 * abs(normal @ SUN)
            color = tuple(int(v * shade) for v in landmark_layer.wall_colour(oid,(222,219,207)))
            mid = tri.mean(0)
            base = float(dtm.at(mid[0] + cx, mid[1] + cy)[0])
            shader = Wall(normal,base)
            shader = landmark_layer.shader(oid,normal,base,shader)
            for spec,support,middle,top in linear_details:
                along = {"x":0,"y":1}[spec["along_axis"]]; face = 1-along
                side = -1 if spec["face_side"] == "min" else 1
                if abs(normal[face])>.8 and side*(mid[face]-middle[face])>spec.get("face_offset_metres",10) and abs(mid[along]-middle[along])<spec.get("selection_half_width_metres",spec["length_metres"]/2) and contains_xy(support,mid[0],mid[1]):
                    shader = linear_surface_detail(spec,middle,base,top)
                    linear_count += 1
                    break
            for fragment in landmark_layer.fragments(oid,tri):
                raster.triangle(fragment,color,shader=shader)
            wall_count += 1
        if k % 10000 == 0:
            print("mesh", k, "/", len(xyz), flush=True)
    for f in sorted(dome_parts, key=lambda f: f.height):
        base = base_dome
        roof_rise = f.roof_height if f.roof_shape == "dome" else 0
        walltop = f.height - roof_rise
        if roof_rise:
            rings = [
                {
                    "radius": float(np.cos(a)),
                    "z": float(walltop + roof_rise * np.sin(a)),
                }
                for a in np.linspace(0, np.pi / 2, 17)
            ]
            rings[-1]["radius"] = 0
            roof_faces = radial_surface(
                f,
                {
                    "rings": rings,
                    "ribs": 0,
                    "source_refs": ["https://www.openstreetmap.org/" + f.osm_id],
                },
            )
        else:
            roof_faces = [
                np.column_stack([t, np.full(3, f.height)])
                for t in triangles(f.geometry)
            ]
        bronze = f.tags.get("building:material") == "bronze"
        for ring in [f.geometry.exterior, *f.geometry.interiors]:
            pts = np.array(ring.coords)
            for a, b in zip(pts[:-1], pts[1:]):
                wall = np.array(
                    [
                        [*a, base + f.min_height],
                        [*b, base + f.min_height],
                        [*b, base + walltop],
                        [*a, base + walltop],
                    ]
                )
                normal = np.array([b[1] - a[1], a[0] - b[0], 0])
                normal /= max(np.linalg.norm(normal), 1e-9)
                shade = 0.74 + 0.25 * abs(normal @ SUN)
                color = tuple(
                    int(v * shade)
                    for v in ((63, 71, 70) if bronze else (242, 241, 230))
                )
                shader = None if f.tags.get("building:material") in {"marble", "bronze"} else Wall(normal, base)
                detail = (
                    city["photographic"].get("landmark_materials", {}).get(f.osm_id, {})
                )
                shader = (
                    radial_surface_detail(
                        detail.get("wall"),
                        [f.geometry.centroid.x, f.geometry.centroid.y],
                        base,
                        f.min_height,
                        walltop,
                    )
                    or shader
                )
                raster.triangle(wall[[0, 1, 2]], color, bias=0.04, shader=shader)
                raster.triangle(wall[[0, 2, 3]], color, bias=0.04, shader=shader)
        for tri in roof_faces:
            tri = tri.copy()
            tri[:, 2] += base
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            normal /= max(np.linalg.norm(normal), 1e-9)
            shade = 0.78 + 0.22 * abs(normal @ SUN)
            color = tuple(
                int(v * shade) for v in ((66, 74, 73) if bronze else (245, 244, 233))
            )
            detail = (
                city["photographic"].get("landmark_materials", {}).get(f.osm_id, {})
            )
            shader = radial_surface_detail(
                detail.get("roof"),
                [f.geometry.centroid.x, f.geometry.centroid.y],
                base,
                f.min_height,
                walltop,
            )
            raster.triangle(tri, color, bias=0.05, shader=shader)
    landmark_geometry = landmark_layer.draw(raster,lambda x,y:dtm.at(x,y)[...,0])
    (OUT / "landmark-geometry.json").write_text(json.dumps(landmark_geometry,indent=2)+"\n")
    Image.fromarray(raster.rgb).save(OUT / "source.png")
    # Save a source-grounded footprint map without making painted roof edges into geometry claims.
    a = Image.fromarray(aerial.a[:3].transpose(1, 2, 0).astype(np.uint8)).resize(
        (1200, 1200)
    )
    draw = ImageDraw.Draw(a)
    with rasterio.open(SRC / "aerial.tif") as f:
        bounds = f.bounds
    for f in buildings:
        for poly in (
            [f.geometry]
            if f.geometry.geom_type == "Polygon"
            else list(getattr(f.geometry, "geoms", []))
        ):
            if not isinstance(poly, Polygon):
                continue
            pts = np.array(poly.exterior.coords)
            lon, lat = scene.lonlat(pts[:, 0], pts[:, 1])
            e, n = project.transform(lon, lat)
            px = (e - bounds.left) / (bounds.right - bounds.left) * a.width
            py = (bounds.top - n) / (bounds.top - bounds.bottom) * a.height
            draw.line(list(zip(px, py)), fill=(255, 205, 51), width=1)
    a.save(OUT / "osm-footprints-over-aerial.png")
    report = {
        "status": "source study; not accepted art",
        "landmark_geometry": landmark_geometry,
        "landmark_geometry_recipe": city["photographic"].get("landmark_geometry",{}),
        "credits": city["photographic"].get("attribution", ""),
        "source_refs": city["photographic"].get("source_refs", []),
        "camera": {
            "type": "orthographic",
            "azimuth_degrees": city["photographic"]["camera"]["azimuth_degrees"],
            "elevation_degrees": city["photographic"]["camera"]["elevation_degrees"],
            "width": camera.width,
            "height": camera.height,
            "pixels_per_metre": camera.scale,
            "horizontal": camera.horizontal_vector.tolist(),
            "vertical": camera.vertical_vector.tolist(),
            "depth": camera.depth_vector.tolist(),
            "target_local_m": camera.target.tolist(),
            "origin_projected_metres": [cx, cy],
            "pixel_matrix": [
                list(camera.horizontal_vector * camera.scale)
                + [
                    camera.width / 2
                    - camera.target @ camera.horizontal_vector * camera.scale
                ],
                list(camera.vertical_vector * camera.scale)
                + [
                    camera.height / 2
                    - camera.target @ camera.vertical_vector * camera.scale
                ],
            ],
        },
        "mesh": {
            "triangles_before_deduplication": int(len(geo)),
            "triangles_after_deduplication": len(xyz),
            "osm_dome_parts": [
                {
                    "osm_id": f.osm_id,
                    "height_m": f.height,
                    "min_height_m": f.min_height,
                    "roof_shape": f.roof_shape,
                }
                for f in dome_parts
            ],
            "legacy_dome_triangles_replaced": int(remove.sum()),
            "roof_triangles": roof_count,
            "wall_triangles": wall_count,
            "linear_facade_triangles": linear_count,
            "provider_features": len(np.unique(owners)),
            "lowest_vertex_minus_terrain_check": {
                "terrain_minus_vertex_p10_p50_p90_m": np.percentile(
                    residual, [10, 50, 90]
                ).tolist(),
                "vertical_shift_applied_m": city["photographic"].get(
                    "mesh_vertical_offset_m", 0
                ),
                "interpretation": "Empirical ground agreement, not a declared source datum or survey accuracy guarantee.",
            },
        },
        "osm": {
            "elements": osm_report["osm_elements"],
            "building_footprints": len(buildings),
            "use": "Footprint reference overlay and explicit landmark part geometry; city mesh supplies remaining visible roofs and walls.",
        },
        "aerial": {
            "year": city["photographic"]["rasters"]["aerial"]["year"],
            "source_gsd_m": city["photographic"]["rasters"]["aerial"]["native_metres"],
            "downloaded_gsd_m": city["photographic"]["rasters"]["aerial"][
                "sample_metres"
            ],
            "crs": aerial.crs,
            "use": "World-XY sampling on visible roof faces and terrain only, never on vertical walls.",
        },
        "terrain": {
            "year": city["photographic"]["rasters"]["dtm"]["year"],
            "gsd_m": 1,
            "mesh_cell_m": cell,
            "vertical_exaggeration": 1,
        },
        "landmarks": landmarks,
        "linear_facade_details": city["photographic"].get("linear_facade_details", {}),
        "limitations": [
            "Legacy 2013-era building mesh may omit or simplify ornaments, dome, columns and newer changes.",
            "Illustrative neutral facades with drawn windows, not photographic facades.",
            "Aerial imagery has shadows, trees, relief displacement and date mismatch.",
            "Vegetation is present in ground photography; redacted LiDAR cannot supply current canopy height here.",
            "This patch has not been accepted as art.",
        ] + city["photographic"].get("source_limits", []),
        "seconds": round(time.time() - start, 2),
        "sha256": hashlib.sha256((OUT / "source.png").read_bytes()).hexdigest(),
    }
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "image": str(OUT / "source.png"),
                "seconds": report["seconds"],
                "landmarks": landmarks,
            }
        ),
        flush=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="capitol-hill-dc")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    recipe = Path(args.city)
    if not recipe.is_file():
        recipe = ROOT / "cities" / f"{args.city}.json"
    city = json.loads(recipe.read_text())
    if "photographic" not in city:
        parser.error(
            "This city needs a photographic source section. See docs/sources.md."
        )
    try:
        LandmarkLayer(city)
        validate_part_options(city)
    except ValueError as exc:
        parser.error(str(exc))
    out = (args.out or ROOT / "outputs/photographic" / city["id"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = out / "source"
    if args.offline:
        validate_raster_coverage(city, source)
    source.mkdir(exist_ok=True)
    identity = {
        "id": city["id"],
        "origin": city["origin"],
        "photographic": {k: v for k, v in city["photographic"].items() if k != "style"},
    }
    cache_record = source / "recipe.json"
    if cache_record.exists() and json.loads(cache_record.read_text()) != identity:
        parser.error(
            "Source recipe changed. Use a new --out directory to preserve the cached run."
        )
    previous = out / "replay.json"
    if previous.exists():
        saved = json.loads(previous.read_text())
        for name, expected in saved["input_hashes"].items():
            cached = out / name
            if (
                not cached.is_file()
                or hashlib.sha256(cached.read_bytes()).hexdigest() != expected
            ):
                parser.error(
                    f"Cached input changed: {name}. Use a new output directory or restore it."
                )
    width, height = (
        city["photographic"]["camera"]["width"],
        city["photographic"]["camera"]["height"],
    )
    if (
        any(type(n) is not int or not 256 <= n <= 8192 for n in [width, height])
        or width * height > 33_554_432
    ):
        parser.error(
            "Photographic source renders must fit within 8192 pixels per side and 32 megapixels; model conditioning limits are separate."
        )
    cache_record.write_text(json.dumps(identity, indent=2) + "\n")
    if not args.offline:
        from fetch_osm import fetch

        acquisition = dict(city)
        acquisition["data_dir"] = str(source)
        # This capture uses its configured DTM rather than the authored renderer's Skadi fallback.
        acquisition["terrain"] = {"provider":"lidar", "fallback":"error", "cache":str(source/"dtm.tif")}
        fetch(acquisition)
        acquire_rasters(city, source)
        acquire_mesh(city, source)
    osm = source / "osm.json"
    if not osm.exists():
        import shutil
        shutil.copy2(osm_input(city, out), osm)
    if (
        previous.exists()
        and saved.get("osm_sha256")
        and hashlib.sha256(osm.read_bytes()).hexdigest() != saved["osm_sha256"]
    ):
        parser.error(
            "OSM cache changed. Use a new output directory to retain the original registration."
        )
    render(city, out)
    record = {
        "recipe": city,
        "osm_sha256": hashlib.sha256(
            osm.read_bytes()
        ).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_hashes": {
            str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.rglob("*")
            if p.is_file()
        },
    }
    (out / "replay.json").write_text(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
