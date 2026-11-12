#!/usr/bin/env python3
"""Render a registered orthographic DSM/DTM city source. No image model runs here.

The mesh follows measured roofs and terrain; OSM supplies material classes.
Run --city edinburgh --out outputs/edinburgh-source, then repeat with --offline.
See docs/sources.md for inputs, camera conventions and limits.
"""
import argparse, json, hashlib, math, re, sys, fcntl
import importlib.metadata
from pathlib import Path
import numpy as np
from numba import njit
from PIL import Image, ImageDraw
from pyproj import Transformer
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]

# evidence palette (linear-ish sRGB albedo before lighting)
DEFAULT_PALETTE = dict(
    slate=(112, 116, 126), flat_roof=(168, 170, 168), sandstone=(170, 158, 138), sandstone_dark=(134, 124, 110),
    modern=(196, 196, 190), glass=(150, 178, 190), copper=(112, 164, 142), lead=(150, 156, 160),
    castle_stone=(150, 140, 120), grass=(104, 132, 70), tree=(62, 96, 48), scrub=(88, 110, 60),
    rock=(92, 88, 82), asphalt=(92, 94, 98), sett=(128, 122, 114), pavement=(176, 170, 158),
    ballast=(120, 112, 104), rail=(70, 66, 62), water=(70, 110, 130), parking=(118, 118, 120),
    path=(190, 178, 150), structure=(150, 142, 128), plaza=(184, 176, 160), sand=(200, 188, 150))
COLOURS = {'grey': (128, 130, 134), 'lightgrey': (176, 178, 180), 'black': (60, 60, 64), 'white': (226, 226, 222),
           'blue': (90, 110, 150), 'beige': (200, 186, 150), 'green': (100, 150, 110), 'red': (160, 70, 60), 'yellow': (210, 190, 110)}


def hexcol(v):
    v = (v or '').strip().lower()
    if v in COLOURS: return COLOURS[v]
    m = re.fullmatch(r'#?([0-9a-f]{6})', v)
    return tuple(int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4)) if m else None


def roof_from_tags(t, palette=None):
    P = palette or DEFAULT_PALETTE
    c = hexcol(t.get('roof:colour'))
    m = t.get('roof:material', '')
    if m == 'copper': return P['copper']
    if m == 'glass': return P['glass']
    if m in ('metal', 'lead'): return P['lead']
    if m == 'grass': return P['grass']
    if c and c != (0x70, 0x55, 0x44): return c          # #705544 is a blanket castle-part tag
    if m == 'slate': return P['slate']
    if m in ('concrete', 'tar_paper', 'asphalt'): return P['flat_roof']
    return None


def wall_from_tags(t, palette=None):
    P = palette or DEFAULT_PALETTE
    c = hexcol(t.get('building:colour'))
    m = t.get('building:material', '')
    if c and c not in ((0x70, 0x55, 0x44),): return c
    if m == 'glass': return P['glass']
    if m in ('concrete', 'plaster'): return P['modern']
    if m == 'brick': return (150, 90, 70)
    if t.get('historic') == 'castle' or 'castle' in t.get('name', '').lower(): return P['castle_stone']
    if m in ('sandstone', 'stone'): return P['sandstone']
    try:
        if int(str(t.get('start_date', '0'))[:4]) >= 1950: return P['modern']
    except ValueError:
        pass
    if t.get('building') in ('office', 'commercial', 'retail', 'hotel') and 'start_date' not in t: return None
    return None


def load_osm(path, tf, W, N, res, palette=None):
    els = json.loads(Path(path).read_text())['elements']
    def rings(e):
        if e['type'] == 'way' and 'geometry' in e:
            yield 'outer', e['geometry']
        elif e['type'] == 'relation':
            for m in e.get('members', []):
                if m.get('type') == 'way' and 'geometry' in m: yield m.get('role') or 'outer', m['geometry']
    out = []
    for e in els:
        t = e.get('tags', {})
        if not t: continue
        rr = []
        for role, g in rings(e):
            lon = np.array([p['lon'] for p in g]); lat = np.array([p['lat'] for p in g])
            x, y = tf.transform(lon, lat)
            rr.append((role, np.c_[(x - W) / res, (N - y) / res]))
        if rr: out.append((e, t, rr))
    return out


def raster_layers(feats, shape, resolution=.5, palette=None):
    H, Wd = shape
    def canvas(mode='L', v=0): return Image.new(mode, (Wd, H), v)
    ground = canvas(v=1); gd = ImageDraw.Draw(ground)
    bid = canvas('I', 0); bd = ImageDraw.Draw(bid)
    G = dict(pavement=1, grass=2, wood=3, scrub=4, rock=5, asphalt=6, sett=7, ballast=8, water=9, parking=10, path=11, plaza=12, sand=13)
    attrs = [None]
    areas, lines = [], []
    for e, t, rr in feats:
        closed = any(len(r) > 3 and np.allclose(r[0], r[-1]) for _, r in rr)
        k = None
        lu, nat, le, hw = t.get('landuse'), t.get('natural'), t.get('leisure'), t.get('highway')
        if 'building' in t or 'building:part' in t:
            if closed and t.get('building') not in ('roof',) and t.get('location') != 'underground':
                attrs.append(dict(osm=f"{e['type']}/{e['id']}", name=t.get('name'), roof=roof_from_tags(t, palette), wall=wall_from_tags(t, palette),
                                  part='building:part' in t, levels=t.get('building:levels')))
                for role, r in rr:
                    pts = [tuple(p) for p in r]
                    if len(pts) > 2: bd.polygon(pts, fill=len(attrs) - 1 if role == 'outer' else 0)
            continue
        if closed:
            if lu in ('grass', 'recreation_ground', 'cemetery', 'village_green', 'meadow') or le in ('park', 'garden', 'pitch', 'playground') or nat == 'grassland' or t.get('amenity') == 'grave_yard': k = 'grass'
            elif nat in ('wood', 'tree_row') or lu == 'forest': k = 'wood'
            elif nat in ('scrub', 'heath'): k = 'scrub'
            elif nat in ('bare_rock', 'cliff', 'scree'): k = 'rock'
            elif nat == 'water' or t.get('water') or lu == 'basin': k = 'water'
            elif t.get('amenity') == 'parking': k = 'parking'
            elif hw in ('pedestrian', 'footway') or t.get('place') == 'square' or t.get('area:highway'): k = 'plaza'
            elif nat == 'sand' or le == 'sandpit': k = 'sand'
            elif lu == 'railway': k = 'ballast'
            if k: areas.append((G[k], rr, 2 if k in ('plaza', 'parking') else 1))
        if hw and not (closed and t.get('area') == 'yes'):
            wmap = {'primary': 14, 'secondary': 12, 'tertiary': 10, 'unclassified': 8, 'residential': 8, 'service': 5, 'pedestrian': 7,
                    'living_street': 7, 'footway': 2.5, 'path': 2, 'steps': 2.5, 'cycleway': 2.5, 'trunk': 16, 'primary_link': 8}
            w = wmap.get(hw)
            if w and t.get('tunnel') != 'yes' and t.get('layer', '0') not in ('-1', '-2'):
                if hw in ('footway', 'path', 'steps', 'cycleway'): kind = 'path' if t.get('surface') in ('gravel', 'dirt', 'compacted', 'fine_gravel') else 'pavement'
                elif t.get('surface') in ('sett', 'cobblestone', 'paving_stones', 'unhewn_cobblestone'): kind = 'sett'
                elif hw == 'pedestrian': kind = 'plaza'
                else: kind = 'asphalt'
                lines.append((G[kind], rr, w + (4 if kind == 'asphalt' else 0), kind))
        if t.get('railway') in ('rail', 'light_rail', 'tram') and t.get('tunnel') != 'yes':
            lines.append((G['ballast'], rr, 6 if t['railway'] == 'rail' else 3, 'rail'))
    for lvl in (1, 2):
        for g, rr, l in areas:
            if l != lvl: continue
            for role, r in rr:
                if len(r) > 2: gd.polygon([tuple(p) for p in r], fill=g if role == 'outer' else 1)
    # pavement-edged roads: first a wider pavement stroke, then the carriageway
    for g, rr, w, kind in lines:
        if kind == 'asphalt':
            for _, r in rr: gd.line([tuple(p) for p in r], fill=G['pavement'], width=int(round((w + 6) / resolution)), joint='curve')
    for g, rr, w, kind in sorted(lines, key=lambda l: l[2]):
        for _, r in rr: gd.line([tuple(p) for p in r], fill=g, width=max(1, int(round(w / resolution))), joint='curve')
    return np.asarray(ground), np.asarray(bid).astype(np.int32), attrs, G


@njit
def raster(X, Y, D, Hg, walla, wshade, opn, colr, nz, out, zb, Wd, Hh):
    ny, nx = X.shape
    for i in range(ny - 1):
        for j in range(nx - 1):
            for tri in range(2):
                if tri == 0:
                    a0, a1, b0, b1, c0, c1 = i, j, i + 1, j, i, j + 1
                else:
                    a0, a1, b0, b1, c0, c1 = i + 1, j, i + 1, j + 1, i, j + 1
                xa, ya, xb, yb, xc, yc = X[a0, a1], Y[a0, a1], X[b0, b1], Y[b0, b1], X[c0, c1], Y[c0, c1]
                minx = max(int(math.floor(min(xa, xb, xc))), 0); maxx = min(int(math.ceil(max(xa, xb, xc))), Wd - 1)
                miny = max(int(math.floor(min(ya, yb, yc))), 0); maxy = min(int(math.ceil(max(ya, yb, yc))), Hh - 1)
                if minx > maxx or miny > maxy: continue
                den = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc)
                if abs(den) < 1e-9: continue
                # face normal from grid (world) coordinates handled outside: nz per cell/tri
                n = nz[i, j, tri]
                steep = n < 0.55
                hi0, hi1 = a0, a1
                if Hg[b0, b1] > Hg[hi0, hi1]: hi0, hi1 = b0, b1
                if Hg[c0, c1] > Hg[hi0, hi1]: hi0, hi1 = c0, c1
                for py in range(miny, maxy + 1):
                    for px in range(minx, maxx + 1):
                        fx, fy = px + .5, py + .5
                        wa = ((yb - yc) * (fx - xc) + (xc - xb) * (fy - yc)) / den
                        wb = ((yc - ya) * (fx - xc) + (xa - xc) * (fy - yc)) / den
                        wc = 1 - wa - wb
                        if wa < -1e-4 or wb < -1e-4 or wc < -1e-4: continue
                        d = wa * D[a0, a1] + wb * D[b0, b1] + wc * D[c0, c1]
                        if d < zb[py, px]:
                            zb[py, px] = d
                            if steep:
                                for k in range(3): out[py, px, k] = min(255.0, walla[hi0, hi1, k] * wshade[i, j] * opn[hi0, hi1])
                            else:
                                for k in range(3): out[py, px, k] = colr[i, j, tri, k]


def render(city, surface, osm, grid, out, *, scale=None, variant=None):
    spec = validate(city['surface'])
    camera = spec['camera']
    CENTRE = camera['centre_projected_m']
    U_RANGE, Y_RANGE = camera['u_range_m'], camera['y_range_screen_m']
    Z0 = camera['z0_m']
    elevation = camera['elevation_deg']
    SIN = .5 if elevation == 30 else math.sin(math.radians(elevation))
    COS = math.sqrt(3) / 2 if elevation == 30 else math.cos(math.radians(elevation))
    s = float(scale if scale is not None else camera['pixels_per_metre'])
    variant = variant or spec.get('variant', 'edges')
    if not math.isfinite(s) or not .1 <= s <= 20 or variant not in ('plain', 'edges'):
        raise ValueError('Scale must be .1..20 pixels/m and variant plain or edges')
    width, height = round((U_RANGE[1]-U_RANGE[0])*s), round((Y_RANGE[1]-Y_RANGE[0])*s)
    if not (1 <= width <= 8192 and 1 <= height <= 8192 and width*height <= 32_000_000):
        raise ValueError('Source camera exceeds 8192 sides or 32 MP')
    P = {**DEFAULT_PALETTE, **spec.get('palette', {})}
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    L = np.load(surface)
    dsm, dtm = L['dsm'].astype(np.float32), L['dtm'].astype(np.float32)
    W, S_, E, N = L['bounds']
    res = float(spec['resolution_m']); ny, nx = dsm.shape
    if dsm.shape != dtm.shape or not np.isfinite(dsm).all() or not np.isfinite(dtm).all():
        raise ValueError('DSM/DTM must share a finite, complete grid')
    if not np.array_equal(L['bounds'], np.asarray(spec['bounds_projected_m'])):
        raise ValueError('Surface cache bounds do not match the city recipe')
    if dsm.size > 64_000_000 or dsm.shape != (round((N-S_)/res), round((E-W)/res)):
        raise ValueError('Surface bounds, shape or resolution mismatch, or more than 64M cells')
    tf = transformer(spec, grid)
    feats = load_osm(osm, tf, W, N, res, P)
    ground, bid, attrs, G = raster_layers(feats, dsm.shape, res, P)
    landmarks = None
    if spec.get('landmarks'):
        from landmark_overlay import Overlay
        landmarks = Overlay(spec, L, surface, osm, grid, scale=s)
    ndsm = dsm - dtm
    # building cells: footprint (1-cell dilation for edge pixels) and raised above ground
    foot = bid > 0
    _, (ri, rj) = ndi.distance_transform_edt(~foot, return_indices=True)
    dist = ndi.distance_transform_edt(~foot)
    bidd = np.where(dist <= 2.0, bid[ri, rj], 0)          # 1 m tolerance for OSM/LiDAR offset
    bmask = (bidd > 0) & (ndsm > 1.5)
    # roughness separates canopy from masonry
    hp = dsm - ndi.uniform_filter(dsm, 5)
    rough = np.sqrt(ndi.uniform_filter(hp * hp, 5))
    road = np.isin(ground, [G['asphalt'], G['sett'], G['parking'], G['ballast']])
    canopy = (~bmask) & (dist > 3) & (ndsm > 2.0) & (rough > 0.35)
    canopy = ndi.binary_opening(canopy, iterations=1)
    # smooth crown tops have low roughness: close and fill the canopy so tops stay green
    canopy = ndi.binary_fill_holes(ndi.binary_closing(canopy, iterations=3)) & (~bmask) & (dist > 3) & (ndsm > 1.0)
    structure = (~bmask) & (~canopy) & (ndsm > 1.2) & ~(road & (ndsm < 4.5))
    H = np.where(bmask | canopy | structure, dsm, dtm).astype(np.float32)
    # rounded crowns: taper canopy height toward its edge so trees read as domes, not cliffs
    cd = ndi.distance_transform_edt(canopy) * res
    crown = dtm + np.clip(ndi.gaussian_filter(dsm, 1.2) - dtm, 0, None) * (.45 + .55 * np.sqrt(np.clip(cd / 3.0, 0, 1)))
    H = np.where(canopy, crown, H)
    roof = ndi.median_filter(dsm, 3); H = np.where(bmask, roof, H)
    if landmarks is not None: H = landmarks.heightfield(H, dtm)
    gy, gx = np.gradient(ndi.gaussian_filter(H, .7), res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    # albedo raster
    alb = np.zeros((ny, nx, 3), np.float32)
    inv = {v: k for k, v in G.items()}
    for g, name in inv.items():
        key = {'wood': 'tree'}.get(name, name)
        alb[ground == g] = P[key]
    tslope = np.degrees(np.arctan(np.hypot(*np.gradient(ndi.gaussian_filter(dtm, 2), res))))
    rocky = (~bmask) & (tslope > 38) & ~road
    alb[rocky] = P['rock']
    alb[canopy] = P['tree']; alb[structure] = P['structure']
    # buildings: per-building roof/wall, else defaults from slope and modern/historic evidence
    roofc = np.zeros((len(attrs), 3), np.float32); wallc = np.zeros((len(attrs), 3), np.float32); has_roof = np.zeros(len(attrs), bool)
    for k, at in enumerate(attrs):
        if at is None: continue
        if at['roof'] is not None: roofc[k] = at['roof']; has_roof[k] = True
        wallc[k] = at['wall'] if at['wall'] is not None else P['sandstone']
    # per-building flatness from LiDAR: share of roof cells under 10 degrees
    lab = np.where(bmask, bidd, 0)
    cnt = np.bincount(lab.ravel(), minlength=len(attrs)); flat = np.bincount(lab.ravel(), weights=(slope.ravel() < 10), minlength=len(attrs))
    flat_share = flat / np.maximum(cnt, 1)
    modern_wall = np.all(wallc == P['modern'], axis=1)
    bl = lab[bmask]
    rc = np.where(has_roof[bl, None], roofc[bl], np.where((slope[bmask] < 10)[:, None], np.where(modern_wall[bl, None], P['modern'], P['flat_roof']), P['slate']))
    # pitched cells on mostly-flat modern roofs keep membrane colour
    alb[bmask] = rc
    # flat modern buildings without tags: pale walls when roof is >80% flat and no historic material tag
    pale = (flat_share > .8) & (cnt > 200)
    for k in np.nonzero(pale)[0]:
        if attrs[k] and attrs[k]['wall'] is None: wallc[k] = P['modern']
    walla = np.where(bmask[..., None], wallc[lab], alb)
    walla[structure] = P['structure']; walla[canopy] = np.array(P['tree']) * .7; walla[rocky | ((~bmask) & (slope > 45) & ~canopy & ~structure)] = P['rock']
    if landmarks is not None: landmarks.materials(alb, walla)
    # projection: u across, v forward (NE), camera at SW
    jj, ii = np.meshgrid(np.arange(nx), np.arange(ny))
    e = W + (jj + .5) * res - CENTRE[0]; n = N - (ii + .5) * res - CENTRE[1]
    u, v = horizontal(e, n, camera['bearing_deg'])
    X = ((u - U_RANGE[0]) * s).astype(np.float64)
    Y = ((-(SIN * v + COS * (H - Z0)) - Y_RANGE[0]) * s).astype(np.float64)
    D = (COS * v - SIN * H).astype(np.float64)
    Wd, Hh = int(round((U_RANGE[1] - U_RANGE[0]) * s)), int(round((Y_RANGE[1] - Y_RANGE[0]) * s))
    # per triangle normals in world space and lighting
    Lv = np.array([-.62, -.30, .72]); Lv /= np.linalg.norm(Lv)
    def tri_normals():
        z = H.astype(np.float64)
        nzs = np.zeros((ny, nx, 2)); shade = np.zeros((ny, nx, 2))
        # tri0: a=(i,j) b=(i+1,j) c=(i,j+1); world x=east (j), y=north (-i)
        for t, (A, B, C) in enumerate((((0, 0), (1, 0), (0, 1)), ((1, 0), (1, 1), (0, 1)))):
            def P3(o):
                zz = np.pad(z, ((0, 1), (0, 1)), mode='edge')[o[0]:o[0] + ny, o[1]:o[1] + nx]
                return np.stack([o[1] * res + 0 * zz, -o[0] * res + 0 * zz, zz], -1)
            pa, pb, pc = P3(A), P3(B), P3(C)
            nn = np.cross(pb - pa, pc - pa); nn *= np.sign(nn[..., 2:3] + 1e-12)
            nn /= np.linalg.norm(nn, axis=-1, keepdims=True)
            nzs[..., t] = nn[..., 2]
            shade[..., t] = .36 + .78 * np.clip(nn @ Lv, 0, 1)
        return nzs, shade
    nzs, shade = tri_normals()
    # soft openness: courtyards/closes darker
    open_ = np.clip(1 - np.clip(ndi.gaussian_filter(H, 6) - H, 0, 8) / 8 * .35, .65, 1)
    colr = np.clip(alb[:, :, None, :] * shade[..., None] * open_[:, :, None, None], 0, 255).astype(np.uint8)
    # wall shading from smoothed surface orientation (one tone per facade, no stair-step stripes)
    hs = ndi.gaussian_filter(H, 2.0); sgy, sgx = np.gradient(hs, res)
    hn = np.stack([-sgx, sgy, np.zeros_like(sgx)], -1)   # outward horizontal normal (east=+x, north=+y; rows go south)
    hn /= np.maximum(np.linalg.norm(hn, axis=-1, keepdims=True), 1e-6)
    wshade = (.40 + .80 * np.clip(hn @ Lv, 0, 1)).astype(np.float32)
    img = np.zeros((Hh, Wd, 3), np.uint8); img[:] = (150, 150, 150); zb = np.full((Hh, Wd), 1e18)
    raster(X, Y, D, H.astype(np.float32), walla.astype(np.float32), wshade, open_.astype(np.float32), colr, nzs, img, zb, Wd, Hh)
    if variant == 'edges':
        # silhouette lines from depth discontinuities (real geometry), 1 px, darkened
        dz = np.maximum(np.abs(np.diff(zb, axis=0, prepend=zb[:1])), np.abs(np.diff(zb, axis=1, prepend=zb[:, :1])))
        edge = dz > 2.5
        img[edge] = (img[edge] * .62).astype(np.uint8)
    if landmarks is not None: img, zb = landmarks.geometry(img, zb)
    Image.fromarray(img).save(out / 'source.png', optimize=False)
    # landmark pixels
    lms = {}
    for e_, t, rr in feats:
        nm = t.get('name')
        if nm in set(spec.get('landmark_names', LANDMARKS)) and nm not in lms:
            r = np.concatenate([q for _, q in rr]); cj, ci = r.mean(0)
            ee = W + cj * res - CENTRE[0]; nn_ = N - ci * res - CENTRE[1]
            iz = int(np.clip(ci, 0, ny - 1)); jz = int(np.clip(cj, 0, nx - 1))
            uu, vv = horizontal(ee, nn_, camera['bearing_deg'])
            px = (uu - U_RANGE[0]) * s; py = (-(SIN * vv + COS * (dtm[iz, jz] - Z0)) - Y_RANGE[0]) * s
            lms[nm] = dict(osm=f"{e_['type']}/{e_['id']}", bng=[round(W + cj * res, 1), round(N - ci * res, 1)], px=[round(px), round(py)],
                           in_frame=bool(0 <= px < Wd and 0 <= py < Hh))
    meta = dict(
        source=spec['description'],
        camera=dict(type='orthographic surface', crs=spec['crs'], bearing_deg=camera['bearing_deg'], elevation_deg=elevation, px_per_m=s,
                    width=Wd, height=Hh, pixel_matrix=pixel_matrix(camera, scale=s),
                    pixel_matrix_coordinates=['projected_easting_m', 'projected_northing_m', 'elevation_m'],
                    centre_projected_m=CENTRE, u_range_m=U_RANGE, y_range_screen_m=Y_RANGE, z0_m=Z0,
                    formula='u=dE*cos(b)-dN*sin(b), v=dE*sin(b)+dN*cos(b); X=(u-u0)*s; Y=(-(sin(e)*v+cos(e)*(z-z0))-y0)*s'),
        size=[Wd, Hh], variant=variant, palette=P, light=Lv.round(3).tolist(),
        colour_evidence=spec['colour_evidence'],
        facades='No window evidence in source; walls are plain shaded planes.',
        landmarks=lms,
        attribution=spec['attribution'],
        osm_snapshot=json.loads(Path(osm).read_text()).get('osm3s', {}).get('timestamp_osm_base'),
        input_sha256={'surface': sha(surface), 'osm': sha(osm), 'horizontal_grid': sha(grid) if grid else None},
        city_id=city['id'], surface_spec=spec,
        source_sha256=sha(out / 'source.png'))
    if landmarks is not None: meta['landmark_overlay'] = landmarks.provenance
    meta['renderer_sha256']=sha(__file__)
    meta['packages']={k:importlib.metadata.version(k) for k in ('numpy','pillow','scipy','pyproj','numba')}
    if spec.get('reference_source_sha256'):
        meta['reference_replay']={'expected_source_sha256':spec['reference_source_sha256'],
                                  'exact_png_match':meta['source_sha256']==spec['reference_source_sha256']}
    (out/'source.json').write_text(json.dumps(meta,indent=1)+'\n')
    print(json.dumps({'size': [Wd, Hh], 'source_sha256': meta['source_sha256'], 'out': str(out)}), flush=True)
    return meta


LANDMARKS = {'Edinburgh Castle', "St Giles' Cathedral", 'Scott Monument', 'Scottish National Gallery', 'Royal Scottish Academy',
             'Old College', 'McEwan Hall', 'National Museum of Scotland', 'Greyfriars Kirk', 'Edinburgh Waverley', 'The Hub',
             'Ramsay Garden', 'Grassmarket', 'Princes Street Gardens', 'Balmoral Hotel', 'Assembly Hall', 'Camera Obscura and World of Illusions',
             'Ross Fountain', 'Bank of Scotland', 'Central Library', 'Usher Hall', 'Victoria Street', 'Signet Library', 'Teviot Row House',
             "St Cuthbert's Church", "St John's Church", 'George Heriot\'s School', 'Princes Street', 'Royal Mile'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def horizontal(e, n, bearing):
    # Keep the exact original 45-degree arithmetic for byte-identical Edinburgh replay.
    if bearing == 45:
        return (e - n) / math.sqrt(2), (e + n) / math.sqrt(2)
    b = math.radians(bearing)
    return e*math.cos(b)-n*math.sin(b), e*math.sin(b)+n*math.cos(b)


def pixel_matrix(camera, *, scale=None):
    """Absolute projected easting/northing/elevation to native image pixels."""
    b, e = map(math.radians, (camera['bearing_deg'], camera['elevation_deg']))
    s = camera['pixels_per_metre'] if scale is None else scale
    east, north = camera['centre_projected_m']
    u0, y0 = camera['u_range_m'][0], camera['y_range_screen_m'][0]
    cb, sb, se, ce = math.cos(b), math.sin(b), math.sin(e), math.cos(e)
    return [[s * cb, -s * sb, 0, s * (-cb * east + sb * north - u0)],
            [-s * se * sb, -s * se * cb, -s * ce,
             s * (se * (sb * east + cb * north) + ce * camera['z0_m'] - y0)]]


def validate(spec):
    if not isinstance(spec, dict):
        raise ValueError('City needs a surface source recipe; see docs/sources.md')
    res = spec['resolution_m']
    if not math.isfinite(res) or not .1 <= res <= 10:
        raise ValueError('Surface resolution must be .1..10 metres')
    b = spec['bounds_projected_m']
    if len(b) != 4 or not all(math.isfinite(v) for v in b) or not (b[0]<b[2] and b[1]<b[3]):
        raise ValueError('Surface bounds must be [west,south,east,north]')
    for extent in (b[2]-b[0], b[3]-b[1]):
        if abs(extent/res-round(extent/res)) > 1e-6:
            raise ValueError('Surface bounds must align to its resolution')
    if round((b[2]-b[0])/res)*round((b[3]-b[1])/res)>64_000_000:
        raise ValueError('Surface exceeds 64M cells')
    c = spec['camera']
    if not (0<c['elevation_deg']<90) or not math.isfinite(c['bearing_deg']):
        raise ValueError('Camera needs finite bearing and elevation strictly between 0 and 90')
    for key in ('u_range_m','y_range_screen_m'):
        v=c[key]
        if len(v)!=2 or not all(math.isfinite(p) for p in v) or v[0]>=v[1]:
            raise ValueError('Camera bounds must be finite increasing pairs')
    if len(c['centre_projected_m'])!=2 or not all(math.isfinite(p) for p in c['centre_projected_m']) or not math.isfinite(c['z0_m']):
        raise ValueError('Camera centre and height must be finite')
    return spec


def transformer(spec, grid):
    if grid:
        pipeline = spec['geographic_pipeline'].replace('{grid}', str(Path(grid).resolve()))
        return Transformer.from_pipeline(pipeline)
    return Transformer.from_crs(4326, spec['crs'], always_xy=True)


def acquire(city, cache, *, offline=False, surface=None, osm=None, grid=None):
    """Reuse explicit files, or acquire aligned COG windows and a dated OSM snapshot."""
    import requests
    import rasterio
    from rasterio.windows import from_bounds
    spec=validate(city['surface']); cache=Path(cache); cache.mkdir(parents=True, exist_ok=True)
    surface=Path(surface) if surface else cache/'surface.npz'
    osm=Path(osm) if osm else cache/'osm.json'
    gspec=spec.get('horizontal_grid')
    grid=Path(grid) if grid else (cache/gspec['filename'] if gspec else None)
    if grid:
        if not grid.is_file():
            if offline: raise ValueError(f'Offline horizontal grid missing: {grid}')
            r=requests.get(gspec['url'], timeout=120); r.raise_for_status(); grid.write_bytes(r.content)
        if sha(grid)!=gspec['sha256']:
            raise ValueError('Horizontal transformation grid checksum mismatch')
    if not surface.is_file():
        if offline: raise ValueError(f'Offline DSM/DTM cache missing: {surface}')
        W,S,E,N=spec['bounds_projected_m']; res=spec['resolution_m']
        shape=(round((N-S)/res), round((E-W)/res))
        arrays={kind:np.full(shape, np.nan, np.float32) for kind in ('dsm','dtm')}
        for record in spec['rasters']:
            with rasterio.open('/vsicurl/'+record['url']) as d:
                if d.crs != rasterio.crs.CRS.from_user_input(spec['crs']) or abs(d.transform.a-res)>1e-8 or abs(d.transform.e+res)>1e-8 or d.transform.b or d.transform.d:
                    raise ValueError('Source raster must match the configured projected grid')
                b=d.bounds; left,bottom,right,top=max(W,b.left),max(S,b.bottom),min(E,b.right),min(N,b.top)
                if right<=left or top<=bottom: continue
                win=from_bounds(left,bottom,right,top,d.transform)
                if any(abs(v-round(v))>1e-6 for v in (win.col_off,win.row_off,win.width,win.height)):
                    raise ValueError('COG window does not align with native pixels')
                arr=d.read(1,window=win).astype(np.float32)
                if d.nodata is not None: arr[arr==d.nodata]=np.nan
                row,col=round((N-top)/res),round((left-W)/res)
                arrays[record['product']][row:row+arr.shape[0],col:col+arr.shape[1]]=arr
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise ValueError('DSM/DTM source has missing cells. Fix source coverage before rendering.')
        surface.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(surface, **arrays, bounds=np.array([W,S,E,N]))
        (cache/'surface-acquisition.json').write_text(json.dumps({'spec':spec,'sha256':sha(surface)},indent=2)+'\n')
    if not osm.is_file():
        if offline: raise ValueError(f'Offline OSM cache missing: {osm}')
        from fetch_osm import ENDPOINTS
        bbox=','.join(map(str,spec.get('osm_bbox',city['bbox'])))
        date=spec.get('osm_snapshot')
        date=f'[date:"{date}"]' if date else ''
        query=f'[out:json][timeout:180]{date}[bbox:{bbox}];(nwr[building];nwr["building:part"];way[highway];relation[highway];way[railway];nwr[leisure];nwr[landuse];nwr[natural];nwr[water];way[waterway];way[barrier];nwr[historic];nwr[tourism=attraction];);out body geom;'
        for endpoint in ENDPOINTS:
            try:
                r=requests.get(endpoint,params={'data':query},timeout=210);r.raise_for_status(); payload=r.json()
                if payload.get('remark') or not payload.get('elements'): raise ValueError('Incomplete OSM response')
                osm.parent.mkdir(parents=True,exist_ok=True); osm.write_text(json.dumps(payload,indent=2)+'\n')
                (cache/'osm-acquisition.json').write_text(json.dumps({'endpoint':endpoint,'query':query,'sha256':sha(osm)},indent=2)+'\n')
                break
            except (requests.RequestException,ValueError): continue
        else: raise ValueError('No OSM endpoint returned complete geometry')
    return surface,osm,grid


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--city',default='edinburgh',help='City slug or JSON recipe')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--cache',type=Path,help='Input cache; defaults to outputs/cache/surface/CITY')
    ap.add_argument('--surface',type=Path,help='Existing NPZ with dsm,dtm,bounds; no resampling')
    ap.add_argument('--osm',type=Path,help='Existing Overpass body-geom JSON')
    ap.add_argument('--grid',type=Path,help='Existing pinned horizontal transformation grid')
    ap.add_argument('--offline',action='store_true')
    ap.add_argument('--scale',type=float)
    ap.add_argument('--variant',choices=['plain','edges'])
    a=ap.parse_args(argv)
    try:
        path=Path(a.city)
        if not path.is_file(): path=ROOT/'cities'/f'{a.city}.json'
        city=json.loads(path.read_text()); validate(city['surface'])
        cache=a.cache or ROOT/'outputs/cache/surface'/city['id']
        a.out.mkdir(parents=True,exist_ok=True)
        with (a.out/'.source.lock').open('a') as lock:
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: raise ValueError('Another surface renderer owns this output')
            Path(cache).mkdir(parents=True,exist_ok=True)
            with (Path(cache)/'.inputs.lock').open('a') as cache_lock:
                fcntl.flock(cache_lock,fcntl.LOCK_EX)
                inputs=acquire(city,cache,offline=a.offline,surface=a.surface,osm=a.osm,grid=a.grid)
            render(city,*inputs,a.out,scale=a.scale,variant=a.variant)
        return 0
    except (ValueError,OSError,KeyError) as error:
        print(json.dumps({'status':'failed','reason':str(error)}),file=sys.stderr); return 1


if __name__ == '__main__':
    raise SystemExit(main())
