#!/usr/bin/env python3
"""Cache OSM geometry and every Skadi degree tile required by a city config."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
from geometry import hgt_tiles
from project import digest, recipe_path, read

ENDPOINTS = ['https://overpass.openstreetmap.fr/api/interpreter','https://overpass-api.de/api/interpreter','https://overpass.private.coffee/api/interpreter','https://overpass.kumi.systems/api/interpreter']
DEM_ATTRIBUTION = 'Mapzen Terrain Tiles. USGS SRTM/GMTED2010, Copernicus EU-DEM, UK © Environment Agency 2015; exact per-pixel source not identified.'


def fetch(city, refresh=False, offline=False):
    folder=Path(city['data_dir'])
    folder.mkdir(parents=True,exist_ok=True)
    session=requests.Session()
    session.headers['User-Agent']='PortableIsometricCity/0.2 (OSM geometry renderer)'
    bbox=','.join(str(v) for v in city['bbox'])
    query=f'''[out:json][timeout:180][bbox:{bbox}];
    (nwr[building];nwr["building:part"];way[highway];relation[highway];way[railway];
     nwr[leisure];nwr[landuse];nwr[natural];nwr[water];way[waterway];
     way[barrier];nwr[historic];nwr[tourism=attraction];);out body geom;'''
    query_path=folder/'overpass-query.txt'
    meta_path=folder/'provenance.json'
    old=json.loads(meta_path.read_text()) if meta_path.exists() else {}
    osm=folder/'osm.json'
    if osm.exists() and not refresh and old.get('bbox_south_west_north_east') not in (None,city['bbox']):
        raise ValueError('OSM cache belongs to different bounds; use a fresh cache or refresh explicitly')
    if not refresh:
        for name,record in old.get('files',{}).items():
            cached=folder/name
            if not cached.is_file() or digest(cached)!=record['sha256']:
                raise ValueError(f'Cached source input changed: {cached}. Use a fresh cache or refresh explicitly')
    imported=None
    if not osm.exists() and city.get('osm_source') and not refresh:
        original=Path(city['osm_source']).expanduser()
        if not original.is_absolute(): original=ROOT/original
        if original.exists():
            shutil.copy2(original,osm)
            imported=str(original)
            original_meta=original.parent/'provenance.json'
            if original_meta.exists():
                old=json.loads(original_meta.read_text())
    endpoint=old.get('osm_endpoint')
    retrieved=old.get('retrieved_at')
    if refresh or not osm.exists():
        if offline:
            raise RuntimeError(f'Offline mode requires an OSM cache: {osm}')
        for candidate in ENDPOINTS:
            try:
                print(f'Fetching OSM from {candidate}',flush=True)
                response=session.get(candidate,params={'data':query},timeout=200)
                response.raise_for_status()
                payload=response.json()
                if payload.get('remark') or not payload.get('elements'):
                    raise RuntimeError(payload.get('remark','Empty Overpass result'))
                osm.write_text(json.dumps(payload,separators=(',',':')))
                query_path.write_text(query+'\n')
                endpoint=candidate
                retrieved=datetime.now(timezone.utc).isoformat()
                break
            except (requests.RequestException,ValueError,RuntimeError) as error:
                print(f'OSM endpoint failed: {error}',flush=True)
        else:
            raise RuntimeError('No Overpass endpoint returned a complete result')
        reused_osm=False
    else:
        reused_osm=True
    spec=city['terrain']
    need_skadi=spec['provider']=='skadi' or spec['provider']=='lidar' and spec.get('fallback','skadi')=='skadi'
    sources={Path(p).name:(Path(p) if Path(p).is_absolute() else ROOT/p) for p in city.get('dem_sources',[])}
    dem_records=[]
    for tile in hgt_tiles(city['bbox']) if need_skadi else []:
        path=folder/f'{tile}.hgt.gz'
        url=f'https://s3.amazonaws.com/elevation-tiles-prod/skadi/{tile[:3]}/{tile}.hgt.gz'
        if not path.exists() and path.name in sources and not refresh:
            shutil.copy2(sources[path.name],path)
        error=None
        if (refresh or not path.exists()) and not offline:
            try:
                print(f'Fetching DEM from {url}',flush=True)
                response=session.get(url,timeout=90)
                response.raise_for_status()
                path.write_bytes(response.content)
            except requests.RequestException as failure:
                error=str(failure)
                print(f'DEM unavailable; manifest will record flat/missing terrain: {failure}',flush=True)
        dem_records.append({'tile':tile,'url':url,'cached':path.exists(),'error':error})
    header=json.loads(osm.read_text())
    files=[osm]+[folder/f"{r['tile']}.hgt.gz" for r in dem_records if r['cached']]
    # A cache can contain documented supplemental geometry. Keep its acquisition
    # evidence when reusing it; a fresh base download invalidates that evidence.
    meta={**(old if reused_osm else {}),
          'city_id':city['id'],'retrieved_at':retrieved,'checked_at':datetime.now(timezone.utc).isoformat(),
          'bbox_south_west_north_east':city['bbox'],'osm_endpoint':endpoint,'osm_imported_from':imported or old.get('osm_imported_from'),
          'osm_data_timestamp':header.get('osm3s',{}).get('timestamp_osm_base'),
          'osm_copyright':'© OpenStreetMap contributors, ODbL 1.0','osm_attribution_url':'https://www.openstreetmap.org/copyright',
          'terrain_provider':spec['provider'],'dem_tiles':dem_records,'dem_attribution':DEM_ATTRIBUTION,
          'dem_attribution_url':'https://github.com/tilezen/joerd/blob/master/docs/attribution.md',
          'files':{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}}
    if query_path.exists():
        meta['osm_query_sha256']=hashlib.sha256(query_path.read_bytes()).hexdigest()
    meta_path.write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({'city':city['id'],'osm':str(osm),'osm_elements':len(header['elements']),'dem_tiles':dem_records},indent=2),flush=True)
    return meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='capitol-hill-dc')
    parser.add_argument('--refresh',action='store_true')
    parser.add_argument('--offline',action='store_true')
    args=parser.parse_args()
    city=read(recipe_path(args.config))
    city.setdefault('data_dir',str(ROOT/'outputs/cache/osm'/city['id']))
    fetch(city,args.refresh,args.offline)


if __name__=='__main__':
    main()
