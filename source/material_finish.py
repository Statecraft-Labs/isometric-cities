"""Registered colour finishing and reviewed native patches. No resampling or inference."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from project import digest, read, write
from scipy import ndimage as ndi

LUMA=np.array([.2126,.7152,.0722],np.float32)


def local_ink(pixels, *, luminance_max=70, strength=.65):
    """Lighten only thin dark strokes toward the neighbouring material's dark tone."""
    if type(luminance_max) not in (int,float) or not 0 < luminance_max <= 255 or type(strength) not in (int,float) or not 0 <= strength <= 1:
        raise ValueError('Ink threshold must be in 0..255 and strength in 0..1')
    f=pixels.astype(np.float32);lum=f@LUMA;bright=lum>82
    weight=ndi.gaussian_filter(bright.astype(np.float32),2.)
    local=np.stack([ndi.gaussian_filter(f[:,:,c]*bright,2.)/np.maximum(weight,.001) for c in range(3)],-1)
    thin=ndi.distance_transform_edt(lum<luminance_max)<=1.45
    mask=(lum<luminance_max)&(weight>.13)&thin&((local@LUMA)>lum*1.7)
    target=np.maximum(f,local*.48);out=f.copy()
    out[mask]=(1-strength)*f[mask]+strength*target[mask]
    return np.clip(out+.5,0,255).astype(np.uint8),mask


def polygon_mask(size, polygon):
    if not isinstance(polygon,list) or len(polygon)<3 or any(not isinstance(p,list) or len(p)!=2 or any(type(v) not in (int,float) or not np.isfinite(v) for v in p) or not (0<=p[0]<size[0] and 0<=p[1]<size[1]) for p in polygon):
        raise ValueError('Material polygon must have at least three finite native image points inside the frame')
    layer=Image.new('1',size);ImageDraw.Draw(layer).polygon([tuple(p) for p in polygon],fill=1)
    return np.array(layer,dtype=bool)


def region_mask(size, region):
    """Native scanline runs preserve source-visible surfaces and their holes."""
    if 'runs' not in region:
        return polygon_mask(size,region['polygon'])
    runs=region['runs']
    if 'polygon' in region or not isinstance(runs,list) or len(runs)>size[0]*size[1]:
        raise ValueError('Supply either a polygon or bounded native runs')
    mask=np.zeros((size[1],size[0]),bool);previous=(-1,0)
    for run in runs:
        if not isinstance(run,list) or len(run)!=3 or any(type(v) is not int for v in run):
            raise ValueError('Mask runs need integer [row,start,end] triples')
        y,x0,x1=run
        if not 0<=y<size[1] or not 0<=x0<x1<=size[0] or y<previous[0] or (y==previous[0] and x0<previous[1]):
            raise ValueError('Mask runs must be ordered, nonoverlapping and inside the image')
        mask[y,x0:x1]=True;previous=(y,x1)
    return mask


def apply_patches(pixels, patches, recipe_path):
    """Replay reviewed pixels only after checking their asset, location and support."""
    changes=[]
    for patch in patches:
        path=(recipe_path.parent/patch['image']).resolve()
        if not path.is_relative_to(recipe_path.parent.resolve()) or digest(path)!=patch['image_sha256']:
            raise ValueError('Repair asset is outside the recipe folder or its hash changed')
        with Image.open(path) as image:
            if image.format!='PNG' or image.mode!='RGB' or image.getexif().get(274,1)!=1:
                raise ValueError('Repair assets must be native RGB PNGs')
            w,h=image.size; replacement=np.array(image)
        position=patch['position']
        if not isinstance(position,list) or len(position)!=2 or any(type(v) is not int for v in position):
            raise ValueError('Repair position needs two integer native coordinates')
        x,y=position
        if min(x,y)<0 or x+w>pixels.shape[1] or y+h>pixels.shape[0]:
            raise ValueError('Repair rectangle must lie inside the map')
        before=pixels[y:y+h,x:x+w]
        if hashlib.sha256(before.tobytes()).hexdigest()!=patch['before_pixels_sha256']:
            raise ValueError('Repair location or prior pixels changed')
        allowed=region_mask((w,h),patch);changed=np.any(before!=replacement,axis=2)
        if np.any(changed&~allowed):raise ValueError('Repair changes pixels outside its reviewed support')
        before[:]=replacement
        changes.append({'name':patch['name'],'pixels':int(changed.sum()),'image_sha256':patch['image_sha256'],'position':position})
    return changes


def source_chroma(art, source, hull, settings):
    """Use registered source colour classes, with explicit protection of vegetation and masonry."""
    for key in ('chroma_max', 'luminance_min', 'luminance_max', 'green_excess'):
        if type(settings[key]) not in (int, float) or not 0 <= settings[key] <= 255:
            raise ValueError('Source mask thresholds must be finite byte values')
    for key in ('close', 'open', 'protect_source_kernel'):
        if type(settings[key]) is not int or not 1 <= settings[key] <= 31 or settings[key] % 2 != 1:
            raise ValueError('Morphology kernels must be odd integers in 1..31')
    if not 0 <= settings['feather'] <= 16 or not 0 < settings['gain'] <= 2:
        raise ValueError('Source feather must be in 0..16 and gain in 0..2')
    vector = np.asarray(settings['channel_scale'], np.float32)
    if vector.shape != (3,) or not np.isfinite(vector).all() or vector.min() < 0 or vector.max() > 2:
        raise ValueError('Source channel scales need three values in 0..2')
    f, src = art.astype(np.float32), source.astype(np.float32)
    luma = np.float32([.299, .587, .114])
    lum, alum = src @ luma, f @ luma
    mask = (src.max(2)-src.min(2) < settings['chroma_max']) & (lum > settings['luminance_min']) & (lum < settings['luminance_max'])
    mask &= ~(src[..., 1] > src[..., 0] + settings['green_excess']) & hull
    def morph(value, operation, size):
        return np.asarray(Image.fromarray((value*255).astype(np.uint8)).filter(operation(size))) > 127
    mask = morph(morph(mask, ImageFilter.MaxFilter, settings['close']), ImageFilter.MinFilter, settings['close'])
    mask = morph(morph(mask, ImageFilter.MinFilter, settings['open']), ImageFilter.MaxFilter, settings['open'])
    soft = np.asarray(Image.fromarray((mask*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(settings['feather']))).astype(np.float32)/255
    moss = (f[..., 1] > f[..., 0]+settings['protect_green_red']) & (f[..., 1] > f[..., 2]+settings['protect_green_blue'])
    wall = (alum > settings['protect_warm_luminance']) & (f[..., 0] > f[..., 2]+settings['protect_warm_red_blue'])
    source_wall = morph(lum > settings['protect_source_luminance'], ImageFilter.MaxFilter, settings['protect_source_kernel'])
    weight = soft * ~moss * ~wall * ~source_wall * hull
    grey = np.stack([alum*value for value in vector], -1)*settings['gain']
    result = f*(1-weight[..., None])+grey*weight[..., None]
    return np.clip(result+.5, 0, 255).astype(np.uint8), weight > 0


def finish(image,source,recipe_path,out):
    image,source,recipe_path,out=map(Path,(image,source,recipe_path,out))
    recipe=read(recipe_path)
    json.dumps(recipe,allow_nan=False)
    if recipe['input_sha256']!=digest(image) or recipe['source_sha256']!=digest(source):
        raise ValueError('Finishing recipe belongs to different source/art pixels')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new or empty finishing directory')
    with Image.open(image) as opened,Image.open(source) as reference:
        if opened.mode!='RGB' or reference.mode!='RGB' or opened.size!=reference.size or opened.width*opened.height>64_000_000:
            raise ValueError('Source and art must be matching RGB images within 64 megapixels')
        original=np.array(opened); src=np.array(reference)
    current=original.copy();patches=apply_patches(current,recipe.get('patches',[]),recipe_path);changes=list(patches)
    if recipe.get('local_material_ink'):
        ink=recipe['local_material_ink']
        if ink is not True and not isinstance(ink,dict):raise ValueError('Ink settings must be true or an object')
        current,_=local_ink(current,**(ink if isinstance(ink,dict) else {}));changes.append({'name':'local material outline tones','pixels':int(np.any(current!=original,axis=2).sum())})
    before_material=current.copy();support=np.zeros(original.shape[:2],bool)
    source_f=src.astype(np.float32)
    for region in recipe.get('regions',[]):
        mask=region_mask((original.shape[1],original.shape[0]),region)
        for excluded in region.get('exclude_polygons',[]):mask&=~polygon_mask((original.shape[1],original.shape[0]),excluded)
        if 'source_mask' in region:
            changed, mask = source_chroma(current, src, mask, region['source_mask'])
            changes.append({'name':region['name'],'pixels':int(np.any(changed!=current,axis=2).sum())})
            current=changed;support|=mask
            continue
        f=current.astype(np.float32);lum=f@LUMA
        kind=region.get('selection','all')
        if kind=='mint':mask&=(f[:,:,1]-f[:,:,0]>5)&(f[:,:,2]-f[:,:,0]>-12)&(lum>100)
        elif kind=='warm':mask&=(f[:,:,0]>f[:,:,1]*1.025)&(f[:,:,1]>f[:,:,2]*1.025)&(lum>35)
        elif kind=='bright_neutral':mask&=(lum>170)&(f.max(2)-f.min(2)<42)
        elif kind=='source_neutral':mask&=(source_f.max(2)-source_f.min(2)<22)&(source_f@LUMA>45)
        elif kind!='all':raise ValueError('Unknown material selection')
        scale=float(region.get('luminance_scale',1.));feather=float(region.get('feather',2))
        if not np.isfinite(scale) or not 0<scale<=2 or not np.isfinite(feather) or not 0<=feather<=32:
            raise ValueError('Use a finite luminance scale in 0..2 and feather in 0..32')
        desired=lum*scale
        if region.get('two_tone'):
            if len(region['two_tone'])!=2 or any(type(v) not in (int,float) or not 0<=v<=255 for v in region['two_tone']):
                raise ValueError('two_tone needs two luminance values in 0..255')
            desired=np.where(lum>95,float(region['two_tone'][1]),float(region['two_tone'][0]))
        if 'saturation' in region:
            saturation=region['saturation']
            if type(saturation) not in (int,float) or not np.isfinite(saturation) or not 0<=saturation<=2:
                raise ValueError('Saturation must be finite in 0..2')
            if 'target_rgb' in region:raise ValueError('Choose saturation or a target colour')
            result=(lum[:,:,None]+(f-lum[:,:,None])*saturation)*scale
        else:
            target=np.array(region['target_rgb'],np.float32)
            if target.shape!=(3,) or not np.isfinite(target).all() or target.min()<0 or target.max()>255 or target@LUMA<=0:
                raise ValueError('target_rgb must have three finite bytes with nonzero luminance')
            result=target[None,None,:]*desired[:,:,None]/float(target@LUMA)
        alpha=np.clip(ndi.distance_transform_edt(mask)/max(1,feather),0,1)
        strength=region.get('strength',1.)
        if type(strength) not in (int,float) or not np.isfinite(strength) or not 0<=strength<=1:
            raise ValueError('Material strength must be finite in 0..1')
        alpha*=mask*strength
        changed=np.clip(current.astype(np.float32)*(1-alpha[:,:,None])+result*alpha[:,:,None]+.5,0,255).astype(np.uint8)
        count=int(np.any(changed!=current,axis=2).sum());current=changed;support|=mask
        changes.append({'name':region['name'],'pixels':count})
    if np.any(np.any(current!=before_material,axis=2)&~support):raise ValueError('Material edits escaped their masks')
    final_patches=apply_patches(current,recipe.get('post_patches',[]),recipe_path)
    patches+=final_patches;changes+=final_patches
    out.mkdir(parents=True,exist_ok=True);Image.fromarray(current).save(out/'map.png')
    record={'method':'native colour-only finishing; no resampling or geometry warp' if not patches else 'reviewed native patches and colour finishing; no global map resampling','input_sha256':digest(image),'source_sha256':digest(source),
            'recipe_sha256':digest(recipe_path),'code_sha256':digest(__file__),'image_sha256':digest(out/'map.png'),'changes':changes,
            'material_changes_outside_masks':0,'dimensions':[original.shape[1],original.shape[0]]}
    if patches:record['patch_changes_outside_masks']=0
    write(out/'proof.json',record);write(out/'recipe.json',recipe)
    return record


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['image','source','recipe','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(argv)
    try:print(json.dumps(finish(a.image,a.source,a.recipe,a.out),indent=2));return 0
    except (ValueError,OSError,KeyError,TypeError) as e:print(str(e),file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
