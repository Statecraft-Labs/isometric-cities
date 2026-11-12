"""Cut a Deep Zoom pyramid from finished pixels; return its metadata."""
from __future__ import annotations
import math
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image
from project import digest


def build(image, descriptor, folder, *, resample='box'):
    image, descriptor, folder = map(Path, (image, descriptor, folder))
    if resample not in ('box', 'nearest'):
        raise ValueError('Tile resampling must be box or nearest')
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('Tile directory must be new or empty')
    folder.mkdir(parents=True, exist_ok=True)
    with Image.open(image) as opened:
        if opened.mode != 'RGB' or opened.width*opened.height > 64_000_000:
            raise ValueError('Tiles require RGB pixels within 64 megapixels')
        original = opened.copy()
    maximum = math.ceil(math.log2(max(original.size)))
    levels, count = {}, 0
    for level in range(maximum, -1, -1):
        divisor = 2**(maximum-level)
        size = (math.ceil(original.width/divisor), math.ceil(original.height/divisor))
        reduced = original if divisor == 1 else original.resize(size, Image.Resampling.BOX if resample == 'box' else Image.Resampling.NEAREST)
        target = folder/str(level)
        target.mkdir(exist_ok=True)
        columns, rows = math.ceil(size[0]/256), math.ceil(size[1]/256)
        levels[str(level)] = {'width': size[0], 'height': size[1], 'columns': columns, 'rows': rows}
        for row in range(rows):
            for column in range(columns):
                box = (max(0, column*256-1), max(0, row*256-1),
                       min(size[0], (column+1)*256+1), min(size[1], (row+1)*256+1))
                reduced.crop(box).save(target/f'{column}_{row}.png')
                count += 1
    root = ET.Element('Image', TileSize='256', Overlap='1', Format='png', xmlns='http://schemas.microsoft.com/deepzoom/2008')
    ET.SubElement(root, 'Size', Width=str(original.width), Height=str(original.height))
    ET.ElementTree(root).write(descriptor, encoding='utf-8', xml_declaration=True)
    return {'tile_size': 256, 'overlap': 1, 'format': 'png', 'max_level': maximum,
            'tile_count': count, 'levels': levels, 'resampling': resample, 'image_sha256': digest(image)}
