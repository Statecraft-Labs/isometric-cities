"""Integer-pixel orthographic rasterizer with deterministic depth and texture."""
from __future__ import annotations

import numpy as np


def tint(color, factor):
    return tuple(int(np.clip(c * factor, 0, 255)) for c in color)


class Raster:
    def __init__(self, camera, palette=None):
        self.camera = camera
        self.palette = palette or {}
        self.rgb = np.full((camera.height, camera.width, 3), self.palette.get('background_rgb',(224,220,198)), np.uint8)
        self.depth = np.full((camera.height, camera.width), -np.inf, np.float32)
        self.triangle_count = 0

    def triangle(self, xyz, color, texture=0, bias=0, opacity=1.0, shader=None):
        """Optional shader(world_xyz, color, pixel_xy, edge_distance) returns visible RGB."""
        p = self.camera.project(xyz)
        x0 = max(0, int(np.floor(p[:,0].min())))
        x1 = min(self.camera.width-1, int(np.ceil(p[:,0].max())))
        y0 = max(0, int(np.floor(p[:,1].min())))
        y1 = min(self.camera.height-1, int(np.ceil(p[:,1].max())))
        if x1 < x0 or y1 < y0:
            return
        a,b,c = p
        denom = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
        if abs(denom) < 1e-8:
            return
        yy,xx = np.mgrid[y0:y1+1, x0:x1+1]
        xx,yy = xx+0.5,yy+0.5
        w0 = ((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/denom
        w1 = ((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/denom
        w2 = 1-w0-w1
        z = w0*a[2]+w1*b[2]+w2*c[2]+bias
        zbuffer = self.depth[y0:y1+1,x0:x1+1]
        mask = (w0 >= -1e-6)&(w1 >= -1e-6)&(w2 >= -1e-6)&(z >= zbuffer)
        if not mask.any():
            return
        view = self.rgb[y0:y1+1,x0:x1+1]
        if opacity<1:
            view[mask]=np.clip(view[mask]*(1-opacity)+np.asarray(color)*opacity,0,255).astype(np.uint8)
        elif shader is not None:
            source=np.asarray(xyz,float)
            world=(w0[mask,None]*source[0]+w1[mask,None]*source[1]+w2[mask,None]*source[2])
            double_area=abs(denom)
            altitudes=double_area/np.maximum(np.array([np.linalg.norm(b[:2]-c[:2]),np.linalg.norm(c[:2]-a[:2]),np.linalg.norm(a[:2]-b[:2])]),1e-12)
            edge_distance=np.minimum.reduce((w0[mask]*altitudes[0],w1[mask]*altitudes[1],w2[mask]*altitudes[2]))
            view[mask]=shader(world,np.asarray(color,np.uint8),np.column_stack((xx[mask],yy[mask])),edge_distance)
        elif texture:
            noise = ((xx.astype(int)*13+yy.astype(int)*7)%11-5)*texture
            view[mask] = np.clip(np.asarray(color)+noise[mask,None],0,255).astype(np.uint8)
        else:
            view[mask] = color
        zbuffer[mask] = z[mask]
        self.triangle_count += 1

    def quad(self, xyz, color, texture=0, bias=0):
        self.triangle(np.asarray(xyz)[[0,1,2]], color, texture, bias)
        self.triangle(np.asarray(xyz)[[0,2,3]], color, texture, bias)
