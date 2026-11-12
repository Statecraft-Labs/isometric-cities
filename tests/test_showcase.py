"""Exercise the actual offline export, tile pixels and explorer interactions without inference."""
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source'))
import showcase


class Showcase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        y, x = np.indices((656, 2048))
        self.pixels = np.stack((x % 256, y % 256, (x+y) % 256), axis=-1).astype('uint8')
        self.image, self.source = self.folder/'art.png', self.folder/'source.png'
        Image.fromarray(self.pixels).save(self.image)
        Image.fromarray(255-self.pixels).save(self.source)
        self.metadata = {'credits': 'Actual source author, CC BY-SA 4.0', 'licence': 'CC BY-SA 4.0',
                         'status': 'Display study with model drift', 'fictional_adjacency': 'Fictional river join',
                         'regions': [{'id': 'dc', 'name': 'DC', 'box': [0, 0, 900, 656]},
                                     {'id': 'edi', 'name': 'Edinburgh', 'box': [1000, 0, 1048, 656]}],
                         'landmarks': [{'name': 'Capitol', 'x': 530, 'y': 150},
                                       {'name': 'Castle', 'x': 1380, 'y': 260}]}

    def test_real_export_native_tiles_and_self_contained_credit(self):
        out = self.folder/'export'
        result = showcase.export(self.image, self.source, self.metadata, out)
        self.assertEqual(result['dimensions'], [2048, 656])
        self.assertEqual((out/'map.png').read_bytes(), self.image.read_bytes())
        self.assertEqual((out/'source.png').read_bytes(), self.source.read_bytes())
        record = json.loads((out/'manifest.json').read_text())
        self.assertEqual(record['source_sha256'], showcase.digest(self.source))
        self.assertEqual(record['export']['native_upscale'], 1)
        self.assertEqual(record['deep_zoom']['resampling'], 'box')
        descriptor = ET.parse(out/'map.dzi').getroot()
        self.assertEqual(descriptor.find('{*}Size').attrib, {'Width': '2048', 'Height': '656'})
        rebuilt = Image.new('RGB', (2048, 656))
        maximum = record['deep_zoom']['max_level']
        for row in range(3):
            for col in range(8):
                tile = Image.open(out/f'map_files/{maximum}/{col}_{row}.png')
                dx, dy = int(col > 0), int(row > 0)
                rebuilt.paste(tile.crop((dx, dy, dx+min(256, 2048-col*256), dy+min(256, 656-row*256))), (col*256, row*256))
        self.assertTrue(np.array_equal(np.asarray(rebuilt), self.pixels))
        page = (out/'viewer.html').read_text()
        self.assertIn('url=viewer.html', (out/'index.html').read_text())
        self.assertIn('data:image/webp;base64,', page)
        self.assertLess(len(page), 2_000_000)
        self.assertTrue((out/'source.dzi').is_file())
        self.assertIn('Actual source author', page)
        self.assertNotIn('src="https://', page)
        self.assertNotIn('__DATA__', page)

    def test_default_demo_selects_a_new_edition_without_overwriting_previous_art(self):
        examples=self.folder/'examples';examples.mkdir()
        for city in ('dc','edinburgh'):
            (examples/f'{city}.png').write_bytes(self.image.read_bytes())
            (examples/f'{city}-source.png').write_bytes(self.source.read_bytes())
            (examples/f'{city}.json').write_text(json.dumps(self.metadata))
        with patch.object(showcase,'ROOT',self.folder),redirect_stdout(io.StringIO()):
            showcase.demo_main([])
            original=next((self.folder/'outputs/demo').iterdir());before=(original/'dc/map.png').read_bytes()
            showcase.demo_main([])
            self.assertEqual(len(list((self.folder/'outputs/demo').iterdir())),1)
            changed=self.pixels.copy();changed[10,10]=[1,2,3];Image.fromarray(changed).save(examples/'dc.png')
            showcase.demo_main([])
        self.assertEqual(len(list((self.folder/'outputs/demo').iterdir())),2)
        self.assertEqual((original/'dc/map.png').read_bytes(),before)

    def test_tampered_hash_and_mismatched_source_rejected_before_export(self):
        metadata = dict(self.metadata, image_sha256='0'*64)
        with self.assertRaisesRegex(ValueError, 'image_sha256'):
            showcase.export(self.image, self.source, metadata, self.folder/'bad')
        self.assertFalse((self.folder/'bad').exists())
        Image.new('RGB', (2048, 640)).save(self.source)
        with self.assertRaisesRegex(ValueError, 'identical'):
            showcase.validate(self.image, self.source, self.metadata)

    def test_phone_server_exposes_only_the_current_export(self):
        examples=self.folder/'examples';examples.mkdir()
        for city in ('dc','edinburgh'):
            (examples/f'{city}.png').write_bytes(self.image.read_bytes())
            (examples/f'{city}-source.png').write_bytes(self.source.read_bytes())
            (examples/f'{city}.json').write_text(json.dumps(self.metadata))
        with patch.object(showcase,'ROOT',self.folder),patch('http.server.ThreadingHTTPServer') as server,redirect_stdout(io.StringIO()):
            server.return_value.__enter__.return_value.server_port=8765
            showcase.demo_main(['--serve','--host','0.0.0.0'])
        address,handler=server.call_args.args
        self.assertEqual(address,('0.0.0.0',8765))
        folder=Path(handler.keywords['directory'])
        self.assertEqual(folder.parent,(self.folder/'outputs/demo').resolve())
        self.assertTrue((folder/'dc/viewer.html').is_file())
        self.assertFalse((folder/'examples').exists())
        server.return_value.__enter__.return_value.serve_forever.assert_called_once()

    def test_invalid_regions_anchors_and_credits(self):
        for mutation in (
            {'regions': [{'name': 'Outside', 'box': [2000, 0, 100, 100]}]},
            {'regions': [{'name': 'Broken', 'box': [0, 0, float('nan'), 100]}]},
            {'regions': [{'id': 'a', 'name': 'A', 'box': [0, 0, 1, 1]}, {'id': 'a', 'name': 'B', 'box': [0, 0, 1, 1]}]},
            {'landmarks': [{'x': -1, 'y': 100}]}, {'credits': ''}, {'width': 12}, {'opening_region':'missing'},
            {'opening_landmark':'missing'}, {'hero_zoom':float('inf')},
            {'landmarks':[{'id':'a','x':1,'y':1},{'id':'a','x':2,'y':2}]},
            {'landmarks':[{'x':1,'y':1,'view_width':0}]},
            {'landmarks':[{'x':1,'y':1,'look_up':float('nan')}]},
            {'navigation':[{'name':'DC','href':'javascript:alert(1)'}]},
            {'navigation':[{'name':'DC','href':'https://example.com/viewer.html'}]},
        ):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                showcase.validate(self.image, self.source, dict(self.metadata, **mutation))

    def test_safe_output_and_rgb_contract(self):
        with self.assertRaisesRegex(ValueError, 'contain either input'):
            showcase.export(self.image, self.source, self.metadata, self.folder)
        out = self.folder/'keep'
        out.mkdir()
        (out/'original.txt').write_text('original')
        with self.assertRaisesRegex(ValueError, 'new or empty'):
            showcase.export(self.image, self.source, self.metadata, out)
        self.assertEqual((out/'original.txt').read_text(), 'original')
        Image.new('RGBA', (2048, 656)).save(self.image)
        with self.assertRaisesRegex(ValueError, 'RGB PNG'):
            showcase.validate(self.image, self.source, self.metadata)

    def test_script_text_is_escaped_and_arbitrary_locations_have_no_fake_regions(self):
        payload = '</script><img src=x onerror=alert(1)>& __ART__'
        encoded = showcase.script_json({'credits': payload})
        self.assertNotIn('<', encoded)
        self.assertNotIn('&', encoded)
        self.assertEqual(json.loads(encoded)['credits'], payload)
        metadata = {'credits': payload}
        out = self.folder/'escaped'
        showcase.export(self.image, self.source, metadata, out)
        page = (out/'viewer.html').read_text()
        self.assertNotIn(payload, page)
        self.assertIn('__ART__', page)  # User data remains literal, not a template marker.
        self.assertNotIn('"name": "Edinburgh"', page)

    def test_browser_interaction_math_and_motion_preferences(self):
        metadata=dict(self.metadata,width=2048,height=656,city_id='dc',navigation=[{'id':'dc','name':'DC','href':'../dc/viewer.html'},{'id':'edinburgh','name':'Edinburgh','href':'../edinburgh/viewer.html'}])
        page=showcase.HTML.replace('__ART__','"art"').replace('__SOURCE__','"source"').replace('__DATA__',showcase.script_json(metadata))
        script=re.search(r'<script>(.*?)</script>',page,re.S).group(1)
        harness=r'''const vm=require('node:vm'),assert=require('node:assert/strict');
class Element{
 constructor(){this.style={setProperty(k,v){this[k]=v}};this.children=[];this.events={};this.dataset={};this.attrs={};this.hidden=true;this.clientWidth=728;this.clientHeight=410;this.classList={add(){},remove(){}}}
 setAttribute(k,v){this.attrs[k]=v}focus(){this.focused=true}appendChild(b){this.children.push(b);b.parent=this}remove(){this.parent.children=this.parent.children.filter(x=>x!==this)}
 addEventListener(k,f){this.events[k]=f}getBoundingClientRect(){return {left:0,top:0,width:170,height:54}}setPointerCapture(){}click(){this.onclick()}matches(){return false}}
const ids={},events={};for(const id of ['view','art-world','source-world','picture','source-picture','source-clip','divider','pill','mini','mini-picture','extent','cities','places','places-toggle','place-list','tour','source','fit','about','info','status','credits','licence','method','download','previous','next','overview'])ids[id]=new Element();
const media={matches:false,addEventListener(k,f){this.change=f}},frames=[],links=[];
const context={document:{getElementById:id=>ids[id],createElement:()=>new Element(),addEventListener(k,f){events[k]=f}},history:{replaceState(a,b,c){links.push(c)}},location:{hash:''},addEventListener(){},performance:{now:()=>0},devicePixelRatio:1,setTimeout:()=>1,clearTimeout(){},matchMedia:()=>media,requestAnimationFrame:f=>{frames.push(f);return frames.length},ResizeObserver:class{constructor(f){this.f=f}observe(){this.f()}},encodeURIComponent,decodeURIComponent,console};vm.createContext(context);vm.runInContext(SCRIPT,context);
const read=expr=>vm.runInContext(expr,context),run=src=>vm.runInContext(src,context);
assert.equal(read('s'),728/1024);assert.equal(ids.cities.children.length,2);assert.equal(ids.cities.children[0].attrs['aria-current'],'page');
ids['places-toggle'].click();assert.equal(ids.places.hidden,false);assert.equal(ids['places-toggle'].attrs['aria-expanded'],'true');assert.equal(ids['place-list'].children[0].focused,true);ids['place-list'].children[1].click();assert.equal(ids.places.hidden,true);assert.equal(read('current'),1);assert.equal(ids['place-list'].children[1].attrs['aria-current'],'location');run('visit(0,false)');
const before=read('JSON.stringify(pose())');ids.source.click();assert.equal(ids['source-clip'].hidden,false);run('setReveal(72)');assert.equal(ids['source-clip'].style.clipPath,'inset(0 28% 0 0)');assert.equal(read('JSON.stringify(pose())'),before);ids.source.click();assert.equal(ids['source-clip'].hidden,true);
run('x=1e9;y=-1e9;draw()');assert(read('x<=0 && y>=Math.min(0,vh-H*s)'));
ids.next.click();assert.equal(read('current'),1);assert.equal(links.at(-1),'#dc/2');assert.equal(ids.pill.textContent,'Castle');
ids.tour.click();assert.equal(read('playing'),true);while(frames.length) {const frame=frames.shift();frame(0);if(read('playing'))break}assert(read('Number.isFinite(x)&&Number.isFinite(y)&&Number.isFinite(s)'));
ids.view.events.wheel({preventDefault(){},deltaY:-10,clientX:364,clientY:205});assert.equal(read('playing'),false);
ids.view.events.pointerdown({pointerType:'touch',pointerId:1,clientX:200,clientY:200});ids.view.events.pointerdown({pointerType:'touch',pointerId:2,clientX:300,clientY:200});const old=read('s');ids.view.events.pointermove({pointerId:2,clientX:400,clientY:200});assert(read('s')>old);ids.view.events.pointercancel({pointerId:1});ids.view.events.pointerup({pointerId:2});assert.equal(read('pointers.size'),0);
media.matches=true;media.change();assert.equal(ids.tour.disabled,true);ids.tour.click();assert.equal(read('playing'),false);events.keydown({key:'1',target:ids.view,preventDefault(){}});assert.equal(read('current'),0);
run('fit(false)');assert.equal(read('s'),728/2048);assert.equal(ids['art-world'].style.transform,ids['source-world'].style.transform);
ids.view.clientWidth=390;ids.view.clientHeight=844;run('resize()');assert.equal(ids.mini.hidden,true);ids.overview.click();assert.equal(ids.mini.hidden,false);
console.log('viewer interactions passed');'''.replace('SCRIPT',json.dumps(script))
        result=subprocess.run(['node','-e',harness],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('viewer interactions passed',result.stdout)


if __name__ == '__main__':
    unittest.main()
