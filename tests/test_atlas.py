"""The bundled examples work without downloads and refuse damaged cached exports."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AtlasWorkflow(unittest.TestCase):
    def test_demo_exports_both_examples_and_checks_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)/'demo'
            command = [sys.executable, str(ROOT/'atlas.py'), 'demo', '--out', str(out)]
            def run(ok=True):
                result = subprocess.run(command, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode == 0, ok, result.stderr)
                return result
            first = json.loads(run().stdout)
            self.assertEqual(list(first), ['dc', 'edinburgh'])
            self.assertIn('url=dc/viewer.html', (out/'index.html').read_text())
            for city in first:
                self.assertEqual((out/city/'map.png').read_bytes(), (ROOT/f'examples/{city}.png').read_bytes())
                self.assertEqual((out/city/'source.png').read_bytes(), (ROOT/f'examples/{city}-source.png').read_bytes())
                self.assertIn('Compare source', (out/city/'viewer.html').read_text())
            self.assertTrue(all(value['cached'] for value in json.loads(run().stdout).values()))
            tile = next((out/'dc/map_files').rglob('*.png'))
            original = tile.read_bytes()
            tile.write_bytes(original+b'changed')
            self.assertIn('changed', run(False).stderr)
            tile.write_bytes(original)
            page = out/'edinburgh/viewer.html'
            page.write_text(page.read_text()+'changed')
            self.assertIn('explorer', run(False).stderr)

    def test_help_exposes_one_current_workflow(self):
        result = subprocess.run([sys.executable, str(ROOT/'atlas.py'), '--help'], capture_output=True, text=True, check=True)
        for command in ('demo', 'make', 'paint', 'finish', 'showcase', 'film', 'tease'):
            self.assertIn(command, result.stdout)
        self.assertNotIn('training', result.stdout)


if __name__ == '__main__':
    unittest.main()
