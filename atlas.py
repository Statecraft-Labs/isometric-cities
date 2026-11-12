#!/usr/bin/env python3
"""Local isometric maps. Explore DC and Edinburgh, or make another place."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source'))
from project import interpreter

COMMANDS = {
    'demo': ('showcase', 'demo_main', 'Explore both bundled examples offline'),
    'make': ('make_location', 'main', 'Resolve a place and prepare its registered source'),
    'photo': ('photographic_patch', 'main', 'Capture aerial colour, terrain and building geometry'),
    'surface': ('surface_patch', 'main', 'Render a measured DSM/DTM surface'),
    'paint': ('context_infill', 'main', 'Grow a map with neighbouring local image edits'),
    'finish': ('material_finish', 'main', 'Apply source-reviewed colour corrections'),
    'showcase': ('showcase', 'main', 'Export an offline explorer and Deep Zoom tiles'),
    'film': ('media', 'main', 'Export a quiet map fly-through'),
    'social': ('social', 'main', 'Cut a portrait feed film with a source-to-art wipe'),
    'tease': ('media', 'tease_main', 'Make a lightweight README landmark GIF'),
}


def doctor():
    packages = {}
    for line in (ROOT/'source/requirements.txt').read_text().splitlines():
        name = line.split('==')[0].strip()
        if name and not name.startswith('#'):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
    checks = {'python_dependencies': all(packages.values()),
              **{name: bool(shutil.which(name)) for name in ('ffmpeg', 'ffprobe')}}
    print(json.dumps({'checks': checks, 'packages': packages,
                      'next': 'python3 atlas.py demo' if all(checks.values()) else './setup.sh'}, indent=2))
    return 0 if all(checks.values()) else 1


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in COMMANDS:
        module, entry, _ = COMMANDS[args[0]]
        return getattr(importlib.import_module(module), entry)(args[1:])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n'.join(f'  {name:10} {description}' for name, (_, _, description) in COMMANDS.items()))
    parser.add_argument('command', nargs='?', choices=[*COMMANDS, 'doctor'], default='demo')
    parsed = parser.parse_args(args)
    return doctor() if parsed.command == 'doctor' else main(['demo'])


if __name__ == '__main__':
    runtime = interpreter()
    if sys.flags.no_site or Path(sys.prefix).resolve() != (ROOT/'.venv').resolve() and runtime != Path(sys.executable):
        os.execv(str(runtime), [str(runtime), str(Path(__file__).resolve()), *sys.argv[1:]])
    try:
        raise SystemExit(main())
    except ModuleNotFoundError as error:
        print(f'Missing dependency {error.name!r}. Run ./setup.sh, then repeat the command.', file=sys.stderr)
        raise SystemExit(1)
    except (ValueError, OSError) as error:
        print(json.dumps({'status': 'failed', 'reason': str(error)}), file=sys.stderr)
        raise SystemExit(1)
