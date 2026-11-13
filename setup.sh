#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --models ) ]]; then
  echo 'Usage: ./setup.sh [--models]'; exit 1
fi
if ! command -v uv >/dev/null; then
  if command -v brew >/dev/null; then brew install uv
  else echo 'Install uv from https://docs.astral.sh/uv/getting-started/installation/ then run ./setup.sh'; exit 1; fi
fi
if command -v brew >/dev/null && ! command -v ffmpeg >/dev/null; then brew install ffmpeg; fi
if [[ ! -x "$repo_dir/.venv/bin/python" ]]; then uv venv --python 3.13 "$repo_dir/.venv"; fi
uv pip install --python "$repo_dir/.venv/bin/python" -r "$repo_dir/source/requirements.txt"
if [[ "${1:-}" == --models ]]; then
  if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    echo 'Local image inference needs Apple Silicon macOS.'; exit 1
  fi
  image_env="$repo_dir/outputs/image-runtime"
  if [[ ! -x "$image_env/bin/python" ]]; then uv venv --python 3.12 "$image_env"; fi
  uv pip install --python "$image_env/bin/python" -r "$repo_dir/source/image-requirements.txt"
  echo 'Fetching the pinned model and adapter into the Hugging Face cache (about 9 GB on first run).'
  "$image_env/bin/python" - "$repo_dir" <<'PY'
import json,sys
from pathlib import Path
from huggingface_hub import snapshot_download,hf_hub_download
root=Path(sys.argv[1]);preset=json.loads((root/'model-presets/pixelart-redmond-klein9b.json').read_text())
base,adapter=preset['base'],preset['adapter']
snapshot_download(base['repo'],revision=base['revision'])
hf_hub_download(adapter['repo'],adapter['filename'],revision=adapter['revision'])
sys.path.insert(0,str(root/'source'))
from local_style import model_paths
print('Pinned model and adapter ready:',model_paths(preset))
PY
fi
# A missing FFmpeg only stops films and GIFs. The demo runs without it, so setup still succeeds.
"$repo_dir/.venv/bin/python" "$repo_dir/atlas.py" doctor || true
