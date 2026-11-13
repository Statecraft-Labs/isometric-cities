# Make a place

Apple Silicon macOS. `./setup.sh` installs source/export dependencies. Add `--models` to install a separate Python 3.12 MFlux runtime under `outputs/image-runtime` and download the pinned model and adapter, about 9 GB, public and ungated. Weights remain in the Hugging Face cache. The tested image environment, including transitive dependencies, is pinned in `source/image-requirements.txt`. FFmpeg is required for films and GIFs. Node is optional, used by the UI tests.

Development used one 128 GB Mac. A native 1024-pixel window peaked at 18.3 GB of MLX memory and took about two and a half minutes, model load included. Smaller machines are untested.

## Choose and inspect the source

```sh
python3 atlas.py make "Glasgow city centre"
```

Bundled aliases resolve first. "Capitol Hill" means Washington, DC. Any other place uses a cached Nominatim search, then a bounded OSM/Skadi starter. Explicit `"55.8609,-4.2514"` resolves offline. Sources record terrain gaps, tagged versus inferred heights, courtyard holes, camera and credits. Unknown places do not acquire photographic coverage automatically.

The starter is an untextured block model, 1024×640, small enough to style as one image. For a larger map, copy the generated `recipe.json`, add `"frame": {"padding_metres": 400}` under `camera` (2560×1600 at the default scale), and run `make` on that file with a new `--out`. `paint` needs both sides to reach 1024 pixels.

Use `--plan-only` to resolve without downloading/rendering sources or invoking a model. Source caches and receipts live under ignored `outputs/`. Repeat the same command to resume. `--offline` prohibits acquisition and needs cached inputs.

Improve the source before styling. Find primary licensed aerial, oblique, DSM or mesh data. Roof colour comes from tagged materials or registered imagery, never random variation. [Provider formats and examples](sources.md).

A supplied image can start a pilot immediately:

```sh
python3 atlas.py make "My location" --source /absolute/source.png \
  --manifest /absolute/source.json --crop 64 112 1024 1024 \
  --out outputs/my-pilot --plan-only
```

Crop values are literal X Y WIDTH HEIGHT. Inputs must be native RGB with no pending EXIF orientation. No resizing or out-of-bounds shifting occurs. A manifest binds the original image hash and camera; cropping translates only its matrix offsets. Omit it for an explicitly unregistered image study.

The full source remains intact up to 64 MP. The selected preset styles native 1024-square windows. Larger or unaligned images stay source-ready until you choose an exact crop or use `paint`.

Reuse a bundled source directly, without downloads:

```sh
python3 atlas.py make Edinburgh --source examples/edinburgh-source.png \
  --manifest examples/edinburgh.json --crop 240 779 1024 1024 --out outputs/castle
```

The supplied source is already registered. Acquiring a fresh Edinburgh surface can change the source hashes and needs remeasuring the optional landmark layer.

## Try one treatment

Repeat the pilot command with `--style --wait`. Use `--caption-file PATH` for visible materials and `--preset PATH` for another pinned model. The default is distilled Klein 9B q4 with Pixel Art Redmond at 0.7, four steps, CFG 1 and seed 4601. Name dominant materials only. Naming a rare material globally can spread it onto other roofs.

Styling runs in a fresh isolated MFlux worker using the downloaded pins. It runs offline, verifies native conditioning dimensions before inference, and serializes GPU workers. It never changes installed model code. There is no cloud fallback. It needs Metal GPU access. If `native-worker.log` reports `No Metal device available`, as it did inside one coding agent's command sandbox, move to a shell with GPU access. The failed job is kept and never relaunched, so give a pilot a fresh `--out`; for `paint`, use the failed-window recovery below.

The example maps were made differently: through `imagine`, a private local job queue running the same pinned model, adapter and settings. A compatible queue client is still used when `ATLAS_IMAGINE` names one or one is installed, and the worker then stands down so only one model loads. An unrelated program that merely shares the name is ignored. The native worker has repainted the full Edinburgh example from a fresh clone on the stock pinned runtime: 10 windows in 24 minutes. The result has the same style and layout as the queue's paint, with small differences in detail. It is not byte-identical, so the bundled finishing recipe refuses it, as its hash guard should. DC through the worker is untested.

Receipts advance through resolved, source-ready, queued, art-ready and reviewed. `--wait` checks an existing job for up to 45 seconds. Repeat to resume. Interrupted submissions recover their ID from the saved log; ambiguous submissions require inspecting that log and never submit twice automatically. A native worker that exits without a result is reported as failed at once. Changed recipes, source pixels, model settings and captions require a fresh `--out`.

Open the generated `viewer.html` directly. Inspect a landmark, an ordinary street and a park at native size. Compare colours, buildings, courtyards, light and outlines against the source. Model completion does not establish geographic acceptance. To record a review, add `--review PATH`. The JSON needs reviewer, decision, source_sha256, image_sha256, findings and four checks: landmarks, materials, geometry, native_detail. Acceptance requires all four to be `pass`; `reject` preserves the study.

## Expand the same treatment

```sh
python3 atlas.py paint --source examples/dc-source.png \
  --caption-file examples/dc-caption.txt --out outputs/dc-art
```

Substitute `edinburgh` for the other example. `--plan-only` reports the window count without resolving models or submitting work. Native 1024-square windows have at most an 800 px stride. Each next input includes finished neighbours, and a 128 px distance feather composes overlaps. This is sequential context infill, not shared-latent denoising. Cut display tiles only after the complete raster passes visual review.

Where a window holds nothing to redraw, the model invents: a window of open sea came back as an island of buildings. `--keep MASK.png` takes an 8-bit greyscale mask at the source size. White pixels keep the source and are never redrawn, and an all-white window gets no job. Draw the kept area into the source first, for example as flat water tones.

Every window keeps its input, raw prediction, settings and durable job receipt. Resume binds the source, prompt, preset, composition code and backend. A failed/ambiguous job is never replaced automatically. After reading its log, move that one folder out of `windows/` and repeat `paint`: finished windows are verified and reused, and only the missing one is submitted. `--adopt DIRECTORY` can replay original experimental windows, validating their exact neighbour context and pinned commands before composition.

The bundled recipes bind the original raw prediction hashes. Those raw paints are too large to bundle, so the `finish` command below reproduces the examples only from them. Re-inference can differ across Metal/runtime versions. If the fresh raw hash differs, review it and make a new finishing recipe. Do not disable a hash guard to reuse old masks.

```sh
python3 atlas.py finish --image outputs/dc-art/map.png \
  --source examples/dc-source.png --recipe examples/dc-finish.json \
  --out outputs/dc-finished
```

The finish preserves the native pixel grid. DC repairs source-open Monument lawn and the White House North Lawn fountain. Edinburgh applies its base-paint ink/material finish, then replaces the wrong Scott Monument using corrected measured source geometry and a calibrated local Gothic asset, and restores the St Giles crown finial. Source revisions and illustrative detail are recorded explicitly. Reviewed replacement PNGs, native support runs and hashes are bundled. `patches` run before material finishing; `post_patches` run after it and bind the finished prior pixels, so their accepted colours remain exact. Native scanline masks protect vegetation, neighbouring walls and unrelated landmarks. Each recipe binds raw/source hashes and records changed-pixel counts. Another city needs its own reviewed masks or repair asset.

## Explore and film

```sh
python3 atlas.py demo --open
python3 atlas.py showcase --image examples/dc.png \
  --source examples/dc-source.png --metadata examples/dc.json --out outputs/my-viewer
python3 atlas.py film --image examples/dc.png --manifest examples/dc.json \
  --output outputs/dc-flythrough.mp4 --seconds 60 --fps 60 --width 1920 --height 1080
```

The offline explorer loads only the visible tiles, with small previews for instant navigation. Drag to pan, pinch or scroll to zoom, and use the divider to compare source and art in the same viewport. The places menu jumps directly to named landmarks. Phone controls have 44-pixel tap targets; Scott Monument and the White House open at native scale. Number keys also hop to landmarks; any input stops the tour. The minimap supports click and drag. Credits remain in the viewer, and reduced motion disables automatic movement. Deep Zoom lower levels use box filtering; native tiles reconstruct the map exactly. Default `demo` selects an immutable edition under `outputs/demo/`, so updated examples open correctly while old exports survive. An explicit `--out` stays strict: use a fresh directory when inputs change.

For iPhone Safari on the same Wi-Fi, run `python3 atlas.py demo --serve --host 0.0.0.0`. Open `http://YOUR-MAC-LAN-ADDRESS:8765/` on the phone. Only the current exported map folder is served, and Ctrl-C stops it. The default `--serve` host is loopback; no site is deployed. A real iOS Safari device still needs its own acceptance check.

The supplied film routes use 1:1 pixels, with no title, HUD or callouts. Credits travel in MP4 metadata and `.credits.txt`; retain them in a sharing description. Optional `--callouts` allows at most seven on-screen words. Its leader tips use original map anchors, independent of camera clamping. The exporter checks every planned viewport and writes route/bounds sidecars. Before sharing, inspect encoded frames and fully decode the movie: `ffmpeg -v error -xerror -i outputs/dc-flythrough.mp4 -f null -`.

```sh
python3 atlas.py social --manifest examples/dc.json --output outputs/dc-social.mp4
```

`social` cuts a 1080×1350 portrait film for a phone feed: keyed shots, no text, and a last frame that returns to the first so it loops. A shot list lives beside the manifest as `NAME-social.json`. The DC list opens on finished art of the Capitol, cuts to the Washington Monument and the White House, then sweeps the registered source into art at three seconds. Without a list, the manifest landmarks make a default that opens split between source and art and wipes to art. Each shot has `seconds`, a `path` of keys (`at` a landmark ID or map X,Y, optional `offset` and `width` in map pixels) and optional `ease` and `wipe`. Shots cut unless neighbours share a key. Views stay within 1.25× of native unless the recipe sets `pixel_zoom`; closer views square the pixels, never smooth them. `--plan-only` checks every frame and writes the poster and a phone-sized contact sheet. Inspect those before encoding. The portrait master is retained at full quality. For a smaller 720×900 upload copy with a silent AAC track:

```sh
ffmpeg -i outputs/dc-social.mp4 -f lavfi -i anullsrc=r=48000:cl=stereo \
  -map 0:v:0 -map 1:a:0 -map_metadata 0 -vf scale=720:900:flags=lanczos \
  -c:v libx264 -crf 18 -c:a aac -shortest -movflags +faststart outputs/dc-x.mp4
```

```sh
python3 atlas.py tease --image examples/dc.png --manifest examples/dc.json \
  --output outputs/dc.gif --centres 6650,1110 886,2670 2688,3335 4300,2530
```

README GIFs hold four native 800×450 crops for two seconds each. No overlays or interpolation. Edinburgh centres are `760,1330 1850,660 860,470 1450,390`. Static fallbacks are the first native crops.

## Change a recipe

Add one JSON file under `cities/`, or give `make` its path. The recipe owns aliases, origin, bounds, landmarks and provider/camera settings. The two bundled recipes show each provider. Copy a generated recipe outside its run before editing, then use a new output directory. For high-altitude places, terrain establishes camera focus; a measured `landmarks[0].elevation_m` can override it. Explicit screen/elevation bounds require your own coverage check.

Nominatim lookups are cached and serialized at one request per second, with attribution and an identifying user agent. No autocomplete or bulk searching. Set `ATLAS_GEOCODER_URL` for another compatible endpoint. Cached failures require inspection before deliberate retry. Geocoding establishes a centre, not imagery coverage.

Run `python3 atlas.py doctor`, `.venv/bin/python -m unittest discover -s tests -v`, and optionally `ruff check atlas.py source tests --select F`. Keep weights, captures, failures and private coordinates out of commits. New scenes need their own source and art review.
