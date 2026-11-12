# Registered sources

Every source uses one camera and pixel lattice for the whole neighbourhood. Imagery colours surfaces in world coordinates before that camera projects them. Rotating a flat satellite photo cannot reveal facades.

## Washington DC

```sh
python3 atlas.py photo --city capitol-hill-dc --out outputs/dc-source
# Reuse that output's captured inputs, with network disabled:
python3 atlas.py photo --city capitol-hill-dc --out outputs/dc-source --offline
```

The default 8192×4096 capture uses 2.5 px/m and 35.264° equal-axis isometric elevation. It includes the White House, Eisenhower Executive Office Building, Washington Monument, National Mall museums, Capitol, Supreme Court, Library of Congress and Botanic Garden. Union Station is clipped at the top; the Jefferson Memorial is outside this frame.

The recipe configures ArcGIS ImageServer aerial/DTM rasters and the documented uncompressed I3S 1.6 building-mesh schema. Native raster chunks align exactly across provider download limits. ImageServer captures also count underlying catalogue rasters and subdivide requests to respect mosaic limits. A successful HTTP response alone does not establish coverage; offline replay requires the captured coverage receipt and matching raster hash. Roofs and ground sample aerial colour by world XY. Vertical walls use illustrative neutral materials and windows, never overhead pixels stretched down a facade. OSM supplies footprint references and explicit Capitol dome parts.

Sources: [DC 2023 aerial](https://catalog.data.gov/dataset/aerial-photography-orthophoto-2023), CC BY 4.0, 8 cm native sampled at 50 cm; [2024 DTM](https://catalog.data.gov/dataset/2024-lidar-hydro-enforced-digital-terrain-model), CC0, NAVD88 metres at 1 m; [DC Office of Planning / CyberCity 3D buildings](https://services.arcgis.com/neT9SoYxizqTHZPH/arcgis/rest/services/DC_3D_Buildings/SceneServer); and [OSM](https://www.openstreetmap.org/copyright), ODbL.

The legacy mesh predates the aerial and terrain. This is a historical mixture, including July 2026 OSM, and does not reconstruct 2025-2026 White House changes. Ornamentation and facade relief are approximate. The government-area DSM is redacted, so photography supplies vegetation appearance without measured current tree heights. Vertical residuals are recorded without guessing a geoid shift. Source-input hashes and assumptions stay in `manifest.json`.

Another city needs compatible licensed data and a new `photographic` recipe. This provider is not a universal photogrammetry downloader.

Optional `photographic.landmark_geometry` records add cited obelisks, arc colonnades and bounded facade corrections. Dimensions, materials, provider feature IDs and references live in the recipe. Invalid counts, coordinates, bands or missing provider features stop the render. These are illustrative source corrections, not surveyed ornament.

## Edinburgh

```sh
python3 atlas.py surface --city edinburgh --out outputs/edinburgh-source
python3 atlas.py surface --city edinburgh --out outputs/edinburgh-source --offline
```

The 3456×1803 source uses Scottish 0.5 m DSM/DTM at 3 px/m, looking northeast from the southwest at 30° elevation. This traditional 2:1 game view is technically dimetric. It includes Castle Rock, Old Town, Princes Street Gardens, Scott Monument, Waverley, Greyfriars, the museum and the Old College/McEwan Hall cluster. Holyrood is outside the frame.

The renderer range-reads configured public COG rasters, captures dated OSM geometry and downloads a SHA-pinned OSTN15 horizontal transformation grid. Preserve `outputs/cache/surface/edinburgh-v3/` for exact source replay. A fresh provider response creates newly recorded inputs; it need not match the delivered snapshot.

Measured roofs, chimneys, spires, terrain and tree canopy form one z-buffered surface. LiDAR contains no RGB. Tagged roof/wall materials take priority; untagged surfaces use slope and documented palette defaults. This does not reconstruct each roof's true colour. The palette is source-informed, not randomly varied. Another city can supply `surface.palette`.

Existing inputs can be supplied with `--surface PATH.npz --osm PATH.json --grid PATH.tif --offline`. NPZ contains finite `dsm`, `dtm` arrays of identical shape and `[west,south,east,north]` bounds, rows north to south. The recipe declares projected CRS and resolution. OSM must be Overpass `out body geom` JSON. `source.json` records all hashes, camera matrix, palette and landmarks.

For another location, a `surface` recipe supplies aligned DSM/DTM COG URLs, CRS, bounds, resolution, projected camera centre, bearing, elevation and screen bounds. Source data remains Scottish Crown copyright/Fugro under OGL v3 and OSM ODbL. [Full notices](../THIRD_PARTY_NOTICES.md).

An optional `surface.landmarks` file can replace smeared heightfield buildings with footprint meshes, measured roof planes and authored landmark structures. The file binds the input hashes, CRS and exact camera; changed inputs require remeasurement. Edinburgh's example is under `cities/landmarks/`. Fine architectural details remain illustrative. Other cities can omit this layer or have their agent author a source-bound one.

For the bundled examples, use their source PNGs for styling without reacquiring data. Fresh Edinburgh provider responses can differ from the pinned landmark inputs and require the agent to remeasure that layer. The bundled source preserves the exact pixels used by the example.

## Any other neighbourhood

`atlas.py make "LOCATION"` produces a bounded OSM/Skadi starter when no photographic/surface recipe exists. It preserves source footprint holes, reconciles building parts, records numeric versus inferred heights, uses roof material/colour tags and declares untextured facades and terrain gaps. Missing terrain falls back explicitly. The starter is a place to inspect and improve, not photographic coverage or finished art.

Keep the exact source PNG, immutable source metadata and captured inputs. The bundled PNGs preserve the pixels used for each example even if a provider changes. Source geometry checks and an attractive model output are separate evidence; inspect landmarks and ordinary blocks before accepting either.
