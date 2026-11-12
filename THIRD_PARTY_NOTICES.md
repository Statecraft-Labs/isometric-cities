# Source credits

The MIT licence covers this repository's original code. Source datasets, model weights and third-party libraries retain their own licences.

Maps use © OpenStreetMap contributors, [ODbL 1.0](https://www.openstreetmap.org/copyright). Acquired geometry and exported footprint databases retain source and licence records. Credits are included in each map manifest, website and film's associated description.

The Edinburgh recipe uses Scottish phase 5 LiDAR under [OGL v3](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/). Credit: DTM: Crown copyright Scottish Government and Fugro (2021). DSM: Crown copyright Scottish Government and Fugro (2020).

The default Capitol Hill, Washington DC photographic recipe uses [DC 2023 aerial imagery](https://catalog.data.gov/dataset/aerial-photography-orthophoto-2023), CC BY 4.0; [2024 terrain](https://catalog.data.gov/dataset/2024-lidar-hydro-enforced-digital-terrain-model), CC0; and the older DC Office of Planning / CyberCity 3D building layer. Keep the District of Columbia Office of Planning credit with its geometry. OSM supplies footprint references and explicit dome parts. See [the source method](docs/sources.md).

The city examples retain source credits in `examples/dc.json` and `examples/edinburgh.json`. Edinburgh's geometry is Scottish LiDAR, not a composited photograph. Its material palette was informed by [Scotland-2016-Aerial-Edinburgh Castle](https://commons.wikimedia.org/wiki/File:Scotland-2016-Aerial-Edinburgh_Castle.jpg), Andrew Shiva / Wikipedia, CC BY-SA 4.0. Edinburgh imagery and the derivative pipeline diagram are distributed under CC BY-SA 4.0. DC imagery retains CC BY 4.0 attribution.

The style uses [FLUX.2 Klein 9B q4](https://huggingface.co/mflux-community/flux2-klein-9b-mflux-q4), FLUX Non-Commercial, with [Pixel Art Redmond](https://huggingface.co/artificialguybr/PIXELART-REDMOND-FLUXKLEIN9B), Apache-2.0. Exact revisions and the adapter SHA are in `model-presets/pixelart-redmond-klein9b.json`. Model terms remain separate from source-image and code licences. No weights are bundled.

The portable coarse terrain option uses [Mapzen Skadi](https://github.com/tilezen/joerd/blob/master/docs/attribution.md), with underlying USGS, Copernicus and Environment Agency source credits recorded in the manifest. It is a mixed-source surface, not a surveyed building model.

Local image models are optional and downloaded separately. Record the model repository, exact revision and licence in each experiment. This code licence does not relicense those weights. No third-party style-reference image or model weight is bundled.

The local generation experiments were informed by the authors' primary writeups: [Isometric NYC](https://cannoneyed.com/projects/isometric-nyc), [Tiny London](https://tiny.london/how/), [Isometric SF](https://www.justinchavez.com/blog/isometric-sf/) and [Isometric Toronto](https://iso.sloptoronto.ca/how). This repository does not redistribute their maps or trained weights.
