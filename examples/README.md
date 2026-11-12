# Separate neighbourhoods

[DC map](dc.png) · [Edinburgh map](edinburgh.png) · [Pipeline](pipeline.png)

DC covers the White House, Washington Monument, National Mall museums, Capitol, Library of Congress, Supreme Court, Senate and House offices, Botanic Garden and American Indian museum. Edinburgh covers the Castle and Rock, Royal Mile, St Giles, Princes Street, Waverley, Greyfriars, National Museum, Old College, McEwan Hall and Teviot. Holyrood and the Jefferson Memorial are outside these frames; Union Station is clipped.

Each city has its own native source, art PNG, caption, finishing recipe and JSON provenance. `dc.json` and `edinburgh.json` bind the actual file hashes, camera, local model, window composition and geographic limitations. Sources and art have identical dimensions. The map is completed before display tiles are cut.

## What was actually done

Real 3D sources → one fixed camera per city → native 1024-square edits with a public pixel-art LoRA → completed neighbours included in each next input → 128 px overlap feather → bounded local colour finish → explorer and silent film.

All image inference ran on one Mac through `imagine`, a private local job queue, using Klein 9B q4, four steps, guidance 1, Redmond scale 0.7 and seeds starting at 4601. The map-generation method is **context infill, not shared-latent sampling**. There is no local training or cloud image teacher in this recipe.

DC includes bounded repairs for invented buildings on the Monument lawn and a tree that replaced the White House North Lawn fountain. Edinburgh gets a gentle outline lift and Great Hall slate tint. Its Scott Monument was rebuilt from the real mapped footprint and measured height, then given a local Gothic art asset calibrated to that corrected silhouette. Its four outer turrets stand free of the arched base, as in the source. St Giles keeps its source-height crown finial. Fine tracery and the statue are illustrative. Replacement PNGs, source/prior-pixel hashes and native support runs are bundled. Exact finishing replay also needs the original raw paints, which are not. The base-paint source hash remains recorded separately from the corrected source. Some facade patterns and tiny ornaments are invented. See each JSON's limitations and compare the source in the viewer.

## Credits

Edinburgh geometry: Scottish Government and Fugro Phase 5 LiDAR, Crown copyright 2020–2021, [OGL v3](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/). © OpenStreetMap contributors, [ODbL](https://www.openstreetmap.org/copyright). Material palette evidence: Andrew Shiva / Wikipedia, [Scotland-2016-Aerial-Edinburgh Castle](https://commons.wikimedia.org/wiki/File:Scotland-2016-Aerial-Edinburgh_Castle.jpg), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). No aerial photo pixels enter the Edinburgh source. Its roof geometry is measured, but untagged colours are material fallbacks. Edinburgh examples and their derivative diagram are distributed under CC BY-SA 4.0.

DC: DC GIS / OCTO 2023 aerial imagery, CC BY 4.0; DC 2024 terrain, CC0; District of Columbia Office of Planning / CyberCity 3D building geometry. © OpenStreetMap contributors, ODbL. Roof and ground colour comes from the registered aerial raster; facade windows and column relief are illustrative. DC examples retain CC BY 4.0 attribution.

Klein 9B uses FLUX Non-Commercial terms. The public [Pixel Art Redmond adapter](https://huggingface.co/artificialguybr/PIXELART-REDMOND-FLUXKLEIN9B) is Apache-2.0. No weights are included. [Full notices](../THIRD_PARTY_NOTICES.md).
