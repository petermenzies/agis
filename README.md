# agis

Run simple GIS tasks by describing them in plain English. Everything runs on your machine with a local model; no API keys, and your data never leaves your computer.

```console
$ agis "clip roads.gpkg to county.shp and reproject to web mercator"
Clip roads to the county and reproject to Web Mercator (EPSG:3857).
#  operation  details                           output
1  clip       input=roads.gpkg mask=county.shp  (intermediate)
2  reproject  input=$step1 crs=EPSG:3857        roads_clipped_3857.gpkg
Run this plan? [y/n] (y):
Wrote roads_clipped_3857.gpkg  reprojected to EPSG:3857 (WGS 84 / Pseudo-Mercator)
```

## Install (macOS)

You need a Mac with Apple Silicon and at least 8 GB of memory (16 GB recommended). Intel Macs work, but models run on the CPU and are noticeably slower.

```bash
brew install uv ollama        # or: curl -LsSf https://astral.sh/uv/install.sh | sh, and the Ollama app from ollama.com
uv tool install agis-spatial   # or: pipx install agis-spatial
agis setup
```

`agis setup` checks that Ollama is running (and offers to start it), downloads `qwen2.5:3b`, and saves that choice to `~/.config/agis/config.toml`. Pass `--model` to use a different one.

You don't need Conda, a system GDAL, or any particular Python: `uv` installs agis in its own environment with its own Python, and the geospatial libraries ship with GDAL, PROJ and GEOS built in.

If something isn't working, run `agis doctor` and include its output in bug reports.

## Usage

```bash
agis                          # interactive prompt; later requests can say "now buffer that"
agis "reproject parcels.shp to EPSG:2227"
agis "buffer the streams by 30 m and remove that from parcels, save as buildable.gpkg"
agis "make an MBTiles file from ortho.tif up to zoom 16"
agis "what projection is dem.tif in?"
```

`agis` on its own opens the prompt. Arrow keys move through the line you are typing, and up and down walk requests from this session and from earlier ones (`~/.config/agis/history`). Tab completes dataset names in the working folder. `ls`, `cd`, and `pwd` run locally and are not sent to the model. `exit` or Ctrl-D quits. A quoted request still runs once and exits.

agis reads the files in the current folder (CRS, geometry type, fields, raster bands) so the model knows what it's working with. It always shows the plan before running it.

| Option | Effect |
| --- | --- |
| `-n`, `--dry-run` | Show the plan without running it |
| `-y`, `--yes` | Run without asking |
| `-f`, `--force` | Allow overwriting existing outputs |
| `-m`, `--model` | Use a different Ollama model |
| `-C`, `--dir` | Work in another folder |
| `-v`, `--verbose` | Show the file context and raw plans |

Outputs are named after their inputs and steps (`roads_clipped_3857.gpkg`) unless you name them. agis never overwrites an input. If the output name is already taken, it appends a number (`roads_clipped_3857_2.gpkg`); `--force` overwrites that file instead. Intermediate results of multi-step plans go to a temporary folder that is deleted afterwards.

### Operations

| Operation | Works on |
| --- | --- |
| reproject (EPSG codes, common names like "web mercator", "utm", or "match county.shp") | vector, raster |
| clip (to a mask layer or a bounding box) | vector, raster |
| difference, union | vector |
| merge (append features from other vector files; columns line up by name) | vector |
| buffer (meters, kilometers, feet, miles; correct in any CRS) | vector |
| dissolve (all, or by a field) | vector |
| filter (keep rows by attribute: equals, comparisons, contains, in, null) | vector |
| select (keep named columns; geometry is always kept) | vector |
| join (attach a CSV, TSV, or another layer's columns on a shared field) | vector + table |
| info | vector, raster, table |
| convert (gpkg, shp, geojson, fgb, kml, tif, Cloud Optimized GeoTIFF) | vector, raster |
| tile_raster (MBTiles of 256px PNG tiles) | raster |

Requests that need anything else (slope, contours, and so on) get a short explanation instead of a guess.

## How it works

agis uses a plan-then-execute design built for small local models:

1. **Context:** summarize the files in the folder and any the request mentions.
2. **Plan:** one call to the model through Ollama, with output constrained to a JSON schema generated from the operation definitions, so the model can only produce well-formed plans.
3. **Validate:** check every file, `$stepN` reference, CRS and output path. If the plan can't run, the problems go back to the model for one corrected attempt.
4. **Preview and confirm**, then **execute** with GeoPandas, pyogrio, shapely, pyproj and rasterio. If a step fails, files from that run are removed and the error goes back to the model for one re-plan.

The model never writes or runs code; it only fills in arguments for a fixed set of operations.

```
src/agis/
  cli.py          command line (run, repl, setup, doctor, eval)
  session.py      plan -> validate -> preview -> confirm -> run, with re-planning
  context.py      file summaries given to the model
  planner.py      system prompt, few-shot examples, model call
  plan.py         Plan schema built from the operation registry
  executor.py     validation, output naming, running steps
  llm.py          model backend (Ollama)
  ops/            one Pydantic model per operation, with its implementation
  setup.py        agis setup / agis doctor
  evals.py        agis eval
```

To add an operation, subclass `OpArgs` in `agis/ops/` with an `op: Literal["name"]` field, describe its arguments with `Field(description=...)`, and implement `run`. It's registered automatically and shows up in the schema and prompt.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
.venv/bin/pytest                  # unit tests, no model needed
.venv/bin/agis eval -m qwen2.5:3b   # plan accuracy against tests/evals/cases.yaml
```

`agis eval` runs each case in `tests/evals/cases.yaml` against generated sample data and compares the plan to the expected one (CRSs, units and file names are normalized). Use it to compare models or check prompt changes; `--only` filters cases and `--show` prints every plan.

## License

MIT
