# DAT-991: One-line population of example project data

## Context

Trying Datamint's one-line trainers or other features currently requires the user to
already have their own annotated data uploaded. There's no quick way to spin up a working
project with real data to experiment with. The ask: a one-liner like
`datamint.examples.busi_dataset.create('MyBusiDataset')`, plus (per brainstorm) a CLI form.

There's a small existing precedent, `datamint/examples/example_projects.py`'s `ProjectMR`
class — it downloads one tiny pydicom test-fixture DICOM + one hardcoded mask PNG, uploads
them, and creates a project (skipping with a warning if the project already exists). It's
narrow (one hardcoded image, not a real dataset) and not wired into `datamint`'s top-level
lazy-loader, but its shape (check-exists → download → upload → create) is the right pattern
to generalize.

**Key finding from research, discussed with you:** of the three datasets you originally
proposed — BCCD (detection), BUSI (2D segmentation), Synapse (3D segmentation) — BCCD is
genuinely zero-setup (public GitHub zip, MIT license, no auth). BUSI and Synapse are fetched
via `https://www.kaggle.com/api/v1/datasets/download/<owner>/<slug>` in the existing
notebooks. Verified live (2026-07-07): both URLs return the real dataset zip via a bare
`curl -L`/`requests.get`, no `KAGGLE_USERNAME`/`KAGGLE_KEY`, no `~/.kaggle/kaggle.json`, no
cookies — this specific Kaggle download endpoint currently serves public datasets
anonymously, matching exactly what the notebooks already do. So all four datasets are
zero-setup today. Caveat: this is Kaggle's browser-facing download URL, not their documented
authenticated API — it's not a published contract, so it could start requiring auth or get
rate-limited for anonymous/scripted access without notice. Synapse is additionally the most
access-restricted of the three by dataset license (the notebook itself has an "⚠️ Dataset
Access" warning) — it's almost certainly the BTCV/Synapse.org challenge dataset, which
commonly disallows redistribution outside its own registered-access mechanism; that's a
licensing concern independent of the download-auth question.

We're adding a fourth dataset, **FracAtlas** (binary fracture classification), covering the
`classification` task type — it turns out to be the *best* candidate of all four for
zero-setup: hosted on Figshare with a plain public API (`api.figshare.com`), no auth
required at all, same as BCCD. Only caveat is size: ~1.2GB compressed, so it's a slower
download than the other three, worth surfacing to the user up front.

**Decisions confirmed with you:**
- All four datasets stay zero-setup: BUSI/Synapse use the same anonymous Kaggle
  download-endpoint curl the notebooks already use, no Kaggle credentials required. No
  re-hosting, no new licensing exposure beyond what the notebooks already do today.
- CLI: both a standalone `datamint example <name>` subcommand, and a prompt folded into the
  existing `datamint init` wizard.
- Don't refactor the existing notebooks to share code with the new module in this ticket —
  they keep their own inline download/upload logic; the new `datamint.examples.*` code is
  new, independent code modeled on the same approach.
- All four of `datamint init`'s task types now have a matching example dataset: detection →
  BCCD, segmentation (2D) → BUSI, segmentation (3D) → Synapse, classification → FracAtlas.

## Design

### Package layout

```
datamint/examples/
  __init__.py          # existing export (ProjectMR) + new: bccd_dataset, busi_dataset, synapse_dataset, fracatlas_dataset
  example_projects.py  # existing, untouched
  _common.py           # NEW: get_or_create_project(project_name, api) -> (Project, already_existed)
                        #      shared by all three new modules (same skip-with-warning behavior as ProjectMR.create)
  _download.py          # NEW: download_and_extract(url, cache_subdir) -> Path
                        #      one function for all four datasets, including BUSI/Synapse's
                        #      Kaggle URLs — no auth needed (verified live), so no separate
                        #      kaggle_download()/credential handling
  bccd_dataset.py       # NEW: create(project_name='BCCD Detection Example') -> Project
  busi_dataset.py       # NEW: create(project_name='BUSI Segmentation Example') -> Project
  synapse_dataset.py    # NEW: create(project_name='Synapse Segmentation Example') -> Project
  fracatlas_dataset.py  # NEW: create(project_name='FracAtlas Classification Example') -> Project
```

Each `create()` re-implements (not imports from) the relevant notebook's fetch/parse/upload
logic as a proper function:
- `bccd_dataset.create()`: download+extract the BCCD zip via `_download.download_and_extract`,
  parse Pascal VOC XML, upload resources + box annotations (mirrors
  `notebooks/06_end_to_end/slice_based/03_bccd_detection.ipynb`).
- `busi_dataset.create()`: `_download.download_and_extract('https://www.kaggle.com/api/v1/datasets/download/sabahesaraki/breast-ultrasound-images-dataset', ...)`,
  upload resources + PNG mask segmentation annotations (mirrors `02_busi_segmentation.ipynb`).
- `synapse_dataset.create()`: `_download.download_and_extract('https://www.kaggle.com/api/v1/datasets/download/dogcdt/synapse', ...)`,
  convert HDF5 → NIfTI per case, upload resources + volume segmentation annotations (mirrors
  `01_synapse_unetrpp.ipynb` / `02_synapse_nnunet.ipynb`).
- `fracatlas_dataset.create()`: query `https://api.figshare.com/v2/articles/22363012` for the
  download URL, fetch+extract via `_download.download_and_extract` (no auth needed, same as
  BCCD), upload the `Fractured`/`Non_fractured` folders with tags, then
  `api.annotations.create_image_classification(identifier='has_fracture', value='yes'/'no')`
  per resource (mirrors `notebooks/06_end_to_end/slice_based/01_fracatlas_classification.ipynb`).
  Print a heads-up before downloading (~1.2GB compressed, a few minutes) since it's
  noticeably bigger than the other three.

Downloaded/extracted data is cached under `configs.DATAMINT_DATA_DIR/examples/<dataset>/`
(i.e. `~/.datamint/examples/bccd/`, etc.) — a new sibling namespace next to the existing
`resources`/`annotations` cache namespaces, so repeated calls don't re-download, and so it's
naturally covered by `datamint config --list-local-data` / `--clean-local-data examples`
(small addition to `datamint_config.py`'s known-namespace list).

### Console output

Each `create()` prints a summary after upload finishes — not just the CLI wrapper, since
`datamint.examples.busi_dataset.create(...)` called directly (your literal one-liner) should
give the same feedback as `datamint example busi`:
```
FracAtlas Classification Example
  717 files uploaded, 717 annotated (100%)
  cached at ~/.datamint/examples/fracatlas/
  project: MyProject (https://app.datamint.io/projects/<id>)
```
- Dataset name: the module's display name (e.g. "FracAtlas Classification Example").
- File count: total resources uploaded this run.
- Annotated count: resources that got at least one annotation (box/mask/classification
  label), as `n (pct%)` — for these four datasets every uploaded file is annotated, so it'll
  read 100%, but computing it from the actual upload/annotation calls (not hardcoding "100%")
  keeps it honest if a dataset ever has partially-labeled data.
- Cache path: the `configs.DATAMINT_DATA_DIR/examples/<dataset>/` path data was
  downloaded/extracted to.
- Project name + link: reuses whatever `_common.get_or_create_project` already returns.

On the already-exists-skip path, print a shorter variant (name + "already exists, skipping"
+ project link) rather than the full stats block, matching `ProjectMR.create`'s existing
skip-with-warning behavior.

### Public API wiring

`datamint/__init__.py` uses `lazy_loader.attach` with an explicit `submodules` list that
currently omits `examples` entirely — meaning `datamint.examples.busi_dataset.create(...)`
(your literal proposed usage) doesn't actually resolve via the top-level package today; only
an explicit `import datamint.examples` does. Add `examples` to that submodules list so the
literal one-liner works.

### CLI

**New standalone subcommand** — `datamint/client_cmd_tools/datamint_example.py`:
```bash
datamint example bccd
datamint example busi --project MyBusiProject
datamint example fracatlas
```
Positional `dataset` (choices: `bccd`, `busi`, `synapse`, `fracatlas`), optional `--project`
(defaults to each module's own default name). Dispatches to the matching
`datamint.examples.<x>.create()`. Registered only in `datamint/__main__.py`'s `_COMMANDS`
dict (unified form) — **no new hyphenated `datamint-example` script**, since that legacy
pattern (established in DAT-986) is for backward compatibility with pre-existing commands,
not something to add for a brand-new one.

**Folded into `datamint init`** (`datamint_init.py`): after the existing task-type prompt
(`detection`/`segmentation`/`classification`), add a prompt: "Populate this project with
example data instead of your own?". If yes:
- `detection` → BCCD.
- `segmentation` → follow-up prompt "2D or 3D?" → BUSI or Synapse (task-type prompt doesn't
  currently distinguish these; needs the one extra prompt).
- `classification` → FracAtlas.

When example data is chosen, `01_upload_data.py` is generated as a short one-liner instead of
the generic "point this at your own data" template, e.g.:
```python
from datamint.examples import busi_dataset
busi_dataset.create("MyProject")
```

### Docs

Add a short section (in `docs/source/command_line_tools.rst` or a new page, whichever reads
better once drafted) documenting `datamint example` and the four datasets. Note FracAtlas's
larger download size (~1.2GB) up front. For `busi`/`synapse`, note that the download relies
on Kaggle's public anonymous download endpoint (no credentials needed today) and that if
Kaggle starts gating it, the fix is to configure Kaggle credentials
(`KAGGLE_USERNAME`/`KAGGLE_KEY` or `~/.kaggle/kaggle.json`) — not a day-one requirement, just
a documented fallback.

## Tests

Per repo convention (CLAUDE.md: no real network calls in tests), all downloads are mocked:
- `tests/test_examples_bccd.py`, `test_examples_busi.py`, `test_examples_synapse.py`,
  `test_examples_fracatlas.py`: mock `_download.download_and_extract` and the `Api` calls
  (`respx`/`httpx.MockTransport`, matching existing test conventions), assert resources +
  annotations get uploaded, assert the already-exists-skip-with-warning branch, and assert
  the printed summary (capsys) has the right file/annotated counts and cache path.
- `tests/test_datamint_example_cmd.py`: exercises the new CLI module the same way
  `tests/test_datamint_config.py` exercises `datamint config` (sys.argv patching + mocked
  `create()`).
- Extend `tests/test_datamint_init.py` (if it exists) or add coverage for the new
  example-data prompt branch in the init wizard.

## Verification
- Run `datamint example bccd --project TestBCCD` for real once (small, zero-auth dataset) to
  confirm the full download → parse → upload → project-creation path works end-to-end against
  a real Datamint server. Do the same for `fracatlas` at least once too (zero-auth, but budget
  for the larger ~1.2GB download).
- Run `datamint init` interactively, choose "use example data", confirm the generated
  `01_upload_data.py` is the short one-liner and actually runs.
- `pytest tests` — confirm new tests pass and nothing else regresses.
