# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`pfs_design_tool` generates fiber allocation design files (`pfsDesign`) for the Subaru Prime Focus Spectrograph (PFS). It queries science targets and calibrators from PostgreSQL databases, runs netflow optimization to assign fibers to cobra positioners, and writes the result as a FITS file consumed by the PFS instrument control system.

## Package Management

This project uses `uv` for dependency management.

```bash
# Install all dependencies
uv sync

# Install with dev extras
uv sync --extra dev
```

All Subaru-PFS dependencies are installed from GitHub (see `pyproject.toml`). The `pfs-utils` dependency is pinned to a fork branch (`u/monodera/symlink-data`) via `[tool.uv.override-dependencies]`.

## Linting and Formatting

```bash
# Format
uv run black src/

# Lint
uv run ruff check src/

# Fix lint issues
uv run ruff check --fix src/
```

Config is in `pyproject.toml`: line-length=88, Python 3.12 target. Ruff ignores F401, F841, E501, and naming convention rules (N802-N816) to accommodate PFS code conventions.

## Running the Main Scripts

The primary entry point is `subaru_fiber_allocation.py`, which requires a TOML config file:

```bash
uv run python -m pfs_design_tool.subaru_fiber_allocation \
    --conf config.toml \
    --ra 150.0 --dec 2.0 --pa -90.0 \
    --observation_time "2024-01-01T10:00:00Z" \
    --design_dir ./output \
    --n_fluxstd 50 --n_sky 240 \
    --arms br
```

To reconfigure fibers for an existing design:

```bash
uv run python -m pfs_design_tool.reconfigure_fibers 0x<designId> \
    --conf config.toml --observation_time "2024-01-01T10:00:00Z"

# From PPP output CSV:
uv run python -m pfs_design_tool.reconfigure_fibers_ppp input.csv \
    --conf config.toml
```

## Architecture

### Data Flow

1. **`pointing_utils/dbutils.py`** — Fetches targets from two PostgreSQL databases:
   - `targetdb` (via the `targetdb` Python package): science targets, flux standards, sky objects
   - Subaru Gaia DB (via `psycopg2`): guide stars and filler targets
   - Connection parameters come from `[targetdb][db]` and `[gaiadb]` sections of the TOML config.

2. **`pointing_utils/nfutils.py`** — Runs the netflow fiber assignment optimizer (`ets_fiber_assigner.netflow`). Loads the cobra instrument model from `pfs_instdata` (path via `PFS_INSTDATA_DIR` env var or `--pfs_instdata_dir` arg) using `ics_cobraCharmer`/`ics_cobraOps`. Returns fiber visibility (`vis`), target positions (`tp`), telescope pointing (`tel`), target list (`tgt`), and the `Bench` object.

3. **`pointing_utils/designutils.py`** — Assembles the `pfsDesign` FITS object from the netflow output using `pfs.datamodel.makePfsDesign`. Also fetches guide stars from Gaia DB and attaches them as `pfsDesign.guideStars`.

4. **`utils.py`** — Coordinate transform utilities (`pfi2sky`, `sky2pfi`, etc. wrapping `pfs.utils.coordinates.CoordTransp`), fiber-to-spectrograph module mapping, and the `CheckDesign` class for inspecting/plotting existing `pfsDesign` files.

### TOML Config Structure

The config file (e.g., `config.toml`) must have these sections:

```toml
[targetdb.db]      # psycopg2 connection kwargs for targetdb
[gaiadb]           # psycopg2 connection kwargs for Gaia DB
[netflow]
use_gurobi = true  # whether to use Gurobi optimizer
[gurobi]           # Gurobi license/settings (only needed if use_gurobi=true)
```

### Key Environment Variables

- `PFS_INSTDATA_DIR` — Path to the `pfs_instdata` repository (instrument calibration data). Set automatically by `nfutils.getBench()` from `--pfs_instdata_dir` argument.

### Source Layout

```
src/pfs_design_tool/
    subaru_fiber_allocation.py   # main script: full design from scratch
    reconfigure_fibers.py        # reconfigure from existing pfsDesign
    reconfigure_fibers_ppp.py    # reconfigure from PPP CSV output
    utils.py                     # coordinate utils, CheckDesign class
    pointing_utils/
        dbutils.py               # DB queries (targetdb, gaiaDB)
        nfutils.py               # netflow + cobra bench setup
        designutils.py           # pfsDesign assembly + guide stars
```

The `legacy/` directory contains old standalone versions of the same files (no package structure) kept for historical reference — do not modify them.

### External PFS Packages (all from Subaru-PFS GitHub org)

- `ets_fiberalloc` — netflow optimization engine
- `ets_shuffle` — guide star geometry utilities (`guidecam_geometry`, `flag_close_pairs`)
- `ets_target_database` — `targetdb` Python ORM/client
- `ics_cobraCharmer` — cobra positioner model (`PFIDesign`, `CobraCoach`)
- `ics_cobraOps` — cobra operations (`Bench`, `BlackDotsCalibrationProduct`, `CollisionSimulator2`)
- `pfs_utils` — fiber ID mapping (`FiberIds`), coordinate transforms
- `pfs.datamodel` — `PfsDesign` data model and FITS I/O

### pfs_utils Path Resolution

`utils.get_pfs_utils_path()` auto-detects the fiber data directory. If `eups` is available, it defers to `PFS_UTILS_DIR`. Otherwise it locates the `data/fiberids/` directory relative to the installed `pfs.utils` package path.

### Logging

Uses `loguru` throughout (migrated from `logzero`). Import as `from loguru import logger`.
