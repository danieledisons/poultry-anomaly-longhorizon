"""
config.py — single source of truth for all filesystem paths.

WHY THIS EXISTS
---------------
The codebase runs on two machines (laptop + SSH server) whose folders live at
different absolute paths. Instead of editing hardcoded paths in every script,
each machine keeps its own `.env` and everything else derives from ONE root.

USAGE
-----
1. Copy the right template on each machine and symlink it to `.env`:
       ln -sf .env.laptop .env      # on the laptop
       ln -sf .env.server .env      # on the server
   (or just `cp`). `.env` is the only machine-specific file.

2. In any script:
       from config import PROJECT_ROOT, DATA_DIR, FEATURES_DIR, RESULTS_DIR
   and build paths from those — never hardcode an absolute path again.

Resolution order for every setting: real OS environment variable  >  value in
`.env`  >  built-in default derived from PROJECT_ROOT. So you can also override
any path inline, e.g.  RESULTS_DIR=/tmp/run7 python run_pipeline.py
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Tiny .env loader (no external dependency on python-dotenv).
# Parses KEY=VALUE lines; ignores blanks and # comments; strips quotes.
# --------------------------------------------------------------------------
_REPO_DIR = Path(__file__).resolve().parent


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip().strip('"').strip("'")
    return values


_ENV = _load_env_file(_REPO_DIR / ".env")


def _get(key: str, default: str) -> str:
    """OS environment  >  .env file  >  default."""
    return os.environ.get(key, _ENV.get(key, default))


def _expand(p: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(p))).resolve()


# --------------------------------------------------------------------------
# The one setting every machine must provide: PROJECT_ROOT.
# Everything below is derived from it unless individually overridden.
# --------------------------------------------------------------------------
PROJECT_ROOT: Path = _expand(_get("PROJECT_ROOT", str(_REPO_DIR)))

# Raw + processed data live under the project root by convention, but each can
# be relocated independently (e.g. raw video on an external drive / server disk).
DATA_DIR:     Path = _expand(_get("DATA_DIR",     str(PROJECT_ROOT / "data")))
RAW_DIR:      Path = _expand(_get("RAW_DIR",      str(DATA_DIR / "raw")))
FEATURES_DIR: Path = _expand(_get("FEATURES_DIR", str(PROJECT_ROOT / "features")))
RESULTS_DIR:  Path = _expand(_get("RESULTS_DIR",  str(PROJECT_ROOT / "results")))
FIGURES_DIR:  Path = _expand(_get("FIGURES_DIR",  str(PROJECT_ROOT / "figures")))

# Named feature files used by the multimodal pipeline. Filenames are stable;
# only the directory changes per machine, so these ride on FEATURES_DIR.
ENV_FEATURES_ROOM2 = FEATURES_DIR / "env_features_Room2.csv"


def ensure_dirs() -> None:
    """Create the writable output directories if they don't exist."""
    for d in (RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # `python config.py` prints the resolved paths — quick way to confirm a
    # machine's .env is wired correctly.
    for name in ("PROJECT_ROOT", "DATA_DIR", "RAW_DIR", "FEATURES_DIR",
                 "RESULTS_DIR", "FIGURES_DIR"):
        print(f"{name:14s} = {globals()[name]}")
