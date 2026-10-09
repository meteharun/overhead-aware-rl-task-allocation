"""Short names instead of paths.

    platform("3") -> data/platforms/platform_3.json

A full path is passed through unchanged. Task sets live in one store and are
fetched by id through tasksets.load.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
OUT = ROOT / "out"


def platform(arg):
    if str(arg).endswith(".json"):
        return Path(arg)
    return DATA / "platforms" / f"platform_{str(arg).removeprefix('platform_')}.json"


def out(kind, name):
    """kind is tasksets, platforms or allocations."""
    d = OUT / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.png"
