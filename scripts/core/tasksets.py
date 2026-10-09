"""The task set store.

    data/tasksets/tasksets.jsonl    one task set per line
    data/tasksets/index.json        id -> byte offset, so one lookup is a seek

Everything that reads task sets goes through here. Use `stream` to visit many
and `load` to fetch one by id. The index is written by the generator and
rebuilt by scanning if it is missing or stale.

Ids look like `taskset_<platform>_<coupling>_<load>_<instance>`, which is
enough to filter by platform or by cell without parsing any JSON.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DIR = ROOT / "data" / "tasksets"
STORE = DIR / "tasksets.jsonl"
INDEX = DIR / "index.json"

_index = None


def exists():
    return STORE.exists()


class Writer:
    """Append task sets and remember where each one landed.

    Used as a context manager. The index is written on close, so an interrupted
    generation leaves a readable store that rebuilds its index on first use.
    """

    def __init__(self, fresh=False):
        DIR.mkdir(parents=True, exist_ok=True)
        self.offsets = {}
        if fresh and STORE.exists():
            STORE.unlink()
        if INDEX.exists():
            INDEX.unlink()
        if STORE.exists():                     # appending, keep what is there
            self.offsets = _scan()
        self.fh = STORE.open("ab")

    def write(self, record):
        line = (json.dumps(record) + "\n").encode()
        self.offsets[record["taskset_id"]] = [self.fh.tell(), len(line)]
        self.fh.write(line)

    def close(self):
        self.fh.close()
        INDEX.write_text(json.dumps(self.offsets))
        global _index
        _index = self.offsets

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _scan():
    """Byte offsets by reading the store once. The fallback when the index is
    missing or stale, which makes the index a cache rather than a source."""
    out, pos = {}, 0
    if not STORE.exists():
        return out
    with STORE.open("rb") as fh:
        for line in fh:
            if line.strip():
                out[json.loads(line)["taskset_id"]] = [pos, len(line)]
            pos += len(line)
    return out


def index():
    global _index
    if _index is None:
        if INDEX.exists():
            _index = json.loads(INDEX.read_text())
            # a stale index would return the wrong task set, so check that the
            # last offset still lines up with the file size before trusting it
            if STORE.exists() and _index:
                last = max(_index.values(), key=lambda v: v[0])
                if last[0] + last[1] != STORE.stat().st_size:
                    _index = _scan()
        else:
            _index = _scan()
    return _index


def ids():
    """Every id, sorted, so every caller sees the same order."""
    return sorted(index())


def count():
    return len(index())


def load(taskset_id):
    """One task set by id. A seek, not a scan."""
    taskset_id = str(taskset_id)
    if not taskset_id.startswith("taskset_"):
        taskset_id = f"taskset_{taskset_id}"
    idx = index()
    if taskset_id not in idx:
        if not exists():
            raise SystemExit(
                f"no task set store at {STORE}. run scripts/generator/gen_dataset.py first")
        raise SystemExit(f"unknown task set {taskset_id!r}")
    off, n = idx[taskset_id]
    with STORE.open("rb") as fh:
        fh.seek(off)
        return json.loads(fh.read(n))


def parse_id(taskset_id):
    """taskset_7_2_4_31 -> (7, 2, 4, 31), without opening the record."""
    p = str(taskset_id).removeprefix("taskset_").split("_")
    if len(p) != 4:
        raise SystemExit(
            f"task set id {taskset_id!r} should look like 7_2_4_31 "
            "(platform_coupling_load_instance)")
    return tuple(int(x) for x in p)


def select(platforms=None, per_cell=None, keep=None):
    """The ids a run should visit, filtered without reading the store.

    platforms   only these platform numbers
    per_cell    only the first K instances of every (platform, coupling, load)
    keep        a predicate on the id, for the train/eval split
    """
    out = []
    for tid in ids():
        p, _, _, n = parse_id(tid)
        if platforms and p not in set(platforms):
            continue
        if per_cell and n > per_cell:
            continue
        if keep and not keep(tid):
            continue
        out.append(tid)
    return out


def stream(taskset_ids=None):
    """Yield (id, record) in file order, reading the store once.

    One open and one sequential pass, where calling `load` in a loop would
    seek per record. `taskset_ids` filters without changing that.
    """
    if not STORE.exists():
        raise SystemExit(
            f"no task set store at {STORE}. run scripts/generator/gen_dataset.py first")
    want = set(taskset_ids) if taskset_ids is not None else None
    with STORE.open("rb") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if want is None or rec["taskset_id"] in want:
                yield rec["taskset_id"], rec
