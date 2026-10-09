"""Which pool a task set belongs to.

Two values, `train` and `eval`. Anything not listed in data/tasksets/train.txt
is eval, which folds test_workload and test_hardware together. The held-out
platforms are identifiable by platform id, so the analysis can still separate
the two.
"""

import tasksets

SPLITS = tasksets.DIR

TRAIN = "train"
EVAL = "eval"

_train = None


def _load():
    """The train list, checked against what is on disk.

    A split is a list of names, so it goes stale when the dataset is
    regenerated larger, which would relabel training data as held out.
    """
    global _train
    if _train is None:
        f = SPLITS / "train.txt"
        if not f.exists():
            raise SystemExit(f"no {f}. run scripts/generator/split_dataset.py first")
        _train = set(f.read_text().split())
        on_disk = tasksets.count()
        if on_disk and len(_train) < on_disk / 4:
            raise SystemExit(
                f"train.txt lists {len(_train)} task sets but {on_disk} are on "
                "disk. The split predates the dataset: rerun "
                "scripts/generator/split_dataset.py")
    return _train


def set_of(taskset_id):
    return TRAIN if taskset_id in _load() else EVAL


def is_eval(taskset_id):
    return taskset_id not in _load()
