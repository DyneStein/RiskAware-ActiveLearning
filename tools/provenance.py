"""
Provenance capture: record the exact environment every experiment ran in.

WHY THIS EXISTS
---------------
The first 24 experiments were run across many Google Colab sessions over
several weeks, and none of them recorded which PyTorch version, CUDA
version or GPU produced them. Colab has since moved on, so that
information is permanently unrecoverable for those runs. Results can shift
between library versions, and a runtime table is unverifiable without
knowing the hardware — so "these results are reproducible" was a claim we
could not actually support.

Every run from now on writes an `environment.json` next to its results, so
the claim is backed by a file rather than by memory. It costs about a
tenth of a second per experiment.

Nothing here affects training. If any individual probe fails (no GPU, git
not installed, a library missing), that field records the reason and the
run continues — provenance capture must never be the thing that crashes a
six-hour experiment.
"""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

# Libraries whose version can plausibly change a result, so each one is
# worth pinning in the record. Anything not installed is reported as such
# rather than omitted, so a gap is visible instead of silent.
_TRACKED_PACKAGES = [
    "torch", "torchvision", "numpy", "pandas",
    "sklearn", "scipy", "matplotlib", "PIL", "tqdm",
]


def _safe(fn, default="unavailable"):
    """Run a probe, and record why it failed rather than crashing the run."""
    try:
        return fn()
    except Exception as exc:                      # noqa: BLE001 — deliberate
        return f"{default} ({type(exc).__name__}: {exc})"


def _package_versions():
    versions = {}
    for name in _TRACKED_PACKAGES:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = "not installed"
    return versions


def _git_state():
    """
    The commit the code was at. Without this, 'we ran the code in the repo'
    does not identify which version of it.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _run(*args):
        return subprocess.check_output(
            args, cwd=repo_root, stderr=subprocess.DEVNULL, text=True
        ).strip()

    state = {"commit": _safe(lambda: _run("git", "rev-parse", "HEAD")),
             "branch": _safe(lambda: _run("git", "rev-parse", "--abbrev-ref", "HEAD"))}
    # A dirty tree means the running code does not match any commit, which
    # makes the commit hash misleading unless the discrepancy is recorded.
    status = _safe(lambda: _run("git", "status", "--porcelain"))
    state["uncommitted_changes"] = bool(status) if isinstance(status, str) else "unknown"
    return state


def _hardware():
    hw = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.replace("\n", " "),
    }
    try:
        import torch
        hw["cuda_available"] = torch.cuda.is_available()
        hw["cuda_version"] = torch.version.cuda
        hw["cudnn_version"] = _safe(lambda: torch.backends.cudnn.version())
        if torch.cuda.is_available():
            hw["gpu_name"] = torch.cuda.get_device_name(0)
            hw["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            hw["gpu_total_memory_gb"] = round(props.total_memory / 1024 ** 3, 2)
        else:
            hw["gpu_name"] = "none (CPU run)"
    except ImportError:
        hw["cuda_available"] = "torch not installed"
    return hw


def _config_snapshot():
    """
    The hyperparameters that were actually in force. The Colab setup cell
    rewrites config.py at the start of every session, so reading the file
    from GitHub later does not tell you what a given run used — only a
    snapshot taken at run time does.
    """
    try:
        import config
        keys = [
            "NUM_CLASSES", "CLASS_NAMES", "AL_ROUNDS", "QUERY_BUDGET_PER_ROUND",
            "SEED_PER_CLASS", "TEST_SPLIT_RATIO", "BATCH_SIZE", "LEARNING_RATE",
            "WEIGHT_DECAY", "EPOCHS_PER_ROUND", "IMAGE_SIZE", "NUM_WORKERS",
            "USE_DYNAMIC_CLASS_WEIGHTS", "UNCERTAINTY_THRESHOLD",
            "RISK_THRESHOLD", "MC_DROPOUT_PASSES", "RANDOM_SEED", "SPLIT_SEED",
        ]
        return {k: getattr(config, k, "absent") for k in keys}
    except ImportError as exc:
        return f"unavailable ({exc})"


def build_environment_record(**extra):
    """
    Assemble the full provenance record as a dict.

    Any keyword arguments are merged in under "run", which is how the
    caller records seed, experiment_id and anything else specific to this
    particular invocation.
    """
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": extra,
        "git": _git_state(),
        "hardware": _hardware(),
        "packages": _package_versions(),
        "config": _config_snapshot(),
    }


def write_environment_record(output_dir, **extra):
    """
    Write `environment.json` into an experiment's results folder.

    Called at the start of every run. On a resumed run this overwrites the
    previous record, which is the correct behaviour: the file should
    describe the session that most recently touched the experiment, and
    Colab hands out different GPUs on different days.

    Returns the record, or None if it could not be written — a provenance
    failure must never abort an experiment.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        record = build_environment_record(**extra)
        path = os.path.join(output_dir, "environment.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str)
        gpu = record["hardware"].get("gpu_name", "unknown")
        commit = str(record["git"].get("commit", "unknown"))[:8]
        print(f"  Environment recorded: {gpu} | commit {commit} -> {path}")
        return record
    except Exception as exc:                      # noqa: BLE001 — deliberate
        print(f"  WARNING: could not write environment record ({exc}). "
              f"Continuing — the experiment is unaffected.")
        return None


if __name__ == "__main__":
    print(json.dumps(build_environment_record(invoked="manually"),
                     indent=2, default=str))
