"""
Build MANIFEST.csv — one row for every artefact this project has produced.

WHY THIS EXISTS
---------------
"Every result is tracked" has to mean something checkable. Three questions
must have answers at any moment:

    1. What artefacts exist?
    2. Where does each one physically live?
    3. Is this file still the one that produced the number in the paper?

Git answers all three for anything small enough to commit. It cannot
answer them for the 1.8 GB of model checkpoints or the 2.8 GB dataset,
which do not belong in git history. So the manifest answers them instead:
every file gets a row with its size and SHA-256 checksum, whether or not
the bytes themselves are committed.

The manifest itself IS committed. That is the trick — git then versions
the *index*, so any change to any artefact anywhere shows up as a diff on
one small text file. Regenerate the manifest, and a modified figure or a
regenerated table appears immediately as a changed checksum.

STORAGE TIERS
-------------
    git       Committed in full. Code, docs, figures, tables, per-round
              results. Roughly 180 MB.
    release   Too big for git history; published as GitHub Release assets
              and mirrored to Zenodo at submission. The model checkpoints.
    external  Not ours to redistribute. The HAM10000 images.

Usage
-----
    python -m tools.build_manifest
    python -m tools.build_manifest --root /path/to/Research --out MANIFEST.csv
    python -m tools.build_manifest --verify      # re-check every checksum
"""

import argparse
import csv
import hashlib
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The dataset sits outside the repository — see docs/DATA_AND_ARTIFACTS.md.
DATA_ROOT = os.environ.get("DATA_ROOT",
                           os.path.join(os.path.dirname(REPO_ROOT), "archive"))

# (path relative to the root, storage tier, what it is)
# Order affects only the readability of the printed output.
TRACKED_TREES = [
    # --- source ---
    ("active_learning",       "git", "AL loop, escalation policies, baseline strategies"),
    ("data",                  "git", "dataset, transforms, pool manager"),
    ("models",                "git", "ResNet-50, DenseNet-169, EfficientNet-B4"),
    ("escalation",            "git", "uncertainty-only and dual-metric policies"),
    ("uncertainty",           "git", "entropy, margin, least-confidence, MC-dropout"),
    ("risk_score",            "git", "clinical risk score"),
    ("evaluation",            "git", "metrics, visualisation, and the rigor layer"),
    ("tools",                 "git", "provenance capture and this manifest"),
    ("Oracle_Simulated_Doctor", "git", "simulated annotator and pool metadata"),
    ("Seed Data",             "git", "the fixed 490-image starting labelled set"),
    # --- documentation ---
    ("docs",                  "git", "data, artefacts and limitations"),
    ("colab",                 "git", "Colab setup cell and runbooks"),
    # --- results ---
    ("results/experiments",   "git", "per-round metrics, pool predictions, per-experiment plots"),
    ("results/logs",          "git", "cross-experiment combined results"),
    ("results/tables",        "git", "cross-experiment comparison tables"),
    ("results/plots",         "git", "cross-experiment comparison plots"),
    ("analysis",              "git", "rigor-layer figures, tables and findings"),
    ("paper",                 "git", "paper writing kit: methods, maths, figures, results"),
    # --- too large for git history, tracked by checksum ---
    ("results/checkpoints",   "release", "trained model weights (GitHub Release / Zenodo)"),
]

# Files sitting loose in the repository root, which no tree above covers.
ROOT_FILE_SUFFIXES = (".py", ".md", ".txt", ".csv")

# Never worth a row: caches, editor droppings, and git's own internals.
SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", ".venv", "venv"}
SKIP_SUFFIXES = (".pyc", ".pyo")

# Hashing 2.8 GB of JPEGs takes minutes and tells us nothing we do not
# already know from the dataset's own published checksums, so external
# trees are inventoried (name, size, count) without being hashed.
HASH_TIERS = {"git", "release"}


def sha256_of(path, block=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def walk_tree(root, rel_tree, tier, description, do_hash):
    base = os.path.join(root, rel_tree)
    if not os.path.isdir(base):
        print(f"  (absent, skipping): {rel_tree}")
        return []

    rows = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in sorted(filenames):
            if name.endswith(SKIP_SUFFIXES):
                continue
            full = os.path.join(dirpath, name)
            try:
                size = os.path.getsize(full)
            except OSError:
                continue
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            rows.append({
                "path": rel,
                "tree": rel_tree,
                "tier": tier,
                "description": description,
                "size_bytes": size,
                "sha256": sha256_of(full) if do_hash else "",
                "extension": os.path.splitext(name)[1].lower(),
            })
    return rows


def root_files(root, do_hash):
    """Loose files in the repository root — main.py, README.md, LICENSE …"""
    rows = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if not os.path.isfile(full):
            continue
        if not (name.endswith(ROOT_FILE_SUFFIXES) or name == "LICENSE"):
            continue
        # The manifest cannot meaningfully contain its own checksum.
        if name in ("MANIFEST.csv", "MANIFEST_SUMMARY.md"):
            continue
        rows.append({
            "path": name, "tree": "(root)", "tier": "git",
            "description": "top-level entry point, configuration or document",
            "size_bytes": os.path.getsize(full),
            "sha256": sha256_of(full) if do_hash else "",
            "extension": os.path.splitext(name)[1].lower(),
        })
    return rows


def build(root, do_hash=True, data_root=None):
    all_rows = []

    rows = root_files(root, do_hash)
    print(f"Scanning (root files) [git] ...\n  {len(rows)} files, "
          f"{sum(r['size_bytes'] for r in rows) / 1024 ** 2:,.1f} MB")
    all_rows.extend(rows)

    for rel_tree, tier, description in TRACKED_TREES:
        should_hash = do_hash and tier in HASH_TIERS
        print(f"Scanning {rel_tree} [{tier}]"
              f"{'' if should_hash else '  (no checksums for this tier)'} ...")
        rows = walk_tree(root, rel_tree, tier, description, should_hash)
        print(f"  {len(rows)} files, "
              f"{sum(r['size_bytes'] for r in rows) / 1024 ** 2:,.1f} MB")
        all_rows.extend(rows)

    # The dataset lives outside the repository. Inventoried, not hashed:
    # 2.8 GB of JPEGs takes minutes to checksum and tells us nothing the
    # dataset's own published checksums do not already establish.
    data_root = data_root or DATA_ROOT
    if os.path.isdir(data_root):
        print(f"Scanning {data_root} [external]  (no checksums for this tier) ...")
        rows = walk_tree(os.path.dirname(data_root), os.path.basename(data_root),
                         "external",
                         "HAM10000 dataset (CC BY-NC-SA 4.0, not redistributed)",
                         False)
        for r in rows:
            r["tree"] = "(external) archive"
        print(f"  {len(rows)} files, "
              f"{sum(r['size_bytes'] for r in rows) / 1024 ** 2:,.1f} MB")
        all_rows.extend(rows)
    else:
        print(f"Dataset not found at {data_root} — skipping (set DATA_ROOT).")

    return all_rows


def write_manifest(rows, out_path):
    fields = ["path", "tree", "tier", "description", "size_bytes",
              "sha256", "extension"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["path"]))


def write_summary(rows, out_path, root):
    """
    A human-readable companion to the CSV. The CSV is the record; this is
    the page someone actually reads to see what the project contains.
    """
    by_tier = {}
    for r in rows:
        t = by_tier.setdefault(r["tier"], {"n": 0, "bytes": 0})
        t["n"] += 1
        t["bytes"] += r["size_bytes"]

    by_tree = {}
    for r in rows:
        t = by_tree.setdefault(r["tree"], {"n": 0, "bytes": 0, "tier": r["tier"]})
        t["n"] += 1
        t["bytes"] += r["size_bytes"]

    lines = [
        "# Artefact Manifest — Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Root: `{root}`",
        "",
        "Full per-file record with SHA-256 checksums: `MANIFEST.csv`.",
        "",
        "## By storage tier",
        "",
        "| Tier | Files | Size | Meaning |",
        "|---|---:|---:|---|",
    ]
    tier_meaning = {
        "git": "Committed to this repository in full",
        "release": "Published as GitHub Release assets / Zenodo — checksummed here",
        "external": "Not redistributed; download separately and verify",
    }
    for tier in ("git", "release", "external"):
        if tier not in by_tier:
            continue
        t = by_tier[tier]
        lines.append(f"| `{tier}` | {t['n']:,} | {t['bytes'] / 1024**2:,.1f} MB | "
                     f"{tier_meaning[tier]} |")

    lines += ["", "## By tree", "",
              "| Tree | Tier | Files | Size |", "|---|---|---:|---:|"]
    for tree, t in sorted(by_tree.items()):
        lines.append(f"| `{tree}` | `{t['tier']}` | {t['n']:,} | "
                     f"{t['bytes'] / 1024**2:,.1f} MB |")

    total = sum(r["size_bytes"] for r in rows)
    lines += [
        "",
        f"**Total: {len(rows):,} files, {total / 1024**3:,.2f} GB.**",
        "",
        "## Verifying",
        "",
        "```bash",
        "python -m tools.build_manifest --verify",
        "```",
        "",
        "Reports any file whose checksum no longer matches the manifest, any "
        "file that has appeared since it was generated, and any that have "
        "gone missing. Run it before submission, and after regenerating "
        "figures or tables.",
        "",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def verify(root, manifest_path):
    """Re-hash everything and report drift against the recorded manifest."""
    if not os.path.isfile(manifest_path):
        print(f"No manifest at {manifest_path} — build it first.")
        return 1

    with open(manifest_path, newline="", encoding="utf-8") as f:
        recorded = {r["path"]: r for r in csv.DictReader(f)}

    current = {r["path"]: r for r in build(root, do_hash=True)}

    changed, missing, added = [], [], []
    for path, row in recorded.items():
        if path not in current:
            missing.append(path)
        elif row["sha256"] and row["sha256"] != current[path]["sha256"]:
            changed.append(path)
    for path in current:
        if path not in recorded:
            added.append(path)

    print("\n" + "=" * 66)
    print(f"Recorded: {len(recorded):,}   Present now: {len(current):,}")
    print(f"  changed : {len(changed):,}")
    print(f"  missing : {len(missing):,}")
    print(f"  added   : {len(added):,}")
    for title, items in (("CHANGED", changed), ("MISSING", missing),
                         ("ADDED", added)):
        if items:
            print(f"\n{title}:")
            for p in items[:25]:
                print(f"  {p}")
            if len(items) > 25:
                print(f"  ... and {len(items) - 25:,} more")

    if not (changed or missing or added):
        print("\nEverything matches the manifest.")
        return 0
    print("\nDifferences found. If they are expected (you regenerated a "
          "figure, or finished a run), rebuild the manifest and commit it.")
    return 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=REPO_ROOT,
                    help="Repository root containing results/, analysis/ "
                         "and paper/.")
    ap.add_argument("--out", default=None,
                    help="Manifest CSV path (default: <repo>/MANIFEST.csv).")
    ap.add_argument("--no-hash", action="store_true",
                    help="Inventory sizes only — much faster, but the "
                         "result cannot detect a modified file.")
    ap.add_argument("--verify", action="store_true",
                    help="Re-check every checksum against the manifest.")
    args = ap.parse_args()

    out = args.out or os.path.join(REPO_ROOT, "MANIFEST.csv")

    if args.verify:
        sys.exit(verify(args.root, out))

    print(f"Root: {args.root}\n")
    rows = build(args.root, do_hash=not args.no_hash)
    write_manifest(rows, out)

    summary = os.path.splitext(out)[0] + "_SUMMARY.md"
    write_summary(rows, summary, args.root)

    total = sum(r["size_bytes"] for r in rows)
    print(f"\n{len(rows):,} files, {total / 1024**3:,.2f} GB")
    print(f"Manifest -> {out}")
    print(f"Summary  -> {summary}")


if __name__ == "__main__":
    main()
