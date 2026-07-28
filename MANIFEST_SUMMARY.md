# Artefact Manifest — Summary

Generated: 2026-07-28T15:45:32.094845+00:00
Root: `C:\Users\dyssa\Desktop\Research\RiskAware-ActiveLearning`

Full per-file record with SHA-256 checksums: `MANIFEST.csv`.

## By storage tier

| Tier | Files | Size | Meaning |
|---|---:|---:|---|
| `git` | 1,568 | 348.0 MB | Committed to this repository in full |
| `release` | 144 | 1,803.9 MB | Published as GitHub Release assets / Zenodo — checksummed here |
| `external` | 10,020 | 2,769.0 MB | Not redistributed; download separately and verify |

## By tree

| Tree | Tier | Files | Size |
|---|---|---:|---:|
| `(external) archive` | `external` | 10,020 | 2,769.0 MB |
| `(root)` | `git` | 11 | 0.1 MB |
| `Oracle_Simulated_Doctor` | `git` | 2 | 0.5 MB |
| `Seed Data` | `git` | 491 | 128.0 MB |
| `active_learning` | `git` | 9 | 0.1 MB |
| `analysis` | `git` | 120 | 42.0 MB |
| `colab` | `git` | 3 | 0.0 MB |
| `data` | `git` | 4 | 0.0 MB |
| `docs` | `git` | 1 | 0.0 MB |
| `escalation` | `git` | 3 | 0.0 MB |
| `evaluation` | `git` | 16 | 0.2 MB |
| `models` | `git` | 6 | 0.0 MB |
| `paper` | `git` | 75 | 26.6 MB |
| `results/checkpoints` | `release` | 144 | 1,803.9 MB |
| `results/experiments` | `git` | 816 | 150.4 MB |
| `risk_score` | `git` | 2 | 0.0 MB |
| `tools` | `git` | 3 | 0.0 MB |
| `uncertainty` | `git` | 6 | 0.0 MB |

**Total: 11,732 files, 4.81 GB.**

## Verifying

```bash
python -m tools.build_manifest --verify
```

Reports any file whose checksum no longer matches the manifest, any file that has appeared since it was generated, and any that have gone missing. Run it before submission, and after regenerating figures or tables.
