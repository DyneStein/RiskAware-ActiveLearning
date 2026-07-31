# Artefact Manifest — Summary

Generated: 2026-07-31T13:50:42.230886+00:00
Root: `RiskAware-ActiveLearning/`

Full per-file record with SHA-256 checksums: `MANIFEST.csv`.

## By storage tier

| Tier | Files | Size | Meaning |
|---|---:|---:|---|
| `git` | 2,025 | 440.9 MB | Committed to this repository in full |
| `release` | 216 | 2,705.8 MB | Published as GitHub Release assets / Zenodo — checksummed here |
| `external` | 10,020 | 2,769.0 MB | Not redistributed; download separately and verify |

## By tree

| Tree | Tier | Files | Size |
|---|---|---:|---:|
| `(external) archive` | `external` | 10,020 | 2,769.0 MB |
| `(root)` | `git` | 10 | 0.1 MB |
| `Oracle_Simulated_Doctor` | `git` | 2 | 0.5 MB |
| `Seed Data` | `git` | 492 | 128.0 MB |
| `active_learning` | `git` | 9 | 0.1 MB |
| `analysis` | `git` | 173 | 57.3 MB |
| `colab` | `git` | 2 | 0.0 MB |
| `data` | `git` | 4 | 0.0 MB |
| `escalation` | `git` | 3 | 0.0 MB |
| `evaluation` | `git` | 19 | 0.2 MB |
| `models` | `git` | 6 | 0.0 MB |
| `paper` | `git` | 69 | 27.5 MB |
| `results/checkpoints` | `release` | 216 | 2,705.8 MB |
| `results/experiments` | `git` | 1,224 | 227.1 MB |
| `risk_score` | `git` | 2 | 0.0 MB |
| `tools` | `git` | 4 | 0.0 MB |
| `uncertainty` | `git` | 6 | 0.0 MB |

**Total: 12,261 files, 5.78 GB.**

## Verifying

```bash
python -m tools.build_manifest --verify
```

Reports any file whose checksum no longer matches the manifest, any file that has appeared since it was generated, and any that have gone missing. Run it before submission, and after regenerating figures or tables.
