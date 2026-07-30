# Artefact Manifest — Summary

Generated: 2026-07-30T11:38:11.055645+00:00
Root: `C:\Users\dyssa\Desktop\Research\RiskAware-ActiveLearning`

Full per-file record with SHA-256 checksums: `MANIFEST.csv`.

## By storage tier

| Tier | Files | Size | Meaning |
|---|---:|---:|---|
| `git` | 2,059 | 441.3 MB | Committed to this repository in full |
| `release` | 216 | 2,705.8 MB | Published as GitHub Release assets / Zenodo — checksummed here |
| `external` | 10,020 | 2,769.0 MB | Not redistributed; download separately and verify |

## By tree

| Tree | Tier | Files | Size |
|---|---|---:|---:|
| `(external) archive` | `external` | 10,020 | 2,769.0 MB |
| `(root)` | `git` | 11 | 0.1 MB |
| `Oracle_Simulated_Doctor` | `git` | 2 | 0.5 MB |
| `Seed Data` | `git` | 491 | 128.0 MB |
| `active_learning` | `git` | 9 | 0.1 MB |
| `analysis` | `git` | 182 | 57.4 MB |
| `colab` | `git` | 5 | 0.1 MB |
| `data` | `git` | 4 | 0.0 MB |
| `docs` | `git` | 2 | 0.0 MB |
| `escalation` | `git` | 3 | 0.0 MB |
| `evaluation` | `git` | 19 | 0.2 MB |
| `models` | `git` | 6 | 0.0 MB |
| `paper` | `git` | 89 | 27.7 MB |
| `results/checkpoints` | `release` | 216 | 2,705.8 MB |
| `results/experiments` | `git` | 1,224 | 227.1 MB |
| `risk_score` | `git` | 2 | 0.0 MB |
| `tools` | `git` | 4 | 0.0 MB |
| `uncertainty` | `git` | 6 | 0.0 MB |

**Total: 12,295 files, 5.78 GB.**

## Verifying

```bash
python -m tools.build_manifest --verify
```

Reports any file whose checksum no longer matches the manifest, any file that has appeared since it was generated, and any that have gone missing. Run it before submission, and after regenerating figures or tables.
