"""
Run the whole rigor layer in dependency order, with one command.

    python -m evaluation.rigor.run_all                 # everything available
    python -m evaluation.rigor.run_all --skip-dump     # reuse existing dumps
    python -m evaluation.rigor.run_all --with-robustness

Dependency order matters: the calibration, per-class AUC, Grad-CAM and
image-level statistics all consume the test-set prediction dumps, so
`dump_test_predictions` has to finish first. The steps that read only the
per-round CSVs (AL efficiency, decision-level ablation, runtime) have no
such dependency and run in seconds.

External validation on ISIC is deliberately NOT included: it needs a
multi-gigabyte download that has to be fetched deliberately, and its
overlap-exclusion step should be watched rather than buried in a batch run.
Run `evaluation.rigor.external_validation_isic` by hand.
"""

import argparse
import runpy
import sys
import time


def run_step(name, module, argv=()):
    print(f"\n{'='*78}\n>> {name}\n{'='*78}")
    old_argv = sys.argv[:]
    sys.argv = [module, *argv]
    t0 = time.perf_counter()
    try:
        runpy.run_module(module, run_name="__main__")
        ok = True
    except SystemExit as e:
        ok = (e.code in (0, None))
    except Exception as e:  # keep going; one broken step shouldn't sink the rest
        print(f"!! {name} failed: {type(e).__name__}: {e}")
        ok = False
    finally:
        sys.argv = old_argv
    print(f"-- {name}: {'ok' if ok else 'FAILED'} ({time.perf_counter()-t0:.1f}s)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-dump", action="store_true",
                    help="Reuse existing test-prediction dumps.")
    ap.add_argument("--with-robustness", action="store_true",
                    help="Also run the corrupted inference passes (slow).")
    ap.add_argument("--robustness-experiments", nargs="*", default=[
        "resnet50_entropy_dual_metric", "resnet50_entropy_uncertainty_only",
        "densenet169_entropy_dual_metric", "densenet169_entropy_uncertainty_only",
        "efficientnet_b4_entropy_dual_metric", "efficientnet_b4_entropy_uncertainty_only",
    ])
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    results = {}

    if not args.skip_dump:
        results["test-set prediction dump"] = run_step(
            "Test-set prediction dump (needed by calibration / AUC / Grad-CAM)",
            "evaluation.rigor.dump_test_predictions",
            ["--threads", str(args.threads)])

    if args.with_robustness:
        for corruption in ["gaussian_noise_0.05", "blur_1.5", "brightness_0.7",
                           "contrast_0.7", "jpeg_q30"]:
            results[f"robustness dump: {corruption}"] = run_step(
                f"Robustness dump — {corruption}",
                "evaluation.rigor.dump_test_predictions",
                ["--corruption", corruption, "--threads", str(args.threads),
                 "--only", *args.robustness_experiments])

    for name, module, argv in [
        ("AL efficiency (accuracy vs labelled samples)",
         "evaluation.rigor.al_efficiency", []),
        ("Decision-level ablation + risk-threshold sweep",
         "evaluation.rigor.ablation_posthoc", []),
        ("Calibration (ECE, Brier, reliability diagrams)",
         "evaluation.rigor.calibration", []),
        ("Per-class AUC with bootstrap CIs",
         "evaluation.rigor.per_class_auc", []),
        ("Statistical significance (p-values, CIs, effect sizes)",
         "evaluation.rigor.statistics", []),
        ("Runtime analysis", "evaluation.rigor.runtime", ["--benchmark"]),
        ("Explainability (Grad-CAM)", "evaluation.rigor.gradcam", []),
        ("Robustness analysis", "evaluation.rigor.robustness", []),
    ]:
        results[name] = run_step(name, module, argv)

    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    for name, ok in results.items():
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}")
    failed = [n for n, ok in results.items() if not ok]
    print(f"\n{len(results)-len(failed)}/{len(results)} steps succeeded.")
    if failed:
        print("Failed steps (most often: a prerequisite dump is missing):")
        for n in failed:
            print(f"  - {n}")


if __name__ == "__main__":
    main()
