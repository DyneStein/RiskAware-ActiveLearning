# Proposed Methodology Upgrades (Pool-Based with Calibrated Fixed Thresholds)

This document outlines the final approved methodology upgrades for the Risk-Aware Active Learning framework, balancing mathematical purity, baseline stability, and real-time clinical simulation.

## 1. Configuration Toggle & CLI Argument
- **Files Modified:** `config.py` and `main.py`
- Add `USE_DYNAMIC_CLASS_WEIGHTS = False` as a toggle switch in `config.py`.
- Expose this toggle as a command-line argument in `main.py` (e.g., `--use-dynamic-weights`), so experiments can easily be run with or without it (Ablation Study).

## 2. Scale-Invariant Uncertainty Metrics
- **Files Modified:** `uncertainty/entropy.py`, `uncertainty/mc_dropout.py`, `uncertainty/margin.py`, `uncertainty/least_confidence.py`
- Remove all mathematical normalization (e.g., dividing by max values). Let the metrics output their raw, pure mathematical values to preserve their unique information, perfectly resolving the professor's scale-invariance critique.

## 3. Seed-Calibrated Thresholds
- **Files Modified:** `active_learning/al_loop.py`
- Instead of hardcoding generic thresholds like `0.5`, the script will run the 490 Seed Images through the model *before* the active learning loops begin.
- It will find the 90th percentile score for the chosen uncertainty method and the 90th percentile for risk.
- These dynamically discovered numbers (e.g., Entropy > 1.42, Risk > 0.28) will be locked in as the **Fixed Deployment Thresholds** for the rest of the experiment, perfectly simulating calibrating a real-time system for clinical deployment.

## 4. Dynamic Weights Implementation & Pool-Based AL
- **Files Modified:** `active_learning/al_loop.py` and `models/base_model.py`
- The framework remains **Pool-Based**. Auto-accepted images are NOT discarded; they remain in the Unlabeled Pool to be re-evaluated in the next round by the newly-smarter model.
- Because we are using Fixed Thresholds, the number of escalated images will naturally shrink each round as the model gets smarter and its uncertainty scores drop below the fixed line.
- If `USE_DYNAMIC_CLASS_WEIGHTS = True`, calculate the inverse frequency of the current labeled pool and pass it to the loss function before `model.train()` is called.

## 5. Documentation & Notebook Updates
- **Files Modified:** `README.md`, `researcher.md` (or equivalent), and `notebooks/run_experiment.ipynb`
- Update all documentation to reflect the final methodology.
- Document the new `--use-dynamic-weights` argument so anyone using the codebase or the Colab notebook knows how to run the Ablation Study.
