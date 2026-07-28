"""
Cost-matched budgets: making the baseline comparison fair.

THE PROBLEM THIS SOLVES
-----------------------
BADGE, CoreSet, CLUE and VAAL are *acquisition strategies*. They answer:

    "Given a budget of k labels this round, which k images?"

Our dual-metric policy is an *escalation policy*. It answers a different
question:

    "Which images are unsafe to auto-accept?"

and it decides its own budget as a consequence — the risk route is
uncapped by design, precisely so a dangerous case is never skipped because
a quota was full. Over 15 rounds the two policies therefore spend
different numbers of labels (dual-metric spends about 382 more than
uncertainty-only).

That difference makes a naive comparison meaningless in both directions.
Give the baselines a fixed 150 per round and we beat them partly by
spending more. Give ourselves 150 per round and we have disabled the
uncapped risk route, which is the contribution being tested.

THE FIX
-------
Match the cost round by round. In round t, hand each baseline exactly the
number of labels the dual-metric policy spent in round t on that same
backbone. Both then consume an identical annotation budget, on an
identical schedule, and any difference in outcome is attributable to
*which* images were chosen rather than how many.

Those per-round counts are not estimated — they are read from the
completed experiment's own `results.csv`, column `queries_this_round`.

WHAT TO REPORT AFTERWARDS
-------------------------
Two axes, because the baselines and our method optimise different things:

  * Learning — accuracy and F1 per label spent. This is the axis the
    baselines are designed to win, and BADGE may well win it.
  * Safety — unsafe auto-accepts and missed cancers. For a baseline, an
    image it did not select is an image that was auto-accepted, so the
    same metric is computable and means the same thing. None of the four
    baselines has any notion of clinical consequence, which is the point.
"""

import os

import pandas as pd

from config import EXPERIMENTS_DIR, QUERY_BUDGET_PER_ROUND


def reference_experiment_id(model_name, uncertainty_method="entropy",
                            policy="dual_metric"):
    """
    The experiment whose spending the baselines are matched against.

    Defaults to `<model>_entropy_dual_metric` — entropy is the cheapest
    uncertainty method and the one used for every other headline analysis
    (robustness, calibration, Grad-CAM), so matching against it keeps the
    baseline comparison on the same footing as everything else in the
    paper.
    """
    return f"{model_name}_{uncertainty_method}_{policy}"


def load_matched_budgets(model_name, num_rounds, uncertainty_method="entropy",
                         policy="dual_metric", fallback=QUERY_BUDGET_PER_ROUND):
    """
    Per-round label counts to hand a baseline, taken from a finished run.

    Parameters
    ----------
    model_name : str
        Backbone — the match is per-backbone, since dual-metric escalates
        different amounts under different architectures.
    num_rounds : int
        How many rounds the baseline will run.

    Returns
    -------
    (budgets, source) : (list of int, str)
        `budgets[t-1]` is the budget for round t. `source` records where
        the numbers came from, and is written into the results so the
        matching is auditable rather than assumed.

    Raises
    ------
    FileNotFoundError
        If the reference run has not completed. This is deliberate: a
        silent fallback to a flat 150/round would produce a comparison
        that looks cost-matched in the paper but is not.
    """
    exp_id = reference_experiment_id(model_name, uncertainty_method, policy)
    path = os.path.join(EXPERIMENTS_DIR, exp_id, "results.csv")

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Cost-matching needs the reference run '{exp_id}' to have "
            f"finished, but {path} does not exist.\n"
            f"Run it first:\n"
            f"    python main.py --model {model_name} "
            f"--uncertainty {uncertainty_method} --policy {policy} --resume\n"
            f"Or pass --query-budget N to run this baseline at a fixed "
            f"budget instead — but then it is NOT cost-matched, and the "
            f"paper must not describe it as such."
        )

    df = pd.read_csv(path)
    if "queries_this_round" not in df.columns:
        raise ValueError(
            f"{path} has no 'queries_this_round' column; cannot cost-match."
        )

    budgets = [int(v) for v in df.sort_values("round")["queries_this_round"]]

    if len(budgets) < num_rounds:
        # The reference run is shorter than the baseline we are about to
        # run. Extending with its final round's spend is the least
        # surprising choice, but it is a real approximation, so it is
        # announced loudly rather than hidden.
        shortfall = num_rounds - len(budgets)
        pad = budgets[-1] if budgets else fallback
        print(f"  WARNING: reference '{exp_id}' has only {len(budgets)} rounds "
              f"but {num_rounds} are needed. Padding {shortfall} round(s) with "
              f"{pad} (its final round's spend). This is an approximation — "
              f"note it if these rounds appear in the paper.")
        budgets = budgets + [pad] * shortfall

    source = (f"cost-matched per round to {exp_id} "
              f"(total {sum(budgets[:num_rounds])} labels over {num_rounds} rounds)")
    return budgets[:num_rounds], source
