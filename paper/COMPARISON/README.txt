===============================================================================
COMPARISON AGAINST PUBLISHED METHODS
Risk-Aware Active Learning for Skin Lesion Diagnosis  --  HAM10000
===============================================================================

This folder answers one question: how does our method compare to what already
exists in the literature?


-------------------------------------------------------------------------------
WHAT WE COMPARED AGAINST
-------------------------------------------------------------------------------

Four published methods, each run on all three networks, 15 rounds each --
12 new experiments on top of the original 24.

    CoreSet    Sener & Savarese, ICLR 2018
    BADGE      Ash et al., ICLR 2020
    CLUE       Prabhu et al., ICCV 2021
    VAAL       Sinha et al., ICCV 2019

Plus the standard uncertainty-only rule, which was already in the study.


-------------------------------------------------------------------------------
WHY THE COMPARISON IS FAIR
-------------------------------------------------------------------------------

There is a trap here worth being explicit about.

The four published methods are given a budget: "pick the best 300 images". Our
method chooses its own budget: "escalate whatever looks unsafe". Compared
naively, whichever one asks for more labels looks better -- and the result
would be measuring budget, not quality of choice.

So we did not do that. Each published method was given EXACTLY the number of
labels our method spent, in the same round, on the same network. Same budget,
same starting point, same test set. The only thing that differs is which images
get chosen.

Totals: 4,678 labels on ResNet-50, 4,773 on DenseNet-169, 3,976 on
EfficientNet-B4.

This was verified to be exact for all 12 runs, and the script that builds these
tables re-checks it every time. If a single run were off by one label it would
refuse to produce any output at all.

All 36 experiments were also confirmed to share one identical 1,905-image test
set -- we hashed every run's split file and got exactly one distinct value.
That is what makes it valid to compare the models image by image.


-------------------------------------------------------------------------------
RESULT 1 -- SAFETY: WE WIN EVERY COMPARISON
-------------------------------------------------------------------------------

Dangerous cases waved through with no human review, added up over 15 rounds.
Lower is better.

    Network            Ours    CoreSet   BADGE    CLUE     VAAL   Standard
    ---------------- -------  --------  ------  ------  -------  ---------
    ResNet-50          4,945     9,575   8,194   8,481   12,628      9,327
    DenseNet-169       4,495     8,543   6,947   7,397   11,308      8,275
    EfficientNet-B4    7,362     9,893  10,963  11,873   12,745     12,346

15 comparisons out of 15 favour our method, by between 26% and 61%.
No exceptions.

    figures/fig1_safety_headline.png
    tables/02_safety_scoreboard.csv


-------------------------------------------------------------------------------
RESULT 2 -- ACCURACY: NOTHING WAS SACRIFICED
-------------------------------------------------------------------------------

Compared image by image across the 1,905 shared test images, and adjusted for
having made 15 comparisons at once.

    versus VAAL, all 3 networks           +2.6 to +5.4 points   real difference
    versus uncertainty-only, B4           +2.4 points           real difference
    versus CoreSet, BADGE, CLUE           -0.7 to +1.5 points   no difference
    versus uncertainty-only, other two    +0.2 to +0.5 points   no difference

The eleven "no difference" rows are the intended result, not a shortfall.

The claim being made is "safety gained at no accuracy cost". Narrow ranges
sitting on top of zero are exactly the evidence for the words "no cost".
CoreSet, BADGE and CLUE exist specifically to extract the most learning per
label; matching them on the same budget while roughly halving dangerous
auto-accepts is the substantive finding.

    figures/fig2_safety_accuracy_tradeoff.png
    figures/fig4_accuracy_significance.png
    tables/03_significance_image_level.csv


-------------------------------------------------------------------------------
RESULT 3 -- CANCER DETECTION ON NEW PATIENTS: AN HONEST NULL
-------------------------------------------------------------------------------

Of 349 cancerous test images, our method detects 264, 272 or 280 depending on
the network. Restricted to the 209 melanomas, it detects 143, 152 or 157.

The direction favours us in 13 of 15 comparisons. But only 2 of those 15 hold
up once corrected for multiple testing, and both are against VAAL. Against
CoreSet and BADGE the gap is one or two cases -- indistinguishable from chance.

We report this as a limitation, not as a result. With 209 melanomas and the two
models typically disagreeing on only 15 to 30 of them, the study simply is not
large enough to detect a difference of this size.


-------------------------------------------------------------------------------
THE DISTINCTION THAT GOVERNS ALL THE WORDING
-------------------------------------------------------------------------------

Result 1 is measured on the unlabelled pool. Result 3 is measured on the
held-out test set. These are different questions and are kept separate.

Measuring safety on the pool is the right call, not a convenient one: this is
an intervention on a DECISION RULE, not on the model's weights, so the place to
ask "did fewer dangerous cases get waved through?" is the set of cases that
were waved through.

What has not been shown is that the extra labels make the final model safer on
unseen patients. The reason is identified rather than hidden: the two checks
share one underlying network, so they tend to fail on the same hard images.


-------------------------------------------------------------------------------
A NOTE ON STATISTICS
-------------------------------------------------------------------------------

A test run across the three networks has only three data points. With three
data points the smallest possible p-value is 0.250, no matter how large the
effect -- significance is arithmetically unreachable. Reporting such a test
would look like a failure when it is really a limit of the arithmetic.

So every p-value here comes from image-level tests across 1,905 images instead,
which the frozen shared test set makes valid.

tables/04_direction_across_backbones.csv deliberately reports win counts and no
p-values at all, so that 0.250 cannot be mistaken for a failed test.


-------------------------------------------------------------------------------
WHAT IS IN THIS FOLDER
-------------------------------------------------------------------------------

  figures/fig1_safety_headline.png          the headline safety result
  figures/fig2_safety_accuracy_tradeoff.png safety against accuracy
  figures/fig3_learning_curves.png          accuracy against labels spent
  figures/fig4_accuracy_significance.png    accuracy differences with ranges

  tables/01_main_comparison.csv             all methods, all networks
  tables/02_safety_scoreboard.csv           safety reductions, absolute and %
  tables/03_significance_image_level.csv    the statistical tests
  tables/04_direction_across_backbones.csv  consistency (win counts only)
  tables/05_learning_curves_per_round.csv   the data behind figure 3
  tables/06_run_provenance.csv              GPU, versions, commit and seed,
                                            recorded per run
  tables/main_comparison.tex                the main table as LaTeX
  tables/safety_reduction.tex               the safety table as LaTeX

The two LaTeX tables are generated from the same CSVs the figures are built
from, so a manuscript table cannot drift away from the underlying data.


-------------------------------------------------------------------------------
REBUILD IT
-------------------------------------------------------------------------------

From the repository root, no GPU and no dataset needed:

    python -m evaluation.rigor.baseline_comparison
    python -m tools.build_comparison_package

===============================================================================
