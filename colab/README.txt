===============================================================================
colab/  --  running this on a free cloud GPU
===============================================================================

Every experiment in this study was run on Google Colab's free T4 GPU. No local
graphics card was used at any point.

setup_cell.py is the first thing to run in a new Colab notebook. It mounts your
Google Drive, installs anything missing, and points the code at your copy of
the dataset.


-------------------------------------------------------------------------------
THE SHORT VERSION
-------------------------------------------------------------------------------

  1. Put this repository and the HAM10000 dataset in your Google Drive.

  2. New Colab notebook -> Runtime -> Change runtime type -> T4 GPU.

  3. First cell: paste the contents of setup_cell.py and run it.

  4. Then run whatever you want, for example:

         !python main.py --run-all --resume

  5. Colab will disconnect after a few hours. That is expected and harmless.
     Reconnect, re-run the setup cell, and run the same command again. The
     --resume flag makes it continue from the last completed round instead of
     starting over.

Full command reference: HOW_TO_RUN.txt in the repository root.


-------------------------------------------------------------------------------
ROUGH TIMINGS ON A FREE T4
-------------------------------------------------------------------------------

  one experiment, 15 rounds                    about 3 hours
  the same but using mc_dropout                about 7 hours
  all 36 experiments                           about 40 hours

Free Colab gives you a few hours at a time, so the full set takes a couple of
weeks of occasional sessions. Everything is saved after every round, so no
session is ever wasted.

===============================================================================
