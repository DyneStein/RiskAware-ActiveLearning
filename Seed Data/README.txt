===============================================================================
Seed Data/  --  the 490 images every experiment starts from
===============================================================================

Active learning has to start somewhere. Every experiment in this study begins
with the same 490 labelled images -- 70 from each of the 7 diagnosis classes --
and then decides for itself which further images to ask a human about.

seed_metadata.csv is that starting set: 490 rows, each an image ID and its
diagnosis.


-------------------------------------------------------------------------------
WHY THE IMAGES THEMSELVES ARE NOT HERE
-------------------------------------------------------------------------------

The image files are part of HAM10000, which belongs to its authors and is
licensed for non-commercial use. This repository's own code is MIT licensed.
Shipping someone else's non-commercially-licensed images inside an MIT
repository would put the two licences in conflict, so we do not do it.

Nothing is lost. The list of IDs is the scientific fact -- it is what makes the
starting set reproducible. Give anyone this CSV and a copy of HAM10000 and they
reconstruct the exact same 490 images.

The code already handles this: it looks for images in the HAM10000 folders you
point DATA_ROOT at, so as long as you have downloaded the dataset, the seed
images are found automatically.


-------------------------------------------------------------------------------
HOW THIS SET WAS CHOSEN
-------------------------------------------------------------------------------

70 images per class, drawn once with a fixed random seed, then frozen. Equal
counts per class rather than proportional ones, so that the model sees a
reasonable number of the rare cancers from the very first round instead of
almost none.

create_seed_data.py in the repository root is the script that produced it.

Dataset citation:

    Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large
    collection of multi-source dermatoscopic images of common pigmented skin
    lesions. Scientific Data 5, 180161 (2018).
    https://doi.org/10.7910/DVN/DBW86T

===============================================================================
