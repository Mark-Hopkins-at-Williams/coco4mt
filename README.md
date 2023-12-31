# coco4mt

### To install this code, run
    git clone https://github.com/Mark-Hopkins-at-Williams/coco4mt
    pip install -r requirements.txt

*** NOTE: the SimCSE dependency seems not to work with Python 3.10 (however 3.9 should be fine)

### To select lines for the SimCSE similarity (2+ neighbors)
    python fill_budget.py -p 0.2 -u sentence -r uniform

### Use instructions:
To get files containing sentence indices for the longest and random baselinesfrom the coco4mt English training split, use fill_budget.py. Enter a command of the form

    python fill_budget.py -p [BUDGET PERCENTAGE (from 0 to 1)] -u [BUDGET UNIT (either "sentence" or "token")] -r [RANKER ("simcse", "uniform", "weighted", or "length")] -c [PATH TO COCO4MT ENGLISH DATA]


To get a file containing sentence indices for a weighted random sample, use sample_weighted.py. Enter a command of the form

    python sample_weighted.py [file to take lines from] [budget (expressed as a percentage, from 0 to 1)] [number of trials to run]


To get a file containing sentence indices for a sample selected using the delfy algorithm:

    python delfy.py -l [file to get lines from (takes sentences, not indices)] -o [file to write sample to] -b [budget percentage (0 to 1)] -u [budget unit ("sentence" or "token")] -r [number of rounds for the delfy algorithm]


To run all unit tests for the repository, run

    python unittests.py


To train the mbart-large-50-many-to-many-mmt checkpoint on a desired lines file (takes indices), run

    python training.py [source language (as two-letter code)] [target language (as two-letter code)] [lines file] [evaluation split (defaults to "validation")]

