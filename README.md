# coco4mt

# TODO: Expand readme with instructions about how to install and use the code
# ---TODO: Create requirements.txt
# ---TODO: Transfer weighted sampling over to baselines
# ---TODO: Maybe eliminate sample_uniform.py and sample_weighted.py
# ---TODO: Maybe eliminate token_budgets.py
# TODO: Unit tests
# ---TODO: Docstrings



### To install this code, run
git clone https://github.com/Mark-Hopkins-at-Williams/coco4mt
pip install -r requirements.txt

### To select lines for the SimCSE similarity (2+ neighbors)
    python fill_budget.py -p 0.2 -u sentence -r uniform

### Use instructions:
To get files containing sentence indices for the longest and random baselinesfrom the coco4mt English training split, use fill_budget.py. Enter a command of the form

    python fill_budget.py -p [BUDGET PERCENTAGE (from 0 to 1)] -u [BUDGET UNIT (either "sentence" or "token")] -r [RANKER ("simcse", "uniform", or "length")] -c [PATH TO COCO4MT ENGLISH DATA]


To get a file containing sentence indices for a weighted random sample, use sample_weighted.py. Enter a command of the form

    python sample_weighted.py [file to take lines from] [budget (expressed as a percentage, from 0 to 1)] [number of trials to run]


To get a file containing sentence indices for a sample selected using the delfy algorithm:
For a sentence budget:

    python sent_delfy.py -l [file to get lines from] -o [file to write sample to] -b [budget percentage (0 to 1)] -r [number of rounds for the delfy algorithm]

For a token budget:

    python delfy.py -l [file to get lines from] -o [file to write sample to] -b [budget percentage (0 to 1)] -r [number of rounds for the delfy algorithm]


To unit test the delfy algorithm on a token budget, run

    python test_delfy.py


To train the mbart-large-50-many-to-many-mmt checkpoint on a desired lines file, run

    python training.py [source language (as two-letter code)] [target language (as two-letter code)] [lines file] [evaluation split (defaults to "validation")]

