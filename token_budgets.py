import random
from numpy.random import multinomial
import sys
from cocodata import load_coco_english
import argparse


def get_wsample(lines, budget):
    """
    Uses the numpy.random.multinomial function to take a sample of the given
    sentences, weighted by token length, accounting for sentences chosen
    multiple times, and iterating until the budget is approximately met. It
    uses a token budget.
    """
    total_tokens = 0
    tokenized_sents = lines

    weights = []
    for line in tokenized_sents:
        total_tokens += len(line)
    for line in tokenized_sents:
        weights.append(len(line) / total_tokens)

    # get average sentences per token (to determine how many sentences to select)
    avg_spt = len(tokenized_sents) / total_tokens

    tok_selected = 0
    remaining = budget - tok_selected
    selected = []
    for i in range(len(tokenized_sents)):
        selected.append(0)

    while remaining > 0:
        new_selected = multinomial(int(0.8 * remaining * avg_spt), weights).tolist()
        # integrate values selected in this round into cumulative selected values
        for i in range(len(tokenized_sents)):
            selected[i] += new_selected[i]
            tok_selected += new_selected[i] * len(tokenized_sents[i])
        # set weights for selected sentences to zero and correct for sentences that were selected more than once
        for i in range(len(tokenized_sents)):
            if selected[i] > 0:
                weights[i] = 0
            if selected[i] > 1:
                old_freq = selected[i]
                selected[i] = 1
                tok_selected -= (old_freq - selected[i]) * len(tokenized_sents[i])

        # fix weights so they still add up to 1
        new_total_weight = 0
        for w in weights:
            new_total_weight += w
        if new_total_weight > 0:
            weight_adjustor = 1 / new_total_weight
        for i in range(len(weights)):
            weights[i] *= weight_adjustor
        remaining = budget - tok_selected

        # stops selection if the number of tokens selected is very close to the budget
        if remaining <= 20 and remaining >= 0:
            break
        # corrects for overselection
        while remaining < 0:
            for i in range(len(selected)):
                if selected[i] == 1:
                    selected[i] = 0
                    tok_selected -= len(tokenized_sents[i])
                    remaining = budget - tok_selected

    # get lines from selected
    selected_lines = []
    for i in range(len(selected)):
        if selected[i] == 1:
            selected_lines.append(i)

    return selected_lines


def get_lines(filter_type, sort_type, budget):
    """
    Given parameters for how sentences should be selected and a budget
    (expressed as a percentage), gets the desired number of sentences in the
    desired way. Filter type controls whether SimCSE sentence embedding is
    used; specify "none" to bypass SimCSE and "sim2" to use SimCSE to select
    from sentences that are the closest neighbor of at least two other
    sentences as defined by SimCSE. Supports sorting with a weighted or uniform
    random distribution, or a length-based sort.
    """
    length = 0
    selected = []
    if filter_type == "none":
        line_nums = [i for i in range(len(train))]
    elif filter_type == "sim2":
        selector = SimCSESelector(train)
        line_nums = selector.common_neighbors(train)


    if sort_type == "w":
        selected = get_wsample(tokenized_train, budget)
        return selected
    elif sort_type == "r":
        line_nums = random.sample(line_nums, k=len(line_nums))
    elif sort_type == "l":
        line_nums = sorted(line_nums, key=lambda x: -len(tokenized_train[x]))


    for line_num in line_nums:
        len_sent = len(train[line_num].split())
        if length + len_sent < budget:
            selected.append(line_num)
            length += len_sent
        else:
            break

    selected.sort()
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outfile')
    parser.add_argument('-f', "--filter")
    parser.add_argument('-s', "--sort")
    parser.add_argument('-b', "--budget", type=float)
    parser.add_argument('-c', "--coco_eng_path", type=str, required=True)
    args = parser.parse_args()
    train = load_coco_english(args.coco_eng_path, "train")
    tokenized_train = [line.split() for line in train]
    total_num_words = 0
    for tok in tokenized_train:
        total_num_words += len(tok)
    word_budget = args.budget * total_num_words
    lines = get_lines(args.filter, args.sort, word_budget)
    with open(args.outfile, 'w') as sys.stdout:
        for line_num in lines:
            print(line_num)