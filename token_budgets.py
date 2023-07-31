import random
from numpy.random import multinomial
import sys
from cocodata import load_coco_english
import argparse




def get_wsample(lines, budget):
    total_tokens = 0
    tokenized_sents = lines

    weights = []
    for line in tokenized_sents:
        total_tokens += len(line)
    for line in tokenized_sents:
        weights.append(len(line) / total_tokens)

    # get average sentences per token
    avg_spt = len(tokenized_sents) / total_tokens

    tok_selected = 0
    remaining = budget - tok_selected
    selected = []
    for i in range(len(tokenized_sents)):
        selected.append(0)
    
    while remaining > 0:
        print("remaining at top of loop: " + str(remaining))
        print("tokens selected at top of loop: " + str(tok_selected))
        new_selected = multinomial(int(0.8 * remaining * avg_spt), weights).tolist()
        # integrate values selected in this round into cumulative selected values
        for i in range(len(tokenized_sents)):
            selected[i] += new_selected[i]
            tok_selected += new_selected[i] * len(tokenized_sents[i])
        print("naive tokens selected after selection round: " + str(tok_selected))
        # set weights for selected sentences to zero and correct for sentences that were selected more than once
        for i in range(len(tokenized_sents)):
            if selected[i] > 0:
                weights[i] = 0
            if selected[i] > 1:
                old_freq = selected[i]
                selected[i] = 1
                tok_selected -= (old_freq - selected[i]) * len(tokenized_sents[i])
        print("adjusted tokens selected: " + str(tok_selected))

        # fix weights so they still add up to 1
        new_total_weight = 0
        for w in weights:
            new_total_weight += w
        if new_total_weight > 0:
            weight_adjustor = 1 / new_total_weight
        for i in range(len(weights)):
            weights[i] *= weight_adjustor
        remaining = budget - tok_selected
        print("remaining at bottom of loop: " + str(remaining))
        print("tokens selected at bottom of loop: " + str(tok_selected))
        if remaining <= 20 and remaining >= 0:
            break
        while remaining < 0:
            print("remaining at top of negative loop: " + str(remaining))
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
    args = parser.parse_args()
    train = load_coco_english("train")
    tokenized_train = [line.split() for line in train]
    total_num_words = 0
    for tok in tokenized_train:
        total_num_words += len(tok)
    word_budget = args.budget * total_num_words 
    lines = get_lines(args.filter, args.sort, word_budget)
    with open(args.outfile, 'w') as sys.stdout:
        for line_num in lines:
            print(line_num)