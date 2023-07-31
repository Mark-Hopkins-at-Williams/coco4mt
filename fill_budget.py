from cocodata import load_coco_english
import argparse
from simcse_neighbors import SimCSESelector


def fill_sentence_budget(ranker, candidates, max_sents):  
    line_nums = ranker.rank(candidates)
    selected = line_nums[:max_sents]
    selected.sort()
    return selected


def fill_token_budget(ranker, candidates, max_tokens):    
    length = 0
    selected = []
    line_nums = ranker.rank(candidates)
    for line_num in line_nums:
        len_sent = len(candidates[line_num].split())
        if length + len_sent < max_tokens:
            selected.append(line_num)
            length += len_sent
        else:
            break
    selected.sort()
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--budget_pct", type=float, required=True)
    parser.add_argument('-u', "--budget_unit", type=str, required=True)
    args = parser.parse_args()
    train = load_coco_english("train")     
    if args.budget_unit == "sentence":
        sent_budget = int(args.budget_pct * len(train))
        lines = fill_sentence_budget(SimCSESelector(train), train, sent_budget)
    elif args.budget_unit == "token":
        tokenized_train = [line.split() for line in train]
        total_num_tokens = 0
        for tok in tokenized_train:
            total_num_tokens += len(tok)
        word_budget = int(args.budget_pct * total_num_tokens)
        lines = fill_token_budget(SimCSESelector(train), train, word_budget)
    else:
        raise Exception(f"Unrecognized budget unit: {args.budget_unit}")
    for line_num in lines:
        print(line_num)