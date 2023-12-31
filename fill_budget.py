from cocodata import load_coco_english
import argparse
from simcse_rankers import SimCSERanker
from baselines import UniformRandomRanker, LengthRanker, WeightedRandomRanker


def fill_sentence_budget(ranker, candidates, max_sents):
    """
    Using the specified ranker, fills the specified sentence budget.

    Parameters
    ----------
    ranker : Ranker
        the Ranker object to be used to determine which sentences should be selected
    candidates : Iterable[str]
        the sentences to be ranked
    max_sents : int
        max number of sentences that the budget allows

    Returns
    -------
    list[int]
        the indices of the selected sentences
    """
    line_nums = list(ranker.rank(candidates))
    selected = line_nums[:max_sents]
    return selected


def fill_token_budget(ranker, candidates, max_tokens):
    """
    Using the specified ranker, fills the specified token budget.

    Parameters
    ----------
    ranker : Ranker
        the Ranker object to be used to determine which tokens should be selected
    candidates : Iterable[str]
        the sentences to be ranked
    max_tokens : int
        max number of tokens that the budget allows

    Returns
    -------
    list[int]
        the indices of the selected sentences
    """
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
    return selected


def lookup_ranker(ranker_name, budget_unit):
    """
    Returns the ranker associated with the given name, if it exists. Otherwise,
    raises an exception. 
    When requesting the simcse ranker, uses the longest sentence from each
    cluster if using a sentence budget and a random sentence from each cluster
    if using a token budget.

    Parameters
    ----------
    ranker_name : String
        the name used to identify the appropriate ranker. Can be "simcse" or "uniform" or "length" or "weighted"
    budget_unit : String
        the measure for budgeting, either "sentence" or "token"

    Returns
    -------
    Ranker
        the identified Ranker
    """
    if ranker_name == "simcse":
        tiebreaker = "length" if budget_unit == "sentence" else "random"
        ranker = SimCSERanker(train, tiebreaker)
    elif ranker_name == "uniform":
        ranker = UniformRandomRanker()
    elif ranker_name == "length":
        ranker = LengthRanker()
    elif ranker_name == "weighted":
        ranker = WeightedRandomRanker()
    else:
        raise Exception(f"Unrecognized ranker: {ranker_name}")
    return ranker


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--budget_pct", type=float, required=True)
    parser.add_argument('-u', "--budget_unit", type=str, required=True)
    parser.add_argument('-r', "--ranker", type=str, required=True)
    parser.add_argument('-c', "--coco_eng_path", type=str, required=True)
    args = parser.parse_args()
    train = load_coco_english(args.coco_eng_path, "train")
    ranker = lookup_ranker(args.ranker, args.budget_unit)
    if args.budget_unit == "sentence":
        sent_budget = int(args.budget_pct * len(train))
        lines = fill_sentence_budget(ranker, train, sent_budget)
    elif args.budget_unit == "token":
        tokenized_train = [line.split() for line in train]
        total_num_tokens = 0
        for tok in tokenized_train:
            total_num_tokens += len(tok)
        word_budget = int(args.budget_pct * total_num_tokens)
        lines = fill_token_budget(ranker, train, word_budget)
    else:
        raise Exception(f"Unrecognized budget unit: {args.budget_unit}")
    for line_num in sorted(lines):
        print(line_num)