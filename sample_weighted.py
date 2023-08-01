from transformers import AutoTokenizer
from numpy.random import multinomial
import sys


# size-weighted random distribution
def weighted_sample(tokenized_sents, budget_pct):
    """
    Uses the numpy.random.multinomial function to take a sample of the given
    sentences, weighted by token length, accounting for sentences chosen
    multiple times, and iterating until the budget is exactly met. It uses a
    sentence budget.
    """
    total_tokens = 0
    for line in tokenized_sents:
        total_tokens += len(line)
    weights = []
    for line in tokenized_sents:
        weights.append(len(line) / total_tokens)
    budget = int(budget_pct * len(tokenized_sents))
    num_selected = 0
    remaining = budget - num_selected
    selected = []
    for i in range(len(tokenized_sents)):
        selected.append(0)
    while remaining > 0:
        num_selected = 0
        new_selected = multinomial(remaining, weights).tolist()
        # integrate values selected in this round into cumulative selected values
        for i in range(len(tokenized_sents)):
            selected[i] += new_selected[i]
            num_selected += selected[i]
        # set weights for selected sentences to zero and correct for sentences that were selected more than once
        for i in range(len(tokenized_sents)):
            if selected[i] > 0:
                weights[i] = 0
            if selected[i] > 1:
                old_freq = selected[i]
                selected[i] = 1
                num_selected -= old_freq - selected[i]
        # fix weights so they still add up to 1
        new_total_weight = 0
        for w in weights:
            new_total_weight += w
        weight_adjustor = 1 / new_total_weight
        for i in range(len(weights)):
            weights[i] *= weight_adjustor
        remaining = budget - num_selected
    # get lines from selected
    selected_lines = []
    for i in range(len(selected)):
        if selected[i] == 1:
            selected_lines.append(i)
    return selected_lines


if __name__ == "__main__":
    sentence_file = sys.argv[1]
    budget_pct = float(sys.argv[2])
    num_trials = int(sys.argv[3])
    model_checkpoint = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenized_sents = []
    with open(sentence_file) as reader:
        for line in reader:
            tokenized_sents.append(tokenizer.tokenize(line.strip()))
    for i in range(num_trials):
        with open(f"wsample.{i}.txt", 'w') as writer:
            selected_lines = weighted_sample(tokenized_sents, budget_pct)            
            for line in selected_lines:
                writer.write(f"{line}\n")