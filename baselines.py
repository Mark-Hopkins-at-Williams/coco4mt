from random import shuffle
from ranker import Ranker
from transformers import AutoTokenizer

class LengthRanker(Ranker):
    """
    Defines an object which takes a specified list of sentences,
    sorts by length, and returns them in order.
    """

    def __init__(self, tokenizer_fn=lambda sent: sent.split()):
        self.tokenizer_fn = tokenizer_fn

    def rank(self, sents):
        """
        Ranks the provided sentences based on length.

        Parameters
        ----------
        sents : list[String]
            the sentences to be ranked

        Returns
        -------
        Generator[int]
            generates the indices of the selected sentences, in order
        """
        tokenized = [self.tokenizer_fn(sent) for sent in sents]
        sorted_sents = sorted(list(enumerate(tokenized)), key=lambda x: -len(x[1]))
        return [i for (i, _) in sorted_sents]


class UniformRandomRanker(Ranker):
    """
    Defines an object which takes a specified list of sentences,
    shuffles them randomly, and returns them in that order.
    """

    def rank(self, sents):
        """
        Randomly ranks the provided sentences.

        Parameters
        ----------
        sents : list[String]
            the sentences to be ranked using SimCSE

        Returns
        -------
        Generator[int]
            generates the indices of the selected sentences, in random order
        """
        line_nums = list(range(len(sents)))
        shuffle(line_nums)
        for line_num in line_nums:
            yield line_num


class WeightedRandomRanker(Ranker):
    """
    Defines an object which takes a specified list of sentences,
    selects them at random using a weighted distribution by length,
    and returns them in that order.
    """
    def get_weights(self, sents):
        model_checkpoint = "facebook/mbart-large-50-many-to-many-mmt"
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        tokenized_sents = []
        for sent in sents:
            tokenized_sents.append(tokenizer.tokenize(sent.strip()))

        total_tokens = 0
        for line in tokenized_sents:
            total_tokens += len(line)
        weights = []
        for line in tokenized_sents:
            weights.append(len(line) / total_tokens)
        return weights

    def rank(self, sents):
        """
        Ranks the sentences in a weighted random order

        Parameters
        ----------
        sents : list[String]
            the sentences to be ranked

        Returns
        -------
        Generator[int]
            generates the indices of the selected sentences, in weighted random order
        """
        model_checkpoint = "facebook/mbart-large-50-many-to-many-mmt"
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        tokenized_sents = []
        for sent in sents:
            tokenized_sents.append(tokenizer.tokenize(sent.strip()))

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
                yield i