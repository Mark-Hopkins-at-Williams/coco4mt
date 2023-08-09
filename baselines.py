from random import shuffle
from ranker import Ranker

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
    # TODO: support the rank method by borrowing code from sample_weighted.py
    pass