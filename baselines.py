from random import shuffle


class LengthRanker:

    def __init__(self, tokenizer_fn=lambda sent: sent.split()):
        self.tokenizer_fn = tokenizer_fn

    def rank(self, sents):
        tokenized = [self.tokenizer_fn(sent) for sent in sents]
        sorted_sents = sorted(list(enumerate(tokenized)), key=lambda x: -len(x[1]))
        return [i for (i, _) in sorted_sents]


class UniformRandomRanker:

    def rank(self, sents):
        line_nums = list(range(len(sents)))
        shuffle(line_nums)
        for line_num in line_nums:
            yield line_num


class WeightedRandomRanker:

    pass
