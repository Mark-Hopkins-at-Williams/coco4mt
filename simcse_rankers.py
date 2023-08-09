from simcse import SimCSE
from random import shuffle
from ranker import Ranker

class SimCSERanker(Ranker):
    """
    Defines objects which take a specified list of sentences, use SimCSE
    sentence embedding to create a vector space of all these sentences,
    identifies all sentences which are the closest neighbor of at least two
    other sentences, and sorts by a certain criterion. If a sentence budget is
    used, it sorts by length. If a token budget is used, it shuffles (uniform
    random distribution). It returns this as a list in the generated order.
    """

    def __init__(self, all_lines, tiebreaker,
                 model_name="princeton-nlp/sup-simcse-bert-base-uncased"):
        self.model = SimCSE(model_name)
        self.model.build_index(all_lines)
        self.all_lines = all_lines
        self.tiebreaker = tiebreaker

    def rank(self, sents):
        """
        Ranks the provided sentences based on the metric attributed to self.

        Parameters
        ----------
        sents : list[String]
            the sentences to be ranked using SimCSE
        """
        nonempty_sents = [sent for sent in sents if len(sent) > 0]
        neighbor_sents = self.model.search(nonempty_sents, threshold=0.0)
        closests = []
        for neighbors in neighbor_sents:
            if len(neighbors) > 1:
                closests.append(neighbors[1][0])
        centrality = dict()
        for sent in sents:
            centrality[sent] = 0
        for sent in closests:
            centrality[sent] = 1 + centrality[sent]
            if centrality[sent] > 2: # treats everything with centrality >= 2 as equal
                centrality[sent] = 2
        centrality = list(centrality.items())
        if self.tiebreaker == "random":
            shuffle(centrality)
        elif self.tiebreaker == "length":
            centrality.sort(key=lambda x: -len(x[0].split()))
        else:
            raise Exception(f"Unrecognized tiebreaker: {self.tiebreaker}")
        ranked_sents = [x[0] for x in sorted(centrality, key=lambda x: -x[1])]
        line_nums = []
        for sent in ranked_sents:
            line_nums.append(self.all_lines.index(sent))
        return line_nums
