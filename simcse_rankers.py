from simcse import SimCSE
from random import shuffle

class SimCSERanker:

    def __init__(self, all_lines, tiebreaker,
                 model_name="princeton-nlp/sup-simcse-bert-base-uncased"):
        self.model = SimCSE(model_name)
        self.model.build_index(all_lines)
        self.all_lines = all_lines
        self.tiebreaker = tiebreaker

    def rank(self, sents):
        nonempty_sents = [sent for sent in sents if len(sent) > 0]
        neighbor_sents = self.model.search(nonempty_sents, threshold=0.0)
        closests = []
        for neighbors in neighbor_sents:
            if len(neighbors) > 1:
                closests.append(neighbors[1][0])        
        centrality = dict()
        for sent in closests:
            centrality[sent] = 1 + centrality.get(sent, 0)
            #if centrality[sent] > 2:
            #    centrality[sent] = 2
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
