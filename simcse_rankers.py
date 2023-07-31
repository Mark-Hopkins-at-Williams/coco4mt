from simcse import SimCSE
from random import shuffle

class SimCSERanker:

    def __init__(self, all_lines, model_name="princeton-nlp/sup-simcse-bert-base-uncased"):
        self.model = SimCSE(model_name)
        self.model.build_index(all_lines)
        self.all_lines = all_lines

    def rank(self, sents):
        nonempty_sents = [sent for sent in sents if len(sent) > 0]
        neighbor_sents = self.model.search(nonempty_sents, threshold=0.0)
        closests = []
        for neighbors in neighbor_sents:
            if len(neighbors) > 1:
                closests.append(neighbors[1][0])        
        sent_counts = dict()
        for sent in closests:
            sent_counts[sent] = 1 + sent_counts.get(sent, 0)
        sent_counts = list(sent_counts.items())
        shuffle(sent_counts)
        ranked_sents = [x[0] for x in sorted(sent_counts, key=lambda x: -x[1])]
        line_nums = []
        for sent in ranked_sents:
            line_nums.append(self.all_lines.index(sent))
        return line_nums
