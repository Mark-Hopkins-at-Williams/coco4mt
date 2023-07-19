from simcse import SimCSE
from cocodata import load_coco_english


class SimCSESelector:

    def __init__(self, all_lines):
        self.model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
        self.model.build_index(all_lines)
        self.all_lines = all_lines

    def count(self, sents):
        counts = dict()
        for sent in sents:
            counts[sent] = 1 + counts.get(sent, 0)
        return counts

    def get_repeats(self, sents):
        repeats = []
        sent_counts = self.count(sents)
        for sent in sent_counts:
            if sent_counts[sent] >= 2:
                repeats.append(sent)
        return repeats

    def common_neighbors(self, sents):
        nonempty_sents = [sent for sent in sents if len(sent) > 0]
        neighbor_sents = self.model.search(nonempty_sents, threshold=0.0)
        closests = []
        for neighbors in neighbor_sents:
            if len(neighbors) > 1:
                closests.append(neighbors[1][0])
        repeats = self.get_repeats(closests)
        line_nums = []
        for sent in sorted(repeats, key=len):
            line_nums.append(self.all_lines.index(sent))
        return line_nums
