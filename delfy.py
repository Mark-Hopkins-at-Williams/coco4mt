import numpy as np
import sys
import argparse

from transformers import AutoTokenizer
from paralleldata import lines_to_exclude

class DecayLogFrequency:

    def __init__(self, sentences, selected, budget):
        self.sentences = sentences
        self.unselected = set([i for i in range(len(sentences)) if i not in selected])
        self.selected = selected
        self.unselected_tok_counts = self.token_count(self.unselected)
        self.selected_tok_counts = self.token_count(self.selected)
        self.uhat = []
        self.uhat_tok_counts = dict()
        self.l1 = 1.0
        self.l2 = 1.0
        self.budget = budget
        # precompute swugwu (for efficiency)
        self.swugwu = 0
        for w in self.unselected_tok_counts:
            self.swugwu += self.gwu(w)

    def run(self):
        """
        Runs the delfy algorithm as defined in the paper, once. For a single round,
        calculates the delfy metric for each sentence and returns the sentences
        selected in that round to be added to the set of all selected sentences.
        """
        # lines 2-5 of algorithm
        # compute lf(s) for every untranslated sentence s and sort
        lf_slist = list(self.unselected)
        lf_slist.sort(key=lambda sent_index: self.lf(sent_index), reverse=True)
        # line 5 cont.
        delfy_scores = {}
        for sent_index in lf_slist:
            delfy_scores[sent_index] = self.delfy(sent_index)
            self.uhat.append(sent_index)
            deltas = self.token_count([sent_index])
            for tok in deltas:
                self.uhat_tok_counts[tok] = 1 + self.uhat_tok_counts.get(tok, 0)
        # line 9 of algorithm
        lf_slist.sort(key=lambda i: delfy_scores[i], reverse=True)
        # line 9 cont. (take the top budget% of sentences by delfy)
        new_selected = set()
        toks_selected = 0
        for sent_index in lf_slist:
            if toks_selected >= self.budget:
                break
            if sent_index not in self.selected:
                new_selected.add(sent_index)
                toks_selected += len(self.sentences[sent_index])
        return new_selected

    def gwu(self, word):
        """
        Calculates the function G(w|U) as defined in the paper, which is used
        to find the logarithm frequency of that word in the unselected corpus.
        """
        return np.log(self.unselected_tok_counts[word] + 1)

    def fwu(self, word):
        """
        Calculates the logarithm frequency of a word in the unselected corpus
        as defined in the paper.
        """
        return self.gwu(word) / self.swugwu

    def lf(self, sentence_index):
        """
        Calculates the average logarithm frequency of all tokens in the
        sentence at the provided sentence index.
        """
        lf = 0
        sentence = self.sentences[sentence_index]
        k = len(sentence)
        if k == 0:
            return 0
        for tok in sentence:
            lf += self.fwu(str(tok)) * np.exp(-self.l1 * self.selected_tok_counts.get(str(tok), 0))
        return lf / k

    def decay(self, word):
        """
        Calculates the decay term defined in the paper, which is used to reduce
        the delfy scores of sentences that are redundant when considering
        other sentences with higher lf scores.
        """
        csil = self.selected_tok_counts.get(word, 0)
        csius = self.uhat_tok_counts.get(word, 0)
        return np.exp(-self.l1 * csil) * np.exp(-self.l2 * csius)

    def delfy(self, sentence_index):
        """
        Defines the delfy index, used to rank each
        unselected sentence in each round.
        """
        sentence = self.sentences[sentence_index]
        delfy = 0
        k = len(sentence)
        if k == 0:
            return 0
        for tok in sentence:
            delfy += self.fwu(str(tok)) * self.decay(str(tok))
        # print("sent_ind: " + str(sentence_index) + ", delfy: " + str(delfy/k))
        return delfy / k

    def token_count(self, sentence_indices):
        """
        Finds the number of tokens in the entire corpus. Used for budgeting.
        """
        result = dict()
        for sent_index in sentence_indices:
            for tok in self.sentences[sent_index]:
                token = str(tok)
                result[token] = result.get(token, 0) + 1
        return result


def tokenize_all_lines(filename):
    """
    Returns a list of all the lines in the provided file, tokenized.
    """
    model_checkpoint = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    lines = []
    with open(filename) as reader:
        for line in reader:
            line = line.strip()
            line = tokenizer(line)
            lines.append(line['input_ids'])
    return lines


def run_delfy(tokenized_sents, budget_percentage=0.2, num_rounds=20):
    """
    Runs the delfy algorithm as defined in the paper for the provided number of
    rounds, with the provided budget. Returns the final selection set.
    """
    total_tok_count = 0
    for sent in tokenized_sents:
        total_tok_count += len(sent)

    total_budget = int(budget_percentage * total_tok_count)
    selected = set()
    for i in range(1, num_rounds + 1):
        print("Round " + str(i) + ":")
        budget_this_round = i * total_budget // num_rounds - (i - 1) * total_budget // num_rounds
        print(f'Budget this round: {budget_this_round}')
        sys.stdout.flush()
        next_selected = DecayLogFrequency(tokenized_sents, selected, budget_this_round).run()
        selected |= next_selected
    for i in sorted(selected):
        print(f'{i}: {tokenized_sents[i]}')
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lines')
    parser.add_argument('-o', '--outfile')
    parser.add_argument('-b', "--budget", type=float)
    parser.add_argument('-r', '--rounds', type=int)
    args = parser.parse_args()

    sentences = tokenize_all_lines(args.lines)
    to_exclude = lines_to_exclude()
    for index in to_exclude:
        sentences[index] = [250004, 2]
    selected_lines = run_delfy(sentences, args.budget, args.rounds)
    with open(args.outfile, 'w') as writer:
        for line in selected_lines:
            writer.write(f'{line}\n')
