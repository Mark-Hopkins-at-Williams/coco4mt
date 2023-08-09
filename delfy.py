import numpy as np
import sys
import argparse

from transformers import AutoTokenizer
from paralleldata import lines_to_exclude

class DecayLogFrequency:

    def __init__(self, sentences, selected, budget, budget_unit):
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
        self.budget_unit = budget_unit
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
        selected_count = 0
        if self.budget_unit == "token":
            for sent_index in lf_slist:
                if selected_count >= self.budget:
                    break
                if sent_index not in self.selected:
                    if selected_count + len(self.sentences[sent_index]) < self.budget:
                        new_selected.add(sent_index)
                        selected_count += len(self.sentences[sent_index])
        elif self.budget_unit == "sentence":
            for sent_index in lf_slist:
                if selected_count >= self.budget:
                    break
                if sent_index not in self.selected:
                    new_selected.add(sent_index)
                    selected_count += 1
        else:
            raise ValueError(f"Only budget units 'sentence' and 'token' are accepted: {self.budget_unit}")
        return new_selected

    def gwu(self, word):
        """
        Calculates the function G(w|U) as defined in the paper, which is used
        to find the logarithm frequency of that word in the unselected corpus.

        Parameters
        ----------
        word : String
            the word to find G(word|U) for
        """
        return np.log(self.unselected_tok_counts[word] + 1)

    def fwu(self, word):
        """
        Calculates the logarithm frequency of a word in the unselected corpus
        as defined in the paper.

        Parameters
        ----------
        word : String
            the word to find the logarithm frequency of
        """
        return self.gwu(word) / self.swugwu

    def lf(self, sentence_index):
        """
        Calculates the average logarithm frequency of all tokens in the
        sentence at the provided sentence index.

        Parameters
        ----------
        sentence_index : int
            the index of the sentence to find lf for in the lines file
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

        Parameters
        ----------
        word : String
            the word to identify the decay value for
        """
        csil = self.selected_tok_counts.get(word, 0)
        csius = self.uhat_tok_counts.get(word, 0)
        return np.exp(-self.l1 * csil) * np.exp(-self.l2 * csius)

    def delfy(self, sentence_index):
        """
        Defines the delfy index, used to rank each
        unselected sentence in each round.

        Parameters
        ----------
        sentence_index : int
            the index of the sentence to find delfy for in the lines file
        """
        sentence = self.sentences[sentence_index]
        delfy = 0
        k = len(sentence)
        if k == 0:
            return 0
        for tok in sentence:
            delfy += self.fwu(str(tok)) * self.decay(str(tok))
        return delfy / k

    def token_count(self, sentence_indices):
        """
        Finds the number of tokens in the entire corpus. Used for budgeting.

        Parameters
        ----------
        sentence_indices : Iterable[int]
            the indices of all the sentences to search through to find the frequencies of all tokens
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

    Parameters
    ----------
    filename : String
        the name of the file containing the lines to tokenize
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


def run_delfy(tokenized_sents, budget_percentage=0.2, budget_unit="sentence", num_rounds=20):
    """
    Runs the delfy algorithm as defined in the paper for the provided number of
    rounds, with the provided budget. Returns the final selection set.

    Parameters
    ----------
    tokenized_sents : list[list[String]]
        a list of all sentences, each organized as a list of tokens
    budget_percentage : float
        the percentage of the total provided data (in either sentences or tokens) to select
    budget_unit : String
        the measure for budgeting, either "sentence" or "token"
    num_rounds : int
        the number of rounds to execute the delfy algorithm
    """
    if budget_unit == "token":
        total_tok_count = 0
        for sent in tokenized_sents:
            total_tok_count += len(sent)

        total_budget = int(budget_percentage * total_tok_count)
        selected = set()
        for i in range(1, num_rounds + 1):
            budget_this_round = i * total_budget // num_rounds - (i - 1) * total_budget // num_rounds
            # last round is "cleanup," gets unused tokens from previous rounds
            if i == num_rounds:
                toks_selected = 0
                for sent in selected:
                    toks_selected += len(tokenized_sents[sent])
                budget_this_round = total_budget - toks_selected
            next_selected = DecayLogFrequency(tokenized_sents, selected, budget_this_round, budget_unit).run()
            selected |= next_selected
    elif budget_unit == "sentence":
        total_budget = int(budget_percentage * len(tokenized_sents))
        selected = set()
        for i in range(1, num_rounds + 1):
            budget_this_round = i * total_budget // num_rounds - (i - 1) * total_budget // num_rounds
            next_selected = DecayLogFrequency(tokenized_sents, selected, budget_this_round, budget_unit).run()
            selected |= next_selected
    else:
        raise ValueError(f"Only budget units 'sentence' and 'token' are accepted: {budget_unit}")
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lines')
    parser.add_argument('-o', '--outfile')
    parser.add_argument('-b', "--budget", type=float)
    parser.add_argument('-u', "--budget-unit")
    parser.add_argument('-r', '--rounds', type=int)
    args = parser.parse_args()

    sentences = tokenize_all_lines(args.lines)
    to_exclude = lines_to_exclude()
    for index in to_exclude:
        sentences[index] = [250004, 2]
    selected_lines = run_delfy(sentences, args.budget, args.budget-unit, args.rounds)
    with open(args.outfile, 'w') as writer:
        for line in selected_lines:
            writer.write(f'{line}\n')
