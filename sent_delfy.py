import numpy as np
import sys

# from spacy.tokenizer import Tokenizer
# from spacy.lang.en import English
from transformers import AutoTokenizer
from paralleldata import lines_to_exclude

# make dictionary that stores counts (key is token, value is count) up to that point
# sort, then run through

# run on test corpus

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
        # line 9 cont. (take the top 20% of sentences by delfy)
        new_selected = set()
        num_selected = 0
        for sent_index in lf_slist:
            if num_selected >= self.budget:
                break
            if sent_index not in self.selected:
                new_selected.add(sent_index)
                num_selected += 1
        return new_selected

    def gwu(self, word):
        return np.log(self.unselected_tok_counts[word] + 1)

    def fwu(self, word):
        return self.gwu(word) / self.swugwu
    
    def lf(self, sentence_index):
        lf = 0
        sentence = self.sentences[sentence_index]
        k = len(sentence)
        if k == 0:
            return 0      
        for tok in sentence:
            lf += self.fwu(str(tok)) * np.exp(-self.l1 * self.selected_tok_counts.get(str(tok), 0))
        # print("sent_ind: " + str(sentence_index) + ", lf: " + str(lf/k))
        return lf / k

    def decay(self, word):
        csil = self.selected_tok_counts.get(word, 0)
        csius = self.uhat_tok_counts.get(word, 0)
        return np.exp(-self.l1 * csil) * np.exp(-self.l2 * csius)

    def delfy(self, sentence_index):
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
        result = dict()
        for sent_index in sentence_indices:            
            for tok in self.sentences[sent_index]:
                token = str(tok)
                result[token] = result.get(token, 0) + 1
        return result


def tokenize_all_lines(filename):
    # nlp = English()
    # tokenizer = Tokenizer(nlp.vocab)
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
    total_budget = int(budget_percentage * len(tokenized_sents))
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
    sentences = tokenize_all_lines("/home/data/coco4mt-shared-task/hr_dataset/eng/train.txt")
    to_exclude = lines_to_exclude()
    for index in to_exclude:
        sentences[index] = [250004, 2]
    budget_percentage = 0.2
    num_rounds = 20
    selected_lines = run_delfy(sentences, budget_percentage, num_rounds)
    with open('sent_delfy.txt', 'w') as writer:
        for line in selected_lines:
            writer.write(f'{line}\n')
   