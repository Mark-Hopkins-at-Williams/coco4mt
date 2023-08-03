import unittest
from paralleldata import enumerate_lines
from delfy import run_delfy
from sent_delfy import run_delfy as run_sent_delfy
from fill_budget import fill_sentence_budget, fill_token_budget, lookup_ranker

class TestDelfy(unittest.TestCase):

    def test_delfy1(self):
        sents = [['a', 'a', 'a', 'b', 'b'], 
                 ['b', 'c', 'a', 'a', 'b', 'a'],
                 ['a', 'a', 'd', 'b'],
                 ['c', 'c', 'd', 'a', 'b', 'd'],
                 ['a', 'a', 'e', 'a']]
        sent_ids = run_delfy(sents, budget_percentage=0.4, num_rounds=2)
        self.assertEqual({2, 4}, sent_ids)

    def test_delfy2(self):
        sents = [['a', 'a', 'a', 'a', 'a', 'a', 'a'], 
                 ['a', 'b', 'a', 'b'],
                 ['c', 'c', 'b', 'a'],
                 ['d', 'a', 'd', 'c', 'b'],
                 ['a', 'e']]
        sent_ids = run_delfy(sents, budget_percentage=0.4, num_rounds=2)
        self.assertEqual({2, 4}, sent_ids)

    def test_delfy3(self):
        sents = [['a', 'a', 'a', 'a', 'a', 'a', 'a'], 
                 ['a', 'b', 'a', 'c'],
                 ['c', 'c', 'b', 'a'],
                 ['d', 'a', 'd', 'c', 'b'],
                 ['a', 'e']]
        sent_ids = run_delfy(sents, budget_percentage=0.4, num_rounds=2)
        self.assertEqual({2, 4}, sent_ids)

    def test_delfy4(self):
        sents = [['a', 'b', 'c', 'b', 'a', 'a'], 
                 ['b', 'd', 'c', 'a'],
                 ['a', 'a', 'd'],
                 ['a', 'e', 'a', 'a', 'b'],
                 ['a', 'b', 'c', 'd', 'e', 'a', 'a', 'a']]
        sent_ids = run_delfy(sents, budget_percentage=0.4, num_rounds=2)
        self.assertEqual({0, 2}, sent_ids)

class TestSentDelfy(unittest.TestCase):

    def test_sent_delfy1(self):
        sents = [['a', 'a', 'a', 'b', 'b'], 
                 ['b', 'c', 'a', 'a', 'b', 'a'],
                 ['a', 'a', 'd', 'b'],
                 ['c', 'c', 'd', 'a', 'b', 'd'],
                 ['a', 'a', 'e', 'a']]
        sent_ids = run_sent_delfy(sents, budget_percentage=0.4, num_rounds=2)
        self.assertEqual({0, 3}, sent_ids)

    def test_sent_delfy2(self):
        sents = [['a', 'a', 'a', 'a', 'a', 'a', 'a'], 
                 ['a', 'b', 'a', 'b'],
                 ['c', 'c', 'b', 'a'],
                 ['d', 'a', 'd', 'c', 'b'],
                 ['a', 'e']]
        sent_ids = run_sent_delfy(sents, budget_percentage=0.4, num_rounds=2)
        self.assertEqual({0, 2}, sent_ids)

    def test_sent_delfy3(self):
        sents = [['a', 'a', 'a', 'a', 'a', 'a', 'a'], 
                 ['a', 'b', 'a', 'c'],
                 ['c', 'c', 'b', 'a'],
                 ['d', 'a', 'd', 'c', 'b'],
                 ['a', 'e']]
        sent_ids = run_sent_delfy(sents, budget_percentage=0.4, num_rounds=2)
        self.assertEqual({0, 2}, sent_ids)

    def test_sent_delfy4(self):
        sents = [['a', 'b', 'c', 'b', 'a', 'a'], 
                 ['b', 'd', 'c', 'a'],
                 ['a', 'a', 'd'],
                 ['a', 'e', 'a', 'a', 'b'],
                 ['a', 'b', 'c', 'd', 'e', 'a', 'a', 'a']]
        sent_ids = run_sent_delfy(sents, budget_percentage=0.4, num_rounds=2)
        self.assertEqual({0, 2}, sent_ids)

class TestFillBudget(unittest.TestCase):

    def test_fill_budget1(self):
        sents = enumerate_lines("test_fill_budget.txt")
        ranker = lookup_ranker("length", "sentence")
        sent_ids = fill_sentence_budget(ranker, sents, 2)
        self.assertEqual([4, 0], sent_ids)

    def test_fill_budget2(self):
        sents = enumerate_lines("test_fill_budget.txt")
        ranker = lookup_ranker("length", "token")
        sent_ids = fill_token_budget(ranker, sents, 37)
        self.assertEqual([4, 0, 3], sent_ids)

if __name__ == "__main__":
    unittest.main()   
