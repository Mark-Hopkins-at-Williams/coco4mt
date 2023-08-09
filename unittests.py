import unittest
from delfy import run_delfy
from fill_budget import fill_sentence_budget, fill_token_budget, lookup_ranker
from simcse_rankers import SimCSERanker


class TestDelfy(unittest.TestCase):

    def test_tok_delfy1(self):
        sents = [['a', 'a', 'a', 'b', 'b'], 
                 ['b', 'c', 'a', 'a', 'b', 'a'],
                 ['a', 'a', 'd', 'b'],
                 ['c', 'c', 'd', 'a', 'b', 'd'],
                 ['a', 'a', 'e', 'a']]
        sent_ids = run_delfy(sents, budget_percentage=0.4, budget_unit="token", num_rounds=2)
        self.assertEqual({2, 4}, sent_ids)

    def test_tok_delfy2(self):
        sents = [['a', 'a', 'a', 'a', 'a', 'a', 'a'], 
                 ['a', 'b', 'a', 'b'],
                 ['c', 'c', 'b', 'a'],
                 ['d', 'a', 'd', 'c', 'b'],
                 ['a', 'e']]
        sent_ids = run_delfy(sents, budget_percentage=0.4, budget_unit="token", num_rounds=2)
        self.assertEqual({2, 4}, sent_ids)

    def test_tok_delfy3(self):
        sents = [['a', 'a', 'a', 'a', 'a', 'a', 'a'], 
                 ['a', 'b', 'a', 'c'],
                 ['c', 'c', 'b', 'a'],
                 ['d', 'a', 'd', 'c', 'b'],
                 ['a', 'e']]
        sent_ids = run_delfy(sents, budget_percentage=0.4, budget_unit="token", num_rounds=2)
        self.assertEqual({2, 4}, sent_ids)

    def test_tok_delfy4(self):
        sents = [['a', 'b', 'c', 'b', 'a', 'a'], 
                 ['b', 'd', 'c', 'a'],
                 ['a', 'a', 'd'],
                 ['a', 'e', 'a', 'a', 'b'],
                 ['a', 'b', 'c', 'd', 'e', 'a', 'a', 'a']]
        sent_ids = run_delfy(sents, budget_percentage=0.4, budget_unit="token", num_rounds=2)
        self.assertEqual({0, 2}, sent_ids)

    def test_sent_delfy1(self):
        sents = [['a', 'a', 'a', 'b', 'b'], 
                 ['b', 'c', 'a', 'a', 'b', 'a'],
                 ['a', 'a', 'd', 'b'],
                 ['c', 'c', 'd', 'a', 'b', 'd'],
                 ['a', 'a', 'e', 'a']]
        sent_ids = run_delfy(sents, budget_percentage=0.4, budget_unit="sentence", num_rounds=2)
        self.assertEqual({0, 3}, sent_ids)

    def test_sent_delfy2(self):
        sents = [['a', 'a', 'a', 'a', 'a', 'a', 'a'], 
                 ['a', 'b', 'a', 'b'],
                 ['c', 'c', 'b', 'a'],
                 ['d', 'a', 'd', 'c', 'b'],
                 ['a', 'e']]
        sent_ids = run_delfy(sents, budget_percentage=0.4, budget_unit="sentence", num_rounds=2)
        self.assertEqual({0, 2}, sent_ids)

    def test_sent_delfy3(self):
        sents = [['a', 'a', 'a', 'a', 'a', 'a', 'a'], 
                 ['a', 'b', 'a', 'c'],
                 ['c', 'c', 'b', 'a'],
                 ['d', 'a', 'd', 'c', 'b'],
                 ['a', 'e']]
        sent_ids = run_delfy(sents, budget_percentage=0.4, budget_unit="sentence", num_rounds=2)
        self.assertEqual({0, 2}, sent_ids)

    def test_sent_delfy4(self):
        sents = [['a', 'b', 'c', 'b', 'a', 'a'], 
                 ['b', 'd', 'c', 'a'],
                 ['a', 'a', 'd'],
                 ['a', 'e', 'a', 'a', 'b'],
                 ['a', 'b', 'c', 'd', 'e', 'a', 'a', 'a']]
        sent_ids = run_delfy(sents, budget_percentage=0.4, budget_unit="sentence", num_rounds=2)
        self.assertEqual({0, 2}, sent_ids)

        


class TestFillBudget(unittest.TestCase):

    def setUp(self):
        self.mitt = ['My favorite meat is hot dog, by the way.',
                     'That is my favorite meat.',
                     'My second favorite meat is hamburger.',
                     "And, everyone says, oh, don't you prefer steak?",
                     "It's like, I know steaks are great, but I like hot dog best, and I like hamburger next best."]


    def test_fill_budget1(self):
        ranker = lookup_ranker("length", "sentence")
        sent_ids = fill_sentence_budget(ranker, self.mitt, 2)
        self.assertEqual([4, 0], sent_ids)

    def test_fill_budget2(self):
        ranker = lookup_ranker("length", "token")
        sent_ids = fill_token_budget(ranker, self.mitt, 37)
        self.assertEqual([4, 0, 3], sent_ids)

    def test_fill_budget3(self):
        ranker = lookup_ranker("weighted", "token")
        weights = ranker.get_weights(self.mitt)
        self.assertEqual([0.17647058823529413, 0.10294117647058823, 0.11764705882352941, 0.22058823529411764, 0.38235294117647056], weights)


class TestSimCSERankerBudget(unittest.TestCase):

    def setUp(self):
        self.mitt = ['My favorite meat is hot dog, by the way.',
                     'That is my favorite meat.',
                     'My second favorite meat is hamburger.',
                     "And, everyone says, oh, don't you prefer steak?",
                     "It's like, I know steaks are great, but I like hot dog best, and I like hamburger next best."]


    def test_rank1(self):
        ranker = SimCSERanker(self.mitt, "length", "princeton-nlp/sup-simcse-bert-base-uncased")
        ranking = ranker.rank(self.mitt)
        self.assertEqual(ranking, [4, 0, 2, 1, 3])


if __name__ == "__main__":
    unittest.main()   
