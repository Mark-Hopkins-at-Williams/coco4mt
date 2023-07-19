import unittest
from delfy import run_delfy

class TestDelfy(unittest.TestCase):

    def test_delfy1(self):
        sents = [['a', 'a', 'a', 'b', 'b'], 
                 ['b', 'c', 'a', 'a', 'b', 'a'],
                 ['a', 'a', 'd', 'b'],
                 ['c', 'c', 'd', 'a', 'b', 'd'],
                 ['a', 'a', 'e', 'a']]
        sent_ids = run_delfy(sents, budget_percentage=0.4, num_rounds=2)
        self.assertEqual({0, 3}, sent_ids)

    def test_delfy2(self):
        sents = [['a', 'a', 'a', 'a', 'a', 'a', 'a'], 
                 ['a', 'b', 'a', 'b'],
                 ['c', 'c', 'b', 'a'],
                 ['d', 'a', 'd', 'c', 'b'],
                 ['a', 'e']]
        sent_ids = run_delfy(sents, budget_percentage=0.4, num_rounds=2)
        self.assertEqual({0, 2}, sent_ids)

    def test_delfy3(self):
        sents = [['a', 'a', 'a', 'a', 'a', 'a', 'a'], 
                 ['a', 'b', 'a', 'c'],
                 ['c', 'c', 'b', 'a'],
                 ['d', 'a', 'd', 'c', 'b'],
                 ['a', 'e']]
        sent_ids = run_delfy(sents, budget_percentage=0.4, num_rounds=2)
        self.assertEqual({0, 2}, sent_ids)

    def test_delfy4(self):
        sents = [['a', 'b', 'c', 'b', 'a', 'a'], 
                 ['b', 'd', 'c', 'a'],
                 ['a', 'a', 'd'],
                 ['a', 'e', 'a', 'a', 'b'],
                 ['a', 'b', 'c', 'd', 'e', 'a', 'a', 'a']]
        sent_ids = run_delfy(sents, budget_percentage=0.4, num_rounds=2)
        self.assertEqual({2, 1}, sent_ids)

  
       
 

if __name__ == "__main__":
    unittest.main()   