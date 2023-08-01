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

"""
Round 1:
Budget this round: 5
toks selected: 0
just selected sentence 0, length 5
toks selected: 5
Round 2:
Budget this round: 5
toks selected: 0
just selected sentence 3, length 6
toks selected: 6
0: ['a', 'a', 'a', 'b', 'b']
3: ['c', 'c', 'd', 'a', 'b', 'd']
.Round 1:
Budget this round: 4
toks selected: 0
just selected sentence 0, length 7
toks selected: 7
Round 2:
Budget this round: 4
toks selected: 0
just selected sentence 2, length 4
toks selected: 4
0: ['a', 'a', 'a', 'a', 'a', 'a', 'a']
2: ['c', 'c', 'b', 'a']
.Round 1:
Budget this round: 4
toks selected: 0
just selected sentence 0, length 7
toks selected: 7
Round 2:
Budget this round: 4
toks selected: 0
just selected sentence 2, length 4
toks selected: 4
0: ['a', 'a', 'a', 'a', 'a', 'a', 'a']
2: ['c', 'c', 'b', 'a']
.Round 1:
Budget this round: 5
toks selected: 0
just selected sentence 2, length 3
toks selected: 3
just selected sentence 3, length 5
toks selected: 8
Round 2:
Budget this round: 5
toks selected: 0
just selected sentence 1, length 4
toks selected: 4
just selected sentence 0, length 6
toks selected: 10
0: ['a', 'b', 'c', 'b', 'a', 'a']
1: ['b', 'd', 'c', 'a']
2: ['a', 'a', 'd']
3: ['a', 'e', 'a', 'a', 'b']
F
======================================================================
FAIL: test_delfy4 (__main__.TestDelfy)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/mnt/storage/alexroot/coco4mt/test_delfy.py", line 40, in test_delfy4
    self.assertEqual({2, 1}, sent_ids)
AssertionError: Items in the second set but not the first:
0
3

----------------------------------------------------------------------
Ran 4 tests in 0.005s

FAILED (failures=1)
"""