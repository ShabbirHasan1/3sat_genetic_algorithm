import unittest
import src.functions as fn

class TestTrivialCase(unittest.TestCase):
	def test_all_clauses_positive(self):
		# Test that it returns (True, 1) when there are no negated variables in the clauses
		pos_clauses = [[1,2,3],[2,1,4],[3,4,2],[4,3,1]]
		result = fn.trivial_case(pos_clauses)
		self.assertEqual(result, (True, 1))

	def test_all_clauses_negated(self):
		# Test that it returns (False, 0) when all variables in the clauses are negated
		neg_clauses = [[-1,-2,-3],[-2,-1,-4],[-3,-4,-2],[-4,-3,-1]]
		result = fn.trivial_case(neg_clauses)
		self.assertEqual(result, (True, 0))

	def test_mixed_clauses(self):
		clauses = [[1,2,3],[2,1,4],[3,4,2],[4,3,1],[-1,-2,-3],[-2,-1,-4],[-3,-4,-2],[-4,-3,-1]]
		result = fn.trivial_case(clauses)
		self.assertEqual(result, (False, -1))



if __name__ == '__main__':
    unittest.main()