import unittest
import numpy as np
import src.functions as fn

# TODO: Write test functions for:
# 	sliding_window_crossover
#	roulette_selection ??

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


class TestRemoveUnitVars(unittest.TestCase):

	def test_no_unit_clauses(self):
		clauses = [[1,2,3],[2,1,4],[3,4,2],[4,3,1],[-1,-2,-3],[-2,-1,-4],[-3,-4,-2],[-4,-3,-1]]
		num_vars = 4
		set_vars = [fn.infinite]*num_vars
		new_clauses, set_vars = fn.remove_unit_vars(clauses, set_vars)
		self.assertEqual(new_clauses, clauses)
		self.assertEqual(set_vars, [fn.infinite]*num_vars)

	def test_one_positive_unit_clause(self):
		clauses = [[1,2,3],[2,1,4],[4],[4,3,1],[-1,-2,-3],[-2,-1,-4],[-3,-4,-2],[-4,-3,-1]]
		num_vars = 4
		processed_clauses = [[1,2,3],[-1,-2,-3],[-2,-1],[-3,-2],[-3,-1]]
		pset_vars = [fn.infinite, fn.infinite, fn.infinite, 1]
		set_vars = [fn.infinite]*num_vars
		new_clauses, set_vars = fn.remove_unit_vars(clauses, set_vars)
		self.assertEqual(new_clauses, processed_clauses)
		self.assertEqual(set_vars, pset_vars)

	def test_one_negative_unit_clause(self):
		clauses = [[1,2,3],[2,1,4],[-4],[4,3,1],[-1,-2,-3],[-2,-1,-4],[-3,-4,-2],[-4,-3,-1]]
		num_vars = 4
		processed_clauses = [[1,2,3],[2,1],[3,1],[-1,-2,-3]]
		pset_vars = [fn.infinite, fn.infinite, fn.infinite, 0]
		set_vars = [fn.infinite]*num_vars
		new_clauses, set_vars = fn.remove_unit_vars(clauses, set_vars)
		self.assertEqual(new_clauses, processed_clauses)
		self.assertEqual(set_vars, pset_vars)

	def test_multiple_unit_clause(self):
		clauses = [[1,2,3],[2],[4],[4,3,1],[-1,-2,-3],[-2,-1,-4],[-3,-4,-2],[-4,-3,-1]]
		num_vars = 4
		processed_clauses = []
		pset_vars = [0, 1, 0, 1]
		set_vars = [fn.infinite]*num_vars
		new_clauses, set_vars = fn.remove_unit_vars(clauses, set_vars)
		self.assertEqual(new_clauses, processed_clauses)
		self.assertEqual(set_vars, pset_vars)


class TestRemovePureVars(unittest.TestCase):

	def test_remove_single_pure_positive(self):
		clauses = [[1,2,3],[2,1,4],[3,4,2],[4,3,1],[-2,-3],[-2,-4],[-3,-4,-2],[-4,-3]]
		num_vars = 4
		set_vars = [fn.infinite]*num_vars
		processed_clauses = [[3,4,2],[-2,-3],[-2,-4],[-3,-4,-2],[-4,-3]]
		pset_vars = [1, fn.infinite, fn.infinite, fn.infinite]
		new_clauses, set_vars = fn.remove_pure_vars(clauses, set_vars)
		self.assertEqual(new_clauses, processed_clauses)
		self.assertEqual(set_vars, pset_vars)

	def test_remove_single_pure_negative(self):
		clauses = [[2,3],[2,4],[3,4,2],[4,3],[-1,-2,-3],[-1,-2,-4],[-3,-4,-2],[-4,-3,-1]]
		num_vars = 4
		set_vars = [fn.infinite]*num_vars
		processed_clauses = [[2,3],[2,4],[3,4,2],[4,3],[-3,-4,-2]]
		pset_vars = [0, fn.infinite, fn.infinite, fn.infinite]
		new_clauses, set_vars = fn.remove_pure_vars(clauses, set_vars)
		self.assertEqual(new_clauses, processed_clauses)
		self.assertEqual(set_vars, pset_vars)

	def test_remove_multiple_pure(self):
		clauses = [[2,3],[2,4],[3,4,2],[4,3],[-1,-3],[-1,-4],[-3,-4],[-4,-3,-1]]
		num_vars = 4
		set_vars = [fn.infinite]*num_vars
		processed_clauses = [[4,3],[-3,-4]]
		pset_vars = [0, 1, fn.infinite, fn.infinite]
		new_clauses, set_vars = fn.remove_pure_vars(clauses, set_vars)
		self.assertEqual(new_clauses, processed_clauses)
		self.assertEqual(set_vars, pset_vars)


class TestMaxsatFitness(unittest.TestCase):

	def test_fitness_range(self):
		clauses = [[1],[2],[3],[4]]
		var_arr = [0]*4
		for i in range(len(var_arr)+1):
			if i>0:
				var_arr[i-1] = 1
			fitness = fn.maxsat_fitness(clauses, var_arr)
			self.assertEqual(fitness, i)


class TestSinglePointCrossover(unittest.TestCase):

	def test_single_point_crossover(self):
		p1 = [0,0]
		p2 = [1,1]
		pchildren = [[0,1],[1,0],[1,1],[0,0]]
		children = fn.single_point_crossover(p1,p2)
		self.assertIn(children[0].astype(int).tolist(), pchildren)
		self.assertIn(children[1].astype(int).tolist(), pchildren)


class TestTwoPointCrossover(unittest.TestCase):

	def test_two_point_crossover(self):
		p1 = [0,0,0]
		p2 = [1,1,1]
		pchildren = [[0,1,0],[1,0,1],[0,0,1],[1,1,0],[0,1,1],[1,0,0]]
		children = fn.two_point_crossover(p1,p2)
		self.assertIn(children[0].astype(int).tolist(), pchildren)
		self.assertIn(children[1].astype(int).tolist(), pchildren)


class TestUniformCrossover(unittest.TestCase):

	def test_uniform_crossover(self):
		p1 = [0,0,0,0]
		p2 = [1,1,1,1]
		pchildren = [[0,1,0,1],[1,0,1,0]]
		children = fn.uniform_crossover(p1,p2)
		self.assertIn(children[0], pchildren)
		self.assertIn(children[1], pchildren)


class TestSingleBitFlip(unittest.TestCase):

	def test_single_bit_flip_zero_rate(self):
		pop = [[0,0,0,0]]
		mut_rate = 0
		new_pop = fn.single_bit_flip(pop, mut_rate)
		self.assertEqual(pop[0], new_pop[0])

	def test_single_bit_flip_max_rate(self):
		pop = [[0,0,0,0]]
		flipped_pop = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
		mut_rate = 1
		new_pop = fn.single_bit_flip(pop, mut_rate)
		self.assertIn(new_pop[0], flipped_pop)


class TestMultipleBitFlip(unittest.TestCase):

	def test_multiple_bit_flip_zero_rate(self):
		pop = [[0,0,0,0]]
		mut_rate = 0
		new_pop = fn.single_bit_flip(pop, mut_rate)
		self.assertEqual(pop[0], new_pop[0])

	def test_multiple_bit_flip_max_rate(self):
		pop = [[0,0,0]]
		flipped_pop = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]]
		mut_rate = 1
		new_pop = fn.single_bit_flip(pop, mut_rate)
		self.assertIn(new_pop[0], flipped_pop)


class TestSingleBitGreedy(unittest.TestCase):

	def test_single_bit_greedy_none_better(self):
		pop = [[0,0,0,0]]
		clauses = [[-1,-2,-3],[-2,-3,-4],[-1,-4,-3]]
		new_pop = fn.single_bit_greedy(clauses, pop)
		self.assertEqual(pop[0], new_pop[0])

	def test_single_bit_greedy_one_better(self):
		pop = [[0,0,0,0]]
		clauses = [[-1,-2,-3],[-2,-3,-4],[1,4],[4]]
		flipped_pop = [[1,0,0,0]]
		new_pop = fn.single_bit_greedy(clauses, pop)
		self.assertEqual(flipped_pop[0], new_pop[0])


class TestSingleBitMaxGreedy(unittest.TestCase):

	def test_single_bit_max_greedy_none_better(self):
		pop = [[0,0,0,0]]
		clauses = [[-1,-2,-3],[-2,-3,-4],[-1,-4,-3]]
		new_pop = fn.single_bit_max_greedy(clauses, pop)
		self.assertEqual(pop[0], new_pop[0])

	def test_single_bit_max_greedy_two_better(self):
		pop = [[0,0,0,0]]
		clauses = [[-1,-2,-3],[-2,-3,-4],[1,4]]
		flipped_pop = [[1,0,0,0],[0,0,0,1]]
		new_pop = fn.single_bit_max_greedy(clauses, pop)
		self.assertIn(new_pop[0], flipped_pop)

	def test_single_bit_max_greedy_one_best(self):
		pop = [[0,0,0,0]]
		clauses = [[-1,-2,-3],[-2,-3,-4],[1,4],[4]]
		flipped_pop = [[0,0,0,1]]
		new_pop = fn.single_bit_max_greedy(clauses, pop)
		self.assertEqual(new_pop[0], flipped_pop[0])	


class TestMultipleBitGreedy(unittest.TestCase):

	def test_multiple_bit_greedy_none_better(self):
		pop = [[0,0,0,0]]
		clauses = [[-1,-2,-3],[-2,-3,-4],[-1,-4,-3]]
		new_pop = fn.multiple_bit_greedy(clauses, pop)
		self.assertEqual(pop[0], new_pop[0])

	def test_multiple_bit_greedy_one_better(self):
		pop = [[0,0,0,0]]
		clauses = [[-1,-2,-3],[-2,-3,-4],[1,4]]
		flipped_pop = [[1,0,0,0]]
		new_pop = fn.multiple_bit_greedy(clauses, pop)
		self.assertIn(new_pop[0], flipped_pop)

	def test_multiple_bit_greedy_two_better(self):
		pop = [[0,0,0,0]]
		clauses = [[-1,-2,-3],[-2,-3,-4],[1,4],[4]]
		flipped_pop = [[1,0,0,1]]
		new_pop = fn.multiple_bit_greedy(clauses, pop)
		self.assertEqual(new_pop[0], flipped_pop[0])	


class TestFlipGa(unittest.TestCase):

	def test_flip_ga_none_better(self):
		pop = [[0,0,0,0]]
		clauses = [[-1,-2,-3],[-2,-3,-4],[-1,-4,-3]]
		new_pop = fn.flip_ga(clauses, pop)
		self.assertEqual(pop[0], new_pop[0])

	def test_flip_ga_one_flips(self):
		pop = [[0,0,0,0]]
		clauses = [[-1,-2,-3],[-2,-3,-4],[1,4]]
		flipped_pop = [[1,0,0,0]]
		new_pop = fn.flip_ga(clauses, pop)
		self.assertIn(new_pop[0], flipped_pop)

	def test_flip_ga_two_flips(self):
		pop = [[0,0,0,0]]
		clauses = [[-1,-2,-3],[-2,-3,-4],[1,4],[4]]
		flipped_pop = [[1,0,0,1]]
		new_pop = fn.flip_ga(clauses, pop)
		self.assertEqual(new_pop[0], flipped_pop[0])	

	def test_flip_ga_three_flips(self):
		pop = [[0,0,0,0]]
		clauses = [[-1,-2,-3],[-2,-3,-4],[1,4],[4],[-1]]
		flipped_pop = [[0,0,0,1]]
		new_pop = fn.flip_ga(clauses, pop)
		self.assertEqual(new_pop[0], flipped_pop[0])	


if __name__ == '__main__':
    unittest.main()