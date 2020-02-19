import functions as fn
import numpy as np

# Fitness Functions:
#	maxsat_fitness(clauses, indiv)
#	float_fitness(clauses, indiv) NOT WORKING CORRECTLY

# Selection Functions
#	roulette_selection(population, )

# Crossover Functions:
#	single_point_crossover(p1, p2)
#	two_point_crossover(p1, p2)
#	sliding_window_crossover(p1, p2, wlen)
#	random_map_crossover(p1, p2)
#	uniform_crossove(p1, p2)

# Mutation Functions:
#	single_bit_flip(pop, mut_rate)
#	multiple_bit_flip(pop, mut_rate)
#	single_bit_greedy(pop)
#	single_bit_max_greedy(pop)
#	multi_bit_greedy(pop)
#	flip_ga(pop)

# TODO: Define Genetic Algorithm class with logging and variable parameters
# TODO: Add function to test all different combinations of hyperparameters (like tensorflow hparams)

############# CONSTANTS #############
infinite = 2**31

############# PARAMETERS #############
max_iters = 1000
pop_size = 500
replacement_rate = 0.5
mutation_rate = 0.1
crossover_window_len = 0.4


############# AUXILIARY VARIABLES #############
cur_iters = 0
sol_found = False


############# PROGRAM #############
if __name__ == '__main__':
	num_vars, clauses = fn.read_problem('./data/uf20-91/uf20-02.cnf')
	#num_vars, clauses = read_problem('../data/simple_3sat_problem_u.cnf')
	set_vars = [infinite]*num_vars

	print (clauses)
	print(len(clauses))

	# CNF Simplification
	sol_found, value = fn.trivial_case(clauses)
	if sol_found:
		print ("Solution found!")
		print ("Assign {} to all variables".format(value))
		sys.exit()

	clauses, set_vars = fn.remove_pure_vars(clauses, set_vars)
	clauses, set_vars = fn.remove_unit_vars(clauses, set_vars)

	print(len(clauses))

	# Genetic Algorithm Execution
	population = fn.initial_population(num_vars, set_vars, pop_size)


	solution = []
	while(not sol_found and cur_iters<max_iters):
		pop_fitness = []
		max_fitness = 0
		for pop in population:
			fitness = fn.maxsat_fitness(clauses, pop)
			if fitness >= max_fitness: max_fitness=fitness
			pop_fitness.append([pop, fitness])
			#print (pop, fitness)
			if fn.maxsat_solution_found(clauses, fitness): 
				print(fitness, " Solution found! > ", pop)
				sol_found=True
				solution = pop
				break

		#print(len(pop_fitness))
		# CHANGE when roulette_selection changes
		parents = fn.roulette_selection(pop_fitness, replacement_rate)
		#print(len(pop_fitness))
		children = [] 
		while(len(parents)>=2):
			p1 = np.random.randint(0, len(parents))
			parent_1 = parents.pop(p1)
			p2 = np.random.randint(0, len(parents))
			parent_2 = parents.pop(p2)
			children += fn.single_point_crossover(parent_1, parent_2)
			
		sorted_pop = sorted(pop_fitness, key=lambda x: x[1], reverse=True)

		#print(len(sorted_pop))

		top_pop = [x for x,y in sorted_pop[:int(replacement_rate*len(sorted_pop))]]
		new_pop = children + top_pop

		#print (len(children), len(top_pop), len(new_pop))
		#print (new_pop[:5])

		#print(new_pop[:5])

		population = fn.single_bit_flip(new_pop, mutation_rate)

		#print(population[:5])

		if cur_iters % 1 == 0:
			print ("Generation {}, Population {}, Max Fitness {}".format(cur_iters, len(population), max_fitness))
		cur_iters += 1

	if len(solution)>0:
		print("{} - Raw Solution: {}".format(fn.float_fitness(clauses, solution), solution))
		psol = []
		for num in solution:
			#if max(solution) - num > num - min(solution):
			#	psol.append(0)
			#else:
			#	psol.append(1)
			if num>0.5: psol.append(1)
			else: psol.append(0)
		print("{} - Solution: {}".format(fn.maxsat_fitness(clauses, psol), psol))