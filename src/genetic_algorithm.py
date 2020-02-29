import functions as fn

# TODO: Define Genetic Algorithm class with logging and variable parameters
# TODO: Add function to test all different combinations of hyperparameters (like tensorflow hparams)
# TODO: Add result logging
# TODO: Add code to run all problems in folder and take measurements
# TODO: Add tests for GA Class
############# CONSTANTS #############
infinite = 2**31

############# GA CLASS #############

class GeneticAlgorithm:

	def __init__(self, filename, max_iters=1000, pop_size=100, 
				elitism=2):
		self.num_vars, self.clauses = fn.read_problem(filename)
		self.set_vars = [infinite]*self.num_vars
		self.sol_value = fn.trivial_case(self.clauses)
		self.clauses, self.set_vars = fn.remove_pure_vars(self.clauses, self.set_vars)
		self.clauses, self.set_vars = fn.remove_unit_vars(self.clauses, self.set_vars)

		self.fitness_func = fn.maxsat_fitness
		self.selection_func = fn.roulette_selection
		self.selection_params = ()
		self.crossover_func = fn.single_point_crossover
		self.crossover_params = ()
		self.mutation_func = fn.single_bit_flip
		self.mutation_params = (0.1,)

		self.max_iters = max_iters
		self.pop_size = pop_size
		self.elitism = elitism
		self.log_level = None
		self.ret_cost = True

	def set_log_level(self, log_level=None):
		self.log_level = log_level

	def set_params(self,fitness_func="maxsat", selection_func="roulette", 
				crossover_func="single point", mutation_func="single bit",
				mutation_rate=0.1, tournament_size=5, crossover_window_len=0.4):
		if fitness_func == "maxsat":
			self.fitness_func = fn.maxsat_fitness

		selection_funcs = {
			'roulette':fn.roulette_selection,
			'roulette elimination':fn.roulette_selection_with_elimination,
			'rank':fn.rank_selection,
			'tournament':fn.tournament_selection,
			'boltzmann':fn.boltzmann_tournament_selection
		}

		selection_params = {
			'roulette':(),
			'roulette elimination':(),
			'rank':(),
			'tournament':(tournament_size,),
			'boltzmann':()
		}


		crossover_funcs = {
			'single point':fn.single_point_crossover,
			'two points':fn.two_point_crossover,
			'sliding window':fn.sliding_window_crossover,
			'random map':fn.random_map_crossover,
			'uniform':fn.uniform_crossover
		}

		crossover_params = {
			'single point':(),
			'two points':(),
			'sliding window':(self.clauses, crossover_window_len,),
			'random map':(),
			'uniform':()
		}


		mutation_funcs = {
			'single bit':fn.single_bit_flip,
			'multiple bit':fn.multiple_bit_flip,
			'single bit greedy':fn.single_bit_greedy,
			'single bit max greedy':fn.single_bit_max_greedy,
			'multiple_bit_greedy':fn.multiple_bit_greedy,
			'flip ga':fn.flip_ga
		}
		mutation_params = {
			'single bit':(mutation_rate,),
			'multiple bit':(mutation_rate,),
			'single bit greedy':(self.clauses,),
			'single bit max greedy':(self.clauses,),
			'multiple_bit_greedy':(self.clauses,),
			'flip ga':(self.clauses,)
		}


		self.selection_func = selection_funcs[selection_func]
		self.sel_params = selection_params[selection_func]

		self.crossover_func = crossover_funcs[crossover_func]
		self.cross_params = crossover_params[crossover_func]

		self.mutation_func = mutation_funcs[mutation_func]
		self.mut_params = mutation_params[mutation_func]
		

	def start_ga(self):
		# Generate initial population
		cur_iter = 0
		num_fitness_evals, num_flips = 0, 0
		max_fitness = 0
		population = fn.random_population(self.num_vars, self.set_vars, self.pop_size)
		while (self.sol_value==-1 and cur_iter < self.max_iters):
			# Evaluate population
			pop_fitness = []
			max_fitness = 0
			for pop in population:
				fitness = self.fitness_func(self.clauses, pop)
				num_fitness_evals += 1
				if fitness >= max_fitness:
					max_fitness = fitness
				pop_fitness.append([pop, fitness])
				if fn.maxsat_solution_found(self.clauses, fitness):
					return (True, pop, cur_iter, fitness, num_fitness_evals, num_flips)

			# Select parents
			parents = self.selection_func(pop_fitness, len(pop_fitness)-elitism, *self.sel_params)

			# Generate children through crossover and mutation
			children = []
			while len(parents)>=2:
				p1 = fn.get_random_int(0, len(parents))
				parent_1 = parents.pop(p1)[0]
				p2 = fn.get_random_int(0, len(parents))
				parent_2 = parents.pop(p2)[0]
				prechildren = self.crossover_func(parent_1, parent_2, *self.cross_params, self.ret_cost)
				if self.ret_cost:
					num_fitness_evals += prechildren[1]
					num_flips += prechildren[2]
					prechildren = prechildren[0]
				prechildren = self.mutation_func(prechildren, *self.mut_params, self.ret_cost)
				if self.ret_cost:
					num_fitness_evals += prechildren[1]
					num_flips += prechildren[2]
					prechildren = prechildren[0]
				children += prechildren

			# Replace population
			sorted_pop = sorted(pop_fitness, key=lambda x: x[1], reverse=True)
			population = children + [x for x,y in sorted_pop[:elitism]]

			cur_iter += 1

			#if cur_iter % 1 == 0:
			#	print ("Generation {}, Population {}, Max Fitness {}".format(cur_iter, len(population), max_fitness))
		return (False, [], cur_iter, max_fitness, num_fitness_evals, num_flips)

	def get_run_average(self, num_runs=10):
		succ_runs, fail_runs = 0,0
		total_iters, total_fitness_evals, total_flips = [],[],[]
		for i in range(num_runs):
			sol_found, sol, num_iters, fitness, num_fitness_evals, num_flips = self.start_ga()
			if sol_found: 
				succ_runs += 1
				total_iters += [num_iters]
				total_fitness_evals += [num_fitness_evals]
				total_flips += [num_flips]

		print ("Success Rate: ", succ_runs/num_runs)
		print ("Average num iterations: ", sum(total_iters)/num_runs)
		print ("Average num fitness evaulations: ", sum(total_fitness_evals)/num_runs)
		print ("Average num flips: ", sum(total_flips)/num_runs)


############# PROGRAM #############
filename = './data/uf20-91/uf20-01.cnf'
foldername = './data/uf20-91'
fitness_funcs = ['maxsat']
selection_funcs = ['roulette', 'roulette elimination', 'rank', 'tournament', 'boltzmann']
crossover_funcs = ['single point', 'two points', 'sliding window', 'random map', 'uniform']
mutation_funcs = ['single bit', 'multiple bit', 'single bit greedy', 'single bit max greedy', 'multiple bit greedy', 'flip ga']
max_iters = 1000
pop_size = 100
elitism = 2
mutation_rate = 0.1
crossover_window_len = 0.4
tournament_size = 5
boltzmann_threshold = "Ni idea"
boltzmann_control_param = "Ni idea"



gen_alg = GeneticAlgorithm(filename=filename, max_iters=max_iters, pop_size=pop_size, elitism=elitism)
gen_alg.set_params(fitness_func=fitness_funcs[0], selection_func=selection_funcs[3], crossover_func=crossover_funcs[1], 
				  mutation_func=mutation_funcs[1],mutation_rate=mutation_rate, tournament_size=tournament_size, 
				  crossover_window_len=crossover_window_len)

sol_found, sol, iteration, fitness, num_fitness_evals, num_flips = gen_alg.start_ga()

if sol_found:
	print ("Solution found in iteration {} > {}".format(iteration, sol))
	print ("Num fitness evals: {}, Num bit flips: {}".format(num_fitness_evals, num_flips))
else:
	print ("Solution not found in {} iterations".format(iteration))
	print ("Max fitness found: ", fitness)

#gen_alg.get_run_average(10)

files = fn.read_folder(foldername)
print (len(files))
