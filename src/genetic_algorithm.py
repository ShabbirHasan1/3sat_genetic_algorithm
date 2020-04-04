import functions as fn
import time

# TODO: Add function to test all different combinations of hyperparameters (like tensorflow hparams)
# TODO: Add logging
# TODO: Add code to run all problems in folder and take measurements
# TODO: Add tests for GA Class??
# TODO: Change elitism and tournament_size to work as a percentages of the total population?
# TODO: Add scripts/tool to visualize the data
# TODO: Time the different parts of the program to see where it looses the most amount and multithread that shit
############# CONSTANTS #############
infinite = 2**31

############# GA CLASS #############

class GeneticAlgorithm:

	def __init__(self, filename, max_iters=1000, pop_size=100, 
				elitism=2, save_to_db = True):
		self.filename = filename
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
		self.save_to_db = save_to_db
		self.db_conn = None
		if self.save_to_db:
			self.db_conn = fn.get_db_connection(dbname='genetic_algorithm', user='postgres', password='changeme')


	def set_log_level(self, log_level=None):
		self.log_level = log_level

	def set_params(self, initial_pop_func="random", fitness_func="maxsat", selection_func="roulette", 
				crossover_func="single point", mutation_func="single bit",
				mutation_rate=0.1, tournament_size=5, crossover_window_len=0.4):
		if fitness_func == "maxsat":
			self.fitness_func = fn.maxsat_fitness

		self.mutation_rate=mutation_rate
		self.tournament_size=tournament_size
		self.crossover_window_len = crossover_window_len
		self.initial_pop_func_str = initial_pop_func
		self.fitness_func_str = fitness_func
		self.selection_func_str = selection_func
		self.crossover_func_str = crossover_func
		self.mutation_func_str = mutation_func

		initial_pop_funcs = {
			'random':fn.random_population,
			'binary range':fn.binary_range_population,
			'satisfy clauses':fn.satisfy_clauses_population
		}

		initial_pop_params = {
			'random':(self.num_vars, self.set_vars, self.pop_size,),
			'binary range':(self.num_vars, self.set_vars, self.pop_size,),
			'satisfy clauses':(self.num_vars, self.set_vars, self.pop_size,self.clauses,)
		}

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
			'single point':(self.ret_cost,),
			'two points':(self.ret_cost,),
			'sliding window':(self.clauses, crossover_window_len,self.ret_cost,),
			'random map':(self.ret_cost,),
			'uniform':(self.ret_cost,)
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
			'single bit':(mutation_rate,self.ret_cost,),
			'multiple bit':(mutation_rate,self.ret_cost,),
			'single bit greedy':(self.clauses,self.ret_cost,),
			'single bit max greedy':(self.clauses,self.ret_cost,),
			'multiple_bit_greedy':(self.clauses,self.ret_cost,),
			'flip ga':(self.clauses,self.ret_cost,)
		}

		self.initial_pop_func = initial_pop_funcs[initial_pop_func]
		self.initial_pop_params = initial_pop_params[initial_pop_func]

		self.selection_func = selection_funcs[selection_func]
		self.sel_params = selection_params[selection_func]

		self.crossover_func = crossover_funcs[crossover_func]
		self.cross_params = crossover_params[crossover_func]

		self.mutation_func = mutation_funcs[mutation_func]
		self.mut_params = mutation_params[mutation_func]
		

	def start_ga(self):
		# Generate initial population
		cur_iter = 0
		t_fitness_evals, t_num_flips = 0,0 
		max_fitness = 0

		if self.log_level=="time":
			t0 = time.time()
		
		population = self.initial_pop_func(*self.initial_pop_params)

		if self.log_level=="time":
			t1 = time.time()
			print ("Initial population function: ", t1-t0)

		if self.save_to_db:
			ga_run_id = fn.add_ga_run(conn=self.db_conn, problem=self.filename.split('/')[-1],
				max_iterations=self.max_iters, pop_size=len(population), elitism=self.elitism, 
				fitness_function=self.fitness_func_str, initial_population_function=self.initial_pop_func_str,
				selection_function=self.selection_func_str, crossover_function=self.crossover_func_str, mutation_function=self.mutation_func_str,
				mutation_rate=self.mutation_rate, tournament_size=self.tournament_size, crossover_window_len=self.crossover_window_len)
			fn.add_ga_run_population(conn=self.db_conn, ga_run_id=ga_run_id, population=population, observation="Initial population")
		
		while (self.sol_value==-1 and cur_iter < self.max_iters):

			num_fitness_evals, num_flips = 0, 0

			if self.log_level=="time":
				t0 = time.time()
			pop_fitness, max_fitness, max_fit_indiv = fn.evaluate_population(population, self.clauses, self.fitness_func)
			num_fitness_evals += len(population) # Since we evaluated the whole population
			if fn.maxsat_solution_found(self.clauses, max_fitness):
				if self.save_to_db:
					fn.add_ga_run_result(self.db_conn, ga_run_id, True, max_fit_indiv, cur_iter, max_fitness, t_fitness_evals, t_num_flips)
					fn.close_db_connection(self.db_conn)
				return (True, max_fit_indiv, cur_iter, max_fitness, t_fitness_evals, t_num_flips)
			if self.log_level=="time":
				t1 = time.time()
				print ("Calculate population fitness: ", t1-t0)
			
			# Select parents
			if self.log_level=="time":
				t0 = time.time()

			parents = self.selection_func(pop_fitness, len(pop_fitness)-self.elitism, *self.sel_params)

			if self.log_level=="time":
				t1 = time.time()
				print ("Selection function: ", t1-t0)

			
			#Generate parent pairs
			if self.log_level=="time":
				t0 = time.time()

			parent_pairs = []
			while len(parents)>=2:
				p1 = fn.get_random_int(0, len(parents))
				parent_1 = parents.pop(p1)[0]
				p2 = fn.get_random_int(0, len(parents))
				parent_2 = parents.pop(p2)[0]
				parent_pairs.append([parent_1, parent_2])

			if self.log_level=="time":
				t1 = time.time()
				print ("Generate parent pairs: ", t1-t0)

			if self.log_level=="time":
				t0 = time.time()
			# Generate children through crossover
			children = []
			for parent_pair in parent_pairs:
				prechildren = self.crossover_func(parent_pair, *self.cross_params)
				if self.ret_cost:
					num_fitness_evals += prechildren[1]
					num_flips += prechildren[2]
					prechildren = prechildren[0]
				children+=prechildren

			if self.log_level=="time":
				t1 = time.time()
				print ("Generate children: ", t1-t0)

			if self.log_level=="time":
				t0 = time.time()
			# Mutate children
			children = self.mutation_func(children, *self.mut_params)
			if self.ret_cost:
				num_fitness_evals += children[1]
				num_flips += children[2]
				children = children[0]
			if self.log_level=="time":
				t1 = time.time()
				print ("Mutate population: ", t1-t0)

			if self.log_level=="time":
				t0 = time.time()
			# Replace population
			sorted_pop = sorted(pop_fitness, key=lambda x: x[1], reverse=True)
			population = children + [x for x,y in sorted_pop[:self.elitism]]
			if self.log_level=="time":
				t1 = time.time()
				print ("Replace population: ", t1-t0)

			cur_iter += 1
			t_fitness_evals += num_fitness_evals
			t_num_flips += num_flips

			if cur_iter % 1 == 0:
				if self.log_level=='all':
					print ("Generation {}, Max Fitness {}".format(cur_iter, max_fitness))
					print ("Pop Len {}, Pop Set Len {}".format(len(population), fn.get_pop_set_len(population)))
				if self.save_to_db:
					fn.add_ga_run_generation(conn=self.db_conn, ga_run_id=ga_run_id, generation_num=cur_iter,
						max_fitness=max_fitness, population_length=len(population), population_set_length=fn.get_pop_set_len(population),
						num_fitness_evals=num_fitness_evals, num_bit_flips=num_flips)

		if self.save_to_db:
			fn.add_ga_run_result(self.db_conn, ga_run_id, False, None, cur_iter, max_fitness, t_fitness_evals, t_num_flips)
			fn.close_db_connection(self.db_conn)
		return (False, [], cur_iter, max_fitness, t_fitness_evals, t_num_flips)

	def get_run_average(self, num_runs=10):
		succ_runs, fail_runs = 0,0
		prev_ret = self.ret_cost
		self.ret_cost = True
		total_iters, total_fitness_evals, total_flips = [],[],[]
		for i in range(num_runs):
			sol_found, sol, num_iters, fitness, num_fitness_evals, num_flips = self.start_ga()
			if sol_found: 
				succ_runs += 1
				total_iters += [num_iters]
				total_fitness_evals += [num_fitness_evals]
				total_flips += [num_flips]

		print ("-----------Problem-----------")
		print ("Filename: ", self.filename.split('/')[-1])
		print ("Location: ", '/'.join(self.filename.split('/')[:-1])+'/')
		print ("----------Parameters---------")
		print ("Fitness function: ", self.fitness_func_str)
		print ("Initial population function: ",self.initial_pop_func_str)
		sel_str = "Selection function: "+self.selection_func_str
		if self.selection_func_str=='tournament':
			sel_str += '(tournament_size = '+str(self.sel_params[0])+')'
		print (sel_str)
		cross_str = "Crossover function: " + self.crossover_func_str
		if self.crossover_func_str=='sliding window':
			cross_str += '(window_len = '+str(self.cross_params[1])+')'
		print (cross_str)
		mut_str = "Mutation function: " + self.mutation_func_str
		if self.mutation_func_str in ['single bit', 'multiple bit']:
			mut_str += '(mutation_rate = '+str(self.mut_params[0])+')'
		print (mut_str)
		print ("------------Results----------")
		print ("Success Rate: ", succ_runs/num_runs)
		print ("Average num iterations: ", sum(total_iters)/num_runs)
		print ("Average num fitness evaulations: ", sum(total_fitness_evals)/num_runs)
		print ("Average num flips: ", sum(total_flips)/num_runs)
		print ("-----------------------------")
		self.ret_cost = prev_ret


############# PROGRAM #############
filename = 'uf200-01.cnf'
foldername = './data/uf20-91'
log_levels = ['all']
initial_pop_funcs = ['random', 'binary range', 'satisfy clauses']
fitness_funcs = ['maxsat']
selection_funcs = ['roulette', 'roulette elimination', 'rank', 'tournament', 'boltzmann']
crossover_funcs = ['single point', 'two points', 'sliding window', 'random map', 'uniform']
mutation_funcs = ['single bit', 'multiple bit', 'single bit greedy', 'single bit max greedy', 'multiple bit greedy', 'flip ga']
max_iters = 10000
pop_size = 100
elitism = 2
mutation_rate = 0.1
crossover_window_len = 0.4
tournament_size = 5
boltzmann_threshold = "Ni idea"
boltzmann_control_param = "Ni idea"



gen_alg = GeneticAlgorithm(filename=foldername+'/'+filename, max_iters=max_iters, pop_size=pop_size, elitism=elitism, save_to_db=False)
gen_alg.set_params(initial_pop_func=initial_pop_funcs[2], fitness_func=fitness_funcs[0], selection_func=selection_funcs[3], crossover_func=crossover_funcs[1], 
				  mutation_func=mutation_funcs[2],mutation_rate=mutation_rate, tournament_size=tournament_size, 
				  crossover_window_len=crossover_window_len)
gen_alg.set_log_level('time')

sol_found, sol, iteration, fitness, num_fitness_evals, num_flips = gen_alg.start_ga()

if sol_found:
	print ("Solution found in iteration {} > {}".format(iteration, sol))
	print ("Num fitness evals: {}, Num bit flips: {}".format(num_fitness_evals, num_flips))
else:
	print ("Solution not found in {} iterations".format(iteration))
	print ("Max fitness found: {}/{}".format(fitness, len(gen_alg.clauses)))

#gen_alg.get_run_average(10)

#files = fn.read_folder(foldername)
#print (len(files))
