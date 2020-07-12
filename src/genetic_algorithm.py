import functions as fn
import time
from datetime import datetime

# TODO: Add function to test all different combinations of hyperparameters (like tensorflow hparams)
# TODO: Add code to run all problems in folder??
############# CONSTANTS #############
infinite = 2**31

############# GA CLASS #############

class GeneticAlgorithm:

	def __init__(self, filename, max_iters=1000, pop_size=100, elitism=0.1, allow_duplicates=True, 
		steady_state_replacement=False, save_to_db = True, max_workers = 1):
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
		self.elitism = int(elitism*pop_size)
		self.log_level = None
		self.ret_cost = True
		self.steady_state_replacement = steady_state_replacement
		self.allow_duplicates = allow_duplicates
		self.save_to_db = save_to_db
		self.db_conn = None
		self.max_workers = max_workers
		if self.save_to_db:
			self.db_conn = fn.get_db_connection(dbname='genetic_algorithm', user='postgres', password='changeme')


	def set_log_level(self, log_level=None):
		self.log_level = log_level

	def set_params(self, initial_pop_func="random", fitness_func="maxsat", selection_func="roulette", 
				crossover_func="single point", mutation_func="single bit", replacement_func="generational",
				mutation_rate=0.1, tournament_size=5, crossover_window_len=0.4, num_individuals=0.5):
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

		self.replacement_func = replacement_func
		if self.steady_state_replacement:
			self.num_individuals = 1
		else:
			self.num_individuals = int(self.pop_size*num_individuals)


		initial_pop_funcs = {
			'random':fn.random_population,
			'binary range':fn.binary_range_population,
			'satisfy clauses':fn.satisfy_clauses_population
		}

		initial_pop_params = {
			'random':(self.num_vars, self.set_vars, self.pop_size, self.allow_duplicates,),
			'binary range':(self.num_vars, self.set_vars, self.pop_size, self.allow_duplicates,),
			'satisfy clauses':(self.num_vars, self.set_vars, self.pop_size,self.clauses, self.allow_duplicates,)
		}

		selection_funcs = {
			'random': fn.random_selection,
			'roulette':fn.roulette_selection,
			'roulette elimination':fn.roulette_selection_with_elimination,
			'rank':fn.rank_selection,
			'tournament':fn.tournament_selection,
			'stochastic':fn.stochastic_universal_sampling_selection,
			'annealed': fn.annealed_selection
		}

		selection_params = {
			'random':[],
			'roulette':[],
			'roulette elimination':[],
			'rank':[],
			'tournament':[tournament_size,],
			'stochastic':[],
			'annealed':[self.max_iters,  0]
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


	def get_filename(self):
		filename = self.filename.split("/")[-1].replace('.cnf','') + "_"
		if self.allow_duplicates == False:
			filename += "set_"
		if self.steady_state_replacement:
			filename += "steady_state_"
		filename += self.initial_pop_func_str.replace(' ','_') + "_"
		filename += self.fitness_func_str.replace(' ','_') + "_"
		filename += self.selection_func_str.replace(' ','_') + "_"
		filename += self.crossover_func_str.replace(' ','_') + "_"
		filename += self.mutation_func_str.replace(' ','_') + "_"
		filename += self.replacement_func.replace(' ','_') + "_"
		filename += datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
		return filename

	def start_ga(self):
		# Generate initial population
		cur_iter = 0
		t_fitness_evals, t_num_flips = 0,0 
		max_fitness = 0

		phenotype_distributions = []
		genotype_distributions = []
		populations = []
		pop_fitness_dict = {}

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

			if self.selection_func_str=="annealed":
				self.sel_params[1] = cur_iter

			num_fitness_evals, num_flips = 0, 0

			if self.log_level=="time":
				t0 = time.time()
			pop_fitness, max_fitness, max_fit_indiv = fn.evaluate_population(population, self.clauses, self.fitness_func, pop_fitness_dict, self.max_workers)
			for i, indiv in enumerate(population):
				if type(indiv) == type(()):
					pop_fitness_dict[indiv]=pop_fitness[i][1]
				else:
					pop_fitness_dict[tuple(indiv)]=pop_fitness[i][1]
			fitness_arr, genes_arr = [], []
			for x in pop_fitness:
				fitness_arr.append(x[1])
				genes_arr.append(''.join([str(y) for y in x[0]]))

			phenotype_distributions.append(fitness_arr)
			genotype_distributions.append(genes_arr)
			populations.append(population)

			num_fitness_evals += len(population) # Since we evaluated the whole population
			if fn.maxsat_solution_found(self.clauses, max_fitness):
				if self.save_to_db:
					fn.add_ga_run_result(self.db_conn, ga_run_id, True, max_fit_indiv, cur_iter, max_fitness, t_fitness_evals, t_num_flips)
					#fn.add_ga_run_population(conn=self.db_conn, ga_run_id=ga_run_id, population=population, observation="Solution found")
					fn.close_db_connection(self.db_conn)

				self.plot_distributions(populations, genotype_distributions, phenotype_distributions)
				return (True, max_fit_indiv, cur_iter, max_fitness, t_fitness_evals, t_num_flips)
			if self.log_level=="time":
				t1 = time.time()
				print ("Calculate population fitness: ", t1-t0)
			

			if self.replacement_func=="mu lambda offspring":
				num_children = len(pop_fitness)*2
			elif self.steady_state_replacement or self.replacement_func in ["random replacement", "parents", "weak parents"]:
				num_children = 2
			else:
				num_children = len(pop_fitness)-self.elitism

			new_children = set()
			while len(new_children) < num_children:
				# Select parents
				if self.log_level=="time":
					t0 = time.time()

				parents = self.selection_func(pop_fitness, num_children, *self.sel_params)

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
					children += prechildren

				if self.log_level=="time":
					t1 = time.time()
					print ("Generate children: ", t1-t0)

				if self.log_level=="time":
					t0 = time.time()
				# Mutate children
				children = fn.mutate_population(children, self.mutation_func, self.mut_params, self.ret_cost, self.max_workers)
				if self.ret_cost:
					num_fitness_evals += children[1]
					num_flips += children[2]
					children = children[0]
				if self.log_level=="time":
					t1 = time.time()
					print ("Mutate population: ", t1-t0)

				if self.log_level=="time":
					t0 = time.time()

				if self.allow_duplicates:
					new_children = children
				else:
					for child in children:
						if tuple(child) not in population:
							new_children.add(tuple(child))
						if len(new_children)>=len(pop_fitness)-self.elitism:
							break


			# Replace population
			if self.replacement_func=="generational":
				population = fn.generational_replacement(self.allow_duplicates, self.elitism, new_children, pop_fitness)
			else:
				children_fitness, _, _ = fn.evaluate_population(new_children, self.clauses, self.fitness_func, pop_fitness_dict, self.max_workers)
				for i, indiv in enumerate(new_children):
					if type(indiv) == type(()):
						pop_fitness_dict[indiv]=children_fitness[i][1]
					else:
						pop_fitness_dict[tuple(indiv)]=children_fitness[i][1]
				num_fitness_evals += len(new_children)

				if self.replacement_func=="mu lambda offspring":
					population = fn.mu_lambda_replacement(self.allow_duplicates, children_fitness, self.pop_size)
				elif self.replacement_func=="mu lambda":
					population = fn.mu_lambda_replacement(self.allow_duplicates, pop_fitness+children_fitness, self.pop_size)
				elif self.replacement_func == 'delete n':
					population = fn.delete_n(self.allow_duplicates, children_fitness, pop_fitness, self.num_individuals, self.selection_func, self.sel_params)
				elif self.replacement_func == 'random replacement':
					population = fn.random_replacement(self.allow_duplicates, children_fitness, pop_fitness)
				elif self.replacement_func == 'parents':
					population = fn.parent_replacement(self.allow_duplicates, children_fitness, parent_pairs, pop_fitness_dict, pop_fitness)
				elif self.replacement_func == 'weak parents':
					population = fn.weak_parent_replacement(self.allow_duplicates, children_fitness, parent_pairs, pop_fitness_dict, pop_fitness)
				
			"""
			sorted_pop = sorted(pop_fitness, key=lambda x: x[1], reverse=True)
			if self.allow_duplicates:
				population = new_children + [x for x,y in sorted_pop[:self.elitism]]
			else:
				population = new_children
				for x,y in sorted_pop[:self.elitism]:
					population.add(x)
			"""
			if self.log_level=="time":
				t1 = time.time()
				print ("Replace population: ", t1-t0)
				print("------------------------------------")

			cur_iter += 1
			t_fitness_evals += num_fitness_evals
			t_num_flips += num_flips

			if cur_iter % 1 == 0:
				if self.log_level=='all':
					print ("Generation {}, Max Fitness {}/{}".format(cur_iter, max_fitness,  len(self.clauses)))
					if self.allow_duplicates:
						pop_set = fn.get_pop_set(population)
						print ("Pop Len {}, Pop Set Len {}".format(len(population), len(pop_set)))
						"""
						if len(pop_set)<10:
							for i, pop in enumerate(pop_set):
								print ("{} - {} copies".format(i, fn.get_indiv_count(population, pop)))
						"""
					else:
						print ("Pop Len {}".format(len(population)))
				if self.save_to_db:
					fn.add_ga_run_generation(conn=self.db_conn, ga_run_id=ga_run_id, generation_num=cur_iter,
						max_fitness=max_fitness, population_length=len(population), population_set_length=len(pop_set),
						num_fitness_evals=num_fitness_evals, num_bit_flips=num_flips)
					fn.add_ga_run_population(conn=self.db_conn, ga_run_id=ga_run_id, population=population)

		if self.save_to_db:
			fn.add_ga_run_result(self.db_conn, ga_run_id, False, None, cur_iter, max_fitness, t_fitness_evals, t_num_flips)
			#fn.add_ga_run_population(conn=self.db_conn, ga_run_id=ga_run_id, population=population, observation="No solution")
			fn.close_db_connection(self.db_conn)

		self.plot_distributions(populations, genotype_distributions, phenotype_distributions)
		return (False, [], cur_iter, max_fitness, t_fitness_evals, t_num_flips)


	def plot_distributions(self, populations, genotype_distributions, phenotype_distributions):
		#gen_distribs = fn.get_genotypic_distribution(genotype_distributions, max_workers=10)
		#norm_gen_distrib = fn.normalize_distributions(gen_distribs, max([max(x) for x in gen_distribs]))
		#norm_phen_distrib = fn.normalize_distributions(phenotype_distributions, len(self.clauses))
		#fn.plot_violin_graph(norm_phen_distrib, "Fitness percentage", "Phenotypic Distributions", self.get_filename(), "phenotype")
		#fn.plot_violin_graph(norm_gen_distrib, "Mean Hamming Distance percentage", "Genotypic Distributions", self.get_filename(), "genotype")
		#fn.plot_means(norm_gen_distrib, norm_phen_distrib, self.get_filename())
		fn.animate_3d_distributions(populations, self.get_filename(), "PCA")
		fn.animate_3d_distributions(populations, self.get_filename(), "SVD")
		fn.animate_3d_distributions(populations, self.get_filename(), "NMF")
		#fn.animate_3d_distributions(populations, self.get_filename(), "FactorAnalysis")
		fn.animate_3d_distributions(populations, self.get_filename(), "KernelPCA")
		#fn.animate_phenotypic_distributions(norm_phen_distrib, self.get_filename())
		#fn.animate_genotypic_distributions(norm_gen_distrib, self.get_filename())

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
filename = 'uf50-01.cnf'
foldername = './data/uf20-91'
log_levels = ['all', 'time']
initial_pop_funcs = ['random', 'binary range', 'satisfy clauses']
fitness_funcs = ['maxsat']
selection_funcs = ['random', 'roulette', 'roulette elimination', 'rank', 'tournament', 'stochastic', 'annealed']
crossover_funcs = ['single point', 'two points', 'sliding window', 'random map', 'uniform']
mutation_funcs = ['single bit', 'multiple bit', 'single bit greedy', 'single bit max greedy', 'multiple bit greedy', 'flip ga']
replacement_funcs = ['generational', 'mu lambda', 'mu lambda offspring', 'delete n', 'random replacement', 'parents', 'weak parents']
max_iters = 200
pop_size = 1000
elitism = 0.1
num_individuals_to_replace = 0.4
mutation_rate = 0.1
crossover_window_len = 0.4
tournament_size = 5

gen_alg = GeneticAlgorithm(filename=foldername+'/'+filename, max_iters=max_iters, pop_size=pop_size, elitism=elitism, 
	steady_state_replacement = False, allow_duplicates=True, save_to_db=True, max_workers=1)
gen_alg.set_params(initial_pop_func=initial_pop_funcs[0], fitness_func=fitness_funcs[0], selection_func=selection_funcs[1], crossover_func=crossover_funcs[1], 
				  mutation_func=mutation_funcs[0], replacement_func=replacement_funcs[0], mutation_rate=mutation_rate, tournament_size=tournament_size, 
				  crossover_window_len=crossover_window_len, num_individuals=num_individuals_to_replace)
gen_alg.set_log_level('all')

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
