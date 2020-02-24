import numpy as np
import random
import sys
from os import listdir

# TODO: Add tests for selection methods
# TODO: Change mutation and crossover functions to prevent set variables from changing
# TODO: Add more functions to generate initial population, so they are better distributed across the solution space
# TODO: Research and add speciation??
############# CONSTANTS #############
infinite = 2**31

############# FUNCTIONS #############

def get_random_int(bot, top):
	return np.random.randint(bot, top)

def read_folder(folder_path):
	files = []
	for file in listdir(folder_path):
		if len(file)>4 and file[-4:]=='.cnf':
			files.append(folder_path+'/'+file)
	return files

def read_problem(filepath):
	with open(filepath, 'r') as fp:
		all_clauses = ""
		for line in fp.readlines():
			if line[0]=='c': 
				continue
			if line[0]=='p':
				num_vars = int(line.split()[2])
				num_clauses = int(line.split()[3])
			else:
				all_clauses += " "+line.replace('\n','')

		clauses = [[] for x in range(num_clauses)]
		clause = 0
		for num in all_clauses.split():
			if num=='0': 
				clause+=1
			elif num=='%': 
				break
			else:
				clauses[clause].append(int(num))

	return num_vars, clauses

############# CNF SIMPLIFICATION #############

def trivial_case(clauses):
	num_pos = 0
	num_neg = 0
	for clause in clauses:
		for num in clause:
			if num > 0: num_pos += 1
			else: num_neg += 1
	if num_neg == 0:
		return 1
	if num_pos == 0:
		return 0
	return -1

def remove_unit_vars(clauses, set_vars):
	"""
		remove_unit_vars
		
		Finds clauses with a single variable in them
		and sets the variable so the clause equals True
	"""
	unit_clauses = 0
	new_clauses = clauses[:]
	for i, clause in enumerate(clauses):
		if len(clause)==1:
			# Clause found
			unit_clauses += 1
			#new_clauses = clauses[:i]+clauses[i+1:]
			new_clauses.pop(i)
			# Set Variable
			if clause[0]>=0: set_vars[clause[0]-1] = 1
			else: set_vars[abs(clause[0])-1] = 0
			for j, new_clause in enumerate(new_clauses):
				if 0-clause[0] in new_clause:
					# Remove negative value from clauses since its False
					new_clauses[j].remove(0-clause[0])
			for new_clause in clauses:
				if clause[0] in new_clause:
					if new_clause in new_clauses:
						# Remove clause since its solved
						new_clauses.remove(new_clause)
			break
	if unit_clauses==0:
		return new_clauses, set_vars
	else:
		return remove_unit_vars(new_clauses, set_vars)

def remove_pure_vars(clauses, set_vars):
	"""
		remove_pure_vars
		
		For every variable it searches through all the clauses,
		if the variable only appears in negated form, it sets it to False
		if the variable only appears in "positive" form, it sets it to True
		if the variable doesn't appear in the clauses, any value will do
	"""
	new_clauses = clauses[:]
	for i in range(len(set_vars)):
		pos_var, neg_var = 0, 0
		for clause in clauses:
			if i+1 in clause: pos_var += 1
			elif -(i+1) in clause: neg_var += 1
		if pos_var > 0 and neg_var == 0:
			set_vars[i] = 1
		elif neg_var > 0 and pos_var == 0:
			set_vars[i] = 0
		elif neg_var == 0 and pos_var == 0:
			# Any value will do, since the variable doesn't appear in the formulas
			set_vars[i] = 0
		if set_vars[i] != infinite:
			for clause in clauses:
				if i+1 in clause:
					new_clauses.remove(clause)
				elif -(i+1) in clause: 
					new_clauses.remove(clause)
	return new_clauses, set_vars

############# GENERATE INITIAL POPULATION #############

def random_population(num_vars, set_vars, pop_size=1000, ptype="bits"):
	population = []
	for p in range(pop_size):
		if ptype=='bits': rpop = np.random.randint(2, size=num_vars)
		elif ptype=='floats': rpop = np.array([random.random() for i in range(num_vars)])
		for j, var in enumerate(set_vars):
			if var != infinite:
				rpop[j] = var
		population.append(rpop)
	return population

############# FITNESS FUNCTIONS #############

def maxsat_fitness(clauses, var_arr):
	# Since CNF formulas are of the shape (x1 OR x2 OR x3) AND (x3 OR -x2 OR -x1)
	# As soon as we find any True value inside a clause that clause is satisfied
	t_clauses = 0
	for clause in clauses:
		for num in clause:
			if num >= 0:
				if var_arr[num-1]==1:
					t_clauses += 1
					break
			elif num <= 0:
				if var_arr[abs(num)-1]==0:
					t_clauses += 1
					break
	return t_clauses

def float_fitness(clauses, var_arr):
	t_res = 1
	for clause in clauses:
		tmp_r = 0
		for num in clause:
			if num<0:
				if var_arr[abs(num)-1] == 0:
					tmp_r += 1
				else:
					tmp_r += 0
			else:
				tmp_r += var_arr[num-1]
		#if tmp_r >= 1: tmp_r = 1
		t_res *= tmp_r
	return t_res

def maxsat_solution_found(clauses, fitness):
	if fitness >= len(clauses): return True
	return False

############# SELECTION FUNCTIONS #############

def roulette_selection_with_elimination(population, num_parents):
	tmp_pop = population[:]
	parents = []
	for _ in range(num_parents):
		if len(tmp_pop)==0: tmp_pop = population[:]
		total_fitness = sum([y for x,y in tmp_pop])
		probabilities = [y/total_fitness if total_fitness>0 else 0 for x,y in tmp_pop]
		if total_fitness>0:
			rnum = random.random()
		else:
			rnum = 0
		sprob = 0
		for i, prob in enumerate(probabilities):
			if rnum >= sprob and rnum <= sprob+prob:
				parents.append(tmp_pop[i])
				tmp_pop.pop(i)
				break
			sprob += prob
	return parents 

def roulette_selection(population, num_parents):
	parents = []
	total_fitness = sum([y for x,y in population])
	probabilities = [y/total_fitness for x,y in population]	
	for _ in range(num_parents):
		rnum = random.random()
		sprob = 0
		for i, prob in enumerate(probabilities):
			if rnum >= sprob and rnum <= sprob+prob:
				parents.append(population[i])
				break
			sprob += prob
	return parents 

def rank_selection(population, num_parents):
	parents = []
	sorted_pop = sorted(population, key=lambda x: x[1])
	pop_rank = [[x, i+1] for i,x in enumerate(sorted_pop)]
	total_rank = int((pop_rank[-1][1]*(pop_rank[-1][1]-1))/2)
	probabilities = [y/total_rank for x,y in pop_rank]
	for _ in range(num_parents):
		rnum = random.random()
		sprob = 0
		for i, prob in enumerate(probabilities):
			if rnum >= sprob and rnum <= sprob+prob:
				parents.append(pop_rank[i][0])
				break
			sprob += prob
	return parents


def tournament_selection(population, num_parents, tournament_size):
	parents = []
	for i in range(num_parents):
		twinner, tfitness = 0, 0
		for i in range(tournament_size):
			rnum = np.random.randint(len(population))
			if population[rnum][1] > tfitness:
				twinner = population[rnum][0]
				tfitness = population[rnum][1]
		parents.append([twinner,tfitness])
	return parents


def boltzmann_tournament_selection(population, num_parents, threshold, control_param):
	parents=[]
	alt_flag = 0
	for i in range(num_parents):
		if alt_flag: alt_flag = 0
		else: alt_flag = 1
		# Choose individual uniformly at random
		indiv_one = population[int(np.random.uniform(0, len(population)))]
		# Choose second individual
		indiv_two = population[int(np.random.uniform(0, len(population)))]
		while indiv_two[1]>=indiv_one[1]-threshold and indiv_two[1]<=indiv_one[1]+threshold:
			indiv_two = population[int(np.random.uniform(0, len(population)))]
		# Choose third individual
		if alt_flag:
			# Strict choice
			big_one, small_one = 0, 0
			if indiv_one[1] > indiv_two[1]:
				big_one = indiv_one[1]
				small_one = indiv_two[1]
			else:
				big_one = indiv_two[1]
				small_one = indiv_one[1]
			indiv_three = population[int(np.random.uniform(0, len(population)))]
			while indiv_three[1]>=small_one-threshold and indiv_three[1]<=big_one+threshold:
				indiv_three = population[int(np.random.uniform(0, len(population)))]
		else:
			# Relaxed choice
			indiv_three = population[int(np.random.uniform(0, len(population)))]
			while indiv_three[1]>=indiv_one[1]-threshold and indiv_three[1]<=indiv_one[1]+threshold:
				indiv_three = population[int(np.random.uniform(0, len(population)))]

		# Anti-acceptance tournament
		chosen_indiv = 0
		prob_2 = np.exp(-indiv_three[1]/control_param)/(np.exp(-indiv_two[1]/control_param)+np.exp(-indiv_three[1]/control_param))
		rnum = random.random()
		if rnum <= prob_2:
			chosen_indiv = indiv_two
		else:
			chosen_indiv = indiv_three

		# Acceptance tournament
		acc_indiv = 0
		prob_1 = np.exp(-indiv_one[1]/control_param)/(np.exp(-indiv_one[1]/control_param)+np.exp(-chosen_indiv[1]/control_param))
		rnum = random.random()
		if rnum <= prob_1:
			acc_indiv = indiv_two
		else:
			acc_indiv = chosen_indiv

		parents.append()
	return parents


############# CROSSOVER FUNCTIONS #############

def single_point_crossover(parent1, parent2, ret_cost=False):
	fitness_evals, bit_flips = 0, 0
	cut_point = np.random.randint(len(parent1))
	children = [
		np.concatenate((parent1[:cut_point],parent2[cut_point:])),
		np.concatenate((parent2[:cut_point],parent1[cut_point:]))
		]
	if ret_cost:
		return children, fitness_evals, bit_flips
	else:
		return children

def two_point_crossover(parent1, parent2, ret_cost=False):
	fitness_evals, bit_flips = 0, 0
	cut_point_1 = np.random.randint(len(parent1)-1)
	cut_point_2 = np.random.randint(cut_point_1+1, len(parent1))
	children = [
		np.concatenate((parent1[:cut_point_1],parent2[cut_point_1:cut_point_2],parent1[cut_point_2:])),
		np.concatenate((parent2[:cut_point_1],parent1[cut_point_1:cut_point_2],parent2[cut_point_2:]))
		]
	if ret_cost:
		return children, fitness_evals, bit_flips
	else:
		return children

def sliding_window_crossover(parent1, parent2, clauses, crossover_window_len=0.4, ret_cost=False):
	fitness_evals, bit_flips = 0, 0
	window_len = int(crossover_window_len*len(parent1))
	max_fitness, max_i = (0,0), (0,0)
	bad_children = [[],[]]
	for i in range(len(parent1)-window_len):
		bad_children[0].append(np.concatenate((parent1[:i],parent2[i:i+window_len],parent1[i+window_len:])))
		bad_children[1].append(np.concatenate((parent2[:i],parent1[i:i+window_len],parent2[i+window_len:])))
		for t in range(2):
			fitness = maxsat_fitness(clauses, bad_children[t][i])
			fitness_evals += 1
			if fitness >= max_fitness[0]:
				max_fitness[t] = fitness
				max_i[t] = i
	if ret_cost:
		return [bad_children[0][max_i[0]], bad_children[1][max_i[1]]], fitness_evals, bit_flips
	else:
		return [bad_children[0][max_i[0]], bad_children[1][max_i[1]]]

def random_map_crossover(parent1, parent2, ret_cost=False):
	fitness_evals, bit_flips = 0, 0
	rand_map = np.random.randint(2, size=len(parent1))
	child_1 = parent1[:]
	child_2 = parent2[:]
	for i, elem in enumerate(rand_map):
		if elem == 0:
			child_1[i] = parent2[i]
			child_2[i] = parent1[i]
			bit_flips += 2
	if ret_cost:
		return [child_1, child_2], fitness_evals, bit_flips
	else:
		return [child_1, child_2]

def uniform_crossover(parent1, parent2, ret_cost=False):
	# Uses alternating bits, maybe change to a normal distribution
	fitness_evals, bit_flips = 0, 0
	child_1 = parent1[:]
	child_2 = parent2[:]
	for i in range(len(parent1)):
		if i%2==0:
			child_1[i] = parent2[i]
			child_2[i] = parent1[i]
			bit_flips += 2
	if ret_cost:
		return [child_1, child_2], fitness_evals, bit_flips
	else:
		return [child_1, child_2]

############# MUTATION FUNCTIONS #############

def single_bit_flip(population, mutation_rate, ret_cost=False):
	fitness_evals, bit_flips = 0, 0
	new_pop = []
	for i, pop in enumerate(population):
		indiv = pop
		rmut = random.random()
		if rmut <= mutation_rate:
			ind = np.random.randint(len(indiv))
			if indiv[ind] == 1: indiv[ind] = 0
			else: indiv[ind] = 1
			bit_flips += 1
		new_pop.append(indiv)
	if ret_cost:
		return new_pop, fitness_evals, bit_flips
	else:
		return new_pop

def multiple_bit_flip(population, mutation_rate, ret_cost=False):
	fitness_evals, bit_flips = 0, 0
	new_pop = []
	for i, pop in enumerate(population):
		indiv = pop
		rmut = random.random()
		if rmut <= mutation_rate:
			num_bits = np.random.randint(len(indiv))
			for x in range(num_bits):
				ind = np.random.randint(len(indiv))
				if indiv[ind] == 1: indiv[ind] = 0
				else: indiv[ind] = 1
				bit_flips += 1
		new_pop.append(indiv)
	if ret_cost:
		return new_pop, fitness_evals, bit_flips
	else:
		return new_pop

def single_bit_greedy(population, clauses, ret_cost=False):
	fitness_evals, bit_flips = 0, 0
	new_pop = []
	for i, pop in enumerate(population):
		ind_fitness = maxsat_fitness(clauses, pop)
		fitness_evals += 1
		for j in range(len(pop)):
			t_indiv = pop
			if t_indiv[j]==1: t_indiv[j]=0
			elif t_indiv[j]==0: t_indiv[j]=1
			bit_flips += 1
			fitness_evals += 1
			if maxsat_fitness(clauses, t_indiv)>ind_fitness:
				break
		new_pop.append(t_indiv)
	if ret_cost:
		return new_pop, fitness_evals, bit_flips
	else:
		return new_pop

def single_bit_max_greedy(population, clauses, ret_cost=False):
	fitness_evals, bit_flips = 0, 0
	new_pop = []
	for i, pop in enumerate(population):
		ind_fitness = maxsat_fitness(clauses, pop)
		fitness_evals += 1
		max_ind, max_fit = 0, 0
		for j in range(len(pop)):
			t_indiv = pop[:]
			if t_indiv[j]==1: t_indiv[j]=0
			elif t_indiv[j]==0: t_indiv[j]=1
			tfit = maxsat_fitness(clauses, t_indiv)
			fitness_evals += 1
			if tfit>ind_fitness and tfit>=max_fit:
				max_ind = t_indiv
				max_fit = tfit
		if max_fit>0:
			new_pop.append(max_ind)
		else:
			new_pop.append(pop)
	if ret_cost:
		return new_pop, fitness_evals, bit_flips
	else:
		return new_pop

def multiple_bit_greedy(population, clauses, ret_cost=False):
	fitness_evals, bit_flips = 0, 0
	new_pop = []
	for i, pop in enumerate(population):
		ind_fitness = maxsat_fitness(clauses, pop)
		indiv = pop[:]
		for j in range(len(indiv)):
			t_indiv = indiv[:]
			new_bit = 1
			if indiv[j]==1: new_bit=0
			t_indiv[j] = new_bit
			t_fitness = maxsat_fitness(clauses, t_indiv)
			if t_fitness > ind_fitness:
				ind_fitness = t_fitness
				indiv = t_indiv
		new_pop.append(indiv)
	if ret_cost:
		return new_pop, fitness_evals, bit_flips
	else:
		return new_pop

def flip_ga(population, clauses, ret_cost=False):
	fitness_evals, bit_flips = 0, 0
	new_pop = []
	for i, pop in enumerate(population):
		indiv = pop[:]
		ind_fitness = maxsat_fitness(clauses, indiv)
		prev_fitness = ind_fitness-1
		while(prev_fitness<ind_fitness):
			prev_fitness = ind_fitness
			for j in range(len(indiv)):
				t_indiv = indiv[:]
				new_bit = 1
				if t_indiv[j]==1: new_bit=0
				t_indiv[j] = new_bit
				t_fitness = maxsat_fitness(clauses, t_indiv)
				if t_fitness > ind_fitness:
					ind_fitness = t_fitness
					indiv = t_indiv
		new_pop.append(indiv)
	if ret_cost:
		return new_pop, fitness_evals, bit_flips
	else:
		return new_pop