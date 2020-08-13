import numpy as np
import random
import sys
import psycopg2
from datetime import datetime as dt
import os
from concurrent import futures
import itertools
import seaborn as sns
import matplotlib as mpl
mpl.rc('figure', max_open_warning = 0)
import matplotlib.pyplot as plt
import glob
import re
import subprocess
import scipy.stats as sp
import hashlib
from multiprocessing import Pool
import pandas as pd
import sklearn.decomposition as sd


# TODO: Explain why we rejected float fitness
# TODO: Explain why boltzmann selection failed (we loose too much phenotypic diversity fast, and so it becomes impossible to find contestants with
# a phenotypical distance equal to the boltzman_threshold amongst them)
############# CONSTANTS #############
infinite = 2**31
levene_constant = 0.05
fp_in = "./gifs/image_*.png"
fp_out = "./gifs/"


############# FUNCTIONS #############

def get_random_int(bot, top):
	return np.random.randint(bot, top)

def read_folder(folder_path):
	files = []
	for file in os.listdir(folder_path):
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

def get_pop_set(population):
	unique_pop = set()
	for indiv in population:
		unique_pop.add(str(indiv))
	return unique_pop

def get_indiv_count(population, indiv):
	str_pop = [str(x) for x in population]
	return str_pop.count(indiv)

def append_to(dest_list, elem):
	if type(dest_list) == type(set()):
		dest_list.add(elem)
	elif type(dest_list) == type([]):
		dest_list.append(elem)
	return dest_list

def animate_phenotypic_distributions(population_distributions, filename):
	min_f, max_f = 2**31, 0
	for distribution in population_distributions:
		max_d, min_d = max(distribution), min(distribution)
		if max_d > max_f: max_f = max_d
		if min_d < min_f: min_f = min_d

	for i, distribution in enumerate(population_distributions):
		fig, ax = plt.subplots()
		plot = sns.distplot(distribution, bins=20, norm_hist=True, ax=ax)
		ax.set_xlim(min_f, max_f)
		#ax.set_ylim(0, 100)
		plt.title("Generation {}".format(i))
		plt.xlabel("Fitness")
		plt.ylabel("Bin frequency")

		textstr = ['variance=%.3f' % (np.var(distribution), )]
		if i>0 and np.var(distribution)>0:
			statistic, result = sp.levene(distribution, population_distributions[i-1])
			# if levene result (p value) < 0.05 there is a difference between the variance of the populations
			textstr.append(r'levene p=%.3f' % (result))
			ttest,pval = sp.ttest_ind(distribution, population_distributions[i-1], equal_var=result > levene_constant)
			textstr.append(r't-test p=%.3f' % (pval))
			# if pval < threshold (0.05, 0.1) reject the null hypothesis of equal averages

		textstr = '\n'.join(textstr)
		# these are matplotlib.patch.Patch properties0
		props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

		# place a text box in upper left in axes coords
		ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
			verticalalignment='top', bbox=props)

		plt.savefig("gifs/image_{}.png".format(i))

	file_out = fp_out + 'phenotype_' + filename + '.gif'
	filenames = sorted(glob.glob( fp_in),key=lambda x:float(re.findall("([0-9]+?)\.png",x)[0]))
	subprocess.call("convert -delay 50 " + " ".join(filenames) + " -loop 5 " + file_out, shell=True)
	for f in sorted(glob.glob(fp_in)):
		os.remove(f)


def animate_3d_distributions(populations, filename, dimensionality_reduction):

	if dimensionality_reduction == "PCA":
		pca = sd.PCA(n_components=3)
	elif dimensionality_reduction == "SVD":
		pca = sd.TruncatedSVD(3)
	elif dimensionality_reduction == "NMF":
		pca = sd.NMF(n_components=3, max_iter=100000)
	elif dimensionality_reduction == "FactorAnalysis":
		pca = sd.FactorAnalysis(n_components=3)
	elif dimensionality_reduction == "KernelPCA":
		pca = sd.KernelPCA(n_components=3)

	pandas_dfs = []
	max_ind = [0,0,0] 
	min_ind = [2**31,2**31,2**31]
	for i, distribution in enumerate(populations):
		principalComponents = pca.fit_transform(distribution)
		dataframe = pd.DataFrame(data = principalComponents, columns=['principal component 1', 
			'principal component 2', 'principal component 3'])
		pandas_dfs.append(dataframe)
		if dataframe.min()[0]<min_ind[0]: min_ind[0] = dataframe.min()[0]
		if dataframe.min()[1]<min_ind[1]: min_ind[1] = dataframe.min()[1]
		if dataframe.min()[2]<min_ind[2]: min_ind[2] = dataframe.min()[2]

		if dataframe.max()[0]>max_ind[0]: max_ind[0] = dataframe.max()[0]
		if dataframe.max()[1]>max_ind[1]: max_ind[1] = dataframe.max()[1]
		if dataframe.max()[2]>max_ind[2]: max_ind[2] = dataframe.max()[2]


	for i, principalDf in enumerate(pandas_dfs):

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.set_ylim(min_ind[1]-1, max_ind[1]+1)
		ax.set_xlim(min_ind[0]-1, max_ind[0]+1)
		ax.set_zlim(min_ind[2]-1, max_ind[2]+1)
		plt.title("Generation {}".format(i))
		ax.set_xlabel("Principal Component 1")
		ax.set_ylabel("Principal Component 2")
		ax.set_zlabel("Principal Component 3")

		ax.scatter(principalDf['principal component 1'],principalDf['principal component 2'],
           principalDf['principal component 3'], c='r', marker='o')

		plt.savefig("gifs/image_{}.png".format(i))

	file_out = fp_out + dimensionality_reduction + '_analysis_' + filename + '.gif'
	filenames = sorted(glob.glob( fp_in),key=lambda x:float(re.findall("([0-9]+?)\.png",x)[0]))
	subprocess.call("convert -delay 50 " + " ".join(filenames) + " -loop 5 " + file_out, shell=True)
	for f in sorted(glob.glob(fp_in)):
		os.remove(f)

def plot_violin_graph(pop_distributions, y_axis, title, filename, dist_type):
	fig, ax = plt.subplots(figsize=(30,15))
	ax.violinplot(pop_distributions, showmeans=True, widths=0.7)
	plt.title(title)
	plt.xlabel("Generation number")
	plt.ylabel(y_axis)
	file_out = fp_out + dist_type + '_' + filename + '.png'
	plt.savefig(file_out)

def plot_means(genotype_distributions, phenotype_distributions, filename):
	fig, ax = plt.subplots(figsize=(30,15))

	ind_arr = [x for x in range(len(genotype_distributions))]

	gen_mean, gen_dev = get_mean_distributions(genotype_distributions)
	gen_lower, gen_upper = [], []
	for i in range(len(gen_mean)):
		gen_lower += [gen_mean[i]-gen_dev[i]]
		gen_upper += [gen_mean[i]+gen_dev[i]]
	
	phen_mean, phen_dev = get_mean_distributions(phenotype_distributions)
	phen_lower, phen_upper = [], []
	for i in range(len(phen_mean)):
		phen_lower += [phen_mean[i]-phen_dev[i]]
		phen_upper += [phen_mean[i]+phen_dev[i]]
	
	ax.fill_between(ind_arr, gen_upper, gen_lower, color="lightcyan")
	ax.plot(ind_arr, gen_mean, color="blue", lw=2, label="Genotype mean")
	#plt.plot(ind_arr, gen_mean, color ='g')

	ax.fill_between(ind_arr, phen_upper, phen_lower, color="lightyellow", alpha=0.7)
	ax.plot(ind_arr, phen_mean, color="brown", lw=2, label="Phenotype mean")

	ax.set_title("Genotypic vs Phenotypic mean evolution")
	ax.set_xlabel("Generation number")
	ax.set_ylabel("Mean percentage")
	ax.legend()
	file_out = fp_out + 'bbands_' + filename + '.png'
	plt.savefig(file_out)

def get_mean_distributions(pop_distributions):
	mean_dists = []
	deviation_dists = []
	for distrib in pop_distributions:
		mean_dists += [np.mean(distrib)]
		deviation_dists += [np.std(distrib)]
	return mean_dists, deviation_dists

def normalize_distributions(pop_distributions, max_value):
	normalized_dists = []
	for distribution in pop_distributions:
		new_dist = []
		for indiv in distribution:
			new_dist += [indiv/max_value]
		normalized_dists += [new_dist]
	return normalized_dists


def hamming_distance(chain1, chain2):
	chaine1 = hashlib.md5(chain1.encode()).hexdigest()
	chaine2 = hashlib.md5(chain2.encode()).hexdigest()
	return sum(c1 != c2 for c1, c2 in zip(chaine1, chaine2))

def get_hamming_distances(population, hamming_dict=None):
	pop = []
	for indiv in population:
		dist = 0
		for indiv2 in population:
			if indiv!=indiv2:
				if hamming_dict != None:
					try:
						ld = hamming_dict[tuple(indiv)+tuple(indiv2)]
					except:
						ld = hamming_distance(indiv, indiv2)
						hamming_dict[tuple(indiv)+tuple(indiv2)] = ld
				else:
					ld = hamming_distance(indiv, indiv2)
				dist += ld
		dist /= len(population)
		pop.append(dist)

	if hamming_dict != None:
		return pop, hamming_dict
	return pop


def get_genotypic_distribution(pop_distributions, max_workers=1):
	genotype_distributions = []

	if max_workers == 1:
		hamming_dict = {}
		for population in pop_distributions:
			pop, hamming_dict = get_hamming_distances(population, hamming_dict)
			genotype_distributions.append(pop)
	
	else:
		for i in range(0,len(pop_distributions), max_workers):
			p = Pool(processes=max_workers)
			data = p.map(get_hamming_distances, pop_distributions[i:i+10])
			p.close()
			genotype_distributions += data

	return genotype_distributions

def animate_genotypic_distributions(genotype_distributions, filename):
	min_f, max_f = 2**31, 0
	for pop in genotype_distributions:
		tmin = min(pop)
		tmax = max(pop)
		if tmin < min_f: min_f = tmin
		if tmax > max_f: max_f = tmax

	for i, distribution in enumerate(genotype_distributions):
		fig, ax = plt.subplots()
		plot = sns.distplot(distribution, bins=20, ax=ax)
		ax.set_xlim(min_f,max_f)
		#ax.set_ylim(0, len(distribution))
		plt.title("Generation {}".format(i))
		plt.xlabel("Hamming distance")
		plt.ylabel("Bin frequency")

		textstr = ['variance=%.3f' % (np.var(distribution), )]
		if i>0 and np.var(distribution)>0:
			statistic, result = sp.levene(distribution, genotype_distributions[i-1])
			# if levene result (p value) < 0.05 there is a difference between the variance of the populations
			textstr.append(r'levene p=%.3f' % (result))
			ttest,pval = sp.ttest_ind(distribution, genotype_distributions[i-1], equal_var=result > levene_constant)
			textstr.append(r't-test p=%.3f' % (pval))
			# if pval < threshold (0.05, 0.1) reject the null hypothesis of equal averages

		textstr = '\n'.join(textstr)
		# these are matplotlib.patch.Patch properties0
		props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

		# place a text box in upper left in axes coords
		ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
			verticalalignment='top', bbox=props)
		plt.savefig("gifs/image_{}.png".format(i))

	file_out = fp_out + 'genotype_' + filename + '.gif'
	filenames = sorted(glob.glob( fp_in),key=lambda x:float(re.findall("([0-9]+?)\.png",x)[0]))
	subprocess.call("convert -delay 50 " + " ".join(filenames) + " -loop 5 " + file_out, shell=True)
	for f in sorted(glob.glob(fp_in)):
		os.remove(f)



############# DATABASE HELPER FUNCTIONS #############

def get_db_connection(dbname='genetic_algorithm', user='pyuser', password='123456'):
	conn, cursor = None, None
	try:
		conn = psycopg2.connect(host='postgres_tfg',database=dbname, user=user, password=password)
	except (Exception, psycopg2.DatabaseError) as error:
		print(error)

	return conn

def close_db_connection(conn):
	try:
		if conn is not None:
			conn.close()
	except (Exception, psycopg2.DatabaseError) as error:
		print(error)

def add_ga_run(conn, problem, max_iterations, pop_size, elitism, selection_function, crossover_function, 
	mutation_function,	mutation_rate, tournament_size, crossover_window_len, population_replacement_function, 
	num_individuals, truncation_proportion, num_clauses):
	new_id = -1
	try:	
		with conn:
			cursor = conn.cursor()
			sql_string = "INSERT INTO ga_run (created_timestamp, problem, max_iterations, pop_size, elitism, \
					 selection_function, crossover_function, mutation_function,\
					 population_replacement_function, mutation_rate, tournament_size, crossover_window_len, \
					 num_individuals, truncation_proportion, num_clauses) VALUES \
					 (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;"
			cursor.execute(sql_string, (dt.now(), problem, max_iterations, pop_size, elitism, selection_function, crossover_function, mutation_function, population_replacement_function, mutation_rate, tournament_size, crossover_window_len, num_individuals, truncation_proportion, num_clauses))
			new_id = cursor.fetchone()[0]

	except (Exception, psycopg2.DatabaseError) as error:
		print (error)

	return new_id

def add_ga_run_result(conn, ga_run_id, sol_found, solution, num_iterations, max_fitness, num_fitness_evals, num_bit_flips):
	try:
		with conn:
			cursor = conn.cursor()
			sql_string = "UPDATE ga_run SET updated_timestamp = %s, sol_found = %s, solution = %s, \
				num_iterations = %s, max_fitness = %s, num_fitness_evals = %s, num_bit_flips = %s WHERE id = %s"
			cursor.execute(sql_string, (dt.now(), sol_found, solution, num_iterations, 
				max_fitness, num_fitness_evals, num_bit_flips, ga_run_id))
	except (Exception, psycopg2.DatabaseError) as error:
		print (error)

def add_ga_run_generation(conn, ga_run_id, generation_num, max_fitness, population_length, population_set_length, num_fitness_evals, num_bit_flips, num_clauses, fitness_array):
	mean = float(numpy.mean(fitness_array))
    median = float(numpy.median(fitness_array))
    mode = stats.mode(fitness_array)
    mode_value = float(mode[0][0])
    mode_count = int(mode[1][0])
    deviation = float(numpy.std(fitness_array))
	try:
		with conn:
			cursor = conn.cursor()
			sql_string = "INSERT INTO ga_run_generations (time_stamp, ga_run_id, generation_num, max_fitness, \
						 population_length, population_set_length, num_fitness_evals, num_bit_flips, num_clauses, \
						 mean, median, mode, mode_count, standard_deviation) VALUES \
						 (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
			cursor.execute(sql_string, (dt.now(), ga_run_id, generation_num, max_fitness, population_length,
						  population_set_length, num_fitness_evals, num_bit_flips, num_clauses, mean, median, mode_value,
						  mode_count, deviation))
	except (Exception, psycopg2.DatabaseError) as error:
		print (error)

def add_ga_run_population(conn, ga_run_id, population, ga_run_generation_id=None, observation=None):
	try:
		with conn:
			cursor = conn.cursor()
			sql_string = "INSERT INTO ga_run_population (ga_run_id, ga_run_generation_id, population, observations) VALUES (%s, %s, %s, %s)"
			cursor.execute(sql_string, (ga_run_id, ga_run_generation_id, population, observation))
	except (Exception, psycopg2.DatabaseError) as error:
		print (error)


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

def random_population(num_vars, set_vars, pop_size, allow_duplicates):
	if allow_duplicates:
		population = []
	else:
		population = set()
	while(len(population)<pop_size):
		for p in range(pop_size):
			rpop = np.random.randint(2, size=num_vars)
			for j, var in enumerate(set_vars):
				if var != infinite:
					rpop[j] = var
			if allow_duplicates:
				population.append(rpop.tolist())
			else:
				population.add(tuple(rpop.tolist()))
	return population

def satisfy_clauses_population(num_vars, set_vars, pop_size, clauses, allow_duplicates):
	#NOT USED
	if allow_duplicates:
		population = []
	else:
		population = set()
	while len(population)<pop_size:
		for p in range(pop_size):
			indiv = [0]*num_vars
			for clause in clauses:
				rind = np.random.randint(len(clause))
				if clause[rind]>=0: indiv[clause[rind]-1] = 1
				else: indiv[abs(clause[rind])-1] = 0
			if allow_duplicates:
				population.append(indiv)
			else:
				population.add(indiv)
	return population

############# EVALUATE POPULATION #############
def evaluate_population(population, clauses, fitness_function, fitness_dict, max_workers=1):
	pop_fitness = []
	max_fitness = 0
	max_fit_indiv = None
	if max_workers == 1:
		for pop in population:
			try:
				if type(pop)==type(()):
					fitness = fitness_dict[pop]
				else:
					fitness = fitness_dict[tuple(pop)]
			except:
				fitness = fitness_function(clauses, pop)
			if fitness >= max_fitness:
				max_fitness = fitness
				max_fit_indiv = pop
			pop_fitness.append([pop, fitness])
	else:
		ex = futures.ProcessPoolExecutor(max_workers=max_workers)
		results = ex.map(fitness_function, itertools.repeat(clauses, len(population)), population)
		for i, fitness in enumerate(results):
			pop_fitness.append([population[i], fitness])
			if fitness >= max_fitness:
				max_fitness = fitness
				max_fit_indiv = population[i]
	return pop_fitness, max_fitness, max_fit_indiv


############# FITNESS FUNCTIONS #############
fitness_dict = {}
def maxsat_fitness(clauses, var_arr):
	# Since CNF formulas are of the shape (x1 OR x2 OR x3) AND (x3 OR -x2 OR -x1)
	# As soon as we find any True value inside a clause that clause is satisfied
	try:
		x = fitness_dict[str(var_arr)]
	except:
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
		fitness_dict[str(var_arr)] = t_clauses
	return fitness_dict[str(var_arr)]

def float_fitness(clauses, var_arr):
	#NOT USED
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

def random_selection(population, num_parents):
	tmp_pop = population[:]
	parents = []
	for _ in range(num_parents):
		ind = np.random.randint(0, len(tmp_pop))
		parents.append(tmp_pop.pop(ind))
	return parents

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
				parents.append(tmp_pop.pop(i))
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
	total_rank = sum([y for x,y in pop_rank])
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

def truncation_selection(population, num_parents, proportion):
	num_indivs = int(len(population)*proportion)
	parents = []
	sorted_pop = sorted(population, key=lambda x: x[1])
	if num_indivs >= num_parents:
		for i in range(num_parents):
			parents.append(sorted_pop[i])
	elif num_indivs < num_parents:
		while len(parents)<num_parents:
			for i in range(num_indivs):
				parents.append(sorted_pop[i])
				if len(parents)>=num_parents:
					break
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


def stochastic_universal_sampling_selection(population, num_parents):
	total_fitness = sum([x[1] for x in population])
	point_distance = total_fitness/num_parents
	start_point = int(np.random.uniform(0, point_distance))
	points = [start_point + i * point_distance for i in range(num_parents)]

	parents = []
	while len(parents) < num_parents:
		random.shuffle(population)
		fit_sum, point = 0, 0
		for indiv in population:
			if len(parents)==num_parents:
				break
			elif fit_sum+indiv[1]>points[point]:
				point+=1
				parents.append(indiv)
			fit_sum += indiv[1]
	return parents


def annealed_selection(population, num_parents, max_generations, generation_num):
	# Combination if rank and roulette wheel selection
	# Rank selection
	sorted_pop = sorted(population, key=lambda x: x[1])
	pop_rank = [[x, i+1] for i,x in enumerate(sorted_pop)]
	total_rank = int((pop_rank[-1][1]*(pop_rank[-1][1]-1))/2)
	# Roulette wheel selection
	total_fitness = sum([x[1] for x in sorted_pop])
	# Combination
	factor = (1/max_generations)*generation_num
	annealed_pop = []
	total_a_fitness = 0
	for i,x in enumerate(pop_rank):
		annealed_fitness = (sorted_pop[i][1]/total_fitness)*(1-factor)+(x[1]/total_rank)*(0+factor)
		annealed_pop.append([x[0], annealed_fitness])
		total_a_fitness += annealed_fitness
	parents = []

	while len(parents)<num_parents:
		rnum = np.random.uniform(0, total_a_fitness)
		fsum = 0
		for indiv in annealed_pop:
			if fsum+indiv[1]>=rnum:
				parents.append(indiv[0])
				break
			fsum += indiv[1]

	return parents


############# CROSSOVER FUNCTIONS #############

def single_point_crossover(parent_pair, ret_cost=False):
	parent1, parent2 = parent_pair
	fitness_evals, bit_flips = 0, 0
	cut_point = np.random.randint(len(parent1))
	children = [
		np.concatenate((parent1[:cut_point],parent2[cut_point:])).tolist(),
		np.concatenate((parent2[:cut_point],parent1[cut_point:])).tolist()
		]
	if ret_cost:
		return children, fitness_evals, bit_flips
	else:
		return children

def two_point_crossover(parent_pair, ret_cost=False):
	parent1, parent2 = parent_pair
	fitness_evals, bit_flips = 0, 0
	cut_point_1 = np.random.randint(len(parent1)-1)
	cut_point_2 = np.random.randint(cut_point_1+1, len(parent1))
	children = [
		np.concatenate((parent1[:cut_point_1],parent2[cut_point_1:cut_point_2],parent1[cut_point_2:])).tolist(),
		np.concatenate((parent2[:cut_point_1],parent1[cut_point_1:cut_point_2],parent2[cut_point_2:])).tolist()
		]
	if ret_cost:
		return children, fitness_evals, bit_flips
	else:
		return children

def sliding_window_crossover(parent_pair, clauses, crossover_window_len=0.4, ret_cost=False):
	parent1, parent2 = parent_pair
	fitness_evals, bit_flips = 0, 0
	window_len = int(crossover_window_len*len(parent1))
	max_fitness, max_i = [0,0], [0,0]
	bad_children = [[],[]]
	for i in range(len(parent1)-window_len):
		bad_children[0].append(np.concatenate((parent1[:i],parent2[i:i+window_len],parent1[i+window_len:])).tolist())
		bad_children[1].append(np.concatenate((parent2[:i],parent1[i:i+window_len],parent2[i+window_len:])).tolist())
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

def random_map_crossover(parent_pair, ret_cost=False):
	parent1, parent2 = parent_pair
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

def uniform_crossover(parent_pair, ret_cost=False):
	parent1, parent2 = parent_pair
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

def mutate_population(population, mutation_function, mutation_params, ret_cost=False, max_workers=1):
	fitness_evals, bit_flips = 0, 0
	new_pop = []
	if max_workers == 1:
		for i, pop in enumerate(population):
			newindiv = mutation_function(pop, *mutation_params)
			if ret_cost:
				fitness_evals += newindiv[1]
				bit_flips += newindiv[2]
				newindiv = newindiv[0]
			new_pop.append(newindiv)
	else:
		ex = futures.ProcessPoolExecutor(max_workers=max_workers)
		results = ex.map(mutation_function, population, itertools.repeat(mutation_params[0], len(population)), itertools.repeat(mutation_params[1], len(population)))
		if ret_cost:
			for indiv, fit_ev, bit_f in results:
				new_pop.append(indiv)
				fitness_evals += fit_ev
				bit_flips += bit_f
		else:
			new_pop=results

	if ret_cost:
		return new_pop, fitness_evals, bit_flips
	else:
		return new_pop


def single_bit_flip(individual, mutation_rate, ret_cost=False):
	fitness_evals, bit_flips = 0, 0
	indiv = individual
	rmut = random.random()
	if rmut <= mutation_rate:
		ind = np.random.randint(len(indiv))
		if indiv[ind] == 1: indiv[ind] = 0
		else: indiv[ind] = 1
		bit_flips += 1
	if ret_cost:
		return indiv, fitness_evals, bit_flips
	else:
		return indiv

def multiple_bit_flip(individual, mutation_rate, ret_cost=False):
	fitness_evals, bit_flips = 0, 0
	indiv = individual
	rmut = random.random()
	if rmut <= mutation_rate:
		num_bits = np.random.randint(len(indiv))
		for x in range(num_bits):
			ind = np.random.randint(len(indiv))
			if indiv[ind] == 1: indiv[ind] = 0
			else: indiv[ind] = 1
			bit_flips += 1
	if ret_cost:
		return indiv, fitness_evals, bit_flips
	else:
		return indiv

def single_bit_greedy(individual, mutation_rate, clauses, ret_cost=False):
	fitness_evals, bit_flips = 0, 0
	ind_fitness = maxsat_fitness(clauses, individual)
	fitness_evals += 1
	max_ind = individual
	rmut = random.random()
	if rmut <= mutation_rate:
		for j in range(len(individual)):
			indiv = individual[:]
			if indiv[j]==1: indiv[j]=0
			elif indiv[j]==0: indiv[j]=1
			bit_flips += 1
			fitness_evals += 1
			if maxsat_fitness(clauses, indiv)>ind_fitness:
				max_ind = indiv
				break
	if ret_cost:
		return max_ind, fitness_evals, bit_flips
	else:
		return max_ind

def single_bit_max_greedy(individual, mutation_rate, clauses, ret_cost=False):
	fitness_evals, bit_flips = 0, 0
	ind_fitness = maxsat_fitness(clauses, individual)
	fitness_evals += 1
	max_ind, max_fit = individual, ind_fitness
	rmut = random.random()
	if rmut <= mutation_rate:
		for j in range(len(individual)):
			t_indiv = individual[:]
			if t_indiv[j]==1: t_indiv[j]=0
			elif t_indiv[j]==0: t_indiv[j]=1
			tfit = maxsat_fitness(clauses, t_indiv)
			fitness_evals += 1
			if tfit>max_fit:
				max_ind = t_indiv
				max_fit = tfit
	if ret_cost:
		return max_ind, fitness_evals, bit_flips
	else:
		return max_ind

def multiple_bit_greedy(individual, mutation_rate, clauses, ret_cost=False):
	fitness_evals, bit_flips = 0, 0	
	ind_fitness = maxsat_fitness(clauses, individual)
	indiv = individual[:]
	rmut = random.random()
	if rmut <= mutation_rate:
		for j in range(len(indiv)):
			t_indiv = indiv[:]
			new_bit = 1
			if indiv[j]==1: new_bit=0
			t_indiv[j] = new_bit
			t_fitness = maxsat_fitness(clauses, t_indiv)
			if t_fitness > ind_fitness:
				ind_fitness = t_fitness
				indiv = t_indiv
	if ret_cost:
		return indiv, fitness_evals, bit_flips
	else:
		return indiv

def flip_ga(individual, mutation_rate, clauses, ret_cost=False):
	fitness_evals, bit_flips = 0, 0
	indiv = individual[:]
	ind_fitness = maxsat_fitness(clauses, indiv)
	prev_fitness = ind_fitness-1
	rmut = random.random()
	if rmut <= mutation_rate:
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
	if ret_cost:
		return indiv, fitness_evals, bit_flips
	else:
		return indiv


############# POPULATION REPLACEMENT FUNCTIONS #############
def random_replacement(allow_duplicates, children_fitness, pop_fitness):
	new_pop = [x for x,y in pop_fitness]
	for _ in children_fitness:
		rand_ind = np.random.randint(0, len(new_pop))
		new_pop.pop(rand_ind)
	new_pop += [x for x,y in children_fitness]
	if allow_duplicates:
		return new_pop
	else:
		return set(new_pop)


def weak_parent_replacement(allow_duplicates, children_fitness, parent_pairs, pop_fitness_dict, pop_fitness):
	new_pop = [x for x,y in pop_fitness]
	tmp_children = list(children_fitness)
	new_children = []
	for ppair in parent_pairs:
		new_pop.remove(ppair[0])
		aux = tmp_children[:2] + [[ppair[0],pop_fitness_dict[tuple(ppair[0])]], [ppair[1],pop_fitness_dict[tuple(ppair[1])]]]
		tmp_children = tmp_children[2:]
		sorted_pop = sorted(aux, key=lambda x: x[1], reverse=True)
		if ppair[0]==ppair[1]:
			new_children += [x for x,y in sorted_pop[:1]]
		else:
			new_pop.remove(ppair[1])
			new_children += [x for x,y in sorted_pop[:2]]
	new_pop += new_children
	if allow_duplicates:
		return new_pop
	else:
		return set(new_pop)

def parent_replacement(allow_duplicates, children_fitness, parent_pairs, pop_fitness_dict, pop_fitness):
	new_pop = [x for x,y in pop_fitness]
	remove_children = 0
	for ppair in parent_pairs:
		try:
			new_pop.remove(ppair[0])
		except: pass
		try:
			new_pop.remove(ppair[1])
		except: pass
	new_pop += [x for x,y in sorted(children_fitness, key=lambda x: x[1], reverse=True)][:len(pop_fitness)-len(new_pop)]
	if allow_duplicates:
		return new_pop
	else:
		return set(new_pop)

def delete_n(allow_duplicates, children_fitness, pop_fitness, num_individuals, selection_method, selection_params):
	tmp_pop = pop_fitness[:]
	tmp_children = children_fitness[:]
	new_children = []
	to_kill = selection_method(tmp_pop, num_individuals, *selection_params)[0]
	for indiv, fitness in to_kill:
		tmp_pop.remove(indiv)

	to_add = selection_method(tmp_children, num_individuals, *selection_params)[0]
	for indiv, fitness in to_add:
		tmp_children.remove(indiv)
		new_children += [indiv]

	new_pop = [x for x,y in tmp_pop] + [x for x,y in new_children]
	if allow_duplicates:
		return new_pop
	else:
		return set(new_pop)

"""
def delete_n(allow_duplicates, children_fitness, pop_fitness, num_individuals, selection_method, selection_params):
	tmp_pop = pop_fitness[:]
	tmp_children = children_fitness[:]
	new_children = []
	for _ in range(num_individuals):
		to_kill = selection_method(tmp_pop, 1, *selection_params)[0]
		if len(tmp_children)>0:
			tmp_pop.remove(to_kill)
			to_add = selection_method(tmp_children, 1, *selection_params)[0]
			tmp_children.remove(to_add)
			new_children += [to_add]
		else:
			break

	new_pop = [x for x,y in tmp_pop] + [x for x,y in new_children]
	if allow_duplicates:
		return new_pop
	else:
		return set(new_pop)
"""

def mu_lambda_replacement(allow_duplicates, pop_fitness, pop_size):
	new_population = []
	sorted_pop = sorted(pop_fitness, key=lambda x: x[1], reverse=True)
	new_population = [x for x,y in sorted_pop[:pop_size]]
	if not allow_duplicates: new_population = set(new_population)
	return new_population

def generational_replacement(allow_duplicates, elitism, children, pop_fitness):
	if elitism == 0:
		return children

	if allow_duplicates:
		new_population = []
	else:
		new_population = set()
	sorted_pop = sorted(pop_fitness, key=lambda x: x[1], reverse=True)
	if allow_duplicates:
		new_population = children + [x for x,y in sorted_pop[:elitism]]
	else:
		new_population = children
		for x,y in sorted_pop[:elitism]:
			new_population.add(x)
	return new_population
