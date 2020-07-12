CREATE USER pyuser WITH ENCRYPTED PASSWORD '123456';
CREATE DATABASE genetic_algorithm
  WITH
  OWNER=pyuser
  ENCODING='UNICODE';

ALTER USER pyuser CREATEDB;

\c genetic_algorithm;

CREATE TABLE ga_run (
	id SERIAL PRIMARY KEY,
	created_timestamp TIMESTAMP NOT NULL,
	updated_timestamp TIMESTAMP,
	problem VARCHAR NOT NULL,
	sol_found Boolean,
	solution INT[],
	max_iterations INT NOT NULL,
	pop_size INT NOT NULL,
	elitism INT NOT NULL,
	max_fitness INT,
	num_iterations INT,
	num_fitness_evals INT,
	num_bit_flips INT,
	fitness_function VARCHAR NOT NULL,
	initial_population_function VARCHAR NOT NULL,
	selection_function VARCHAR NOT NULL,
	crossover_function VARCHAR NOT NULL,
	mutation_function VARCHAR NOT NULL,
	mutation_rate NUMERIC(4,3),
	tournament_size INT,
	crossover_window_len NUMERIC(4,3)
);

CREATE TABLE ga_run_generations (
	id SERIAL PRIMARY KEY,
	time_stamp TIMESTAMP NOT NULL,
	ga_run_id INT NOT NULL,
	generation_num INT NOT NULL,
	max_fitness INT NOT NULL,
	population_length INT NOT NULL,
	population_set_length INT NOT NULL,
	num_fitness_evals INT NOT NULL,
	num_bit_flips INT NOT NULL,
	FOREIGN KEY (ga_run_id) REFERENCES ga_run(id)
);

CREATE TABLE ga_run_population (
	id SERIAL PRIMARY KEY,
	ga_run_id INT NOT NULL,
	ga_run_generation_id INT,
	population INT[][] NOT NULL,
	observations VARCHAR,
	FOREIGN KEY (ga_run_id) REFERENCES ga_run(id),
	FOREIGN KEY (ga_run_generation_id) REFERENCES ga_run_generations(id)
);
