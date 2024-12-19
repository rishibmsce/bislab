import random

def define_problem():
    """Define a sample optimization problem: Sphere function"""
    def sphere_function(genes):
        return sum(x**2 for x in genes)
    return sphere_function

def initialize_parameters():
    """Set parameters for the Gene Expression Algorithm"""
    return {
        'population_size': 50,
        'num_generations': 100,
        'num_genes': 5,
        'mutation_rate': 0.1,
        'crossover_rate': 0.8,
        'lower_bound': -10,
        'upper_bound': 10,
    }

def initialize_population(size, num_genes, lower_bound, upper_bound):
    """Generate an initial random population"""
    return [[random.uniform(lower_bound, upper_bound) for _ in range(num_genes)] for _ in range(size)]

def evaluate_fitness(population, fitness_function):
    """Evaluate fitness of each individual"""
    return [fitness_function(individual) for individual in population]

def selection(population, fitness, num_parents):
    """Select individuals based on fitness (tournament selection)"""
    selected = []
    for _ in range(num_parents):
        candidates = random.sample(list(zip(population, fitness)), k=3)
        selected.append(min(candidates, key=lambda x: x[1])[0])
    return selected

def crossover(parent1, parent2, crossover_rate):
    """Perform crossover between two parents"""
    if random.random() < crossover_rate:
        point = random.randint(1, len(parent1) - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
    return parent1, parent2

def mutate(individual, mutation_rate, lower_bound, upper_bound):
    """Mutate an individual by modifying genes"""
    return [
        gene + random.uniform(-1, 1) if random.random() < mutation_rate else gene
        for gene in individual
    ]

def gene_expression_algorithm():
    """Main function to execute the Gene Expression Algorithm"""
    fitness_function = define_problem()
    params = initialize_parameters()
    population = initialize_population(
        params['population_size'],
        params['num_genes'],
        params['lower_bound'],
        params['upper_bound']
    )

    for generation in range(params['num_generations']):
        fitness = evaluate_fitness(population, fitness_function)
        parents = selection(population, fitness, params['population_size'] // 2)
        next_generation = []

        for i in range(0, len(parents), 2):
            p1, p2 = parents[i], parents[min(i + 1, len(parents) - 1)]
            offspring1, offspring2 = crossover(p1, p2, params['crossover_rate'])
            next_generation.extend([
                mutate(offspring1, params['mutation_rate'], params['lower_bound'], params['upper_bound']),
                mutate(offspring2, params['mutation_rate'], params['lower_bound'], params['upper_bound']),
            ])

        population = next_generation
        best_fitness = min(fitness)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

    final_fitness = evaluate_fitness(population, fitness_function)
    best_solution = population[final_fitness.index(min(final_fitness))]
    print("Final Best Solution:", best_solution)

gene_expression_algorithm()
