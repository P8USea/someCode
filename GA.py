import numpy as np
import time
import math
def griewank_function(x):
   
    sum_term = sum(xi ** 2 / 4000 for xi in x)
    product_term = math.prod(math.cos(xi / math.sqrt(i + 1)) for i, xi in enumerate(x))
    result = 1 + sum_term - product_term
    return result

def rastrigin_function(position):
  fitnessVal = 0.0
  for i in range(len(position)):
    xi = position[i]
    fitnessVal += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
  return fitnessVal

def ackley_function(x):
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    result = term1 + term2 + 20 + np.exp(1)
    return result

def initialize_population(population_size, dim, minx, maxx):
    np.random.seed(10)
    return np.random.uniform(low=minx, high=maxx, size=(population_size, dim))



def evaluate_fitness(population):
    return np.array([fitness(individual) for individual in population])

def roulette_wheel_selection(population, fitness_values):
    probabilities = fitness_values / np.sum(fitness_values)
    selected_indices = np.random.choice(np.arange(len(population)), size=len(population), p=probabilities)
    return population[selected_indices]

def crossover(parent1, parent2):
    crossover_points = np.random.choice(np.arange(len(parent1)), size=np.random.randint(1, len(parent1)), replace=False)
    child1, child2 = parent1.copy(), parent2.copy()
    child1[crossover_points] = parent2[crossover_points]
    child2[crossover_points] = parent1[crossover_points]
    return child1, child2

def mutate(individual, mutation_rate):
    mutation_mask = np.random.rand(len(individual)) < mutation_rate
    individual[mutation_mask] += np.random.uniform(low=-0.5, high=0.5, size=np.sum(mutation_mask))
    return individual

def clip_individual(individual, lower_bound, upper_bound):
    return np.clip(individual, lower_bound, upper_bound)

def clip_population(population, lower_bound, upper_bound):
    return np.clip(population, lower_bound, upper_bound)

def bound_individual(individual, lower_bound, upper_bound):
    return np.minimum(upper_bound, np.maximum(lower_bound, individual))

def genetic_algorithm(population_size, dim, generations, mutation_rate,minx, maxx):
    lower_bound = minx
    upper_bound = maxx
    
    population = initialize_population(population_size, dim, minx, maxx)
    population = clip_population(population, lower_bound, upper_bound)
    
    best_solution = None
    best_fitness = float('inf')
    
    for generation in range(generations):

        
        with open(file_name, "a") as file:
            file.write(str(best_fitness) + '\n')

        fitness_values = evaluate_fitness(population)
        average_fitness = np.mean(fitness_values)
        best_individual = population[np.argmin(fitness_values)]
        
        if average_fitness < best_fitness:
            best_solution = best_individual
            best_fitness = average_fitness
        
        selected_population = roulette_wheel_selection(population, fitness_values)
        
        children = []
        for i in range(0, len(selected_population), 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1]
            
            child1, child2 = crossover(parent1, parent2)
            
            children.append(child1)
            children.append(child2)
        
        for i in range(len(children)):
            children[i] = mutate(children[i], mutation_rate)
            children[i] = bound_individual(children[i], lower_bound, upper_bound)
        
        population = np.vstack((population, np.array(children)))
        population = clip_population(population, lower_bound, upper_bound)
        
        fitness_values = evaluate_fitness(population)
        selected_indices = np.argsort(fitness_values)[:population_size]
        population = population[selected_indices] 
    
    return best_solution, best_fitness



#fitness = ackley_function
fitness = griewank_function
#fitness = rastrigin_function
file_name = "GA.txt"

population_size = 300
dim = 50
generations = 100
mutation_rate = 0.1

minx, maxx = -10.0, 10.0

with open(file_name, "w") as file:
    file.write("")
print("GA Setup Succeeded!")

start_time = time.time()
best_solution, best_fitness = genetic_algorithm(population_size, dim, generations, mutation_rate, minx, maxx)

end_time = time.time()

elapsed_time = end_time - start_time
print("Time: " + str(elapsed_time))
#so sanh bang khoi tao ngau nhien seed()


