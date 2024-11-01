import random
from SetCoveringProblemCreator import * # type: ignore
import time
import json
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

def fitness(solution, subsets, universe):
    covered = set()
    num_subsets_used = sum(solution)  
    for i, included in enumerate(solution):
        if included:
            covered.update(subsets[i])
    return len(covered) - len(universe - covered) * 5 - num_subsets_used * 0.1 # Adjust penalty as needed

def initialize_population(pop_size, num_subsets):
    return [[random.choice([0, 1]) for _ in range(num_subsets)] for _ in range(pop_size)]

def selection(population, fitness_values):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitness_values)), 3)
        winner = max(tournament, key=lambda x: x[1])[0]
        selected.append(winner)
    return random.sample(selected, 2)

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 2)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

def mutate(offspring, mutation_rate = 0.01):
    return [(gene if random.random() > mutation_rate else 1 - gene) for gene in offspring]

def genetic_algorithm(subsets, universe, pop_size = 50, generations = 50, time_limit = 45):
    start_time = time.time()
    population = initialize_population(pop_size, len(subsets))
    best_solution = None
    best_fitness = -float('inf')
    fitness_history = []
    
    for gen in range(generations):
        if time.time() - start_time > time_limit:
            break
        
        fitness_values = [fitness(indiv, subsets, universe) for indiv in population]
        fitness_history.append(max(fitness_values))
        
        best_idx = np.argmax(fitness_values)
        if fitness_values[best_idx] > best_fitness:
            best_fitness = fitness_values[best_idx]
            best_solution = population[best_idx]
        
        new_population = [x for _, x in sorted(zip(fitness_values, population), reverse=True)][:1]
        while len(new_population) < pop_size:
            parent1, parent2 = selection(population, fitness_values)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])
        
        population = new_population[:pop_size]
    
    return best_solution, best_fitness, fitness_history, time.time() - start_time

scp = SetCoveringProblemCreator() # type: ignore
all_results = {}
num_runs = 10
# listOfSubsets = scp.ReadSetsFromJson("scp_test.json")
listOfSubsets = scp.Create(usize = 100, totalSets = 150)
universe = set(range(1, 101))
mean_fitness_over_gens = np.zeros(50)
fitnesses = []
chromosomes=[]
for i in range(num_runs):
    best_solution, best_fitness, fitness_history, total_time = genetic_algorithm(listOfSubsets, universe)
    chromosomes.append(best_solution)
    fitnesses.append(best_fitness)
    mean_fitness_over_gens += np.array(fitness_history)
    mean_fitness_over_gens /= num_runs
    all_results[50] = (mean_fitness_over_gens, np.mean(fitnesses), np.std(fitnesses))

print("Roll no : 2022A7PS0019G")
print("Number of subsets in scp_test. file :" , len(listOfSubsets))
print("Solution : ")
print(*[f'{i + 1}:{value}' for i, value in enumerate(best_solution)], sep = ', ') # type: ignore
print("Fitness value of the best state :", best_fitness)
print("The minimum number of subsets that can cover the entire Universe-set :", sum(best_solution)) # type: ignore
print(f"Time taken : {round(total_time, 2)} seconds")


mean = sum(fitness_history) / len(fitness_history)
std = np.std(fitness_history)
plt.text(16, best_fitness, f"Standard Deviation {round(std, 2)}", fontsize=10)
plt.plot(fitness_history)
plt.plot([mean for i in range(len(fitness_history))], color = 'r', label = 'Mean')
plt.legend()
plt.show()