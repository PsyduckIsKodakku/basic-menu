import numpy
import pandas as pd
import numpy as np
import random
import pygad

menu = pd.read_csv("example.csv")
menu1 = menu.to_numpy()
name = menu1[:, 0]
data = menu1[:, 1:]
ref = [1, 2, 1, 1]


def fitness_func(solution, solution_idx):
    data7 = np.zeros([35, 4])

    count = 0

    for i in solution:
        data7[count] = data[int(i)]
        count += 1

    total = 0
    a1 = 0
    b1 = 5
    for j in range(7):
        for i in range(4):
            total += numpy.abs(ref[i] - numpy.sum(data7[a1:b1, i])) * ((i+1) * 4)
        a1 += 5
        b1 += 5

    a1 = 0
    b1 = 5
    for j in range(7):
        s = np.sort(solution[a1:b1], axis=None)
        s1 = np.delete(s[:-1][s[1:] == s[:-1]], np.where(s[:-1][s[1:] == s[:-1]] == 0))
        total += len(s1)
        a1 += 5
        b1 += 5

    a1 = 0
    b1 = 5
    for j in range(6):
        s1 = np.sort(solution[a1:b1], axis=None)
        s2 = np.sort(solution[a1 + 5:b1 + 5], axis=None)
        if np.array_equal(s1, s2):
            total += 5

    fitness = 1 / (total + 1.0e-10)
    return fitness


def generate_menu(m):
    menu2 = np.zeros(int(len(data) * len(data[0])))
    for i in range(len(menu2)):
        menu2[i] = random.randint(0, len(m) - 1)
    return menu2


def print_menu(n, m):
    p = np.chararray([7, 5], itemsize=10000)
    c = 0
    for i in range(len(p)):
        for j in range(len(p[0])):
            p[i][j] = n[int(m[c])].encode("UTF-8")
            c += 1
    print(p.decode("UTF-8"))
    return p.decode("UTF-8")


fitness_function = fitness_func

num_generations = 500
num_parents_mating = 4

sol_per_pop = 5000
num_genes = 35

init_range_low = 0
init_range_high = 15

parent_selection_type = "sss"
keep_parents = 100

mutation_type = "adaptive"
mutation_percent_genes = 15

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       crossover_type="two_points",
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       mutation_percent_genes=mutation_percent_genes,
                       gene_type=int
                       )

ga_instance.run()
ga_instance.plot_result()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

print_menu(name, solution)
