# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 10:19:34 2021

@author: Valen
"""
#Hello

import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
#from math import fabs,sqrt
import glob, os

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'individual_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[8],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment

env.state_to_log() # checks environment state


####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker


# genetic algorithm params

run_mode = 'train' # train or test

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
print("Number of weights: " + str(n_vars))


# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

# DEAP - own code starts here 
from deap import base, creator
import random
from deap import tools
#maximizing fitness
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#creator.create("Individual", list, fitness=creator.FitnessMax)
from array import array
creator.create("Individual", array, typecode="d",
               fitness=creator.FitnessMax, strategy=None)
creator.create("Strategy", array, typecode="d")

def initES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind

IND_SIZE = n_vars
MIN_VALUE, MAX_VALUE = -1., 1.
MIN_STRAT, MAX_STRAT = -1., 1. 

toolbox = base.Toolbox()
toolbox.register("individual", initES, creator.Individual,
                 creator.Strategy, IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRAT, 
                 MAX_STRAT)


#uniform = random.uniform(-1, 1)
toolbox.register("attribute", np.random.uniform, -1, 1)
#random.uniform(-1.,1.)
#toolbox.register("individual", tools.initRepeat, creator.Individual,
#                 toolbox.attribute, n= n_vars)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#toolbox.register("population", tools.initRepeat, np.array([]), toolbox.individual)



toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutESLogNormal, c = 1, indpb=0.1)
toolbox.register("parentselection", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)
toolbox.register("survivalselection", tools.selBest)

#to compile statistics
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)
logbook = tools.Logbook()


pop_size = 20

#test

def main():
    pop = toolbox.population(n=pop_size)
    CXPB, MUTPB, NGEN = 0.7, 0.2, 10

    # Evaluate the entire population
    #fitnesses = map(toolbox.evaluate, np.array(pop))
    #for ind, fit in zip(pop, fitnesses):
    #    ind.fitness.values = fit
    for i in range(pop_size):
        pop[i].fitness.values = toolbox.evaluate(np.array([pop[i]]))

    # show first random generation
    record = stats.compile(pop)
    print("Gen 0: ")
    print(record)

    for g in range(NGEN):
        # Select the next generation individuals - parent selection 
        parents = toolbox.parentselection(pop, (2*len(pop)))
        # Clone the selected individuals
        parents = list(map(toolbox.clone, parents))

        
        offspring = []
        # Apply crossover on the offspring
        
        #like this
        for parent1, parent2 in zip(parents[::2], parents[1::2]): 
            if random.random() < CXPB:
                child1, child2 = toolbox.mate(parent1, parent2)
                del child1.fitness.values
                del child2.fitness.values
                offspring.append(child1)
                offspring.append(child2)
        print(len(offspring)) 
        
        #or like this
# =============================================================================
#         for child1, child2 in zip(parents[::2], parents[1::2]):
#             if random.random() < CXPB:
#                 toolbox.mate(child1, child2)
#                 del child1.fitness.values
#                 del child2.fitness.values
# =============================================================================
                
        # mutation
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        #invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        #for i in range(len(invalid_ind)):
        #    invalid_ind[i].fitness.values = toolbox.evaluate(np.array([invalid_ind[i]]))
        for i in range(len(offspring)):
            offspring[i].fitness.values = toolbox.evaluate(np.array([offspring[i]]))
            
        
        survivedoffspring = toolbox.survivalselection(offspring, pop_size)
        
        # The population is entirely replaced by the offspring
        pop = survivedoffspring
        print(len(pop))
        
        record = stats.compile(pop)
        print("Gen : " + str(g+1))
        print(record)
        logbook.record(gen= (g +1), evals=30, **record)
        
    return pop

logbooks = []
for i in range(5):
    lastgen = main()
    logbooks.append(logbook)
    
for logbook in logbooks:
    print(logbook)


