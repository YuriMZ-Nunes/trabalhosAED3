import numpy as np
import random
from time import time

def firstPopulation(populationSize, numGenes):
    return [random.sample(range(numGenes), numGenes) for _ in range(populationSize)]

def calculateFitness(individual, matrix):
    totalFitness = 0
    for i in range(len(individual) - 1):
        totalFitness += matrix[individual[i]][individual[i + 1]]
    totalFitness += matrix[individual[-1]][individual[0]]
    return 1 / totalFitness

def crossover(firstParent, secondParent):
    crossoverPoint = random.randint(0, len(firstParent) - 1)
    child = firstParent[:crossoverPoint] + [gene for gene in secondParent if gene not in firstParent[:crossoverPoint]]
    return child

def mutate(individual):
    index1, index2 = random.sample(range(len(individual)), 2)
    individual[index1], individual[index2] = individual[index2], individual[index1]
    return individual

def geneticAlgorithm(matrix, populationSize, crossoverRate, mutationRate, maxGenerations) :
    
    begin = time()
    
    numGenes, _ = matrix.shape
    
    population = firstPopulation(populationSize, numGenes)
    
    for _ in range(maxGenerations):
        fitnessScore = [calculateFitness(individual, matrix) for individual in population]
        
        parents = random.choices(population,weights=fitnessScore, k=populationSize)

        newPopulation = []
        for i in range(0, populationSize, 2):
            if random.random() < crossoverRate:
                child1 = crossover(parents[i], parents[i + 1])
                child2 = crossover(parents[i + 1], parents[i ])
                newPopulation.extend([child1, child2])
            else:
                newPopulation.extend([parents[i], parents[i + 1]])
        
        for i in range(populationSize):
            if random.random() < mutationRate:
                newPopulation[i] = mutate(newPopulation[i])
        
        population = newPopulation
    
    bestIndividual = max(population, key=lambda x: calculateFitness(x, matrix))
    bestFitness = 1 / calculateFitness(bestIndividual, matrix)
    
    return bestFitness, time() - begin