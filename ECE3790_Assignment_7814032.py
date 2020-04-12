'''
ECE 3790    -   ASSIGNMENT 1

Completed by Ryan Bate 7814032
'''

#if it ain't broke, don't fix it

from numpy.random import random_sample as ran
import numpy as np
import math as m
from copy import deepcopy

import matplotlib.pyplot as plt



class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class TSP:
    def __init__(self, n, popSize=10, predefinedMap=None):
        #Overall Problem constructor.
        # If predefinedMap is not defined then it will create a map using

        if (n <= 1) and (predefinedMap is None):
            print("ERROR: Invalid parameters!")

        elif predefinedMap is not None:
            self.cities = deepcopy(predefinedMap)
            self.cSize = len(self.cities)

        else:
            self.cSize = n
            self.cities = []
            
            for i in range(n):
                self.cities.append(City(ran(), ran())) #Append a new city with a random location


def fitness(path, cities):
    fitness = 0.0
    for i in range(1, len(path)):
        fitness += m.sqrt(((cities[i].x-cities[i-1].x)**2) + ((cities[i].y-cities[i-1].y)**2))

    #Add the last section of the path to complete the loop
    fitness +=  m.sqrt(((cities[len(path)].x-cities[0].x)**2) + ((cities[len(path)].y-cities[0].y)**2))

    return 1/fitness #Get the inverse since GA is a maximization alg.

def fitnessVector(pop, cities):
    #Gets vector of all fitnesses in the population, sorted in descending order
    fitnesses = np.zeros(len(pop))
    for i in range(len(pop)):
        fitnesses[i] = fitness(pop[i], cities)
    
    return fitnesses
        

def selection(fitnesses, numSelected, maxits=100):
    #Selection method for genetic algorithms
    # Creates cumulative distribution of all fitnesses and selects
    # random genes out of this distribution.
    # Prevents getting stuck in a loop using default value maxits=100. If loop
    # exceeds this, it will take the sorted values up to when the loop broke.
    fitnessCopy = deepcopy(fitnesses)
    sortedFitness = (np.sort(fitnesses)[::-1]) + 1
    fitnesscdf = np.cumsum(sortedFitness / np.sum(sortedFitness))
    picked = []
    it = 0 #Keep track of number of iterations
    while(len(picked) < numSelected and it < maxits):
        #Pick population to keep based on fitness
        it += 1
        val = ran()
        j = 0
        while val > fitnesscdf[j] and j != len(fitnesscdf)-1:
            j += 1
        if j not in picked:
            picked.append(j)
    
    for i in range(len(picked)):
        picked[i] = fitnessCopy.index(picked[i])
    
    return picked

def selectfirst(fitnesses, numSelected, desc=False):
    fitnessCopy = deepcopy(fitnesses)
    sortedFitness = np.sort(fitnesses)
    if desc:
        sortedFitness = sortedFitness[::-1]
    picked = []
    for i in range(numSelected):
        picked.append(sortedFitness[i])

    for i in range(len(picked)):
        picked[i] = fitnessCopy.index(picked[i])
    return picked

def crossover(gene1, gene2):
    #Performs gene crossover using Order 1 permutation crossover
    child = np.zeros(len(gene1))
    startpoint = np.random.randint(0, len(child)-1)
    endpoint = np.random.randint(startpoint+1, len(child))

    for i in range(startpoint, endpoint):
        child[i] = gene1[i] #Copy gene1's selected values over to the child
    i1 = 0
    while i1 < startpoint:
        #Append values of gene 2 not found in child in order in which they
        # appear in gene 2 BEFORE the startpoint.
        for j in gene2:
            if j not in child:
                child[i1] = j
                i1 += 1
    
    i1 = endpoint
    while i1 < len(child):
        #Append values of gene 2 not found in child in order in which they
        # appear in gene 2 AFTER the endpoint
        for j in gene2:
            if j not in child:
                child[i1] = j
                i1 += 1
    return child

def round2(x):
    return 2 * round(x/2)

def mutation(gene):
    #Mutates gene by swapping two elements in the array
    aIndex = np.random.randint(0, len(gene))
    bIndex = aIndex

    while bIndex == aIndex:
        #Ensure that the two swap points are not equal
        bIndex = np.random.randint(0, len(gene))

    #Swap them
    temp = gene[aIndex]
    gene[aIndex] = gene[bIndex]
    gene[bIndex] = temp
    return gene

def select2(from_here):
    assert len(from_here) >=2
    aIndex = np.random.randint(0, len(from_here))
    bIndex = aIndex

    while bIndex == aIndex:
        #Ensure that the two swap points are not equal
        bIndex = np.random.randint(0, len(from_here))

    return aIndex, bIndex

    

        

def main(maxits = 100):

    #USER DEFINED VARIABLES
    popSize = 10
    numCities = 5
    
    
    repr_chance = 0.5 #Chance to reproduce
    mutation_chance = 0.2 #Chance for a gene to mutate

    lottery_win_percent = 0.3 #Percentage of genes which will win the 'lottery'

    #---------------------------------------------------------------------------
    #                           INITIALIZATION
    #---------------------------------------------------------------------------
    
    
    cities = TSP(numCities)
    bestFitnessSoFar = 0

    #Generate a starting population - Each gene is a different path
    population = np.meshgrid(range(numCities), range(popSize))[0]
    for j in population:
        #Shuffle values so we get random paths
        np.random.shuffle(j)

    numLottoWinners = round2(lottery_win_percent*len(population))


    its = 0
    while its < maxits:

        fitnesses = fitnessVector(population, cities.cities)

        #-----------------------------------------------------------------------
        #                                SELECTION
        #-----------------------------------------------------------------------
        winners = selection(fitnesses, numLottoWinners)


        if(max(fitnesses) > bestFitnessSoFar):
            bestFitnessSoFar = max(fitnesses)
            bestPathSoFar = deepcopy(population[list(fitnesses).index(max(fitnesses))])


        

        
        #-----------------------------------------------------------------------
        #                                REPRODUCTION
        #-----------------------------------------------------------------------
        winnersCpy = deepcopy(winners)
        parentOrder = []
        while len(winnersCpy) > 0:
            parentA, parentB = select2(winnersCpy)
            parentOrder.append(parentA)
            parentOrder.append(parentB)
            winnersCpy.pop(parentA)
            winnersCpy.pop(parentB)
        
        children = []
        for i in range(0, len(parentOrder), 2):
            diceroll = ran()
            if diceroll < repr_chance:
                children.append(crossover(parentA, parentB))
            else:
                pass
        
        if len(children) > 0:
            weakest = selectfirst(fitnesses, len(children))

            for i, item in enumerate(weakest):
                #Replace weakest in the population with children
                population[item] = children[i]

        #-----------------------------------------------------------------------
        #                                MUTATION
        #-----------------------------------------------------------------------
        
        for i in range(len(population)):
            diceroll = ran()
            if diceroll < mutation_chance:
                mutation(population[i])
                

    #FINAL LOOP
    fitnesses = fitnessVector(population, cities.cities)

    if(max(fitnesses) > bestFitnessSoFar):
            bestFitnessSoFar = max(fitnesses)
            bestPathSoFar = deepcopy(population[list(fitnesses).index(max(fitnesses))])

    

    
main()