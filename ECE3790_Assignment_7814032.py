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
        fitness += m.sqrt(((cities[path[i]].x-cities[path[i-1]].x)**2) + 
                        ((cities[path[i]].y-cities[path[i-1]].y)**2))

    #Add the last section of the path to complete the loop
    fitness +=  m.sqrt(((cities[path[len(path)-1]].x-cities[path[0]].x)**2) + 
                ((cities[path[len(path)-1]].y-cities[path[0]].y)**2))

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
    sortedFitness = (np.sort(fitnesses)[::-1])
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
        picked[i] = list(fitnessCopy).index(sortedFitness[picked[i]])
    
    return picked

def selectfirst(fitnesses, numSelected, desc=False):

    '''TODO: Figure out why I'm getting THIS error:
    File "c:\Users\ryanb\Documents\Schoolwork-University\Eng Alg\Assignment 1\ECE3790_Assignment_7814032.py", line 267, in main
    weakest = selectfirst(fitnesses, len(children))
  File "c:\Users\ryanb\Documents\Schoolwork-University\Eng Alg\Assignment 1\ECE3790_Assignment_7814032.py", line 100, in selectfirst
    picked[i] =  list(fitnessCopy).index(sortedFitness[picked[i]])
IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
'''

#That will be the key to solving life's mysteries...


    fitnessCopy = list(deepcopy(fitnesses))
    sortedFitness = np.sort(fitnesses)
    if desc:
        sortedFitness = sortedFitness[::-1]
    picked = []
    for i in range(numSelected):
        picked.append(sortedFitness[i])

    for i in range(len(picked)):
        picked[i] =  fitnessCopy.index(sortedFitness[picked[i]])
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
                if i1 >= startpoint:
                    break
    
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
    fig, ((ax1, ax2)) = plt.subplots(1, 2)
    
    cities = TSP(numCities)
    bestFitnessSoFar = 0

    #Generate a starting population - Each gene is a different path
    population = np.meshgrid(range(numCities), range(popSize))[0]
    for j in population:
        #Shuffle values so we get random paths
        np.random.shuffle(j)

    numLottoWinners = round2(lottery_win_percent*len(population))

    xbest = []
    ybest = []

    its = 0
    while its < maxits:

        fitnesses = fitnessVector(population, cities.cities)

        #-----------------------------------------------------------------------
        #                                SELECTION
        #-----------------------------------------------------------------------
        winners = selection(fitnesses, numLottoWinners)



        xplot = []
        yplot = []
        if(max(fitnesses) > bestFitnessSoFar):
            bestFitnessSoFar = max(fitnesses)
            bestPathSoFar = deepcopy(population[list(fitnesses).index(max(fitnesses))])
            xbest = []
            ybest = []
            for i in bestPathSoFar:
                xbest.append(cities.cities[i].x)
                ybest.append(cities.cities[i].y)

        bestPathCurrently = deepcopy(population[list(fitnesses).index(max(fitnesses))])
        
        
        for i in bestPathCurrently:
            xplot.append(cities.cities[i].x)
            yplot.append(cities.cities[i].y)
            
        np.append(xplot, xplot[0])
        np.append(yplot, yplot[0])

        np.append(xbest, xbest[0])
        np.append(ybest, ybest[0]) 

        ax1.cla()
        ax2.cla()
        ax1.plot(xplot, yplot)
        ax2.plot(xplot, yplot)
        ax1.set_title("Best at current iteration")
        ax2.set_title("Best overall iteration")
        plt.pause(0.005)

        #-----------------------------------------------------------------------
        #                                REPRODUCTION
        #-----------------------------------------------------------------------
        winnersCpy = deepcopy(winners)
        parentOrder = []
        while len(winnersCpy) > 1:
            parentA, parentB = select2(winnersCpy)
            parentOrder.append(parentA)
            parentOrder.append(parentB)
            if(parentA > parentB):
                winnersCpy.pop(parentA)
                winnersCpy.pop(parentB)
            else:
                winnersCpy.pop(parentB)
                winnersCpy.pop(parentA)
            
        
        children = []
        for i in range(0, len(parentOrder), 2):
            diceroll = ran()
            if diceroll < repr_chance:
                children.append(crossover(population[parentA], population[parentB]))
                print("hello")
            else:
                pass
        
        if len(children) > 0:
            weakest = selectfirst(fitnesses, len(children))
            print("hello2)")

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
plt.show()