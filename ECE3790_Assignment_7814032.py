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
    def __init__(self, n, predefinedMap=None):
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


def selection(fitnesses, numSelected, maxits=100):
    #Selection method for genetic algorithms
    # Creates cumulative distribution of all fitnesses and selects
    # random genes out of this distribution.
    # Prevents getting stuck in a loop using default value maxits=100. If loop
    # exceeds this, it will take the sorted values up to when the loop broke.
    sortedFitness = np.sort(fitnesses)[::-1]
    fitnesscdf = np.cumsum(sortedFitness / np.sum(sortedFitness))
    picked = []
    it = 0
    while(len(picked) < numSelected and it < maxits):
        #Pick population to keep based on fitness
        it += 1
        val = ran()
        j = 0
        while val > fitnesscdf[j] and j != len(fitnesscdf)-1:
            j += 1
        if j not in picked:
            picked.append(j)
            
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



def mutation():
    pass
            

def pathPrint(path, cities):
    mpX = []
    mpY = []
    for visited in path:
        mpX.append(cities[visited].x)
        mpY.append(cities[visited].y)
        
    mpX.append(cities[path[0]].x)
    mpY.append(cities[path[0]].y)
    print(mpX)
    print(mpY)
    plt.plot(mpX, mpY)
    plt.show()

        

def main():
    popSize = 10
    numCities = 5
    cities = TSP(numCities)


    #Generate a starting population - Each gene is a different path
    population = np.meshgrid(range(numCities), range(popSize))[0]
    for j in population:
        #Shuffle values so we get random paths
        np.random.shuffle(j)

    b = np.sort(population[0])[::-1]
    p = selection(b, 2)
    print(b)
    print(p)
    print("SELECTED:")
    print(b[p[0]])
    print(b[p[1]])
    
    

    
    
main()