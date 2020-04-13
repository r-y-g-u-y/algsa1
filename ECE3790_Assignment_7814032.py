'''
ECE 3790    -   ASSIGNMENT 1

Completed by Ryan Bate 7814032
'''


from numpy.random import random_sample as ran
import numpy as np
import math as m
from copy import deepcopy
import matplotlib.pyplot as plt

def uni(arr):
    #Checks if array is unique (doesn't have duplicate values)
    return len(np.unique(arr)) == len(arr)



class City:
    #City class
    #
    #Variables:
    # x - the x position of the city
    # y - the y position of the city
    def __init__(self, x, y):
        self.x = x
        self.y = y

class TSP:
    def __init__(self, n, popSize=10, predefinedMap=None, predefinedX=None, predefinedY=None):
        #Overall Problem constructor.
        # If predefinedMap is not defined then it will create a map using random values of size n
        # if predefined map is anything but None then it will create a map using the given
        # arrays of x and y coordinates.

        if (n <= 1) and (predefinedMap is None):
            print("ERROR: Invalid parameters!")

        elif predefinedMap is not None:
            #Check if the user input a predefined map, then use that one.
            self.cities = []
            for i in range(len(predefinedX)):
                self.cities.append(City(predefinedX[i], predefinedY[i]))
            self.cSize = len(self.cities)

        else:
            self.cSize = n
            self.cities = []
            for i in range(n):
                self.cities.append(City(ran(), ran())) #Append a new city with a random location


def fitness(path, cities):
    #Calculates the fitness of the population
    fitness = 0.0
    for i in range(1, len(path)):
        #Calculate distance between two points
        fitness += m.sqrt(((cities[path[i]].x-cities[path[i-1]].x)**2) + 
                        ((cities[path[i]].y-cities[path[i-1]].y)**2))

    #Add the last section of the path to complete the loop
    fitness +=  m.sqrt(((cities[path[len(path)-1]].x-cities[path[0]].x)**2) + 
                ((cities[path[len(path)-1]].y-cities[path[0]].y)**2))

    return 1/fitness #Get the inverse since GA is a maximization alg.

def fitnessVector(pop, cities):
    #Gets vector of all fitnesses in the population
    fitnesses = np.zeros(len(pop))
    for i in range(len(pop)):
        fitnesses[i] = fitness(pop[i], cities)
    
    return fitnesses
        

def selection(fitnesses, numSelected, maxits=100, asc=False):
    #Selection method for genetic algorithms
    # Creates cumulative distribution of all fitnesses and selects
    # random genes out of this distribution.
    # Prevents getting stuck in a loop using default value maxits=100. If loop
    # exceeds this, it will take the sorted values up to when the loop broke.
    fitnessCopy = deepcopy(fitnesses)
    sortedFitness = (np.sort(fitnesses)[::-1])

    #get cumulative distribution of fitnesses
    fitnesscdf = np.cumsum(sortedFitness / np.sum(sortedFitness))

    picked = []
    it = 0 #Keep track of number of iterations
    while(len(picked) < numSelected and it < maxits):
        #Pick population to keep based on fitness
        # Higher fitness = more likely to get picked
        it += 1
        val = ran()
        j = 0
        while val > fitnesscdf[j] and j != len(fitnesscdf)-1:
            j += 1
        if j not in picked:
            picked.append(j)
    
    if(asc):
        #If chosen in ascending order, reverse the picked order
        #Used in picking of which children are to die off.
        for i in range(len(picked)):
            picked[i] = len(fitnesscdf) - 1 - picked[i]

    for i in range(len(picked)):
        #Change index of each picked value to represent the unorderd population
        for count, k in enumerate(fitnessCopy):
            if k == sortedFitness[picked[i]]:
                picked[i] = count
                break
    
    return picked


def crossover(gene1, gene2):
    #Performs gene crossover using Order 1 permutation crossover
    #
    # Picks indices low, high which are somewhere between 0 and len(gene1)
    # Values in between low, high are copied over to the child from gene1
    # Left over values are copied in order to the child from gene 2

    #If the values are identical then it will clone the gene into a child
    clone=True
    for i in range(len(gene1)):
        if gene1[i] != gene2[i]:
            clone = False
            break
    if clone:
        return gene1
        

    child = np.zeros(len(gene1))

    #Choose start and end points
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
    while i1 < len(child) and not uni(child):
        #Append values of gene 2 not found in child in order in which they
        # appear in gene 2 AFTER the endpoint
        for j in gene2:
            if j not in child:
                child[i1] = j
                i1 += 1
                if i1 >= len(child)-1:
                    break
    return child

def round2(x):
    #Rounds input value to nearest even number
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
    #Selects two random values from given list at a time. Asserts list is at least
    # 2 elements long.
    assert len(from_here) >=2
    aIndex = np.random.randint(0, len(from_here))
    bIndex = aIndex

    while bIndex == aIndex:
        #Ensure that the two swap points are not equal
        bIndex = np.random.randint(0, len(from_here))

    return aIndex, bIndex

def randomeval(citySpace, maxits=100):
    #Evaluates the Traveling Salesman Problem by continuously testing the path
    # length of random paths, while keeping track of the best path so far.
    #Default number of iterations is 100, but can be changed on call.
    path = np.arange(len(citySpace))
    np.random.shuffle(path) #Start with random path
    bestFitness = 0
    bestPath = []
    it = 0
    while it < maxits:
        pathFitness = fitness(path, citySpace)
        if pathFitness > bestFitness:
            #Update if fitness is best so far
            bestPath = np.copy(path)
            bestFitness = pathFitness
        np.random.shuffle(path) #Randomize after every iteration
        it += 1 #Keep track of loop
    return bestFitness, bestPath



def main(maxits = 200):
    #Main method
    # Everything in user-defined variables is modifiable.
    # Use the user defined variables to alter how the program trends towards an 
    #  optimized value

    #---------------------------------------------------------------------------
    #USER DEFINED VARIABLES
    #---------------------------------------------------------------------------
    
    #Plotting
    plotting = False

    #Population Size:
    popSize = 160

    #Number of cities in TSP
    numCities = 20

    #Optional Function Declaration here
    phi = np.linspace(0, 2*m.pi, numCities, endpoint=False)
    x = np.cos(phi)
    y = np.sin(phi)
    
    cities = TSP(numCities, popSize, True, x, y) #Uncomment to use test function
   # cities = TSP(numCities, popSize) #Uncomment to use randomization


    repr_chance = 0.5 #Chance to reproduce
    mutation_chance = .5 #Chance for a gene to mutate
    mutation_strength = 1 #How many times does something have the chance to mutate
    lottery_win_percent = 0.1 #Percentage of genes which will win the 'lottery'

    #Choose if children are immune to mutations
    children_vaccinated = True

    #---------------------------------------------------------------------------
    #                           INITIALIZATION
    #---------------------------------------------------------------------------
    fig, ((ax1, ax2)) = plt.subplots(1, 2)
    
    
    bestFitnessSoFar = 0

    #Generate a starting population - Each gene is a different path
    population = np.meshgrid(range(numCities), range(popSize))[0]
    for j in population:
        #Shuffle values so we get random paths
        np.random.shuffle(j)

    #Round lottery winners to nearest even number. Useful for determining
    # parents of children.
    numLottoWinners = round2(lottery_win_percent*len(population))

    

    xbest = []
    ybest = []

    its = 0
    while its < maxits:
        its += 1

        fitnesses = fitnessVector(population, cities.cities)

        bestPathCurrently = deepcopy(population[list(fitnesses).index(max(fitnesses))])
        bestFitnessCurrently = max(fitnesses)
        
        xplot = []
        yplot = []

        for i in bestPathCurrently:
            xplot.append(cities.cities[i].x)
            yplot.append(cities.cities[i].y)
            
        xplot.append(xplot[0])
        yplot.append(yplot[0])

        if(bestFitnessCurrently> bestFitnessSoFar):
            bestFitnessSoFar = max(fitnesses)
            bestPathSoFar = deepcopy(population[list(fitnesses).index(max(fitnesses))])
            xbest = []
            ybest = []
            for i in bestPathSoFar:
                xbest.append(cities.cities[i].x)
                ybest.append(cities.cities[i].y)
            xbest.append(xbest[0])
            ybest.append(ybest[0]) 

        if(plotting):
            ax1.cla()
            ax2.cla()
            ax1.plot(xplot, yplot)
            ax2.plot(xbest, ybest)
            ax1.set_title("Best at current iteration")
            ax2.set_title("Best overall iteration")
            fig.suptitle("Iteration Number " + str(its))
            plt.draw()
            plt.pause(0.001)


        #-----------------------------------------------------------------------
        #                                SELECTION
        #-----------------------------------------------------------------------
        winners = selection(fitnesses, numLottoWinners)



        

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
            else:
                pass
        

        if len(children) > 0:
            weakest = selection(fitnesses, len(children), asc=True)

            for i, item in enumerate(weakest):
                #Replace weakest in the population with children
                population[item] = children[i]
        #-----------------------------------------------------------------------
        #                                MUTATION
        #-----------------------------------------------------------------------
        for j in range(mutation_strength):
            for i in range(len(population)):
                diceroll = ran()
                if diceroll < mutation_chance:
                    if len(children) > 0 and (not ((children_vaccinated ) and (any(np.equal(children, population[i]).all(1))))): 
                    #Can choose to allow for children not to mutate
                        mutation(population[i])
                    elif len(children)==0:
                        mutation(population[i])
                    

    #FINAL LOOP
    fitnesses = fitnessVector(population, cities.cities)

    if(max(fitnesses) > bestFitnessSoFar):
            bestFitnessSoFar = max(fitnesses)
            bestPathSoFar = deepcopy(population[list(fitnesses).index(max(fitnesses))])

    rFitness, rPath = randomeval(cities.cities)
    print("Final Stats:")
    print("Number of iterations:")
    print("Best fitness at current iteration:  %f" %(max(fitnesses)))
    print("Best fitness overall: \t\t    %f" %(bestFitnessSoFar))
    print("Shortest path using best fitness:  %f" %(1/bestFitnessSoFar))
    print("\nBest Random fitness: \t\t    %f" %(rFitness))
    print("Shortest Path using Random Fitness:%f" %(1/rFitness))


    if(plotting):
        plt.show()
    

    
main()
print("DONE")

