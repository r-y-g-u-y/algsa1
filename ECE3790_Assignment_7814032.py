'''
ECE 3790    -   ASSIGNMENT 1

Completed by Ryan Bate 7814032
'''

#if it ain't broke, don't fix it

from numpy.random import random_sample as ran
import numpy as np
import math as m
from copy import deepcopy



class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class TravelingSalesman:
    def __init__(self, n, predefinedMap=None):
        #Overall Problem constructor.
        # If predefinedMap is 

        if (n <= 1) and (predefinedMap is None):
            print("ERROR: Invalid parameters!")

        elif(predefinedMap is not None):
            self.cities = deepcopy(predefinedMap)
            self.cSize = len(cities)

        else:
            self.cSize = n
            self.cities = []
            for i in range(n):
                cities.append(City(ran(), ran())) #Append a new city with a random location

def fitness(path, cities):
    fitness = 0.0
    for i in range(1, len(path)):
        fitness += m.sqrt(((cities[i].x-cities[i-1].x)**2) + ((cities[i].y-cities[i-1].y)**2))

    #Add the last section of the path to complete the loop
    fitness +=  m.sqrt(((cities[len(path)].x-cities[0].x)**2) + ((cities[len(path)].y-cities[0].y)**2))

    return 1/fitness #Get the inverse since GA is a maximization alg.

        
        

def main():
    n = 10
    cityMap = []
    for i in range(n):
        cityMap = City()