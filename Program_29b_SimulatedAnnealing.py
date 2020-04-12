import numpy as np
import sys
import math
import copy
from matplotlib import pyplot as plt
from timeit import default_timer as timer
from datetime import timedelta

fig, ((ax1, ax2)) = plt.subplots(1, 2) #make global for convenience - not recommended in general

def energyfunction(state, x, y):
    n = len(state)
    energy = 0
    for i in range(0,n-1):
        
        #if we don't want to include the square root (it is just extra calculations) remove it here
        #energy += (x[state[i+1]] - x[state[i]])**2 + (y[state[i+1]] - y[state[i]])**2
        
        
        #including square root just so we can compare to circle circumference - in general not required
        energy += math.sqrt((x[state[i+1]] - x[state[i]])**2 + (y[state[i+1]] - y[state[i]])**2)
        
    #energy += (x[state[n-1]] - x[state[0]])**2 + (y[state[n-1]] - y[state[0]])**2 #add return to first state (without square root)
    energy += math.sqrt((x[state[n-1]] - x[state[0]])**2 + (y[state[n-1]] - y[state[0]])**2) #add return to first state (with square root)
    
    return energy

def neighbourfunction(state):
    n = len(state)
    indexes = np.random.randint(n, size=2)
    newstate = np.copy(state)
    
    #flip two entries
    temp = newstate[indexes[0]]
    newstate[indexes[0]] = newstate[indexes[1]]
    newstate[indexes[1]] = temp;

    return newstate

def travelingSalesmanSA(s0, x, y, E, S, maxits, innermaxits):

    c = 100
    iteration = 0
    state = np.copy(s0)
    stateE = E(s0, x, y)
    beststate = state
    beststateE = stateE

    best_E_history = np.zeros(maxits)
    E_history = np.zeros(maxits)
    
    while (iteration < maxits):
        
        iteration += 1

        #inner iterations keep c constant
        for inner_iteration in range(0,innermaxits):

            neighbour = S(state)
            neighbourE = E(neighbour, x, y)

            if (neighbourE < stateE):
                state = np.copy(neighbour)
                stateE = neighbourE #accept the solution
            else:
            
                deltaE = neighbourE - stateE
                Paccept = np.exp(-deltaE/c)
                chance = np.random.rand(1)[0]
                if chance < Paccept:
                    state = np.copy(neighbour)
                    stateE = neighbourE #accept the solution

                #temperature = float(max_its)/float(iteration + 1)
                
            #can't hurt to keep the best solution, especially for plotting purposes
            if stateE < beststateE:
                beststateE = stateE
                beststate = np.copy(state)
        
        #update history - outer iterations only
        best_E_history[iteration-1] = beststateE
        E_history[iteration-1] = stateE
    
        c = 0.9*c #simple cooling
        
        #update the user
        print("Outer iteration ", iteration, " Energy = ", stateE)
        
        #add loop back to first state for plotting
        xplot = np.append(x[state],x[state[0]])
        yplot = np.append(y[state],y[state[0]])
        xbestplot = np.append(x[beststate],x[beststate[0]])
        ybestplot = np.append(y[beststate],y[beststate[0]])
        
        #plot the current and best solutions
        ax1.cla()
        ax1.plot(xplot,yplot)
        ax2.cla()
        ax2.plot(xbestplot,ybestplot)
        ax1.set_title('Best at current iteration',fontsize=11)
        ax2.set_title('Best overall',fontsize=11)
        fig.suptitle(['SA Traveling Salesman Iteration # ',iteration])
        plt.pause(0.05)
        
    
    plt.show()

    return beststate, E_history, best_E_history
    
#main

n = 20 #number of cities

#random cities example
"""
x = np.random.rand(n)
y = np.random.rand(n)
s0 = range(0,n) #intial state, start by visiting them in order
"""

#circle example (optimal solution is just to traverse the circle)
phi = np.linspace(0, 2*math.pi, n, endpoint=False)
x = np.cos(phi)
y = np.sin(phi)
s0 = np.random.permutation(n)

maxits = 100
innermaxits = 500

beststate, E_history, best_E_history = travelingSalesmanSA(s0,x, y, energyfunction, neighbourfunction,maxits,innermaxits)
#could plot E_history or best_E_history here




