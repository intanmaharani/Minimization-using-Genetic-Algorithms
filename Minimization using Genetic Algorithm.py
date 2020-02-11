# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 19:13:20 2019

@author: OWNER
"""

import numpy as np
import matplotlib.pyplot as plt

RMIN = [-3, -2]
RMAX = [3, 2]
P = 12

def function(x):
    return 4*x[0]**2 - 2.1*x[0]**4 + ((x[0]**6)/3) + x[0]*x[1] - 4*x[1]**2 + 4*x[1]**4

def encodingkromosom(kromosom):
    global RMIN
    global RMAX
    x1 = RMIN[0] + (RMAX[0]-RMIN[0]/(9*(10**-1+ 10**-2 + 10**-3 + 10**-4 + 10**-5 + 10**-6)))*(kromosom[0]*10**-1 + kromosom[1]*10**-2 + kromosom[2]*10**-3 + kromosom[3]*10**-4 + kromosom[4]*10**-5 + kromosom[5]*10**-6)
    x2 = RMIN[1] + (RMAX[1]-RMIN[1]/(9*(10**-1+ 10**-2 + 10**-3 + 10**-4 + 10**-5 + 10**-6)))*(kromosom[6]*10**-1 + kromosom[7]*10**-2 + kromosom[8]*10**-3 + kromosom[9]*10**-4 + kromosom[10]*10**-5 + kromosom[11]*10**-6)
    return x1, x2

def generatekromosom():
    kromosom = np.random.randint(0, 9, 12)
    return list(kromosom)

def generatepopulation(p):
    populasi = []
    for i in range(p):
        individu = generatekromosom()
        populasi.append(individu)
    return populasi


def fungsi_fitness(hasilfungsi):
    return 1/(hasilfungsi+144) #144 adalah worst case nya 12*12
#print (fitness(function(x)))
    
def hitung_fitness(populasi):
    global RMIN
    global RMAX
    nilai_fitness = []
    for individu in populasi:
        x = encodingkromosom(individu)
        fitness = fungsi_fitness(function(x))
        nilai_fitness.append(fitness)
    return nilai_fitness

def roulettewheelselection(populasi, fitness):
    total = np.sum(fitness)
    r = np.random.uniform()
    individu = -1
    while (r>0):
        r -= fitness[individu]/total
        individu += 1
    return populasi[individu]

def crossover(populasi, fitness):
    mating = []
    for i in range(len(fitness)//2):
        
        parent1 = roulettewheelselection(populasi, fitness)
        parent2 = roulettewheelselection(populasi, fitness)
        parent1[3:5+1], parent2[3:5+1] = parent2[3:5+1], parent1[3:5+1] 
        mating.append(parent1)
        mating.append(parent2)
    return mating

def mutation(populasi):
    for i in range(len(populasi)):
        pilihmutasi = np.random.rand()
        if pilihmutasi < 0.01:
            kromosom = populasi[i]
            gen = np.random.randint(len(kromosom))
            if kromosom[gen] == 1: kromosom[gen] = 9
            elif kromosom[gen] == 2: kromosom[gen] = 8
            elif kromosom[gen] == 3: kromosom[gen] = 7
            elif kromosom[gen] == 4: kromosom[gen] = 6
            elif kromosom[gen] == 5: kromosom[gen] = 5
            elif kromosom[gen] == 6: kromosom[gen] = 4
            elif kromosom[gen] == 7: kromosom[gen] = 3
            elif kromosom[gen] == 8: kromosom[gen] = 2
            elif kromosom[gen] == 9: kromosom[gen] = 1
            populasi[i] = kromosom
    return populasi

def generational_replacement(populasi):
    nilai_fitness = hitung_fitness(populasi)
    idx = np.argmax(nilai_fitness)
    populasibaru = [populasi[idx]]
    
    while len(populasibaru) < P:
        offspring = crossover(populasi, nilai_fitness)
        populasibaru.extend(offspring)
    
    populasibaru = mutation(populasibaru)
    return populasibaru
            

def geneticalgorithm():
    list_best = []
    generasi = 0
    populasi = generatepopulation(P)
    while generasi<100:
        generasi += 1
        populasi = generational_replacement(populasi)
        nilai_fitness = hitung_fitness(populasi)
    
        idx = np.argmin(nilai_fitness)
        best_kromosom = populasi[idx]
        x = encodingkromosom(best_kromosom)
        list_best.append(function(x))
    
    print("Total generasi:", generasi)
    x = encodingkromosom(best_kromosom)
    print("x", x)
    print("f(x) =", function(x))
    
    return generasi, best_kromosom, list_best


for i in range(10):
    gen, best, list_best = geneticalgorithm()
    
    # Plot progress
    plt.plot(list_best)
    plt.xlabel('Generation')
    plt.ylabel('Best score (% target)')
    plt.show()
    

# populasi = generatepopulation(P)
# print("Populasi:\n", populasi[:2])

# parent = roulettewheelselection(populasi,nilai_fitness)
# nilai_fitness = hitung_fitness(populasi)
# print("Fitness :\n", nilai_fitness[:2])
#print("Parent :\n", parent)

# populasi = crossover(populasi, nilai_fitness)
# print("Crossover :\n",populasi[:2])

# populasi = mutation(populasi)
# print("Mutation :\n",populasi[:2])



#parent = roulettewheelselection(nilai_fitness)
#print(parent)

#kromosom = generatekromosom()
#x = encodingkromosom(RMIN, RMAX, kromosom)
#print(x)


#fn = [1, 3,5,3,5,7,5,3,2]
#print(np.sum(fn))
#total = 0
#for i in fn: total += i
#print(total)