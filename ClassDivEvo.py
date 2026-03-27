import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
class DivEvo:
    def __init__(self, f, cr, g, fofma, start_population, mutation='rand_to_best', lowlimit=-1, uplimit=1):
        self.F = f
        self.Cr = cr
        self.Sigma = 0.000001           #  точность
        self.fofma = fofma              #  целевая функция
        self.lowlimit = lowlimit
        self.uplimit = uplimit
        self.result = {"Point": None,
                       "Minimum": None,
                       "Time_to_minimum": None,
                       "Minimum_of_Demand": None,
                       "Time_of_Demand": None,
                       "Point_of_Demand": None
                       }
        mutations = { 'rand' : self.mutation_rand,
                      'rand_to_best' : self.mutation_rand_to_best,
                      'current_to_rand' : self.mutation_current_to_rand
                      }
        self.mutation = mutations[mutation]
        self.progressOfFitness = list() # построение графика
        self.CountOfGeneration = g      # количество вычислений целевой функции
        self.popSize = len(start_population)
        self.population = start_population
        self.fitness = np.zeros(self.popSize)


    def clear(self):
        self.result = {"Point": None,
                       "Minimum": None,
                       "Time_to_minimum": None,
                       "Minimum_of_Demand": None,
                       "Time_of_Demand": None,
                       "Point_of_Demand": None
                       }
        self.progressOfFitness = list()
    
    def mutation_rand(self, k):
            # C - i-ый представитель популяции; F - коэффициент силы мутации; A - представитель популяции отличный от первого; B - ещё один другой представитель популяции
            # V - вектор-донор (второй родиель)
            Threeman = list(range(0, self.popSize))
            Threeman.pop(k)
            Threeman = np.random.choice(Threeman, size=3, replace=False)
            V = self.population[Threeman[0]] - self.F * ( self.population[Threeman[1]] - self.population[Threeman[2]] )
            return V
    
    def mutation_rand_to_best(self, k):
            Twoman = list(range(0, self.popSize))
            Twoman.pop(k)
            Twoman = np.random.choice(Twoman, size=2, replace=False)
            V = self.population[k] + self.F*( self.population[self.fitness.argmin()] - self.population[k] ) + self.F*( self.population[Twoman[0]] - self.population[Twoman[1]])
            return V
    
    def mutation_current_to_rand(self, k):
            Threeman = list(range(0, self.popSize))
            Threeman.pop(k)
            Threeman = np.random.choice(Threeman, size=3, replace=False)
            V = self.population[k] + self.F*( self.population[Threeman[0]] - self.population[k] ) + self.F*( self.population[Threeman[1]] - self.population[Threeman[2]])
            return V

    def findOptim(self):


        def shade(lowlimit, uplimit, V):
            while (True):
                Downmask = V <= lowlimit
                Hightmask = V >= uplimit
                if ((Downmask.any() == False) and (Hightmask.any() == False)): break
                V = np.where(Downmask, (lowlimit + V) / 2.2, V)
                V = np.where(Hightmask, (uplimit + V) / 2.2, V)
            return V

        def crossover(V, C, Cr):
            #  V - вектор-донор (второй родитель); C - случайный вектор(родитель); Cr - вероятность передачи мутированных значений; U - потомок
            U = C.copy()
            mask = np.random.rand(len(V)) <= Cr
            mask[np.random.randint(0, len(V))] = True
            U = np.where(mask, V, C)
            return U

        # popSize = len(self.population)
        tak = time.perf_counter()

        population = self.population  # переменная для хранения популяции
        fitness = self.fitness
        for i in range(self.popSize):
            fitness[i] = self.fofma(population[i])
        G = self.popSize
        self.progressOfFitness = [fitness.min()]

        demandmin = True
    
        while (G < self.CountOfGeneration):
            for k in range(self.popSize):
                V = self.mutation(k)     # Создание мутанта
                #V = shade(self.lowlimit, self.uplimit, V)           # Проверка мутанта на выход за границы
                U = crossover(V, population[k], self.Cr)            # Скрещивание
                rezU = self.fofma(U)
                G += 1
                # print(f"Значение старого вектора: {fitness[k]} \n Значение нового вектора: {rezU} ")
                if (fitness[k] >= rezU):
                    population[k] = U
                    fitness[k] = rezU
            self.progressOfFitness.append(fitness.min())
            distance = np.sum(np.sqrt(np.power(population[fitness.argmax()] - population[fitness.argmin()], 2)))
            if demandmin:
                if (distance < self.Sigma):
                    self.result["Minimum_of_Demand"] = fitness.min()
                    self.result["Time_of_Demand"] = time.perf_counter() - tak
                    self.result["Point_of_Demand"] = population[fitness.argmin()]
                    demandmin = False
        self.result["Time_to_minimum"] = time.perf_counter() - tak
        self.result["Point"] = population[fitness.argmin()]
        self.result["Minimum"] = fitness.min()
        #print(f"IS OVER")
