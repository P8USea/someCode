import copy
import math
import time
import numpy as np

from matplotlib.animation import FuncAnimation


def test_function(x):
    return x**4 - 33*x**2 + 7*x + 999
def fitness_function(position):
    x = position[0]
    return test_function(x)

#Hàm có nhiều local minimum kiểm tra thuật toán có dễ bị kẹt tại minima đó không
def griewank_function(x):
   
    sum_term = sum(xi ** 2 / 4000 for xi in x)
    product_term = math.prod(math.cos(xi / math.sqrt(i + 1)) for i, xi in enumerate(x))
    result = 1 + sum_term - product_term
    return result
#Hàm nhiều local minima và có global minimum tại (0, 0, ... 0)
def ackley_function(x):
    
    n = len(x)
    sum1 = sum(xi ** 2 for xi in x)
    sum2 = sum(math.cos(2 * math.pi * xi) for xi in x)
    term1 = -20 * math.exp(-0.2 * math.sqrt(sum1 / n))
    term2 = -math.exp(sum2 / n)
    result = term1 + term2 + 20 + math.exp(1)
    return result


def fitness_rastrigin(position):
  fitnessVal = 0.0
  for i in range(len(position)):
    xi = position[i]
    fitnessVal += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
  return fitnessVal



class Particle:
    def __init__(self, fitness, dim, minx, maxx,seed ):

        self.position = [np.random.uniform(minx, maxx) for i in range(dim)]
        self.velocity = [(-0.3*self.position[i]) for i in range(dim)]
        #Vận tốc ban đầu ở đây có thể tuỳ chọn, không gây ảnh hưởng quá nhiều
        self.fitness = fitness(self.position) 
        self.best_part_pos = copy.copy(self.position)
        self.best_part_fitnessVal = self.fitness
def pso(fitness, max_iter, dim, n, minx, maxx):
    w = 0.7
    c1 = c2 = 2.0
    rnd = np.random
    #Giá trị w, c1, c2 tốt nhất để tối ưu thuật toán
    np.random.seed(10)
    swarm = [Particle(fitness, dim, minx, maxx, i) for i in range(n)]
    
    best_swarm_pos = [0.0 for i in range(dim)]
    best_swarm_fitnessVal = float('inf')
    
    for i in range(n):
        if swarm[i].fitness < best_swarm_fitnessVal:
            best_swarm_fitnessVal = swarm[i].fitness
            best_swarm_pos = swarm[i].best_part_pos
    Iter = 0
    
    while Iter <= max_iter:
        with open(file_name, "a") as file:
          file.write(str(best_swarm_fitnessVal) + '\n')
        for i in range(n):
            r1 = rnd.random()
            r2 = rnd.random()
            for k in range(dim):
                
                
                swarm[i].velocity[k] = (
                (w * swarm[i].velocity[k])
                + r1 * c1 * (swarm[i].best_part_pos[k] - swarm[i].position[k])
                + r2 * c2 * (best_swarm_pos[k] - swarm[i].position[k])
                                        )
                if swarm[i].velocity[k] < minx:
                    swarm[i].velocity[k] = minx
                if swarm[i].velocity[k] > maxx:
                    swarm[i].velocity[k] = maxx 
            for k in range(dim):
                swarm[i].position[k] += swarm[i].velocity[k]
                swarm[i].fitness = fitness(swarm[i].position)
            
            if swarm[i].fitness < swarm[i].best_part_fitnessVal:
                swarm[i].best_part_fitnessVal = swarm[i].fitness
                swarm[i].best_part_pos = copy.copy(swarm[i].position)
                
            
            if swarm[i].fitness < best_swarm_fitnessVal:
                best_swarm_fitnessVal = swarm[i].best_part_fitnessVal
                best_swarm_pos = copy.copy(swarm[i].position)
            #sử dụng copy.copy() để đảm bảo giá trị g_best và p_best
            #là nguyên vẹn khi position cập nhật (passed by value)
  
        Iter += 1






file_name = "PSO.txt"



fitness = griewank_function
#fitness = ackley_function
#fitness = fitness_rastrigin
dim = 50

max_iter = 100
n = 300
minx, maxx = -10.0, 10.0


with open(file_name, "w") as file:
    file.write("")

print("PSO Setup Succeeded!")
start_time = time.time()
pso(fitness, max_iter, dim, n, minx, maxx)



end_time = time.time()
elapsed_time = end_time - start_time
print("Time: " + str(elapsed_time))




              
                    
                
                
        
        