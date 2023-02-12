import numpy as np
from LinReg import LinReg
import random
import configparser
import matplotlib.pyplot as plt

dataset=np.loadtxt(open("dataset.txt", "rb"), delimiter=",")
X=dataset[:,:-1]
Y=dataset[:,-1]

lR=LinReg()

def bitstring2number(bitstring):
    l = bitstring.dot(2**np.arange(bitstring.size)[::-1])
    return l*2**(7-len(bitstring))

def unit_test_bitstring2number():
    print(bitstring2number(np.array([1,0,1])) == 5*2**(7-3))

def fitness_for_sinus(bitstring):
    x = bitstring2number(bitstring)
    return np.sin(x)

def unit_test_fitness_for_sinus():
    x = bitstring2number(np.array([1,0,1]))
    print(np.sin(x) == np.sin(5*2**(7-3)))

def fitness_for_sinus_with_penalty(bitstring,interval,penalty_rate):
    x = bitstring2number(bitstring)
    if interval[0]<=x<=interval[1]:
        return np.sin(x)
    elif x<interval[0]:
        return np.sin(x)-penalty_rate*(interval[0]-x)
    elif x>interval[1]:
        return np.sin(x)-penalty_rate*(x-interval[1])

def population_entropy(population):
    bit_proba = np.mean(population,axis=1)
    return - np.sum(bit_proba.dot(np.log(bit_proba)))

def fitness_for_data_case(bitstring):
    bitstring = np.array(bitstring)

    x_filtered=lR.get_columns(X,bitstring)
    return lR.get_fitness(x_filtered, Y)

def initiate_population(n,bitstring_lentgh):
    """Initiate the population
    
    Args:
        - n: The number of individual

    Returns:
        - A random population of n individual
    """
    p=np.random.randint(2, size=(n,bitstring_lentgh))
    if p.all()==0:
        p=np.random.randint(2, size=(n,bitstring_lentgh))
    return p

def trivial_selection(population,n,fitness_fonction):
    """Trivial selection: takes the best individual
    
    Args:
        - population: The population
        - n: The number of individual we keep

    Returns:
        - A population of the n best individual of the previous population
    """
    
    F=[]
    for i,bitstring in enumerate(population):
        F.append([i,fitness_fonction(bitstring)])
    F=np.array(F)
    F=F[F[:, 1].argsort()[::-1]]
    F=F[:n,:]
    F=F[:,0]
    F=F.astype(int)
    return population[F]

def trivial_compete(i1,i2,fitness_fonction):
    """Compete between two individual

    Args:
        - i1: An individual
        - i2: An individual

    Returns:
        - The best individual
    """
    i1 = np.array(i1)
    i2 = np.array(i2)

    f1=fitness_fonction(i1)
    f2=fitness_fonction(i2)

    if f1>f2:
        return i1
    else :
        return i2

def random_crossover(p1,p2):
    """Crossover between two parent

    Args:
        -p1: A parent
        -p2: A parent
    
    Returns:
        - Two child
    """
    c1=[]
    c2=[]
    for i in range(len(p1)):
        if random.randint(0,1)==0:
            c1.append(p1[i])
            c2.append(p2[i])
        else:
            c1.append(p2[i])
            c2.append(p1[i])
    return c1,c2

def normal_mutation(c):
    """Mutation of an individual

    Args:
        -c: An individual

    Returns:
        An individual with a probability of 1/2 to be a mutation of c else it return c
        (only one elements of c is mutated in case of mutation)
    """
    mutate_c=c

    if random.randint(0,1)==1:
        i=random.randint(0,len(c)-1)
        if mutate_c[i]==0:
            mutate_c[i]=1
        else :
            mutate_c[i]=0
    
    return mutate_c

class Ga:
    def __init__(self,crossover_type,crossover_rate,mutation_type,selection_type,fitness_fonction,crowding_option=False, 
                compete_type='none',interval='none',penalty_rate='0') -> None:
        
        self.crossover_rate = float(crossover_rate)
        self.crowding_option = crowding_option

        if interval !='none':
            I = []

            L = interval.split(' ')
            I.append(int(L[0]))
            I.append(int(L[1]))

            self.interval = I
            self.penalty_rate = float(penalty_rate)
            f_sin = lambda x:fitness_for_sinus_with_penalty(x,self.interval,self.penalty_rate)
        else :
            f_sin = fitness_for_sinus

        if crossover_type == 'random':
            self.crossover = random_crossover
        
        if mutation_type == 'normal':
            self.mutation = normal_mutation

        if selection_type == 'trivial_selection':
            self.selection = trivial_selection
        
        if fitness_fonction == 'fitness_for_sinus':
            self.fitness_function = f_sin

        if fitness_fonction == 'fitness_for_data_case':
            self.fitness_function = fitness_for_data_case

        if compete_type == 'trivial_compete':
            self.compete = trivial_compete

    def one_cycle_function(self,p1,p2):
        """Crossover of parent, mutation of their child and compete between parents and childs

        Args:
            -p1: A parent
            -p2: A parent
        
        Returns:
            Individuals that result after crossover, mutation and compete
        """
        c1,c2=self.crossover(p1,p2)

        c1 = self.mutation(c1)
        c2 = self.mutation(c2)

        if self.crowding_option == True:

            d_p1_c1 = np.abs(np.sum(p1-c1))
            d_p2_c2 = np.abs(np.sum(p2-c2))
            d_p1_c2 = np.abs(np.sum(p1-c2))
            d_p2_c1 = np.abs(np.sum(p2-c1))

            if d_p1_c1+d_p2_c2<d_p1_c2+d_p2_c1:
                i1=self.compete(p1,c1,self.fitness_function)
                i2=self.compete(p2,c2,self.fitness_function)
            else :
                i1=self.compete(p1,c2,self.fitness_function)
                i2=self.compete(p2,c1,self.fitness_function)

            return i1,i2

        else :
            return c1,c2

    def one_cycle(self, population, number_of_individuals, number_of_parents):
        childs=[]
        parents=self.selection(population,number_of_parents,self.fitness_function)
        for k in range(0,number_of_parents,2):
            c1,c2=self.one_cycle_function(parents[k],parents[k+1])
            childs.append(c1)
            childs.append(c2)
        population= np.append(population,np.array(childs),axis=0)
        population=self.selection(population,number_of_individuals,self.fitness_function)

        return population

    def run(self, number_of_individuals, number_of_cycle,bitstring_length):
        number_of_parents = int(number_of_individuals*self.crossover_rate)
        entropy = []
        population=initiate_population(number_of_individuals,bitstring_length)
        entropy.append(population_entropy(population))

        for _ in range(number_of_cycle):
            population = self.one_cycle(population, number_of_individuals,number_of_parents)
            entropy.append(population_entropy(population))
        
        return population,entropy

